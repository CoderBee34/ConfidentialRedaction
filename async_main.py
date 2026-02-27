import os
import io
import re
import base64
import uuid
import asyncio
import logging
import httpx
import fitz 
from typing import List
from dotenv import load_dotenv
from rapidfuzz.distance import Levenshtein
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest


# ── Request Model ─────────────────────────────────────────────────────────────

class RedactRequest(BaseModel):
    apr_name: str = Field(..., description="Applicant name (search value)")
    bank_cust_no: str = Field(..., description="Bank customer number (search value)")
    doc_id: str = Field(..., description="Unique document identifier")
    doc_content: str = Field(..., description="PDF file content encoded as base64")


# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
CALLBACK_URL = os.getenv("CALLBACK_URL")
_INCH_TO_PT = 72.0
MAX_FILE_SIZE = 20 * 1024 * 1024
SCORE_TOLERANCE = 0.05
PADDING_PT = 0

# Testing: save redacted PDFs to this directory (set to empty string to disable)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

if not ENDPOINT or not KEY:
    raise RuntimeError("Missing Azure credentials in environment variables.")

# Initialize client once to reuse connection pools
azure_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global azure_client
    azure_client = DocumentIntelligenceClient(
        endpoint=ENDPOINT, credential=AzureKeyCredential(KEY)
    )
    yield
    try:
        await azure_client.close()
    except Exception:
        pass

app = FastAPI(title="PDF Redactor API - Async Webhook", lifespan=lifespan)

# ── Core Logic (Refactored for Memory Streams) ────────────────────────────────

def normalize(text):
    """Normalize text for comparison."""
    t = text.strip().lower()
    t = re.sub(r"^[^\w]+", "", t)
    t = re.sub(r"[^\w]+$", "", t)
    return t

def _fuzzy_word_match(search_word, target_word, threshold=None):
    """
    Return True if *search_word* fuzzy-matches *target_word*.

    Allowed edit distance scales with word length:
      len <= 3 -> 0,  4-6 -> 1,  7-10 -> 2,  >10 -> 3
    """
    if search_word == target_word:
        return True
    if search_word in target_word or target_word in search_word:
        return True

    if threshold is None:
        max_len = max(len(search_word), len(target_word))
        if max_len <= 3:
            threshold = 0
        elif max_len <= 6:
            threshold = 1
        elif max_len <= 10:
            threshold = 2
        else:
            threshold = 3

    dist = Levenshtein.distance(search_word, target_word, score_cutoff=threshold)
    return dist <= threshold


def _word_match_score(search_word, target_word):
    """Return a match score 0.0-1.0 for a single word pair."""
    if search_word == target_word:
        return 1.0
    if search_word in target_word or target_word in search_word:
        shorter = min(len(search_word), len(target_word))
        longer = max(len(search_word), len(target_word))
        return 0.85 * (shorter / longer)

    max_len = max(len(search_word), len(target_word))
    if max_len <= 3:
        threshold = 0
    elif max_len <= 6:
        threshold = 1
    elif max_len <= 10:
        threshold = 2
    else:
        threshold = 3

    dist = Levenshtein.distance(search_word, target_word, score_cutoff=threshold)
    if dist > threshold:
        return 0.0
    return max(0.0, 1.0 - dist / max(max_len, 1))

def _phrase_fuzzy_match(search_phrase, grouped_words):
    """
    Check if a search phrase (tuple of normalised words) fuzzy-matches
    within a group of words.  Handles merged / split token artefacts.
    """
    n_search = len(search_phrase)
    n_group = len(grouped_words)

    if n_search == 0:
        return True
    if n_search == 1:
        return any(_fuzzy_word_match(search_phrase[0], gw) for gw in grouped_words)

    for start in range(n_group):
        si, gi = 0, start
        while si < n_search and gi < n_group:
            sw, gw = search_phrase[si], grouped_words[gi]
            if _fuzzy_word_match(sw, gw):
                si += 1
                gi += 1
            elif gi + 1 < n_group:
                merged = grouped_words[gi] + grouped_words[gi + 1]
                if _fuzzy_word_match(sw, merged):
                    si += 1
                    gi += 2
                    continue
                if si + 1 < n_search:
                    merged_search = search_phrase[si] + search_phrase[si + 1]
                    if _fuzzy_word_match(merged_search, gw):
                        si += 2
                        gi += 1
                        continue
                break
            else:
                break
        if si >= n_search:
            return True
    return False

def build_search_phrases(search_texts: List[str]):
    """
    Build normalised search phrases.  Multi-word inputs also produce
    adjacent-pair concatenations (handles merged tokens).
    """
    phrases = []
    for text in search_texts:
        words = [normalize(w) for w in text.split() if normalize(w)]
        if words:
            phrases.append(tuple(words))
            if len(words) >= 2:
                for i in range(len(words) - 1):
                    phrases.append((words[i] + words[i + 1],))
    return phrases

def polygon_to_bbox(polygon):
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    return min(xs), min(ys), max(xs), max(ys)

def _row_match_score(row, search_phrases):
    """
    Aggregate match score for a row against all search phrases.

    Each cell's content acts as a phrase group; single-word phrases
    are matched against every word in the row.
    """
    if not search_phrases:
        return 0.0

    # Each cell's normalised words form a phrase group
    row_grouped = []
    for cell_text in row["cell_texts"]:
        norm_words = tuple(normalize(w) for w in cell_text.split() if normalize(w))
        if norm_words:
            row_grouped.append(norm_words)

    all_row_words = [w for group in row_grouped for w in group]

    total = 0.0
    matched_any = False

    for search_phrase in search_phrases:
        best_phrase_score = 0.0

        if len(search_phrase) == 1:
            sw = search_phrase[0]
            for rw in all_row_words:
                s = _word_match_score(sw, rw)
                best_phrase_score = max(best_phrase_score, s)
        else:
            for grouped in row_grouped:
                if not _phrase_fuzzy_match(search_phrase, grouped):
                    continue
                word_scores = []
                for sw in search_phrase:
                    best_w = max(
                        (_word_match_score(sw, gw) for gw in grouped),
                        default=0.0,
                    )
                    word_scores.append(best_w)
                avg = sum(word_scores) / len(word_scores) if word_scores else 0.0
                best_phrase_score = max(best_phrase_score, avg)

        if best_phrase_score > 0.0:
            matched_any = True
        total += best_phrase_score

    return total if matched_any else 0.0

def _word_in_span(word, spans):
    """Check if a word's span falls within any of the given spans."""
    w_off = word.span.offset
    w_end = w_off + word.span.length
    for span in spans:
        if w_off >= span.offset and w_end <= (span.offset + span.length):
            return True
    return False


def extract_table_rows(result, page_number):
    """
    Extract table rows for a given page from the Azure DI result.

    Groups table cells by ``row_index``.  For each row, collects cell
    content and word-level polygons (for precise redaction).

    Returns
    -------
    (rows, page_width, page_height)
        *rows* is a list of dicts sorted by row_index.
    """
    page = result.pages[page_number - 1]
    page_words = page.words or []

    # Find tables that touch this page
    page_tables = []
    if result.tables:
        for table in result.tables:
            if table.bounding_regions:
                for region in table.bounding_regions:
                    if region.page_number == page_number:
                        page_tables.append(table)
                        break

    if not page_tables:
        return [], page.width, page.height

    # Use the largest table on the page
    table = max(page_tables, key=lambda t: len(t.cells))

    # Group cells by row_index
    rows_dict = {}
    for cell in table.cells:
        ri = cell.row_index
        rows_dict.setdefault(ri, []).append(cell)

    rows = []
    for row_idx in sorted(rows_dict.keys()):
        cells = sorted(rows_dict[row_idx], key=lambda c: c.column_index)

        cell_texts = [c.content or "" for c in cells]
        row_text = " ".join(cell_texts)

        # A row is a header if any of its cells has kind == "columnHeader"
        is_header = any(
            getattr(c, "kind", None) == "columnHeader" for c in cells
        )

        # Word-level polygons with column info
        word_polygons = []
        word_infos = []
        for cell in cells:
            col_idx = cell.column_index
            if not cell.spans:
                continue
            for word in page_words:
                if _word_in_span(word, cell.spans):
                    if word.polygon:
                        word_polygons.append(word.polygon)
                        x_min, y_min, x_max, y_max = polygon_to_bbox(word.polygon)
                        bbox_w = x_max - x_min
                        bbox_h = y_max - y_min
                        font_size = min(bbox_w, bbox_h)
                        word_infos.append({
                            "polygon": word.polygon,
                            "column_index": col_idx,
                            "font_size": font_size,
                        })

        # Cell-level polygons (fallback)
        cell_polygons = []
        for cell in cells:
            if cell.bounding_regions:
                for region in cell.bounding_regions:
                    if region.polygon:
                        cell_polygons.append(region.polygon)

        rows.append({
            "row_index": row_idx,
            "text": row_text,
            "cell_texts": cell_texts,
            "is_header": is_header,
            "word_polygons": word_polygons,
            "word_infos": word_infos,
            "cell_polygons": cell_polygons,
        })

    return rows, page.width, page.height

def find_matching_rows(rows, search_phrases, score_tolerance=0.05):
    """
    Score every non-header row and return the indices to keep.

    All rows whose score is within *score_tolerance* (fraction) of
    the best score are kept.

    Returns
    -------
    (keep_indices, scored_rows)
    """
    scored_rows = []
    for i, row in enumerate(rows):
        if row["is_header"]:
            continue
        score = _row_match_score(row, search_phrases)
        scored_rows.append((i, score))

    # Sort descending by score
    scored_rows.sort(key=lambda x: x[1], reverse=True)

    if not scored_rows or scored_rows[0][1] <= 0:
        return set(), scored_rows

    best_score = scored_rows[0][1]
    threshold = best_score * (1.0 - score_tolerance)

    keep_indices = {i for i, score in scored_rows if score >= threshold and score > 0}
    return keep_indices, scored_rows


def process_redaction(pdf_bytes: bytes, search_texts: List[str], azure_result,
                      padding_pt: float, score_tolerance: float = 0.10,
                      font_size_tolerance_pt: float = 1.0):
    """
    Applies redaction to the in-memory PDF bytes.

    Rows whose match score is within *score_tolerance* of the best
    score are all kept (not redacted).
    """
    search_phrases = build_search_phrases(search_texts)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc[page_num]
        rows, _, _ = extract_table_rows(azure_result, page_num + 1)
        if not rows:
            continue

        # Keep all header rows
        keep_indices = {i for i, row in enumerate(rows) if row["is_header"]}

        # Score non-header rows; keep all within tolerance of the best
        match_indices, scored_rows = find_matching_rows(
            rows, search_phrases, score_tolerance=score_tolerance,
        )
        keep_indices |= match_indices

        # Log scoring summary
        if scored_rows:
            best_score = scored_rows[0][1]
            thresh = best_score * (1.0 - score_tolerance) if best_score > 0 else 0
            logger.info(
                f"Page {page_num + 1}: best_score={best_score:.3f}, "
                f"tolerance={score_tolerance:.0%}, keep>={thresh:.3f}"
            )

        # ── Compute average font size in POINTS from columns 0-8 ──
        target_columns = set(range(9))
        font_sizes_pt = []
        for row in rows:
            for wi in row.get("word_infos", []):
                if wi["column_index"] in target_columns:
                    font_sizes_pt.append(wi["font_size"] * _INCH_TO_PT)

        avg_fs_pt = (sum(font_sizes_pt) / len(font_sizes_pt)) if font_sizes_pt else None

        # ── Draw redactions ──
        grey = (0.5, 0.5, 0.5)
        derotation = page.derotation_matrix
        clip_rect = page.cropbox

        for i, row in enumerate(rows):
            if i in keep_indices:
                continue

            word_infos = row.get("word_infos", [])

            if word_infos:
                for wi in word_infos:
                    # Font-size filter: only apply to words outside cols 0-8
                    if avg_fs_pt is not None and wi["column_index"] not in target_columns:
                        word_fs_pt = wi["font_size"] * _INCH_TO_PT
                        if abs(word_fs_pt - avg_fs_pt) > font_size_tolerance_pt:
                            continue

                    poly = wi["polygon"]
                    x_min, y_min, x_max, y_max = polygon_to_bbox(poly)
                    visual_rect = fitz.Rect(
                        x_min * _INCH_TO_PT - padding_pt,
                        y_min * _INCH_TO_PT - padding_pt,
                        x_max * _INCH_TO_PT + padding_pt,
                        y_max * _INCH_TO_PT + padding_pt,
                    )
                    draw_rect = (visual_rect * derotation) & clip_rect
                    page.add_redact_annot(draw_rect, fill=grey)
            else:
                # Fallback: no word_infos, use raw polygons
                polygons = row.get("word_polygons") or row.get("cell_polygons", [])
                for poly in polygons:
                    x_min, y_min, x_max, y_max = polygon_to_bbox(poly)
                    visual_rect = fitz.Rect(
                        x_min * _INCH_TO_PT - padding_pt,
                        y_min * _INCH_TO_PT - padding_pt,
                        x_max * _INCH_TO_PT + padding_pt,
                        y_max * _INCH_TO_PT + padding_pt,
                    )
                    draw_rect = (visual_rect * derotation) & clip_rect
                    page.add_redact_annot(draw_rect, fill=grey)

    for page in doc:
        page.apply_redactions()

    # Save to memory buffer
    output_buffer = io.BytesIO()
    doc.save(output_buffer, garbage=4, deflate=True)
    doc.close()
    return output_buffer.getvalue()

# ── Background Task Worker ────────────────────────────────────────────────────

async def process_and_send_webhook(
    job_id: str,
    doc_id: str,
    apr_name: str,
    bank_cust_no: str,
    pdf_bytes: bytes,
    search_list: List[str],
    padding: float,
    score_tolerance: float,
    filename: str,
    callback_url: str
):
    """
    Background task that processes the PDF and POSTs the result as JSON
    (base64-encoded PDF) to the callback URL.
    """
    logger.info(f"Job {job_id} started. Processing {filename}.")
    
    try:
        # Validate PDF header
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValueError(f"Invalid PDF: missing %PDF header (starts with {pdf_bytes[:20]!r})")
        
        logger.info(f"Job {job_id}: PDF size={len(pdf_bytes)} bytes, header valid")
        
        # 1. Call Azure (Async Call) - matching row_redactor_azure.py format
        poller = await azure_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=bytes(pdf_bytes)),
            locale="tr-TR"
        )
        result = await poller.result()

        # 2. Process PDF (Offload CPU-bound task to thread)
        redacted_pdf_bytes = await asyncio.to_thread(
            process_redaction, pdf_bytes, search_list, result, padding,
            score_tolerance
        )

        # 3. Save result PDF locally for testing
        if OUTPUT_DIR:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, f"redacted_{filename}")
            with open(output_path, "wb") as f:
                f.write(redacted_pdf_bytes)
            logger.info(f"Job {job_id} saved locally to {output_path}")

        # 4. Send Result to Callback URL as JSON with base64-encoded PDF
        logger.info(f"Job {job_id} processing complete. Sending to {callback_url}.")
        payload = {
            "job_id": job_id,
            "doc_id": doc_id,
            "apr_name": apr_name,
            "bank_cust_no": bank_cust_no,
            "status": "completed",
            "doc_content": base64.b64encode(redacted_pdf_bytes).decode("utf-8"),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
            logger.info(f"Job {job_id} successfully delivered to webhook.")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        # Attempt to notify the callback URL about the failure
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    callback_url,
                    json={
                        'job_id': job_id,
                        'doc_id': doc_id,
                        'status': 'failed',
                        'error': str(e),
                    }
                )
        except Exception as webhook_err:
            logger.error(f"Job {job_id} failed to send error webhook: {str(webhook_err)}")


# ── API Endpoint ──────────────────────────────────────────────────────────────

@app.post("/redact", status_code=202)
async def redact_pdf(
    body: RedactRequest,
    background_tasks: BackgroundTasks,
):
    """
    Accept a JSON payload with a base64-encoded PDF and search values.
    Returns immediately with the doc_id. 
    The redacted PDF will be POSTed to the callback URL.
    """
    # Decode base64 PDF content
    try:
        pdf_bytes = base64.b64decode(body.doc_content)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 in doc_content")

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="doc_content is empty")

    # Validate PDF header
    if not pdf_bytes.startswith(b'%PDF'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid PDF: missing %PDF header (got: {pdf_bytes[:20]!r})"
        )

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size allowed is {MAX_FILE_SIZE / (1024 * 1024)}MB."
        )

    # Build search list from apr_name and bank_cust_no
    search_list = [v for v in [body.apr_name.strip(), body.bank_cust_no.strip()] if v]
    if not search_list:
        raise HTTPException(status_code=400, detail="No search values provided")

    # Generate a unique job ID (same doc can be processed multiple times)
    job_id = str(uuid.uuid4())

    # Build output filename from search inputs
    safe_name = re.sub(r'[^\w\s-]', '', body.apr_name.strip()).strip().replace(' ', '_')
    safe_cust = re.sub(r'[^\w-]', '', body.bank_cust_no.strip())
    filename = f"{safe_name}_{safe_cust}_{body.doc_id}.pdf"

    # Dispatch to background task
    background_tasks.add_task(
        process_and_send_webhook,
        job_id=job_id,
        doc_id=body.doc_id,
        apr_name=body.apr_name,
        bank_cust_no=body.bank_cust_no,
        pdf_bytes=pdf_bytes,
        search_list=search_list,
        padding=PADDING_PT,
        score_tolerance=SCORE_TOLERANCE,
        filename=filename,
        callback_url=CALLBACK_URL
    )

    # Return immediately (202 Accepted is the standard for async job creation)
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "doc_id": body.doc_id,
            "status": "processing",
            "message": "File accepted for processing. Result will be sent to the callback URL."
        }
    )
import os
import io
import re
import asyncio
import fitz  # PyMuPDF
from typing import List
from dotenv import load_dotenv
from rapidfuzz.distance import Levenshtein

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
_INCH_TO_PT = 72.0
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

if not ENDPOINT or not KEY:
    raise RuntimeError("Missing Azure credentials in environment variables.")

# ── Lifecycle & Dependency Injection ──────────────────────────────────────────

# Initialize client once to reuse connection pools
azure_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global azure_client
    azure_client = DocumentIntelligenceClient(
        endpoint=ENDPOINT, credential=AzureKeyCredential(KEY)
    )
    yield
    # Clean up if necessary (Azure async client doesn't strictly require close, but good practice)
    try:
        await azure_client.close()
    except Exception:
        pass

app = FastAPI(title="PDF Redactor API", lifespan=lifespan)

# ── Core Logic (Refactored for Memory Streams) ────────────────────────────────

def normalize(text):
    """Normalize text for comparison."""
    t = text.strip().lower()
    t = re.sub(r"^[^\w]+", "", t)
    t = re.sub(r"[^\w]+$", "", t)
    return t

def _word_match_score(search_word, target_word):
    """Return a match score 0.0-1.0 for a single word pair."""
    if search_word == target_word:
        return 1.0
    if search_word in target_word or target_word in search_word:
        shorter = min(len(search_word), len(target_word))
        longer = max(len(search_word), len(target_word))
        return 0.85 * (shorter / longer)

    max_len = max(len(search_word), len(target_word))
    threshold = 0 if max_len <= 3 else 1 if max_len <= 6 else 2 if max_len <= 10 else 3

    dist = Levenshtein.distance(search_word, target_word, score_cutoff=threshold)
    if dist > threshold:
        return 0.0
    return max(0.0, 1.0 - dist / max(max_len, 1))

def _phrase_fuzzy_match(search_phrase, grouped_words):
    """Check if a search phrase fuzzy-matches within a group of words."""
    n_search = len(search_phrase)
    n_group = len(grouped_words)
    
    if n_search == 0: return True
    if n_search == 1:
        return any(_word_match_score(search_phrase[0], gw) > 0.8 for gw in grouped_words)

    # Simplified phrase matching logic for speed in API context
    # (You can paste your full complex logic here if preferred)
    for i in range(n_group - n_search + 1):
        window = grouped_words[i : i + n_search]
        if all(_word_match_score(sw, gw) > 0.8 for sw, gw in zip(search_phrase, window)):
            return True
    return False

def build_search_phrases(search_texts: List[str]):
    phrases = []
    for text in search_texts:
        words = [normalize(w) for w in text.split() if normalize(w)]
        if words:
            phrases.append(tuple(words))
    return phrases

def polygon_to_bbox(polygon):
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    return min(xs), min(ys), max(xs), max(ys)

def _row_match_score(row, search_phrases):
    """Return True if row matches any search phrase (Logic Fix Applied: Boolean Return)."""
    if not search_phrases: return False
    
    # Flatten row text for faster checking
    row_text_normalized = normalize(row["text"])
    
    # Quick check
    for phrase in search_phrases:
        flat_phrase = "".join(phrase)
        if flat_phrase in row_text_normalized:
            return True

    # Deep check (fuzzy)
    row_grouped = []
    for cell_text in row["cell_texts"]:
        norm_words = tuple(normalize(w) for w in cell_text.split() if normalize(w))
        if norm_words: row_grouped.append(norm_words)

    for search_phrase in search_phrases:
        for grouped in row_grouped:
            if _phrase_fuzzy_match(search_phrase, grouped):
                return True
    return False

def extract_table_rows(result, page_number):
    """(Same logic as original, condensed for brevity)"""
    page = result.pages[page_number - 1]
    page_words = page.words or []
    
    # Find table on this page
    page_table = None
    if result.tables:
        for t in result.tables:
            for r in t.bounding_regions:
                if r.page_number == page_number:
                    page_table = t
                    break
            if page_table: break
    
    if not page_table:
        return [], page.width, page.height

    # Group cells
    rows_dict = {}
    for cell in page_table.cells:
        rows_dict.setdefault(cell.row_index, []).append(cell)

    rows = []
    for r_idx in sorted(rows_dict.keys()):
        cells = sorted(rows_dict[r_idx], key=lambda c: c.column_index)
        cell_texts = [c.content or "" for c in cells]
        is_header = any(getattr(c, "kind", None) == "columnHeader" for c in cells)
        
        word_infos = []
        for cell in cells:
            if not cell.spans: continue
            for word in page_words:
                # Simple span check
                w_off = word.span.offset
                w_end = w_off + word.span.length
                c_off = cell.spans[0].offset
                c_end = c_off + cell.spans[0].length
                if w_off >= c_off and w_end <= c_end:
                     if word.polygon:
                        x_min, y_min, x_max, y_max = polygon_to_bbox(word.polygon)
                        word_infos.append({
                            "polygon": word.polygon,
                            "column_index": cell.column_index,
                            "font_size": min(x_max-x_min, y_max-y_min)
                        })

        rows.append({
            "text": " ".join(cell_texts),
            "cell_texts": cell_texts,
            "is_header": is_header,
            "word_infos": word_infos
        })
    return rows, page.width, page.height

def process_redaction(pdf_bytes: bytes, search_texts: List[str], azure_result, padding_pt: float):
    """Applies redaction to the in-memory PDF bytes."""
    search_phrases = build_search_phrases(search_texts)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        rows, _, _ = extract_table_rows(azure_result, page_num + 1)
        if not rows: continue

        # 1. Identify which rows to keep
        # Logic Fix: Keep headers AND any matching row (not just the best one)
        keep_indices = set()
        for i, row in enumerate(rows):
            if row["is_header"]:
                keep_indices.add(i)
            elif _row_match_score(row, search_phrases):
                keep_indices.add(i)

        # 2. Dynamic Font Size Filter
        # Calculate avg font size for THIS table (not hardcoded cols 0-8)
        all_fonts = [wi["font_size"] for row in rows for wi in row["word_infos"]]
        avg_fs_pt = (sum(all_fonts) / len(all_fonts) * _INCH_TO_PT) if all_fonts else None

        # 3. Draw Redactions
        grey = (0.5, 0.5, 0.5)
        derotation = page.derotation_matrix
        clip_rect = page.cropbox

        for i, row in enumerate(rows):
            if i in keep_indices: continue

            for wi in row["word_infos"]:
                # Font size filter
                if avg_fs_pt:
                    w_fs = wi["font_size"] * _INCH_TO_PT
                    if abs(w_fs - avg_fs_pt) > 1.5: # 1.5pt tolerance
                        continue

                x_min, y_min, x_max, y_max = polygon_to_bbox(wi["polygon"])
                visual_rect = fitz.Rect(
                    x_min * _INCH_TO_PT - padding_pt,
                    y_min * _INCH_TO_PT - padding_pt,
                    x_max * _INCH_TO_PT + padding_pt,
                    y_max * _INCH_TO_PT + padding_pt,
                )
                draw_rect = (visual_rect * derotation) & clip_rect
                page.add_redact_annot(draw_rect, fill=grey)

        page.apply_redactions()

    # Save to memory buffer
    output_buffer = io.BytesIO()
    doc.save(output_buffer, garbage=4, deflate=True)
    doc.close()
    return output_buffer.getvalue()

# ── API Endpoint ──────────────────────────────────────────────────────────────

@app.post("/redact")
async def redact_pdf(
    file: UploadFile = File(...),
    search_values: str = Form(..., description="Comma-separated values or multiple form fields"),
    padding: float = Form(1.0)
):
    """
    Upload a PDF and a list of search values.
    Returns the redacted PDF where only matching rows are visible.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Validate file size before reading into memory
    file_size = getattr(file, "size", None)
    if file_size is None:
        # Fallback for systems where size isn't pre-populated
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size allowed is {MAX_FILE_SIZE / (1024 * 1024)}MB."
        )

    # Handle search values (split by comma if single string, or handle list)
    # Simple split for robustness
    search_list = [s.strip() for s in search_values.split(",") if s.strip()]
    
    if not search_list:
        raise HTTPException(status_code=400, detail="No search values provided")

    # 1. Read file to memory
    try:
        pdf_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read file upload")

    # 2. Call Azure (Async Call)
    try:
        poller = await azure_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=pdf_bytes),
            locale="tr-TR"
        )
        result = await poller.result()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Azure Analysis failed: {str(e)}")

    # 3. Process PDF (Offload CPU-bound task to a thread to avoid blocking the event loop)
    try:
        redacted_pdf_bytes = await asyncio.to_thread(process_redaction, pdf_bytes, search_list, result, padding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redaction processing failed: {str(e)}")

    # 4. Return Result
    return Response(
        content=redacted_pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="redacted_{file.filename}"'}
    )
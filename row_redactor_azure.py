"""
PDF Table Row Redaction Tool using Azure Document Intelligence.

Analyzes PDF tables with Azure DI, finds the rows matching search values,
and redacts (greys out) all other rows.  Header rows are always kept.
Rows with scores close to the best match are also kept (controlled by
``--score-tolerance``).

Examples:
    py row_redactor_azure.py input/input.pdf "2013/8024"
    py row_redactor_azure.py input/input2.pdf "şeref aydemir" "7456710" -o redacted.pdf
    py row_redactor_azure.py input/input.pdf "AHMET" --score-tolerance 0.15
"""

import sys
import os
import re
import argparse

import fitz
from dotenv import load_dotenv
from rapidfuzz.distance import Levenshtein

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

load_dotenv()

ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")

# Azure DI coordinates are in inches; PDF uses 72 points per inch.
_INCH_TO_PT = 72.0


# ── Azure Document Intelligence ──────────────────────────────────────────────

def analyze_document(input_path):
    """Analyze a PDF with Azure Document Intelligence (prebuilt-layout)."""
    client = DocumentIntelligenceClient(
        endpoint=ENDPOINT, credential=AzureKeyCredential(KEY)
    )
    with open(input_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=f.read()),
            locale="tr-TR"
        )
    return poller.result()


# ── Coordinate helpers ───────────────────────────────────────────────────────


def polygon_to_bbox(polygon):
    """Convert a flat polygon [x0,y0,x1,y1,...] to (x_min, y_min, x_max, y_max)."""
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    return min(xs), min(ys), max(xs), max(ys)


# ── Text normalisation ──────────────────────────────────────────────────────


def normalize(text):
    """Normalize text for comparison: lowercase, strip edge punctuation."""
    t = text.strip().lower()
    t = re.sub(r"^[^\w]+", "", t)
    t = re.sub(r"[^\w]+$", "", t)
    return t


# ── Fuzzy matching helpers ───────────────────────────────────────────────────

def _fuzzy_word_match(search_word, target_word, threshold=None):
    """
    Return True if *search_word* fuzzy-matches *target_word*.

    Uses rapidfuzz (C-extension) for the Levenshtein distance.

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


def build_search_phrases(search_texts):
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


# ── Row extraction from Azure DI tables ──────────────────────────────────────


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

    Parameters
    ----------
    result : AnalyzeResult
        The Azure DI analysis result.
    page_number : int
        1-based page number.

    Returns
    -------
    (rows, page_width, page_height)
        *rows* is a list of dicts sorted by row_index::

            {"row_index", "text", "cell_texts", "word_polygons", "cell_polygons"}
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

        # Word-level polygons (precise redaction) with column info
        word_polygons = []
        word_infos = []  # richer info: polygon, column_index, font_size
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
                        # Font size ≈ the shorter bbox dimension
                        # (the longer one is word length / char count).
                        # Works regardless of text rotation.
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


# ── Matching ─────────────────────────────────────────────────────────────────


def _word_match_score(search_word, target_word):
    """Return a match score 0.0-1.0 for a single word pair.

    Uses rapidfuzz Levenshtein.
    """
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


def find_matching_rows(rows, search_phrases, score_tolerance=0.05):
    """
    Score every non-header row and return the indices to keep.

    All rows whose score is within *score_tolerance* (fraction) of
    the best score are kept.  For example, with tolerance=0.10 and
    best_score=3.0, every row scoring >= 2.7 is kept.

    Parameters
    ----------
    rows : list[dict]
        Row dicts as returned by :func:`extract_table_rows`.
    search_phrases : list[tuple[str, ...]]
        Normalised search phrases from :func:`build_search_phrases`.
    score_tolerance : float
        Fraction (0.0–1.0).  0.0 keeps only the exact best; 1.0
        keeps every row that scored > 0.

    Returns
    -------
    (keep_indices, scored_rows)
        *keep_indices* is a set of row indices to keep.
        *scored_rows* is a list of ``(row_index, score)`` sorted
        descending by score (excludes header rows).
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


# ── Redaction ────────────────────────────────────────────────────────────────


def redact_rows(page, rows, keep_indices,
                padding_pt=1, font_size_tolerance_pt=1.0):
    """
    Draw grey rectangles directly on a PyMuPDF *page* over words in
    rows whose index is NOT in *keep_indices*.

    Coordinate conversion
    ---------------------
    Azure DI coordinates are in **inches**.  We convert to PDF points
    (× 72) to get visual-space coordinates, then apply the page’s
    ``derotation_matrix`` to map into the internal CropBox coordinate
    system used by PyMuPDF’s drawing operations.  This correctly
    handles pages with any ``/Rotate`` value (0, 90, 180, 270).

    Font-size filter
    ----------------
    Computes the average font size (in **points**) from words in columns
    0-8 across all rows.  Words outside columns 0-8 whose font size
    deviates from the average by more than *font_size_tolerance_pt* are
    **skipped** (not redacted).
    """
    derotation = page.derotation_matrix
    clip_rect = page.cropbox

    # ── Compute average font size in POINTS from columns 0-8 ─────────
    target_columns = set(range(9))  # columns 0-8
    font_sizes_pt = []
    for row in rows:
        for wi in row.get("word_infos", []):
            if wi["column_index"] in target_columns:
                font_sizes_pt.append(wi["font_size"] * _INCH_TO_PT)

    avg_fs_pt = (sum(font_sizes_pt) / len(font_sizes_pt)) if font_sizes_pt else None

    if avg_fs_pt is not None:
        print(f"  Avg font size (cols 0-8): {avg_fs_pt:.2f} pt  "
              f"(tolerance ±{font_size_tolerance_pt} pt)")

    kept_rows = []
    redacted_rows = []
    skipped_words = 0
    grey = (0.5, 0.5, 0.5)  # RGB 0–1

    for i, row in enumerate(rows):
        if i in keep_indices:
            kept_rows.append(i)
        else:
            redacted_rows.append(i)
            word_infos = row.get("word_infos", [])

            if word_infos:
                for wi in word_infos:
                    # Font-size filter: only apply to words outside cols 0-8
                    if avg_fs_pt is not None and wi["column_index"] not in target_columns:
                        word_fs_pt = wi["font_size"] * _INCH_TO_PT
                        if abs(word_fs_pt - avg_fs_pt) > font_size_tolerance_pt:
                            skipped_words += 1
                            continue

                    poly = wi["polygon"]
                    x_min, y_min, x_max, y_max = polygon_to_bbox(poly)
                    # Azure inches → visual points → CropBox coords
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
                polygons = row["word_polygons"] or row["cell_polygons"]
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

    if skipped_words:
        print(f"  Font-size filter: skipped {skipped_words} word(s)")

    return kept_rows, redacted_rows


# ── Processing pipeline ─────────────────────────────────────────────────────

def _process_with_result(input_path, output_path, search_texts, result,
                         padding_pt=0, score_tolerance=0.10):
    """Run the redaction pipeline with a pre-obtained Azure DI result.

    Draws grey rectangles directly on the vector PDF pages — no
    rasterisation, preserves selectable text, tiny file sizes.

    Rows whose match score is within *score_tolerance* of the best
    score are all kept (not redacted).
    """
    search_phrases = build_search_phrases(search_texts)

    doc = fitz.open(input_path)

    for page_num in range(len(doc)):
        page_number = page_num + 1
        print(f"--- Page {page_number}/{len(doc)} ---")

        page = doc[page_num]
        print(f"  Page size: {page.rect.width:.0f}×{page.rect.height:.0f} pt")

        rows, page_width, page_height = extract_table_rows(result, page_number)

        if not rows:
            print("  No table rows found on this page.")
            continue

        print(f"  Table rows detected: {len(rows)}")

        # Keep all header rows (columnHeader kind)
        keep_indices = {i for i, row in enumerate(rows) if row["is_header"]}

        # Score non-header rows; keep all within tolerance of the best
        match_indices, scored_rows = find_matching_rows(
            rows, search_phrases, score_tolerance=score_tolerance,
        )
        keep_indices |= match_indices

        # Print scoring summary
        if scored_rows:
            best_score = scored_rows[0][1]
            threshold = best_score * (1.0 - score_tolerance) if best_score > 0 else 0
            print(f"  Best score: {best_score:.3f}  "
                  f"(tolerance {score_tolerance:.0%}, "
                  f"keep threshold >= {threshold:.3f})")
            for idx, score in scored_rows:
                kept_flag = "KEEP" if idx in keep_indices else "    "
                preview = rows[idx]['text'][:90]
                print(f"    [{kept_flag}] Row {idx}: score {score:.3f}  {preview}")

        kept_rows, redacted_rows_list = redact_rows(
            page, rows, keep_indices,
            padding_pt=padding_pt,
        )

        print(f"  Kept rows ({len(kept_rows)})  |  Redacted rows: {len(redacted_rows_list)}")
    
    for page in doc:
        page.apply_redactions()
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"\nRedacted PDF saved to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Redact rows from a PDF table using Azure Document Intelligence. "
            "The header row is always kept.  Rows containing any of the "
            "given search values are kept; all other rows are greyed out."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  py row_redactor_azure.py input/input.pdf "2013/8024"\n'
            '  py row_redactor_azure.py input/input.pdf "AHMET" "ULVIHANOGLU"\n'
        ),
    )
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument(
        "search_values",
        nargs="*",
        default=[],
        help=(
            "Values to search for in table rows.  Any row containing at "
            "least one of these values will be kept visible. "
            'Use quotes for multi-word values:  "John Doe"'
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PDF path (default: <input>_redacted.pdf)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0,
        help="Extra padding in points around redaction rectangles (default: 0)",
    )
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=0.10,
        help=(
            "Fraction (0.0-1.0) of the best row score used as a proximity "
            "band.  Rows scoring within this fraction of the best score "
            "are also kept un-redacted.  0.0 = only the single best row; "
            "1.0 = every row that scored > 0.  (default: 0.10)"
        ),
    )

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input file '{args.input_pdf}' not found.")
        sys.exit(1)

    if not ENDPOINT or not KEY:
        print("Error: DOCUMENT_INTELLIGENCE_ENDPOINT and DOCUMENT_INTELLIGENCE_KEY "
              "must be set in .env or environment.")
        sys.exit(1)

    # ── Collect search values ─────────────────────────────────────────
    search_texts = list(args.search_values)
    
    if not search_texts:
        print("Error: No search values specified. "
              "Provide them as arguments or via --keep-file.")
        sys.exit(1)

    # ── Output path ───────────────────────────────────────────────────
    if args.output is None:
        base, ext = os.path.splitext(args.input_pdf)
        args.output = f"{base}_redacted{ext}"

    # ── Summary ───────────────────────────────────────────────────────
    search_phrases = build_search_phrases(search_texts)
    print("=" * 60)
    print("PDF Table Row Redaction Tool  (Azure Document Intelligence)")
    print("=" * 60)
    print(f"  Input:           {args.input_pdf}")
    print(f"  Output:          {args.output}")
    print(f"  Padding:         {args.padding} pt")
    print(f"  Score tolerance: {args.score_tolerance:.0%}")
    print(f"  Search for:      {search_texts}")
    print(f"  Phrases:         {search_phrases}")
    print("=" * 60)
    print()

    # ── Analyse document once ─────────────────────────────────────────
    print("Analyzing document with Azure Document Intelligence...")
    result = analyze_document(args.input_pdf)
    print("Analysis complete.\n")

    # ── Actual redaction ──────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _process_with_result(
        args.input_pdf, args.output, search_texts, result,
        padding_pt=args.padding,
        score_tolerance=args.score_tolerance,
    )

if __name__ == "__main__":
    main()

"""
PDF Table Row Redaction Tool using Azure Document Intelligence.

Analyzes PDF tables with Azure DI, finds the row matching search values,
and redacts (greys out) all other rows.  The header row (row 0) is always kept.

Examples:
    py row_redactor_azure.py input/input.pdf "2013/8024"
    py row_redactor_azure.py input/input2.pdf "şeref aydemir" "7456710" -o redacted.pdf
"""

import sys
import os
import re
import argparse

import fitz
from PIL import Image, ImageDraw
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

load_dotenv()

ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")


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
        )
    return poller.result()


# ── PDF page rendering ──────────────────────────────────────────────────────


def pdf_page_to_image(page, dpi=300):
    """Convert a PyMuPDF page to a PIL Image at the specified DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


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

# Common OCR / character confusion pairs
_OCR_CONFUSIONS = {
    '0': 'o', 'o': '0',
    '1': 'l', 'l': '1',
    'i': 'ı', 'ı': 'i',
    '5': 's', 's': '5',
    '8': 'b', 'b': '8',
    'ğ': 'g', 'g': 'ğ',
    'ü': 'u', 'u': 'ü',
    'ö': 'o',
    'ş': 's',
    'ç': 'c', 'c': 'ç',
    'â': 'a',
    'î': 'i',
    'û': 'u',
}


def _char_match(a, b):
    """Return True if characters *a* and *b* are identical or confusable."""
    if a == b:
        return True
    return _OCR_CONFUSIONS.get(a) == b or _OCR_CONFUSIONS.get(b) == a


def _edit_distance(s1, s2, max_dist=None):
    """Edit distance with confusion-aware substitution (cost 0)."""
    len1, len2 = len(s1), len(s2)
    if max_dist is not None and abs(len1 - len2) > max_dist:
        return max_dist + 1
    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        row_min = i
        for j in range(1, len2 + 1):
            cost = 0 if _char_match(s1[i - 1], s2[j - 1]) else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + cost,     # substitution
            )
            row_min = min(row_min, curr[j])
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[len2]


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
    dist = _edit_distance(search_word, target_word, max_dist=threshold)
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

        # Word-level polygons (precise redaction)
        word_polygons = []
        for cell in cells:
            if not cell.spans:
                continue
            for word in page_words:
                if _word_in_span(word, cell.spans):
                    if word.polygon:
                        word_polygons.append(word.polygon)

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
            "cell_polygons": cell_polygons,
        })

    return rows, page.width, page.height


# ── Matching ─────────────────────────────────────────────────────────────────


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

    dist = _edit_distance(search_word, target_word, max_dist=threshold)
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


def row_matches_search(row, search_phrases):
    """Return True if ANY search phrase fuzzy-matches within the row."""
    if not search_phrases:
        return False

    row_grouped = []
    for cell_text in row["cell_texts"]:
        norm_words = tuple(normalize(w) for w in cell_text.split() if normalize(w))
        if norm_words:
            row_grouped.append(norm_words)

    all_row_words = tuple(w for group in row_grouped for w in group)

    for search_phrase in search_phrases:
        if len(search_phrase) == 1:
            if any(_fuzzy_word_match(search_phrase[0], rw) for rw in all_row_words):
                return True
        else:
            for grouped in row_grouped:
                if _phrase_fuzzy_match(search_phrase, grouped):
                    return True
    return False


# ── Redaction ────────────────────────────────────────────────────────────────


def redact_rows(image, rows, keep_indices, page_width, page_height, padding=2):
    """
    Grey out words in every row whose index is NOT in *keep_indices*.

    Uses word-level polygons for precise redaction (preserves table grid
    lines).  Falls back to cell-level polygons when word polygons are
    unavailable.
    """
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    scale_x = img_w / page_width
    scale_y = img_h / page_height

    kept_rows = []
    redacted_rows = []

    for i, row in enumerate(rows):
        if i in keep_indices:
            kept_rows.append(i)
        else:
            redacted_rows.append(i)
            polygons = row["word_polygons"] or row["cell_polygons"]
            for poly in polygons:
                x_min, y_min, x_max, y_max = polygon_to_bbox(poly)
                x0 = max(x_min * scale_x - padding, 0)
                y0 = max(y_min * scale_y - padding, 0)
                x1 = min(x_max * scale_x + padding, img_w)
                y1 = min(y_max * scale_y + padding, img_h)
                draw.rectangle([x0, y0, x1, y1], fill="grey")

    return image, kept_rows, redacted_rows


# ── Processing pipeline ─────────────────────────────────────────────────────


def process_pdf(input_path, output_path, search_texts, dpi=200, padding=2):
    """Main pipeline: analyse with Azure DI, match rows, redact, save."""
    search_phrases = build_search_phrases(search_texts)
    print(f"Search phrases (normalized): {search_phrases}\n")

    print("Analyzing document with Azure Document Intelligence...")
    result = analyze_document(input_path)
    print("Analysis complete.\n")

    _process_with_result(input_path, output_path, search_texts, result,
                         dpi=dpi, padding=padding)


def _process_with_result(input_path, output_path, search_texts, result,
                         dpi=200, padding=2):
    """Run the redaction pipeline with a pre-obtained Azure DI result."""
    search_phrases = build_search_phrases(search_texts)

    doc = fitz.open(input_path)
    redacted_images = []

    for page_num in range(len(doc)):
        page_number = page_num + 1
        print(f"--- Page {page_number}/{len(doc)} ---")

        page = doc[page_num]
        image = pdf_page_to_image(page, dpi=dpi)
        print(f"  Image size: {image.size[0]}x{image.size[1]} px")

        rows, page_width, page_height = extract_table_rows(result, page_number)

        if not rows:
            print("  No table rows found on this page.")
            redacted_images.append(image)
            continue

        print(f"  Table rows detected: {len(rows)}")

        # Keep all header rows (columnHeader kind)
        keep_indices = {i for i, row in enumerate(rows) if row["is_header"]}

        # Score non-header rows and pick the single best match
        best_idx = None
        best_score = 0.0
        for i, row in enumerate(rows):
            if row["is_header"]:
                continue
            score = _row_match_score(row, search_phrases)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            keep_indices.add(best_idx)
            print(f"  Best matching row: {best_idx} (score {best_score:.3f})")

        redacted_img, kept_rows, redacted_rows_list = redact_rows(
            image, rows, keep_indices, page_width, page_height, padding=padding,
        )

        print(f"  Kept rows ({len(kept_rows)}):")
        for ki in kept_rows:
            print(f"    Row {ki}: {rows[ki]['text'][:120]}")
        print(f"  Redacted rows: {len(redacted_rows_list)}")

        redacted_images.append(redacted_img)

    doc.close()

    if redacted_images:
        redacted_images[0].save(
            output_path,
            "PDF",
            save_all=True,
            append_images=redacted_images[1:] if len(redacted_images) > 1 else [],
            resolution=dpi,
        )
        print(f"\nRedacted PDF saved to: {output_path}")
    else:
        print("No pages found in the PDF.")


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
            '  py row_redactor_azure.py input/input.pdf --keep-file rows.txt -o out.pdf\n'
            '  py row_redactor_azure.py input/input.pdf "2014/9870" --dry-run\n'
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
        "--keep-file",
        default=None,
        help="Path to a text file with search values (one per line)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for rendering PDF pages as images (default: 200)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Extra padding (px) around redaction rectangles (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show detected rows and keep/redact decisions without saving",
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
    if args.keep_file:
        if not os.path.exists(args.keep_file):
            print(f"Error: Keep file '{args.keep_file}' not found.")
            sys.exit(1)
        with open(args.keep_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    search_texts.append(line)

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
    print(f"  Input:      {args.input_pdf}")
    print(f"  Output:     {args.output}")
    print(f"  DPI:        {args.dpi}")
    print(f"  Padding:    {args.padding}px")
    print(f"  Search for: {search_texts}")
    print(f"  Phrases:    {search_phrases}")
    if args.dry_run:
        print("  Mode:       DRY RUN (no output will be saved)")
    print("=" * 60)
    print()

    # ── Analyse document once ─────────────────────────────────────────
    print("Analyzing document with Azure Document Intelligence...")
    result = analyze_document(args.input_pdf)
    print("Analysis complete.\n")

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        doc = fitz.open(args.input_pdf)
        for page_num in range(len(doc)):
            page_number = page_num + 1
            print(f"--- Page {page_number}/{len(doc)} ---")

            rows, page_width, page_height = extract_table_rows(
                result, page_number)

            if not rows:
                print("  No table rows found on this page.\n")
                continue

            print(f"  Rows detected: {len(rows)}\n")

            best_idx = None
            best_score = 0.0
            for i, row in enumerate(rows):
                if row["is_header"]:
                    continue
                score = _row_match_score(row, search_phrases)
                if score > best_score:
                    best_score = score
                    best_idx = i

            for i, row in enumerate(rows):
                if row["is_header"]:
                    status = "HEADER (keep)"
                elif i == best_idx:
                    status = "BEST   (keep)"
                else:
                    score = _row_match_score(row, search_phrases)
                    status = f"match  s={score:.2f}" if score > 0 else "REDACT"

                print(f"  Row {i:3d} [{status:14s}]  {row['text'][:100]}")

            if best_idx is not None:
                print(f"\n  -> Best match: Row {best_idx} "
                      f"(score {best_score:.3f})")
        doc.close()
        print("\nDry run complete. No file was saved.")

    # ── Actual redaction ──────────────────────────────────────────────
    else:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        _process_with_result(
            args.input_pdf, args.output, search_texts, result,
            dpi=args.dpi, padding=args.padding,
        )


if __name__ == "__main__":
    main()

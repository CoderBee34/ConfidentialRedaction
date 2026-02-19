"""
Examples:
    py row_redactor.py input/input.pdf "2013/8024"
    py row_redactor.py input/input.pdf "AHMET" "ULVIHANOGLU" -o redacted.pdf
"""

import sys
import os
import re
import argparse

import cv2
import numpy as np
import fitz
from PIL import Image, ImageDraw
import pytesseract
from pytesseract import Output

# Configure Tesseract path for Windows if not on PATH
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def pdf_page_to_image(page, dpi=300):
    """Convert a PyMuPDF page to a PIL Image at the specified DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def get_ocr_words(image):
    """Run OCR on an image and return word-level bounding box data."""
    data = pytesseract.image_to_data(image, lang="tur", output_type=Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if conf < 0 or not text:
            continue
        words.append({
            "text": text,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "conf": conf,
        })
    return words


def normalize(text):
    """Normalize text for fuzzy comparison: lowercase, strip edge punctuation."""
    t = text.strip().lower()
    t = re.sub(r"^[^\w]+", "", t)
    t = re.sub(r"[^\w]+$", "", t)
    return t


def build_search_set(search_texts):
    search_words = set()
    for phrase in search_texts:
        for word in phrase.split():
            n = normalize(word)
            if n:
                search_words.add(n)
    return search_words


# ── Line detection (table grid) ──────────────────────────────────────────────


def _detect_lines(binary, direction, min_ratio=0.10, merge_gap=8):
    if direction == "horizontal":
        dim = binary.shape[1]  # width
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(int(dim * min_ratio), 50), 1))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        coords = np.where(opened.max(axis=1) > 0)[0]  # y-positions
    else:
        dim = binary.shape[0]  # height
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(int(dim * min_ratio), 50)))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        coords = np.where(opened.max(axis=0) > 0)[0]  # x-positions

    if len(coords) == 0:
        return []

    # Merge nearby positions into single line positions
    merged = []
    cluster = [coords[0]]
    for c in coords[1:]:
        if c - cluster[-1] <= merge_gap:
            cluster.append(c)
        else:
            merged.append(int(np.mean(cluster)))
            cluster = [c]
    merged.append(int(np.mean(cluster)))
    return merged


def _binarise(pil_image, threshold=128):
    """Convert a PIL image to a binarised numpy array (ink = 255)."""
    img_np = np.array(pil_image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def detect_lines(pil_image, direction, min_ratio=0.10, merge_gap=8,
                 binarize_threshold=128):
    """
    Public helper: detect lines in a PIL image.

    Parameters
    ----------
    direction : str  ``"horizontal"`` or ``"vertical"``
    """
    binary = _binarise(pil_image, binarize_threshold)
    return _detect_lines(binary, direction, min_ratio, merge_gap)


# ── Orientation auto-detection & normalisation ───────────────────────────────


def detect_and_normalise_orientation(pil_image):
    """
    Auto-detect whether the scanned table is rotated by comparing the
    number of horizontal vs vertical grid lines.

    * More horizontal lines → table is upright (or 180°) → rows are
      separated by horizontal lines → no rotation needed.
    * More vertical lines → table is rotated 90° or 270° → try both
      rotations, pick the one that yields the most horizontal lines.

    Returns
    -------
    (normalised_image, rotation_applied)
        ``rotation_applied`` is the CCW angle (0, 90, 180, 270) applied to
        make `rows` horizontal.  To undo, rotate by ``(360 - angle) % 360``.
    """
    binary = _binarise(pil_image)
    h_lines = _detect_lines(binary, "horizontal")
    v_lines = _detect_lines(binary, "vertical")

    print(f"  Orientation check: {len(h_lines)} horizontal lines, "
          f"{len(v_lines)} vertical lines")

    if len(h_lines) >= len(v_lines):
        # Table rows already run horizontally – no rotation needed.
        # (Could still be 180°-flipped, but line detection works either way
        #  and OCR handles upside-down text poorly; we'll try OSD below.)
        angle = _maybe_detect_180(pil_image)
        if angle == 180:
            return pil_image.rotate(180, expand=True), 180
        return pil_image, 0

    # Vertical lines dominate → table is rotated 90° or 270°.
    # Try both and pick the one giving more horizontal lines.
    rot90 = pil_image.rotate(90, expand=True)
    rot270 = pil_image.rotate(270, expand=True)

    h90 = len(detect_lines(rot90, "horizontal"))
    h270 = len(detect_lines(rot270, "horizontal"))

    if h90 >= h270:
        print(f"  -> Rotating 90 deg CCW  (h_lines after: {h90})")
        return rot90, 90
    else:
        print(f"  -> Rotating 270 deg CCW (h_lines after: {h270})")
        return rot270, 270


def _maybe_detect_180(pil_image):
    """
    Use Tesseract OSD to check if the image is upside-down (180°).
    Returns 180 if so, else 0.  Swallows errors and defaults to 0.
    """
    try:
        osd = pytesseract.image_to_osd(pil_image)
        m = re.search(r"Rotate:\s*(\d+)", osd)
        if m and int(m.group(1)) == 180:
            conf_m = re.search(r"Orientation confidence:\s*([\d.]+)", osd)
            conf = float(conf_m.group(1)) if conf_m else 0
            if conf > 2.0:
                print(f"  -> OSD says 180 deg (confidence {conf:.2f})")
                return 180
    except Exception:
        pass
    return 0


def undo_rotation(pil_image, rotation_applied):
    """Rotate the redacted image back to its original orientation."""
    if rotation_applied == 0:
        return pil_image
    undo_angle = (360 - rotation_applied) % 360
    return pil_image.rotate(undo_angle, expand=True)


# ── Table-region detection (dense-data filter) ───────────────────────────────


def _find_dense_range(values, bin_size, min_density_ratio=0.20):
    """
    Find the contiguous range along one axis that covers the densest
    cluster of values.  Used to locate the table's vertical and
    horizontal extent.

    Parameters
    ----------
    values : list[float]
        1-D coordinate values (e.g. word y-centres).
    bin_size : float
        Width of each histogram bin.
    min_density_ratio : float
        A bin is considered "occupied" if its count ≥ this fraction of
        the peak bin count (and ≥ 1).

    Returns
    -------
    (range_min, range_max) or (None, None) if *values* is empty.
    """
    if not values:
        return None, None

    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    if span < bin_size:
        return vmin, vmax

    n_bins = max(1, int(span / bin_size) + 1)
    bins = [0] * n_bins
    for v in values:
        bi = min(int((v - vmin) / bin_size), n_bins - 1)
        bins[bi] += 1

    peak = max(bins)
    threshold = max(peak * min_density_ratio, 1)

    # Collect contiguous runs of bins above the threshold
    runs = []
    start = None
    for i, count in enumerate(bins):
        if count >= threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, n_bins - 1))

    if not runs:
        return vmin, vmax

    # Pick the run covering the most values
    best_run = max(runs, key=lambda r: sum(bins[r[0]:r[1] + 1]))

    range_min = vmin + best_run[0] * bin_size
    range_max = vmin + (best_run[1] + 1) * bin_size
    return range_min, range_max


def filter_table_words(words, img_width, img_height):
    """
    Separate OCR words into **table words** (inside the dense data
    region) and **non-table words** (isolated text such as page
    headers, footers, watermarks, stamps).

    Table data is tightly clustered on the **y-axis** (rows are close
    together) but can span the full width of the page on x, so
    filtering is based on y-density only.

    Parameters
    ----------
    words : list[dict]
        OCR word dicts.
    img_width, img_height : int
        Image dimensions in pixels.

    Returns
    -------
    (table_words, non_table_words)
    """
    if len(words) < 5:
        return list(words), []

    # Derive bin size from median word height
    heights = sorted(w["height"] for w in words if w["height"] > 0)
    median_h = heights[len(heights) // 2] if heights else 20

    bin_size_y = max(median_h * 2.0, 15)

    y_centres = [w["top"] + w["height"] / 2 for w in words]
    y_lo, y_hi = _find_dense_range(y_centres, bin_size_y)

    if y_lo is None:
        return list(words), []

    # Generous vertical margin so we don't clip table edges
    y_margin = median_h * 3

    table_words = []
    non_table_words = []
    for w in words:
        # Skip single-character words (OCR noise)
        if len(w["text"].strip()) <= 1:
            non_table_words.append(w)
            continue
        cy = w["top"] + w["height"] / 2
        if (y_lo - y_margin) <= cy <= (y_hi + y_margin):
            table_words.append(w)
        else:
            non_table_words.append(w)

    return table_words, non_table_words


# ── Column detection & tilt estimation ────────────────────────────────────────


def _detect_columns(words, x_tolerance=None):
    """
    Cluster words into vertical columns by their horizontal centre.

    Returns a list of columns, each a list of word dicts sorted by y-centre
    (top → bottom).  Columns are sorted left → right.
    """
    if not words:
        return []

    # Sort words by x-centre
    by_x = sorted(words, key=lambda w: w["left"] + w["width"] / 2)

    # Derive x_tolerance from gap distribution
    if x_tolerance is None:
        centres = [w["left"] + w["width"] / 2 for w in by_x]
        if len(centres) >= 2:
            gaps = [centres[i + 1] - centres[i] for i in range(len(centres) - 1)]
            gaps_sorted = sorted(gaps)
            median_gap = gaps_sorted[len(gaps_sorted) // 2]
            # Column separators are significantly larger than within-column gaps
            x_tolerance = max(median_gap * 2.5, 15)
        else:
            x_tolerance = 30

    # Gap-based x-clustering
    columns = []
    current_col = [by_x[0]]
    for w in by_x[1:]:
        cx = w["left"] + w["width"] / 2
        col_mean_x = sum(
            (cw["left"] + cw["width"] / 2) for cw in current_col
        ) / len(current_col)
        if abs(cx - col_mean_x) <= x_tolerance:
            current_col.append(w)
        else:
            columns.append(current_col)
            current_col = [w]
    columns.append(current_col)

    # Sort each column internally by y-centre (top → bottom)
    for col in columns:
        col.sort(key=lambda w: w["top"] + w["height"] / 2)

    return columns


def _estimate_tilt_slope(columns, max_slope=0.27):
    """
    Estimate the tilt slope (dy / dx) of the table from detected columns.

    For every pair of columns that share the same word count, words at the
    same index are assumed to belong to the same table row.  The slope is
    the median of  ``(y_b - y_a) / (x_b - x_a)``  over all such pairs.

    Parameters
    ----------
    columns : list[list[dict]]
        Columns as returned by :func:`_detect_columns`.
    max_slope : float
        Absolute slope cap (≈ tan 15°).  Anything larger is likely noise.

    Returns
    -------
    float   Estimated slope (0.0 when insufficient data).
    """
    if len(columns) < 2:
        return 0.0

    slopes = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a, col_b = columns[i], columns[j]
            n = min(len(col_a), len(col_b))
            if n < 2:
                continue

            # Mean x of each column for dx
            mean_xa = sum(w["left"] + w["width"] / 2 for w in col_a) / len(col_a)
            mean_xb = sum(w["left"] + w["width"] / 2 for w in col_b) / len(col_b)
            dx = mean_xb - mean_xa
            if abs(dx) < 10:
                continue

            # Only pair when word counts are close (allow ±1 difference)
            if abs(len(col_a) - len(col_b)) > 1:
                continue

            for k in range(n):
                ya = col_a[k]["top"] + col_a[k]["height"] / 2
                yb = col_b[k]["top"] + col_b[k]["height"] / 2
                slopes.append((yb - ya) / dx)

    if not slopes:
        return 0.0

    # Median is robust against outlier word pairs
    slopes.sort()
    median_slope = slopes[len(slopes) // 2]

    # Clamp to reasonable range
    return max(-max_slope, min(max_slope, median_slope))


# ── Row building from word coordinates ────────────────────────────────────────


def build_rows_from_words(words, img_height, y_tolerance=None):
    """
    Build table rows by combining column-aware tilt estimation with
    y-centre clustering.  Robust against tilted scans and photos.

    Algorithm
    ---------
    1.  Detect vertical **columns** by clustering word x-centres.
    2.  **Estimate tilt** from corresponding words across columns
        (same positional index → same logical row).
    3.  Compute a **corrected y** for each word:
        ``corrected_y = y_centre − slope × x_centre`` which removes the
        effect of document tilt.
    4.  **Cluster words by corrected y** (gap-based) to form rows.

    Parameters
    ----------
    words : list[dict]
        OCR word dicts with keys: text, left, top, width, height, conf.
    img_height : int
        Height of the image in pixels (used as fallback boundary).
    y_tolerance : float or None
        Maximum corrected-y distance for two words to be in the same
        row.  Defaults to ``0.6 × median_word_height`` (clamped ≥ 5 px).

    Returns
    -------
    list[dict]
        Rows sorted top → bottom, each with keys:
        ``y_min``, ``y_max``, ``words``, ``text``.
    """
    if not words:
        return []

    # ── Step 1 & 2: columns → tilt slope ──────────────────────────────
    columns = _detect_columns(words)
    tilt_slope = _estimate_tilt_slope(columns)

    print(f"    Columns detected: {len(columns)}  "
          f"(sizes {[len(c) for c in columns]})")
    print(f"    Estimated tilt slope: {tilt_slope:.5f}  "
          f"({np.degrees(np.arctan(tilt_slope)):.2f}°)")

    # ── Step 3: corrected y for each word ─────────────────────────────
    annotated = []
    for w in words:
        cx = w["left"] + w["width"] / 2
        cy = w["top"] + w["height"] / 2
        corrected_y = cy - tilt_slope * cx
        annotated.append((corrected_y, w))

    annotated.sort(key=lambda t: t[0])

    # ── y-tolerance from median word height ───────────────────────────
    if y_tolerance is None:
        heights = sorted(w["height"] for w in words if w["height"] > 0)
        if heights:
            median_h = heights[len(heights) // 2]
            y_tolerance = max(median_h * 0.6, 5)
        else:
            y_tolerance = 10

    # ── Step 4: gap-based clustering on corrected y ───────────────────
    clusters = []
    current_cluster = [annotated[0]]
    for item in annotated[1:]:
        cy = item[0]
        cluster_mean_y = sum(c[0] for c in current_cluster) / len(current_cluster)
        if abs(cy - cluster_mean_y) <= y_tolerance:
            current_cluster.append(item)
        else:
            clusters.append(current_cluster)
            current_cluster = [item]
    clusters.append(current_cluster)

    # ── Convert clusters into row dicts ───────────────────────────────
    rows = []
    for cluster in clusters:
        row_words = [item[1] for item in cluster]
        row_words.sort(key=lambda w: w["left"])

        y_min = min(w["top"] for w in row_words)
        y_max = max(w["top"] + w["height"] for w in row_words)
        text = " ".join(w["text"] for w in row_words)

        rows.append({
            "y_min": y_min,
            "y_max": y_max,
            "words": row_words,
            "text": text,
        })

    rows.sort(key=lambda r: r["y_min"])
    return rows


def row_matches_search(row, search_set):
    """Return True if ANY word in the row matches ANY word in search_set."""
    for w in row["words"]:
        n = normalize(w["text"])
        if n and n in search_set:
            return True
    return False


# ── Redaction ────────────────────────────────────────────────────────────────


def redact_rows(image, rows, keep_indices, padding=2):
    """
    Black out words in every row whose index is NOT in keep_indices.
    Each word is redacted individually using its bounding box, preserving
    the table grid lines and surrounding whitespace.
    """
    draw = ImageDraw.Draw(image)

    kept_rows = []
    redacted_rows = []

    for i, row in enumerate(rows):
        if i in keep_indices:
            kept_rows.append(i)
        else:
            redacted_rows.append(i)
            for w in row["words"]:
                x0 = max(w["left"] - padding, 0)
                y0 = max(w["top"] - padding, 0)
                x1 = min(w["left"] + w["width"] + padding, image.size[0])
                y1 = min(w["top"] + w["height"] + padding, image.size[1])
                draw.rectangle([x0, y0, x1, y1], fill="grey")

    return image, kept_rows, redacted_rows


def process_page(image, search_set, padding=2):
    """
    Full pipeline for one page:
      1.  Auto-detect & normalise orientation
      2.  Detect horizontal table lines  →  row boundaries
      3.  OCR  →  words
      4.  Assign words to rows
      5.  Row 0  =  header  (always kept)
      6.  Find rows matching search terms
      7.  Redact everything else
      8.  Rotate back to original orientation
    """
    # ── orientation ──
    normalised, rotation = detect_and_normalise_orientation(image)

    h_lines = detect_lines(normalised, "horizontal")
    all_words = get_ocr_words(normalised)

    # Separate table words from non-table words (headers, footers, etc.)
    table_words, non_table_words = filter_table_words(
        all_words, normalised.size[0], normalised.size[1])
    if non_table_words:
        print(f"    Excluded {len(non_table_words)} non-table word(s): "
              f"{' '.join(w['text'] for w in non_table_words[:10])}")

    rows = build_rows_from_words(table_words, normalised.size[1])

    if not rows:
        result = undo_rotation(normalised, rotation)
        return result, [], [], rows, h_lines, rotation

    # Always keep the header (first row with actual content)
    keep_indices = {0}

    # Find rows that contain any of the search words
    for i, row in enumerate(rows):
        if row_matches_search(row, search_set):
            keep_indices.add(i)

    redacted_img, kept_rows, redacted_rows = redact_rows(
        normalised, rows, keep_indices, padding=padding
    )

    # Rotate back to original orientation
    result = undo_rotation(redacted_img, rotation)
    return result, kept_rows, redacted_rows, rows, h_lines, rotation


def process_pdf(input_path, output_path, search_texts, dpi=200, padding=2):
    """Main processing pipeline: detect grid, OCR, match rows, redact."""
    search_set = build_search_set(search_texts)
    print(f"Search words (normalized): {search_set}\n")

    doc = fitz.open(input_path)
    redacted_images = []

    for page_num in range(len(doc)):
        print(f"--- Page {page_num + 1}/{len(doc)} ---")
        page = doc[page_num]

        image = pdf_page_to_image(page, dpi=dpi)
        print(f"  Image size: {image.size[0]}x{image.size[1]} px")

        redacted_img, kept_rows, redacted_rows, rows, h_lines, rotation = \
            process_page(image, search_set, padding=padding)

        print(f"  Rotation applied: {rotation} deg")
        print(f"  Horizontal lines detected: {len(h_lines)}")
        print(f"  Table rows detected: {len(rows)}")
        print(f"  Kept rows ({len(kept_rows)}):")
        for ki in kept_rows:
            print(f"    Row {ki} [y={rows[ki]['y_min']}-{rows[ki]['y_max']}]: "
                  f"{rows[ki]['text'][:120]}")
        print(f"  Redacted rows: {len(redacted_rows)}")

        redacted_images.append(redacted_img)

    doc.close()

    # Save redacted images as a new PDF
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
            "Redact rows from a scanned PDF table. "
            "The header row is always kept. Rows containing any of the "
            "given search values are kept; all other rows are blacked out."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  py row_redactor.py input/input.pdf "2013/8024"\n'
            '  py row_redactor.py input/input.pdf "AHMET" "ULVIHANOGLU"\n'
            '  py row_redactor.py input/input.pdf --keep-file rows.txt -o out.pdf\n'
            '  py row_redactor.py input/input.pdf "2014/9870" --dry-run\n'
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
        help="DPI for rendering PDF pages (default: 200)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Extra padding (px) above/below redaction rectangles (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show detected rows and keep/redact decisions without saving",
    )

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input file '{args.input_pdf}' not found.")
        sys.exit(1)

    # ── Collect search values ─────────────────────────────────────────────
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

    # ── Output path ───────────────────────────────────────────────────────
    if args.output is None:
        base, ext = os.path.splitext(args.input_pdf)
        args.output = f"{base}_redacted{ext}"

    # ── Summary ───────────────────────────────────────────────────────────
    search_set = build_search_set(search_texts)
    print("=" * 60)
    print("PDF Table Row Redaction Tool")
    print("=" * 60)
    print(f"  Input:      {args.input_pdf}")
    print(f"  Output:     {args.output}")
    print(f"  DPI:        {args.dpi}")
    print(f"  Padding:    {args.padding}px")
    print(f"  Search for: {search_texts}")
    print(f"  Normalized: {search_set}")
    if args.dry_run:
        print("  Mode:       DRY RUN (no output will be saved)")
    print("=" * 60)
    print()

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        doc = fitz.open(args.input_pdf)
        for page_num in range(len(doc)):
            print(f"--- Page {page_num + 1}/{len(doc)} ---")
            page = doc[page_num]
            image = pdf_page_to_image(page, dpi=args.dpi)

            # Auto-detect and normalise orientation
            normalised, rotation = detect_and_normalise_orientation(image)
            print(f"  Rotation applied: {rotation} deg")

            h_lines = detect_lines(normalised, "horizontal")
            print(f"  Horizontal lines at y: {h_lines}")

            all_words = get_ocr_words(normalised)

            table_words, non_table_words = filter_table_words(
                all_words, normalised.size[0], normalised.size[1])
            if non_table_words:
                print(f"  Excluded {len(non_table_words)} non-table word(s): "
                      f"{' '.join(w['text'] for w in non_table_words[:10])}")

            rows = build_rows_from_words(table_words, normalised.size[1])

            print(f"  Rows detected: {len(rows)}\n")
            for i, row in enumerate(rows):
                is_header = (i == 0)
                matches = row_matches_search(row, search_set)
                if is_header:
                    status = "HEADER (keep)"
                elif matches:
                    status = "MATCH  (keep)"
                else:
                    status = "REDACT"
                print(f"  Row {i:3d} [{status:14s}]  "
                      f"y={row['y_min']:4d}-{row['y_max']:4d}  "
                      f"{row['text'][:100]}")
        doc.close()
        print("\nDry run complete. No file was saved.")

    # ── Actual redaction ──────────────────────────────────────────────────
    else:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        process_pdf(args.input_pdf, args.output, search_texts,
                     dpi=args.dpi, padding=args.padding)


if __name__ == "__main__":
    main()

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
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
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


# ── Row building from grid lines ─────────────────────────────────────────────


def build_rows_from_lines(h_lines, words, img_height):
    """
    Given a list of horizontal line y-positions and OCR words,
    build table rows by assigning words to the band between consecutive
    horizontal lines.

    Returns a list of row dicts sorted top → bottom:
        {
            "y_min": int,          # top boundary (line above)
            "y_max": int,          # bottom boundary (line below)
            "words": [word, ...],
            "text":  str,
        }
    """
    if len(h_lines) < 2:
        # Fallback: treat the whole page as a single row
        text = " ".join(w["text"] for w in words)
        return [{
            "y_min": 0,
            "y_max": img_height,
            "words": list(words),
            "text": text,
        }]

    # Bands between consecutive lines define rows
    bands = []
    for i in range(len(h_lines) - 1):
        bands.append((h_lines[i], h_lines[i + 1]))

    # Assign each word to the band whose range contains the word's vertical centre
    rows = []
    for y_top, y_bot in bands:
        row_words = []
        for w in words:
            word_centre_y = w["top"] + w["height"] / 2
            if y_top <= word_centre_y < y_bot:
                row_words.append(w)
        # Skip empty bands (e.g. thin inter-line gaps with no text)
        if not row_words:
            continue
        row_words.sort(key=lambda w: w["left"])
        text = " ".join(w["text"] for w in row_words)
        rows.append({
            "y_min": y_top,
            "y_max": y_bot,
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
    words = get_ocr_words(normalised)
    rows = build_rows_from_lines(h_lines, words, normalised.size[1])

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
        default=1,
        help="Extra padding (px) above/below redaction rectangles (default: 2)",
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

            words = get_ocr_words(normalised)
            rows = build_rows_from_lines(h_lines, words, normalised.size[1])

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

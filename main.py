import sys
import os
import argparse

import fitz
from PIL import Image, ImageDraw
import pytesseract
from pytesseract import Output

# Configure Tesseract path for Windows if not on PATH
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def verify_tesseract():
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def pdf_page_to_image(page, dpi=300):
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def get_ocr_words(image):
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
    import re
    t = text.strip().lower()
    # Strip leading/trailing punctuation but keep internal ones (e.g., hyphens in names)
    t = re.sub(r"^[^\w]+", "", t)
    t = re.sub(r"[^\w]+$", "", t)
    return t


def build_keep_set(keep_texts):
    keep_words = set()
    for phrase in keep_texts:
        for word in phrase.split():
            n = normalize(word)
            if n:
                keep_words.add(n)
    return keep_words


def should_keep(word_text, keep_set):
    n = normalize(word_text)
    if not n:
        return True
    return n in keep_set


def redact_page(image, keep_set, padding=2):
    words = get_ocr_words(image)
    draw = ImageDraw.Draw(image)

    kept = []
    redacted = []

    for w in words:
        if should_keep(w["text"], keep_set):
            kept.append(w["text"])
        else:
            redacted.append(w["text"])
            x0 = w["left"] - padding
            y0 = w["top"] - padding
            x1 = w["left"] + w["width"] + padding
            y1 = w["top"] + w["height"] + padding
            draw.rectangle([x0, y0, x1, y1], fill="black")

    return image, kept, redacted


def process_pdf(input_path, output_path, keep_texts, dpi=300, padding=2):
    keep_set = build_keep_set(keep_texts)
    print(f"Words to keep (normalized): {keep_set}\n")

    doc = fitz.open(input_path)
    redacted_images = []

    for page_num in range(len(doc)):
        print(f"--- Page {page_num + 1}/{len(doc)} ---")
        page = doc[page_num]

        # Convert to image
        image = pdf_page_to_image(page, dpi=dpi)
        print(f"  Image size: {image.size[0]}x{image.size[1]} px")

        # OCR and redact
        redacted_img, kept, redacted = redact_page(image, keep_set, padding=padding)

        print(f"  Kept words ({len(kept)}): {kept[:20]}{'...' if len(kept) > 20 else ''}")
        print(f"  Redacted words ({len(redacted)}): {redacted[:20]}{'...' if len(redacted) > 20 else ''}")

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Redact sensitive information from scanned PDF tables. "
            "All text is redacted EXCEPT the specified values."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  py main.py input/input.pdf "Name" "Date" "Total"\n'
            '  py main.py input/input.pdf "John Doe" "Jane Smith" -o redacted.pdf\n'
            '  py main.py input/input.pdf --keep-file keep_words.txt\n'
        ),
    )
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument(
        "keep_texts",
        nargs="*",
        default=[],
        help=(
            'Text values to keep visible. Use quotes for multi-word phrases: '
            '"First Name" "Last Name". Everything else will be redacted.'
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
        help="Path to a text file with values to keep (one per line)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for rendering PDF pages for OCR (default: 300)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Extra padding (px) around redaction boxes (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be redacted, without saving output",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input file '{args.input_pdf}' not found.")
        sys.exit(1)

    # Verify Tesseract
    if not verify_tesseract():
        print(
            "Error: Tesseract OCR not found.\n"
            "Install it from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Or via winget:  winget install UB-Mannheim.TesseractOCR"
        )
        sys.exit(1)

    # Collect keep texts
    keep_texts = list(args.keep_texts)
    if args.keep_file:
        if not os.path.exists(args.keep_file):
            print(f"Error: Keep file '{args.keep_file}' not found.")
            sys.exit(1)
        with open(args.keep_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    keep_texts.append(line)

    if not keep_texts:
        print("Error: No keep texts specified. Provide them as arguments or via --keep-file.")
        sys.exit(1)

    # Set output path
    if args.output is None:
        base, ext = os.path.splitext(args.input_pdf)
        args.output = f"{base}_redacted{ext}"

    # Summary
    print("=" * 50)
    print("PDF Table Redaction Tool")
    print("=" * 50)
    print(f"  Input:   {args.input_pdf}")
    print(f"  Output:  {args.output}")
    print(f"  DPI:     {args.dpi}")
    print(f"  Padding: {args.padding}px")
    print(f"  Keep:    {keep_texts}")
    if args.dry_run:
        print("  Mode:    DRY RUN (no output will be saved)")
    print("=" * 50)
    print()

    if args.dry_run:
        # Dry run: show OCR results without redacting
        keep_set = build_keep_set(keep_texts)
        print(f"Keep set (normalized): {keep_set}\n")
        doc = fitz.open(args.input_pdf)
        for page_num in range(len(doc)):
            print(f"--- Page {page_num + 1}/{len(doc)} ---")
            page = doc[page_num]
            image = pdf_page_to_image(page, dpi=args.dpi)
            words = get_ocr_words(image)
            for w in words:
                status = "KEEP" if should_keep(w["text"], keep_set) else "REDACT"
                print(f"  [{status:6s}] \"{w['text']}\" (conf={w['conf']})")
        doc.close()
        print("\nDry run complete. No file was saved.")
    else:
        # Ensure output directory exists
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        process_pdf(args.input_pdf, args.output, keep_texts,
                     dpi=args.dpi, padding=args.padding)


if __name__ == "__main__":
    main()

"""
Examples:
    py row_redactor.py input/input.pdf "2013/8024"
    py row_redactor.py input/input2.pdf "şeref aydemir" "7456710" -o redacted.pdf
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
    """Normalize text for comparison: lowercase, strip edge punctuation."""
    t = text.strip().lower()
    t = re.sub(r"^[^\w]+", "", t)
    t = re.sub(r"[^\w]+$", "", t)
    return t


# ── Fuzzy matching helpers ────────────────────────────────────────────────────

# Common OCR confusion pairs (both directions are checked automatically)
_OCR_CONFUSIONS = {
    '0': 'o', 'o': '0',
    '1': 'l', 'l': '1',
    'i': '1', '1': 'i',
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
    """Return True if characters *a* and *b* are identical or OCR-confusable."""
    if a == b:
        return True
    return _OCR_CONFUSIONS.get(a) == b or _OCR_CONFUSIONS.get(b) == a


def _edit_distance(s1, s2, max_dist=None):
    """
    Compute edit distance between *s1* and *s2* with OCR-aware
    character substitutions (OCR-confusable chars cost 0).

    Uses early termination when the distance exceeds *max_dist*.
    """
    len1, len2 = len(s1), len(s2)
    if max_dist is not None and abs(len1 - len2) > max_dist:
        return max_dist + 1

    # Use single-row DP
    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        row_min = i
        for j in range(1, len2 + 1):
            if _char_match(s1[i - 1], s2[j - 1]):
                cost = 0
            else:
                cost = 1
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


def _fuzzy_word_match(search_word, ocr_word, threshold=None):
    """
    Return True if *search_word* fuzzy-matches *ocr_word*.

    The allowed edit distance scales with word length:
      - len ≤ 3:  exact match only (threshold 0)
      - len 4–6:  up to 1 edit
      - len 7–10: up to 2 edits
      - len > 10: up to 3 edits

    OCR-confusable character substitutions cost 0 edits.
    Also returns True if one is a substring of the other (handles
    OCR merging/splitting artefacts).
    """
    if search_word == ocr_word:
        return True

    # Substring containment (OCR sometimes merges or splits words)
    if search_word in ocr_word or ocr_word in search_word:
        return True

    if threshold is None:
        max_len = max(len(search_word), len(ocr_word))
        if max_len <= 3:
            threshold = 0
        elif max_len <= 6:
            threshold = 1
        elif max_len <= 10:
            threshold = 2
        else:
            threshold = 3

    dist = _edit_distance(search_word, ocr_word, max_dist=threshold)
    return dist <= threshold


def _phrase_fuzzy_match(search_phrase, grouped_words):
    """
    Check if a search phrase (tuple of normalised words) fuzzy-matches
    within a group of OCR words (also a tuple of normalised words).

    Tries to find a contiguous or near-contiguous subsequence in
    *grouped_words* that matches all words of *search_phrase* in order.
    Also handles the case where OCR merged two search words into one
    OCR token or split one search word across two OCR tokens.
    """
    n_search = len(search_phrase)
    n_group = len(grouped_words)

    if n_search == 0:
        return True

    if n_search == 1:
        # Single-word phrase: fuzzy match against any word in the group
        sw = search_phrase[0]
        return any(_fuzzy_word_match(sw, gw) for gw in grouped_words)

    # Multi-word phrase: try sliding window + flexible matching
    # Allow the matched span to be slightly wider than the search phrase
    # (OCR may split a word into two tokens)
    for start in range(n_group):
        si = 0  # index into search_phrase
        gi = start  # index into grouped_words
        while si < n_search and gi < n_group:
            sw = search_phrase[si]
            gw = grouped_words[gi]
            if _fuzzy_word_match(sw, gw):
                si += 1
                gi += 1
            elif gi + 1 < n_group:
                # Try matching search word against two merged OCR words
                merged = grouped_words[gi] + grouped_words[gi + 1]
                if _fuzzy_word_match(sw, merged):
                    si += 1
                    gi += 2
                    continue
                # Try matching two search words against one OCR word
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
    Build a list of normalised search phrases.  Each input string is
    treated as a full phrase (e.g. a person's full name) and kept
    intact as a tuple of normalised words.

    For multi-word inputs, adjacent word concatenations are also added
    as single-word phrases because OCR often merges names without a
    space (e.g. "derya aytun" → "deryaaytun").

    Example
    -------
    >>> build_search_phrases(["Lale Bilge Deniz", "2013/8024"])
    [('lale', 'bilge', 'deniz'), ('lalebilde',), ('bilgedeniz',), ('2013/8024',)]
    """
    phrases = []
    for text in search_texts:
        words = [normalize(w) for w in text.split() if normalize(w)]
        if words:
            phrases.append(tuple(words))
            # Add adjacent-pair concatenations for OCR merge handling
            if len(words) >= 2:
                for i in range(len(words) - 1):
                    merged = words[i] + words[i + 1]
                    phrases.append((merged,))
    return phrases


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
    """Convert a PIL image to a binarised numpy array (ink = 255).

    Uses adaptive Gaussian thresholding so uneven lighting, shadows,
    and photo gradients are handled correctly.  The *threshold*
    parameter is retained for API compatibility but unused.
    """
    img_np = np.array(pil_image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
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

    # Filter out rows with fewer than 10 total characters (likely OCR noise)
    MIN_ROW_CHARS = 10
    filtered_rows = []
    for row in rows:
        total_chars = sum(len(w["text"].strip()) for w in row["words"])
        if total_chars >= MIN_ROW_CHARS:
            filtered_rows.append(row)
        else:
            print(f"    Filtered noise row (chars={total_chars}): "
                  f"{row['text'][:80]}")

    return filtered_rows


def _group_words_into_phrases(row_words, gap_factor=2.5):
    """
    Group words in a row into phrases based on x-axis proximity.
    Words that are close together horizontally (like first name,
    middle name, last name) are merged into a single phrase string.

    Parameters
    ----------
    row_words : list[dict]
        Words already sorted left → right.
    gap_factor : float
        A gap larger than ``gap_factor × median_word_width`` starts
        a new phrase.

    Returns
    -------
    list[str]   Normalised phrase strings.
    """
    if not row_words:
        return []

    # Compute gaps between consecutive words
    gaps = []
    for i in range(len(row_words) - 1):
        right_edge = row_words[i]["left"] + row_words[i]["width"]
        left_edge = row_words[i + 1]["left"]
        gaps.append(max(left_edge - right_edge, 0))

    if gaps:
        # Use median word width to set threshold
        widths = sorted(w["width"] for w in row_words if w["width"] > 0)
        median_w = widths[len(widths) // 2] if widths else 20
        gap_threshold = median_w * gap_factor
    else:
        gap_threshold = 0

    # Group words into phrases by splitting at large gaps
    phrases = []
    current_phrase_words = [row_words[0]["text"]]
    for i, gap in enumerate(gaps):
        if gap > gap_threshold:
            phrases.append(" ".join(current_phrase_words))
            current_phrase_words = [row_words[i + 1]["text"]]
        else:
            current_phrase_words.append(row_words[i + 1]["text"])
    phrases.append(" ".join(current_phrase_words))

    # Normalise each phrase into a tuple of normalised words
    result = []
    for p in phrases:
        norm_words = tuple(normalize(w) for w in p.split() if normalize(w))
        if norm_words:
            result.append(norm_words)
    return result


def _word_match_score(search_word, ocr_word):
    """
    Return a match score between 0.0 and 1.0 for a single word pair.
    Higher is better.  0.0 means no match.
    """
    if search_word == ocr_word:
        return 1.0
    if search_word in ocr_word or ocr_word in search_word:
        shorter = min(len(search_word), len(ocr_word))
        longer = max(len(search_word), len(ocr_word))
        return 0.85 * (shorter / longer)

    max_len = max(len(search_word), len(ocr_word))
    if max_len <= 3:
        threshold = 0
    elif max_len <= 6:
        threshold = 1
    elif max_len <= 10:
        threshold = 2
    else:
        threshold = 3

    dist = _edit_distance(search_word, ocr_word, max_dist=threshold)
    if dist > threshold:
        return 0.0
    return max(0.0, 1.0 - dist / max(max_len, 1))


def _row_match_score(row, search_phrases):
    """
    Compute an aggregate match score for a row against all search
    phrases.  The score is the sum of the best per-phrase scores.

    For single-word phrases, the best fuzzy word score across all
    row words is used.  For multi-word phrases, the average of
    per-word best scores within each proximity group is used.

    Returns 0.0 if the row doesn't match at all.
    """
    if not search_phrases:
        return 0.0

    row_grouped = _group_words_into_phrases(row["words"])
    all_row_words = [normalize(w["text"]) for w in row["words"]
                     if normalize(w["text"])]

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
                # Score each search word against the best in the group
                word_scores = []
                for sw in search_phrase:
                    best_w = max((_word_match_score(sw, gw) for gw in grouped),
                                 default=0.0)
                    word_scores.append(best_w)
                avg = sum(word_scores) / len(word_scores) if word_scores else 0.0
                best_phrase_score = max(best_phrase_score, avg)

        if best_phrase_score > 0.0:
            matched_any = True
        total += best_phrase_score

    return total if matched_any else 0.0


def row_matches_search(row, search_phrases):
    """
    Return True if ANY search phrase fuzzy-matches within the row.

    Multi-word phrases are matched against x-proximity word groups
    (so "Lale Bilge Deniz" must appear together).  Single-word
    phrases are matched against every individual word in the row.

    Uses OCR-aware fuzzy matching (edit distance with confusion
    pairs, substring containment, merge/split handling).

    Parameters
    ----------
    row : dict
        Row dict with a ``words`` list (sorted left → right).
    search_phrases : list[tuple[str]]
        Normalised search phrases from :func:`build_search_phrases`.
    """
    if not search_phrases:
        return False

    row_grouped = _group_words_into_phrases(row["words"])
    # Also build a flat list of all normalised words for single-word matching
    all_row_words = tuple(normalize(w["text"]) for w in row["words"]
                          if normalize(w["text"]))

    for search_phrase in search_phrases:
        if len(search_phrase) == 1:
            # Single-word: match against any word in the entire row
            if any(_fuzzy_word_match(search_phrase[0], rw)
                   for rw in all_row_words):
                return True
        else:
            # Multi-word: must match within a single proximity group
            for grouped in row_grouped:
                if _phrase_fuzzy_match(search_phrase, grouped):
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


def process_page(image, search_phrases, padding=2):
    """
    Full pipeline for one page:
      1.  Auto-detect & normalise orientation
      2.  Detect horizontal table lines  →  row boundaries
      3.  OCR  →  words
      4.  Assign words to rows
      5.  Row 0  =  header  (always kept)
      6.  Find rows matching search phrases
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
    keep_indices = set()

    # Score all rows and pick the single best match
    best_idx = None
    best_score = 0.0
    for i, row in enumerate(rows):
        score = _row_match_score(row, search_phrases)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is not None:
        keep_indices.add(best_idx)
        print(f"    Best matching row: {best_idx} (score {best_score:.3f})")

    redacted_img, kept_rows, redacted_rows = redact_rows(
        normalised, rows, keep_indices, padding=padding
    )

    # Rotate back to original orientation
    result = redacted_img #undo_rotation(redacted_img, rotation)
    return result, kept_rows, redacted_rows, rows, h_lines, rotation


def process_pdf(input_path, output_path, search_texts, dpi=200, padding=2):
    """Main processing pipeline: detect grid, OCR, match rows, redact."""
    search_phrases = build_search_phrases(search_texts)
    print(f"Search phrases (normalized): {search_phrases}\n")

    doc = fitz.open(input_path)
    redacted_images = []

    for page_num in range(len(doc)):
        print(f"--- Page {page_num + 1}/{len(doc)} ---")
        page = doc[page_num]

        image = pdf_page_to_image(page, dpi=dpi)
        print(f"  Image size: {image.size[0]}x{image.size[1]} px")

        redacted_img, kept_rows, redacted_rows, rows, h_lines, rotation = \
            process_page(image, search_phrases, padding=padding)

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
    search_phrases = build_search_phrases(search_texts)
    print("=" * 60)
    print("PDF Table Row Redaction Tool")
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
            best_idx = None
            best_score = 0.0
            for i, row in enumerate(rows):
                if i > 0:
                    score = _row_match_score(row, search_phrases)
                    if score > best_score:
                        best_score = score
                        best_idx = i

            for i, row in enumerate(rows):
                is_header = (i == 0)
                if is_header:
                    status = "HEADER (keep)"
                elif i == best_idx:
                    status = f"BEST   (keep)"
                else:
                    score = _row_match_score(row, search_phrases)
                    if score > 0:
                        status = f"match  s={score:.2f}"
                    else:
                        status = "REDACT"
                print(f"  Row {i:3d} [{status:14s}]  "
                      f"y={row['y_min']:4d}-{row['y_max']:4d}  "
                      f"{row['text'][:100]}")
            if best_idx is not None:
                print(f"\n  -> Best match: Row {best_idx} "
                      f"(score {best_score:.3f})")
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

"""
Visual Element Segmenter
Segments diagrams, formulas, tables, and visual regions using SAM 3.1 & Muse Wonder
with intelligent visual-vs-text discrimination and OpenCV contour fallbacks.
Strictly filters out ordinary reading text paragraphs and slide headers.
"""

import io
import base64
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from config import (
    MODEL_SAM,
    MODEL_MUSE_WONDER,
    API_BASE_URL,
    get_headers,
    DEFAULT_TIMEOUT_SECONDS,
    parse_response_content
)

logger = logging.getLogger("VisualScribe.Segmenter")


def image_to_base64_data_uri(image: Image.Image, format: str = "JPEG", max_dim: int = 1600) -> str:
    """
    Converts a PIL Image into a base64 Data URI.
    Automatically scales down massive phone pictures (>1600px) so network payloads
    are swift, robust, and never exceed API buffer limits.
    """
    buffered = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Scale down oversized camera/phone photos proportionally
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    image.save(buffered, format=format, quality=88)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"


def segment_visual_regions(
    image: Image.Image,
    pdf_bytes: bytes = None,
    page_idx: int = 0,
    min_region_area: int = 2500,
    max_regions: int = 6,
    use_sam_api: bool = True,
    focus_mode: str = "balanced",
    user_instruction: str = ""
) -> List[Dict[str, Any]]:
    """
    Identifies and crops genuine visual elements and mathematical calculations.
    
    STRICTLY FILTERS OUT:
      - Textbook decorative page borders, publisher logos, and header/footer ribbons
      - Plain body text paragraphs and reading sentences
      
    Args:
        focus_mode: 'balanced', 'equations_only', or 'diagrams_only'
        user_instruction: Custom user directive or special focus prompt
    """
    regions = []

    # Priority 1: If PDF source is provided, extract mathematical calculation blocks
    # and genuine diagram vector graphics (filtering out textbook decorative ribbons)
    if pdf_bytes and fitz is not None:
        try:
            pdf_regions = _extract_pdf_vector_and_image_regions(
                pdf_bytes, page_idx, target_image_size=image.size, focus_mode=focus_mode, max_regions=max_regions
            )
            if pdf_regions:
                logger.info(f"PyMuPDF extracted {len(pdf_regions)} calculation/visual regions.")
                regions.extend(pdf_regions)
        except Exception as e:
            logger.warning(f"PDF vector extraction notice: {e}")

    # Priority 2: Multimodal AI Vision Locator (Muse Wonder / SAM 3.1)
    # Strictly instructed to find mathematical formulas, derivations, and technical visuals
    if not regions and use_sam_api:
        try:
            ai_regions = _call_vision_segmentation_api(
                image, focus_mode=focus_mode, max_regions=max_regions, user_instruction=user_instruction
            )
            if ai_regions:
                logger.info(f"Vision model segmented {len(ai_regions)} visual/calculation regions.")
                regions.extend(ai_regions)
        except Exception as e:
            logger.warning(f"AI segmentation call notice: {e}. Activating OpenCV visual filter.")

    # Priority 3: Advanced OpenCV contour & entropy fallback with text filtering
    if not regions:
        logger.info("Running OpenCV visual-contour segmentation with text paragraph suppression...")
        regions = _segment_with_opencv_fallback(
            image, min_area=min_region_area, max_regions=max_regions, focus_mode=focus_mode
        )

    # Final post-processing: Clamp, remove duplicates, filter decorative headers, rank calculations
    cleaned_regions = _post_process_regions(regions, image, max_regions=max_regions, focus_mode=focus_mode)
    return cleaned_regions


def _extract_pdf_vector_and_image_regions(
    pdf_bytes: bytes,
    page_idx: int,
    target_image_size: Tuple[int, int],
    focus_mode: str = "balanced",
    max_regions: int = 6
) -> List[Dict[str, Any]]:
    """
    Extracts genuine mathematical formula blocks, derivations, and technical diagrams.
    Strictly excludes decorative textbook banners, publisher logos, and running headers/footers.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_idx >= len(doc):
        doc.close()
        return []

    page = doc.load_page(page_idx)
    page_rect = page.rect
    pw, ph = page_rect.width, page_rect.height
    tw, th = target_image_size
    scale_x = tw / pw
    scale_y = th / ph

    equation_rects = []
    diagram_rects = []

    # 1. PRIMARY TARGET: Detect Mathematical Formulas, Derivations & Calculations
    # In academic textbooks, equations contain mathematical symbols, variables, equals signs, or numbering (1.1)
    MATH_CHARS = set("=+\\-*/^_{}∫∑∏√±×÷≈≠≤≥αβγδθλμπσωΔ∇∂|")
    for b in page.get_text("blocks"):
        # b: (x0, y0, x1, y1, text, block_no, block_type)
        if b[6] == 0:  # text block
            bx1 = int(b[0] * scale_x)
            by1 = int(b[1] * scale_y)
            bx2 = int(b[2] * scale_x)
            by2 = int(b[3] * scale_y)

            # Skip header margin (top 10%) and footer margin (bottom 8%)
            if by2 < th * 0.10 or by1 > th * 0.90:
                continue

            block_text = b[4].strip()
            math_count = sum(1 for c in block_text if c in MATH_CHARS)
            words = block_text.split()
            word_count = len(words)

            # Check if this block is an equation, calculation step, or derivation
            is_calculation = False

            # Condition A: Explicit equals sign with concise mathematical structure
            if "=" in block_text and (math_count >= 1 or any(c.isdigit() for c in block_text)):
                if word_count < 35:
                    is_calculation = True

            # Condition B: High density of mathematical operators or Greek letters
            if math_count >= 2 and word_count < 40:
                is_calculation = True

            # Condition C: Equation numbering pattern like (1), (2.3), [4], Eq. 5
            import re
            if re.search(r"\(\d+(\.\d+)?\)|Eq\.\s*\d+", block_text) and math_count >= 1:
                is_calculation = True

            if is_calculation and (bx2 - bx1) > 40 and (by2 - by1) > 16:
                equation_rects.append((bx1, by1, bx2, by2, "equation", "Mathematical Formula / Calculation"))

    # 2. SECONDARY TARGET: Technical Diagrams, Plots & Circuits (Vector Drawings)
    drawings = page.get_drawings()
    drawing_boxes = []
    for d in drawings:
        r = d.get("rect")
        if r and r.width > 20 and r.height > 20:
            # Skip full-page background or full-width decorative dividing lines
            if r.width > pw * 0.92 or r.height < 5:
                continue
            dx1 = int(r.x0 * scale_x)
            dy1 = int(r.y0 * scale_y)
            dx2 = int(r.x1 * scale_x)
            dy2 = int(r.y1 * scale_y)

            # Skip top and bottom header/footer margin decorations
            if dy2 < th * 0.10 or dy1 > th * 0.90:
                continue

            drawing_boxes.append([dx1, dy1, dx2, dy2])

    if drawing_boxes:
        clustered = _cluster_bounding_boxes(drawing_boxes, proximity=45)
        for box in clustered:
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            # Must be substantial size to be a real scientific diagram
            if bw * bh > 5000 and bw > 70 and bh > 50:
                diagram_rects.append((box[0], box[1], box[2], box[3], "diagram", "Technical Diagram / Schematic"))

    # 3. Embedded Raster Images (FILTER OUT textbook publisher logos, icons, decorative banners)
    for img_info in page.get_images():
        xref = img_info[0]
        for r in page.get_image_rects(xref):
            rx1 = int(r.x0 * scale_x)
            ry1 = int(r.y0 * scale_y)
            rx2 = int(r.x1 * scale_x)
            ry2 = int(r.y1 * scale_y)
            rw = rx2 - rx1
            rh = ry2 - ry1

            # FILTER 1: Exclude running headers & footers (publisher logos live here)
            if ry2 < th * 0.12 or ry1 > th * 0.88:
                continue

            # FILTER 2: Exclude small decorative icons / publisher stamps (< 100px)
            if rw < 100 or rh < 80:
                continue

            # FILTER 3: Exclude thin decorative horizontal divider ribbons
            if rw / float(max(1, rh)) > 6.0:
                continue

            # Genuine scientific visual or plot
            diagram_rects.append((rx1, ry1, rx2, ry2, "diagram", "Scientific Graphic / Plot"))

    doc.close()

    # Prioritize calculations first, then diagrams based on user focus mode
    raw_rects = []
    if focus_mode == "equations_only":
        raw_rects = equation_rects
    elif focus_mode == "diagrams_only":
        raw_rects = diagram_rects
    else:
        # Balanced: Calculations first!
        raw_rects = equation_rects + diagram_rects

    regions = []
    merged = _non_max_suppress_rects(raw_rects, iou_thresh=0.2)
    for idx, (x1, y1, x2, y2, cat, label) in enumerate(merged, start=1):
        pad = 12
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(tw, x2 + pad)
        cy2 = min(th, y2 + pad)
        regions.append({
            "id": idx,
            "box": [cx1, cy1, cx2, cy2],
            "crop": None,
            "category_hint": cat,
            "label": f"{label} #{idx}",
            "confidence": 0.95
        })

    return regions


def _call_vision_segmentation_api(
    image: Image.Image,
    focus_mode: str = "balanced",
    max_regions: int = 6,
    user_instruction: str = ""
) -> List[Dict[str, Any]]:
    """
    Calls Multimodal Vision API (Muse Wonder) explicitly instructed to locate
    mathematical calculations, derivations, formulas, and scientific diagrams.
    Strictly forbids textbook decorative designs, publisher logos, and header/footer ornaments.
    Honors user custom instruction / directive if provided.
    """
    import requests
    import json

    b64_uri = image_to_base64_data_uri(image)
    url = f"{API_BASE_URL}/chat/completions"
    headers = get_headers()

    target_count = max(max_regions, 6)
    system_prompt = (
        "You are an expert technical visual segmenter and STEM document analyzer.\n"
        f"GOAL: Locate up to {target_count} genuine mathematical calculation blocks, equation derivations, formula steps, "
        "scientific plots, graphs, and circuit schematics.\n\n"
        "PRIMARY EXTRACTION TARGETS:\n"
        "1. Mathematical calculations, equation blocks, formulas, integrals, summations, matrices, and algebraic derivations.\n"
        "2. Technical diagrams, scientific graphs, plots, coordinate systems, and circuit schematics.\n\n"
        "CRITICAL NEGATIVE CONSTRAINTS (STRICTLY EXCLUDE):\n"
        "- DO NOT crop textbook decorative designs, publisher logos (e.g., Pearson, Wiley, McGraw-Hill, IEEE), chapter decorative banners, or page header/footer ornaments.\n"
        "- DO NOT crop plain paragraphs of narrative prose, reading text, or bullet points with no calculations.\n\n"
    )

    if user_instruction:
        system_prompt += (
            f"SPECIAL USER DIRECTIVE / FOCUS GOAL:\n"
            f"\"{user_instruction}\"\n"
            f"Strictly prioritize identifying regions and formulas relevant to this user directive.\n\n"
        )

    system_prompt += (
        "Return ONLY a valid JSON array of objects with keys:\n"
        "'box': [ymin, xmin, ymax, xmax] (normalized coordinates 0 to 1000),\n"
        "'category_hint': ('equation' | 'diagram' | 'chart' | 'table'),\n"
        "'label': 'Short title of what the calculation or visual represents'.\n"
        "If no formulas or scientific visuals exist, return []."
    )

    payload = {
        "model": MODEL_MUSE_WONDER,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": b64_uri}}
                ]
            }
        ],
        "max_tokens": 2500,
        "temperature": 0.1
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
    if resp.status_code != 200:
        raise RuntimeError(f"Vision segmentation error HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    raw_text = parse_response_content(data)

    cleaned = raw_text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = parts[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    items = json.loads(cleaned)
    if not isinstance(items, list):
        return []

    w, h = image.size
    results = []

    for idx, item in enumerate(items, start=1):
        raw_box = item.get("box", [])
        if len(raw_box) != 4:
            continue

        ymin, xmin, ymax, xmax = raw_box
        if max(raw_box) <= 1000:
            x1 = int(xmin * w / 1000.0)
            y1 = int(ymin * h / 1000.0)
            x2 = int(xmax * w / 1000.0)
            y2 = int(ymax * h / 1000.0)
        else:
            x1, y1, x2, y2 = int(xmin), int(ymin), int(xmax), int(ymax)

        # Skip textbook header and footer margins
        if y2 < h * 0.08 or y1 > h * 0.92:
            continue

        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(x1 + 10, min(x2, w)), max(y1 + 10, min(y2, h))

        cat = item.get("category_hint", "equation").lower()
        if cat not in ("equation", "diagram", "chart", "table", "visual_element"):
            cat = "equation"

        crop = image.crop((x1, y1, x2, y2))
        results.append({
            "id": idx,
            "box": [x1, y1, x2, y2],
            "crop": crop,
            "category_hint": cat,
            "label": item.get("label", f"Asset #{idx}"),
            "confidence": 0.95
        })

    return results


def _segment_with_opencv_fallback(
    image: Image.Image,
    min_area: int = 3500,
    max_regions: int = 6,
    focus_mode: str = "balanced"
) -> List[Dict[str, Any]]:
    """
    OpenCV visual region segmentation with strict text-paragraph suppression.
    Filters out horizontal text lines and targets genuine 2D diagrams, plots,
    and isolated mathematical expressions.
    """
    w, h = image.size
    total_area = w * h

    if cv2 is None:
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        crop = image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
        return [{
            "id": 1,
            "box": [margin_x, margin_y, w - margin_x, h - margin_y],
            "crop": crop,
            "category_hint": "diagram",
            "label": "Primary Visual Focus",
            "confidence": 0.85
        }]

    img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold
    mean_val = np.mean(gray)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV if mean_val > 127 else cv2.THRESH_BINARY,
        25, 10
    )

    # Morphological gradient to highlight high-entropy graphic boundaries & diagrams
    # Square kernel connects 2D graphics without biasing horizontally across text lines
    sq_size = max(5, int(min(w, h) * 0.012))
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (sq_size, sq_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_sq)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[int, int, int, int, float, str]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        if area < min_area or area > total_area * 0.85:
            continue
        if cw < 40 or ch < 30:
            continue

        # Extract ROI to evaluate text density vs graphic entropy
        roi = thresh[y:y+ch, x:x+cw]
        
        # Text-paragraph suppression check:
        # Paragraphs have uniform horizontal lines of small text with periodic peaks
        is_text, cat_hint = _classify_roi_visual_nature(roi, cw, ch)
        if is_text:
            continue  # REJECT plain text paragraph!

        # Score based on graphic visual entropy
        entropy_score = cv2.countNonZero(roi) / float(area)
        candidates.append((x, y, cw, ch, area, cat_hint))

    # Sort candidates by area and select top non-overlapping regions
    candidates.sort(key=lambda c: c[4], reverse=True)
    merged_boxes = _non_max_suppress_boxes(candidates, iou_thresh=0.25)

    regions: List[Dict[str, Any]] = []
    padding = 12

    for idx, (x, y, cw, ch, cat_hint) in enumerate(merged_boxes[:max_regions], start=1):
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)

        crop = image.crop((x1, y1, x2, y2))
        regions.append({
            "id": idx,
            "box": [x1, y1, x2, y2],
            "crop": crop,
            "category_hint": cat_hint,
            "label": f"{cat_hint.capitalize()} Asset #{idx}",
            "confidence": 0.90
        })

    return regions


def _classify_roi_visual_nature(roi: np.ndarray, w: int, h: int) -> Tuple[bool, str]:
    """
    Analyzes an image region to determine whether it is a regular reading text paragraph
    (which must be excluded) or a genuine diagram/formula (which must be kept).
    
    Returns:
        (is_plain_text_to_exclude: bool, category_hint: str)
    """
    if roi.size == 0:
        return True, "text"

    # 1. Aspect ratio analysis
    ar = w / float(max(1, h))

    # 2. Horizontal projection profile: sum dark pixels across rows
    row_sums = np.sum(roi > 0, axis=1)
    # Count transitions from zero to non-zero (estimating text line count)
    line_threshold = np.max(row_sums) * 0.15
    active_rows = row_sums > line_threshold

    # Find distinct line transitions
    transitions = np.diff(active_rows.astype(int))
    line_count = np.sum(transitions == 1)

    # 3. If it has 4+ repeated horizontal text lines of thin height, it is a text paragraph
    if line_count >= 4 and ar > 1.4:
        # Check if individual line thickness is typical of small font (10-25px)
        avg_line_height = h / float(max(1, line_count))
        if avg_line_height < 32:
            return True, "text"  # Plain text paragraph!

    # 4. If it has very wide aspect ratio with multiple lines and low vertical variance
    if ar > 3.5 and line_count >= 2:
        return True, "text"

    # 5. Check for standalone mathematical formula:
    # Formulas are often isolated, moderate width, and contain distinct horizontal lines (fraction bars, equals)
    if (ar > 2.0 and h < 140) or (line_count <= 2 and ar > 1.5):
        return False, "equation"

    # 6. Otherwise it is a diagram, flowchart, graph, or schematic
    return False, "diagram"


def _cluster_bounding_boxes(boxes: List[List[int]], proximity: int = 35) -> List[List[int]]:
    """Groups adjacent or overlapping drawing bounding boxes into unified diagram clusters."""
    if not boxes:
        return []

    clusters: List[List[int]] = []
    for b in boxes:
        matched = False
        for c in clusters:
            # Check if box b is within proximity distance of cluster c
            if (b[0] <= c[2] + proximity and b[2] >= c[0] - proximity and
                b[1] <= c[3] + proximity and b[3] >= c[1] - proximity):
                c[0] = min(c[0], b[0])
                c[1] = min(c[1], b[1])
                c[2] = max(c[2], b[2])
                c[3] = max(c[3], b[3])
                matched = True
                break
        if not matched:
            clusters.append(list(b))

    return clusters


def _non_max_suppress_rects(
    rects: List[Tuple[int, int, int, int, str, str]],
    iou_thresh: float = 0.25
) -> List[Tuple[int, int, int, int, str, str]]:
    """Suppresses heavily overlapping bounding rectangles."""
    if not rects:
        return []

    # Sort by area descending
    sorted_rects = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    kept = []

    for r in sorted_rects:
        box1 = [r[0], r[1], r[2], r[3]]
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        overlap = False

        for k in kept:
            box2 = [k[0], k[1], k[2], k[3]]
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2], box2[2])
            inter_y2 = min(box1[3], box2[3])

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                iou = inter_area / float(area1 + area2 - inter_area)
                if iou > iou_thresh:
                    overlap = True
                    break

        if not overlap:
            kept.append(r)

    return kept


def _non_max_suppress_boxes(
    candidates: List[Any],
    iou_thresh: float = 0.3
) -> List[Tuple[int, int, int, int, str]]:
    """Suppresses heavily overlapping candidate boxes."""
    boxes = [[c[0], c[1], c[0] + c[2], c[1] + c[3], c[5]] for c in candidates]
    if not boxes:
        return []

    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append((current[0], current[1], current[2] - current[0], current[3] - current[1], current[4]))

        filtered = []
        for b in boxes:
            xx1 = max(current[0], b[0])
            yy1 = max(current[1], b[1])
            xx2 = min(current[2], b[2])
            yy2 = min(current[3], b[3])

            inter_w = max(0, xx2 - xx1)
            inter_h = max(0, yy2 - yy1)
            inter_area = inter_w * inter_h

            area_curr = (current[2] - current[0]) * (current[3] - current[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union_area = area_curr + area_b - inter_area

            iou = inter_area / float(union_area) if union_area > 0 else 0
            if iou < iou_thresh:
                filtered.append(b)
        boxes = filtered

    return keep


def _post_process_regions(
    regions: List[Dict[str, Any]],
    image: Image.Image,
    max_regions: int = 6,
    focus_mode: str = "balanced"
) -> List[Dict[str, Any]]:
    """
    Ensures all crops are populated, boundaries clamped, and strictly non-empty.
    Filters out decorative textbook header/footer ribbons and prioritizes calculations.
    """
    w, h = image.size
    final_list = []
    seen_boxes = []

    # Sort candidates: prioritize equations and calculations first
    sorted_regions = sorted(
        regions,
        key=lambda r: 0 if r.get("category_hint") == "equation" else 1
    )

    for reg in sorted_regions:
        box = reg.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 10, min(x2, w))
        y2 = max(y1 + 10, min(y2, h))

        # Filter out textbook margin ornaments (top 8% and bottom 8%)
        if y2 < h * 0.08 or y1 > h * 0.92:
            continue

        cat = reg.get("category_hint", "diagram")
        if cat in ("text", "plain_text"):
            continue  # Exclude any plain text region

        # Apply user focus mode
        if focus_mode == "equations_only" and cat != "equation":
            continue
        if focus_mode == "diagrams_only" and cat == "equation":
            continue

        # Check for near duplicate box
        is_dup = False
        for sb in seen_boxes:
            if abs(x1 - sb[0]) < 25 and abs(y1 - sb[1]) < 25 and abs(x2 - sb[2]) < 25 and abs(y2 - sb[3]) < 25:
                is_dup = True
                break
        if is_dup:
            continue

        seen_boxes.append([x1, y1, x2, y2])
        crop = reg.get("crop")
        if crop is None or not isinstance(crop, Image.Image):
            crop = image.crop((x1, y1, x2, y2))

        final_list.append({
            "id": len(final_list) + 1,
            "box": [x1, y1, x2, y2],
            "crop": crop,
            "category_hint": cat,
            "label": reg.get("label", f"Asset #{len(final_list) + 1}"),
            "confidence": reg.get("confidence", 0.95)
        })

        if len(final_list) >= max_regions:
            break

    return final_list


def draw_bounding_boxes(
    image: Image.Image,
    regions: List[Dict[str, Any]],
    show_labels: bool = True
) -> Image.Image:
    """Renders high-visibility, color-coded bounding overlays on genuine visual elements."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    color_map = {
        "diagram": (37, 99, 235),      # Meta Indigo / Blue
        "equation": (147, 51, 234),    # Electric Purple
        "chart": (16, 185, 129),       # Emerald Green
        "table": (14, 165, 233),       # Cyan
        "visual_element": (245, 158, 11) # Amber
    }

    for region in regions:
        box = region.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        category = region.get("category_hint", "diagram")
        color = color_map.get(category, (37, 99, 235))

        line_width = max(3, int(min(image.size) * 0.005))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        if show_labels:
            tag = f"#{region['id']} [{category.upper()}]"
            banner_height = 24
            banner_width = len(tag) * 9 + 8
            draw.rectangle([x1, max(0, y1 - banner_height), x1 + banner_width, y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - banner_height) + 4), tag, fill=(255, 255, 255))

    return overlay


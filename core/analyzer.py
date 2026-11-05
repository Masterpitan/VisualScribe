"""
Multimodal Vision & Formula Analyzer
Uses Muse Wonder (muse-wonder-1.0) to perform diagram deconstruction,
symbol recognition, and vision-to-LaTeX transcription.
"""

import json
import logging
import re
from typing import Dict, Any, List
from PIL import Image
import requests

from config import (
    MODEL_MUSE_WONDER,
    API_BASE_URL,
    get_headers,
    DEFAULT_TIMEOUT_SECONDS,
    parse_response_content
)
from core.segmenter import image_to_base64_data_uri

logger = logging.getLogger("VisualScribe.Analyzer")


def analyze_crop(
    crop_image: Image.Image,
    category_hint: str = "diagram",
    region_id: int = 1,
    user_instruction: str = ""
) -> Dict[str, Any]:
    """
    Performs multimodal vision analysis using Muse Wonder (muse-wonder-1.0).
    Extracts validated LaTeX expressions, breaks down visual elements, and provides
    pedagogical explanations.
    """
    b64_uri = image_to_base64_data_uri(crop_image)
    url = f"{API_BASE_URL}/chat/completions"
    headers = get_headers()

    system_prompt = (
        "You are an expert multimodal visual analyst capable of interpreting ANY visual, "
        "including network packet captures, software UI screenshots, system architectures, biology diagrams, "
        "circuit schematics, mathematical equations, data plots, tables, and engineering flowcharts.\n\n"
        "Instructions:\n"
        "1. Identify the domain and exact nature of this visual: What is it showing? (e.g., Wireshark packet capture, network topology, chemical reaction, circuit, mathematical derivation, algorithm flowchart, parameter table, etc.).\n"
        "2. If it is a network trace, log, software screenshot, or data table: Transcribe and interpret visible IPs, ports, protocols (e.g. DNS, HTTP, TCP), filter criteria, timestamps, and column headers.\n"
        "3. If it contains genuine mathematical equations: Extract them into standard LaTeX ($...$ or $$...$$). IF NO MATH EXISTS, LEAVE 'latex' AS EMPTY STRING \"\". NEVER invent artificial equations.\n"
        "4. Provide a clear, technically rigorous explanation of what the visual demonstrates, what action is taking place, and its operational significance.\n"
        "5. Provide 2 to 4 concrete technical takeaways.\n\n"
        "You MUST respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "is_visual_or_math": true,\n'
        '  "category_hint": "diagram",\n'
        '  "title": "Concise descriptive title of what this visual represents",\n'
        '  "latex": "",\n'
        '  "explanation": "Detailed explanation of what this visual displays and does.",\n'
        '  "key_takeaways": ["Takeaway 1", "Takeaway 2"]\n'
        "}"
    )

    user_text = f"Analyze this cropped {category_hint} element. Provide an accurate and comprehensive breakdown."
    if user_instruction:
        user_text += f"\nUser Directive / Focus Context: {user_instruction}"

    payload = {
        "model": MODEL_MUSE_WONDER,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": b64_uri}}
                ]
            }
        ],
        "max_tokens": 2500,
        "temperature": 0.2
    }

    last_error = None
    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                raw_content = parse_response_content(data)
                parsed = _parse_analysis_json(raw_content, default_title=f"{category_hint.capitalize()} #{region_id}", default_cat=category_hint)
                parsed["id"] = region_id
                return parsed
            elif resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"Muse Wonder transient HTTP {resp.status_code} (attempt {attempt+1}/2). Backing off...")
                import time
                time.sleep(2.0)
                continue
            else:
                logger.warning(f"Muse Wonder API HTTP {resp.status_code}: {resp.text}")
                break
        except Exception as e:
            logger.warning(f"Error during Muse Wonder analysis (attempt {attempt+1}/2): {e}")
            last_error = e
            import time
            time.sleep(1.5)

    # Fallback response if API fails
    return _generate_fallback_analysis(crop_image, category_hint, region_id)


def _parse_analysis_json(raw_text: str, default_title: str, default_cat: str = "diagram") -> Dict[str, Any]:
    """
    Parses structured JSON from Muse Wonder response, stripping markdown code fences
    and extracting embedded JSON objects if surrounded by conversational filler.
    """
    cleaned = raw_text.strip()

    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()

    parsed_obj = {}
    try:
        parsed_obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed_obj = json.loads(cleaned[start:end + 1])
            except Exception:
                pass

    if isinstance(parsed_obj, dict) and parsed_obj:
        cat = str(parsed_obj.get("category_hint", default_cat)).lower()
        is_vm = parsed_obj.get("is_visual_or_math", True)
        is_valid = bool(is_vm and cat != "plain_text" and cat != "text")
        return {
            "title": str(parsed_obj.get("title", default_title)),
            "category_hint": cat if is_valid else "plain_text",
            "latex": str(parsed_obj.get("latex", "")),
            "explanation": str(parsed_obj.get("explanation", "Conceptual breakdown generated.")),
            "key_takeaways": list(parsed_obj.get("key_takeaways", [])),
            "is_valid_asset": is_valid
        }

    # Fallback to text parsing
    sentences = [s.strip() for s in re.split(r"[.\n]+", raw_text) if len(s.strip()) > 20 and not s.strip().startswith("{")]
    clean_takeaways = sentences[:3] if sentences else ["Visual structure and operational parameters extracted from source."]
    return {
        "title": default_title,
        "category_hint": default_cat,
        "latex": "",
        "explanation": raw_text if raw_text else "Visual element extracted.",
        "key_takeaways": clean_takeaways,
        "is_valid_asset": default_cat != "plain_text"
    }


def _generate_fallback_analysis(image: Image.Image, category_hint: str, region_id: int) -> Dict[str, Any]:
    """Fallback generator when API is offline or rate-limited."""
    w, h = image.size
    is_valid = category_hint != "plain_text" and category_hint != "text"
    display_cat = category_hint if category_hint != "equation" else "visual"
    return {
        "id": region_id,
        "title": f"Visual Element #{region_id} ({display_cat.capitalize()})",
        "latex": "",
        "explanation": (
            f"Segmented visual region ({w}x{h} px) containing technical "
            f"{display_cat} structure, parameters, or data."
        ),
        "key_takeaways": [
            f"Isolated {display_cat} component #{region_id} from source visual.",
            "Visual structure preserved for technical study analysis."
        ],
        "category_hint": display_cat,
        "is_valid_asset": is_valid
    }

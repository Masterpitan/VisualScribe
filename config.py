"""
VisualScribe Configuration & Core API Client
Provides endpoint constants, model identifiers, exponential backoff retries,
and robust response parsing to prevent null/None returns from reasoning models.
"""

import os
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VisualScribe.Config")

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("META_API_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
META_API_KEY = os.getenv("META_API_KEY", "").strip()

# Model Slugs
MODEL_MUSE_WONDER = "muse-wonder-1.0"      # Multimodal analysis, diagram deconstruction & vision-to-LaTeX
MODEL_MUSE_SPARK = "muse-spark-1.3"        # High-speed conceptual reasoning & study summary compilation
MODEL_SAM = "sam-3.1"                      # Visual element segmentation and bounding boxes

# Document Processing Defaults
DEFAULT_RENDER_DPI = 200                   # DPI for PDF page to image rendering (200-300 standard)
DEFAULT_MAX_TOKENS = 2048                  # Generous token budget to accommodate reasoning tokens + content
DEFAULT_TIMEOUT_SECONDS = 35               # Snappy HTTP timeout (avoids long browser freezes)
MAX_RETRIES = 2                            # Fast-fail retries
BACKOFF_FACTOR = 1.5                       # Retry backoff multiplier


def get_headers() -> Dict[str, str]:
    """Return authorization headers for Meta API requests."""
    key = os.getenv("META_API_KEY", META_API_KEY)
    if not key:
        logger.warning("META_API_KEY is not set or empty in environment!")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }


def parse_response_content(response_json: Dict[str, Any]) -> str:
    """
    Universally extracts generated text from OpenAI-compatible and reasoning model responses.
    Handles:
      1. Plain string `choice['message']['content']`
      2. Structured content block arrays `[{'type': 'text', 'text': '...'}]`
      3. Reasoning models with extended `reasoning_content`
      4. Legacy completion format `choice['text']`
    """
    if not isinstance(response_json, dict):
        return ""

    choices = response_json.get("choices", [])
    if not choices:
        # Check for direct output or error field
        if "output" in response_json:
            return str(response_json["output"])
        return ""

    first_choice = choices[0]
    
    # 1. Standard ChatCompletion message dictionary
    if "message" in first_choice and isinstance(first_choice["message"], dict):
        msg = first_choice["message"]
        content = msg.get("content")

        if isinstance(content, str) and content.strip():
            return content.strip()
        elif isinstance(content, list):
            # Content block format (multimodal or structured text)
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            combined = "\n".join(p for p in parts if p)
            if combined.strip():
                return combined.strip()

        # If content was null/None, check for reasoning_content (common in reasoning models)
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            logger.info("Content was empty; returning extracted reasoning content.")
            return reasoning.strip()

    # 2. Legacy completion format
    if "text" in first_choice and isinstance(first_choice["text"], str):
        return first_choice["text"].strip()

    return ""


def call_chat_completion(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = MAX_RETRIES
) -> str:
    """
    Executes a chat completion call with automatic exponential backoff for rate limits (429/50x).
    Returns the parsed text output or raises an exception if all retries fail.
    """
    url = f"{API_BASE_URL}/chat/completions"
    headers = get_headers()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    last_error = None
    delay = 1.0

    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"Calling {model} (attempt {attempt}/{retries})...")
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

            # If rate-limited or transient server error, back off and retry
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"Transient HTTP {resp.status_code} from {model}: {resp.text[:200]}")
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
                continue

            if resp.status_code == 200:
                data = resp.json()
                content = parse_response_content(data)
                return content
            else:
                resp.raise_for_status()

        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            logger.warning(f"Request exception on attempt {attempt}: {e}")
            last_error = e
            if attempt < retries:
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                break

    raise RuntimeError(f"Failed to call {model} after {retries} attempts: {last_error}")


def verify_api_connection() -> Dict[str, Any]:
    """Quick diagnostic check to verify credentials and connectivity."""
    key = os.getenv("META_API_KEY", META_API_KEY)
    if not key:
        return {"status": False, "message": "META_API_KEY is missing in .env file."}
    try:
        content = call_chat_completion(
            model=MODEL_MUSE_SPARK,
            messages=[{"role": "user", "content": "Reply with 'OK'."}],
            max_tokens=300,
            timeout=15,
            retries=1
        )
        return {"status": True, "message": f"Connection verified. Spark replied: {content[:30]}"}
    except Exception as e:
        return {"status": False, "message": str(e)}

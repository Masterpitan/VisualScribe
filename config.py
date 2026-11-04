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
META_API_KEY = os.getenv("META_API_KEY", "").strip().strip('"').strip("'")

# Baseten High-Throughput Inference Configuration
BASETEN_BASE_URL = os.getenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1").rstrip("/")
BASETEN_API_KEY = os.getenv("BASETEN_API_KEY", "").strip().strip('"').strip("'")
BASETEN_MODEL = os.getenv("BASETEN_MODEL", "meta-models/muse-spark-open-1026").strip().strip('"').strip("'")

# Active Provider: "baseten" or "meta"
raw_provider = os.getenv("LLM_PROVIDER", "baseten").split("#")[0].strip().strip('"').strip("'").lower()
DEFAULT_LLM_PROVIDER = "baseten" if (raw_provider == "baseten" or (not META_API_KEY and BASETEN_API_KEY)) else "meta"

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


def get_baseten_headers(use_bearer: bool = False) -> Dict[str, str]:
    """Return authorization headers for Baseten OpenAI-compatible requests."""
    key = os.getenv("BASETEN_API_KEY", BASETEN_API_KEY).strip().strip('"').strip("'")
    if not key:
        logger.warning("BASETEN_API_KEY is not set or empty in environment!")
    auth_prefix = "Bearer" if use_bearer else "Api-Key"
    return {
        "Authorization": f"{auth_prefix} {key}",
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
    model: Optional[str] = None,
    messages: List[Dict[str, Any]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = MAX_RETRIES,
    provider: Optional[str] = None
) -> str:
    """
    Executes a chat completion call with automatic exponential backoff for rate limits.
    Supports routing to either Baseten (High-Throughput) or Meta Model API with automatic failover.
    """
    if messages is None:
        messages = []

    active_provider = (provider or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)).split("#")[0].strip().strip('"').strip("'").lower()

    if active_provider == "baseten":
        target_url = f"{BASETEN_BASE_URL}/chat/completions"
        target_model = model or BASETEN_MODEL
        headers = get_baseten_headers(use_bearer=False)
        provider_name = f"Baseten [{target_model}]"
    else:
        target_url = f"{API_BASE_URL}/chat/completions"
        target_model = model or MODEL_MUSE_SPARK
        headers = get_headers()
        provider_name = f"Meta [{target_model}]"

    payload = {
        "model": target_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    last_error = None
    delay = 1.0

    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"Calling {provider_name} (attempt {attempt}/{retries})...")
            resp = requests.post(target_url, headers=headers, json=payload, timeout=timeout)

            # If 401 on Baseten with Api-Key, try Bearer
            if resp.status_code == 401 and active_provider == "baseten" and headers["Authorization"].startswith("Api-Key"):
                logger.info("Baseten returned 401 for Api-Key; retrying with Bearer header...")
                headers = get_baseten_headers(use_bearer=True)
                resp = requests.post(target_url, headers=headers, json=payload, timeout=timeout)

            # If rate-limited or transient server error, back off and retry
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"Transient HTTP {resp.status_code} from {provider_name}: {resp.text[:200]}")
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
            logger.warning(f"Request exception on attempt {attempt} for {provider_name}: {e}")
            last_error = e
            if attempt < retries:
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                break

    # Automated Failover: If Baseten failed and Meta credentials exist, attempt Meta fallback
    if active_provider == "baseten" and META_API_KEY:
        logger.warning(f"Baseten failed ({last_error}). Triggering automated failover to Meta Model API...")
        try:
            return call_chat_completion(
                model=MODEL_MUSE_SPARK,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                retries=1,
                provider="meta"
            )
        except Exception as fb_err:
            logger.error(f"Fallback to Meta also failed: {fb_err}")

    # Automated Failover: If Meta failed and Baseten credentials exist, attempt Baseten fallback
    elif active_provider == "meta" and BASETEN_API_KEY:
        logger.warning(f"Meta failed ({last_error}). Triggering automated failover to Baseten...")
        try:
            return call_chat_completion(
                model=BASETEN_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                retries=1,
                provider="baseten"
            )
        except Exception as fb_err:
            logger.error(f"Fallback to Baseten also failed: {fb_err}")

    raise RuntimeError(f"Failed to call {provider_name} after {retries} attempts: {last_error}")


def verify_api_connection() -> Dict[str, Any]:
    """Quick diagnostic check to verify credentials and connectivity for Meta."""
    key = os.getenv("META_API_KEY", META_API_KEY)
    if not key:
        return {"status": False, "message": "META_API_KEY is missing in .env file."}
    try:
        content = call_chat_completion(
            model=MODEL_MUSE_SPARK,
            messages=[{"role": "user", "content": "Reply with 'OK'."}],
            max_tokens=300,
            timeout=15,
            retries=1,
            provider="meta"
        )
        return {"status": True, "message": f"Connection verified. Spark replied: {content[:30]}"}
    except Exception as e:
        return {"status": False, "message": str(e)}


def verify_baseten_connection() -> Dict[str, Any]:
    """Quick diagnostic check to verify Baseten credentials, endpoint, and round-trip speed."""
    key = os.getenv("BASETEN_API_KEY", BASETEN_API_KEY)
    if not key:
        return {"status": False, "message": "BASETEN_API_KEY is missing in .env file."}
    start = time.time()
    try:
        content = call_chat_completion(
            model=BASETEN_MODEL,
            messages=[{"role": "user", "content": "Reply with 'Baseten Connected successfully'."}],
            max_tokens=200,
            timeout=25,
            retries=1,
            provider="baseten"
        )
        elapsed = round(time.time() - start, 2)
        return {
            "status": True,
            "elapsed_seconds": elapsed,
            "message": f"Baseten verified in {elapsed}s. Model: {BASETEN_MODEL}. Output: '{content[:50]}'"
        }
    except Exception as e:
        return {"status": False, "message": str(e)}

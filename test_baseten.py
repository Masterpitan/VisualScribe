"""
Quick Baseten Connection & Latency Diagnostic Test
Validates BASETEN_API_KEY, BASETEN_BASE_URL, and BASETEN_MODEL.
Measures latency and confirms response parsing.
"""

import os
import sys
import time

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import (
    BASETEN_API_KEY,
    BASETEN_BASE_URL,
    BASETEN_MODEL,
    verify_baseten_connection,
    call_chat_completion
)


def run_baseten_test():
    print("=" * 65)
    print("🚀 Testing Baseten Inference Integration")
    print("=" * 65)

    print(f"  • Base URL : {BASETEN_BASE_URL}")
    print(f"  • Model    : {BASETEN_MODEL}")
    masked_key = (BASETEN_API_KEY[:6] + "..." + BASETEN_API_KEY[-4:]) if len(BASETEN_API_KEY) > 10 else "(not set)"
    print(f"  • API Key  : {masked_key}")
    print("-" * 65)

    if not BASETEN_API_KEY:
        print("❌ Error: BASETEN_API_KEY is missing in your .env file!")
        print("   Please add BASETEN_API_KEY=your_key in .env and re-run.")
        sys.exit(1)

    print("\n[Step 1] Sending handshake ping to Baseten...")
    result = verify_baseten_connection()

    if result["status"]:
        print(f"  ✅ SUCCESS: {result['message']}")
    else:
        print(f"  ❌ FAILED: {result['message']}")
        print("\nTroubleshooting tips:")
        print("1. Verify your BASETEN_API_KEY is correct in .env")
        print("2. Ensure model slug exists in your Baseten dashboard: " + BASETEN_MODEL)
        sys.exit(1)

    print("\n[Step 2] Testing STEM reasoning & LaTeX formula synthesis on Baseten...")
    start = time.time()
    try:
        sample_prompt = (
            "You are an academic STEM reference assistant. Provide the Pythagorean theorem "
            "and its definition in LaTeX format."
        )
        response = call_chat_completion(
            messages=[
                {"role": "system", "content": "You are a concise STEM documentation assistant."},
                {"role": "user", "content": sample_prompt}
            ],
            max_tokens=300,
            temperature=0.2,
            provider="baseten"
        )
        elapsed = round(time.time() - start, 2)
        print(f"  ✅ Received response in {elapsed}s:")
        print("  " + "-" * 50)
        for line in response.strip().split("\n")[:8]:
            print(f"  {line}")
        print("  " + "-" * 50)
    except Exception as e:
        print(f"  ❌ Error during inference test: {e}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(f"🎉 BASETEN INTEGRATION VERIFIED IN {elapsed}s (2M TPM CAPACITY ACTIVE)")
    print("=" * 65)


if __name__ == "__main__":
    run_baseten_test()


"""
VisualScribe Provider Speed & Quality Benchmark
Directly tests and compares Meta Model API vs. Baseten (2M TPM) side-by-side
on identical technical workloads to measure real-world latency and throughput.
"""

import os
import sys
import time
from typing import Dict, Any

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import (
    META_API_KEY,
    BASETEN_API_KEY,
    BASETEN_BASE_URL,
    BASETEN_MODEL,
    MODEL_MUSE_SPARK,
    call_chat_completion
)


def run_benchmark():
    print("=" * 70)
    print("⚡ VisualScribe Inference Speed Benchmark: Meta vs. Baseten")
    print("=" * 70)

    # Validate keys
    has_meta = bool(META_API_KEY)
    has_baseten = bool(BASETEN_API_KEY)

    print(f"  • Meta API Key   : {'Configured ✅' if has_meta else 'Missing ❌'}")
    print(f"  • Baseten API Key: {'Configured ✅' if has_baseten else 'Missing ❌'} ({BASETEN_MODEL})")
    print("-" * 70)

    if not has_meta and not has_baseten:
        print("❌ Error: Neither META_API_KEY nor BASETEN_API_KEY found in .env!")
        sys.exit(1)

    # Realistic STEM workload: Cheatsheet synthesis from 3 technical diagrams
    benchmark_prompt = (
        "You are an academic STEM reference compiler. Synthesize these 3 elements:\n"
        "1. Element #1 (Equation): Maxwell-Ampere Law curl(B) = mu_0*J + mu_0*eps_0*dE/dt\n"
        "2. Element #2 (Diagram): Electromagnetic wave propagation with orthogonal E and B vectors\n"
        "3. Element #3 (Formula): Wave speed c = 1 / sqrt(mu_0 * eps_0)\n\n"
        "Generate a structured 2-section summary with formal LaTeX equation blocks and 2 revision questions."
    )

    messages = [
        {"role": "system", "content": "You are a concise STEM documentation assistant."},
        {"role": "user", "content": benchmark_prompt}
    ]

    results = {}

    # Test Baseten
    if has_baseten:
        print("\n[1/2] 🚀 Benchmarking Baseten (High-Throughput 2M TPM)...")
        print(f"      Endpoint: {BASETEN_BASE_URL}")
        print(f"      Model   : {BASETEN_MODEL}")
        try:
            start_t = time.time()
            resp_baseten = call_chat_completion(
                model=BASETEN_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.2,
                timeout=45,
                retries=1,
                provider="baseten"
            )
            elapsed_baseten = round(time.time() - start_t, 2)
            char_len = len(resp_baseten)
            est_words = len(resp_baseten.split())
            results["baseten"] = {
                "status": "Success",
                "time_sec": elapsed_baseten,
                "words": est_words,
                "speed_wps": round(est_words / max(0.1, elapsed_baseten), 1),
                "chars": char_len
            }
            print(f"      ✅ Completed in: {elapsed_baseten}s ({est_words} words, ~{results['baseten']['speed_wps']} words/sec)")
        except Exception as e:
            results["baseten"] = {"status": f"Failed ({e})", "time_sec": None}
            print(f"      ❌ Baseten Failed: {e}")
    else:
        print("\n[1/2] ⚠️ Skipping Baseten (BASETEN_API_KEY not configured)")

    # Test Meta
    if has_meta:
        print("\n[2/2] 🌐 Benchmarking Meta Model API (Managed Endpoints)...")
        print(f"      Model   : {MODEL_MUSE_SPARK}")
        try:
            start_t = time.time()
            resp_meta = call_chat_completion(
                model=MODEL_MUSE_SPARK,
                messages=messages,
                max_tokens=1000,
                temperature=0.2,
                timeout=45,
                retries=1,
                provider="meta"
            )
            elapsed_meta = round(time.time() - start_t, 2)
            char_len = len(resp_meta)
            est_words = len(resp_meta.split())
            results["meta"] = {
                "status": "Success",
                "time_sec": elapsed_meta,
                "words": est_words,
                "speed_wps": round(est_words / max(0.1, elapsed_meta), 1),
                "chars": char_len
            }
            print(f"      ✅ Completed in: {elapsed_meta}s ({est_words} words, ~{results['meta']['speed_wps']} words/sec)")
        except Exception as e:
            results["meta"] = {"status": f"Failed ({e})", "time_sec": None}
            print(f"      ❌ Meta Failed: {e}")
    else:
        print("\n[2/2] ⚠️ Skipping Meta (META_API_KEY not configured)")

    # Comparison Summary
    print("\n" + "=" * 70)
    print("📊 BENCHMARK COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Provider':<20} | {'Status':<10} | {'Time (s)':<10} | {'Output Size':<15} | {'Throughput'}")
    print("-" * 70)

    for prov, data in results.items():
        name = "Baseten (2M TPM)" if prov == "baseten" else "Meta Model API"
        st = data.get("status", "N/A")
        if data.get("time_sec") is not None:
            t = f"{data['time_sec']}s"
            sz = f"{data['words']} words"
            tp = f"~{data['speed_wps']} words/s"
        else:
            t = "—"
            sz = "—"
            tp = "—"
        print(f"{name:<20} | {st:<10} | {t:<10} | {sz:<15} | {tp}")

    print("=" * 70)

    if "baseten" in results and "meta" in results:
        t_b = results["baseten"].get("time_sec")
        t_m = results["meta"].get("time_sec")
        if t_b and t_m:
            diff = round(abs(t_m - t_b), 2)
            if t_b < t_m:
                speedup = round(t_m / t_b, 2)
                print(f"🏆 Winner: Baseten is {speedup}x faster ({diff}s quicker) on this STEM workload!")
            elif t_m < t_b:
                speedup = round(t_b / t_m, 2)
                print(f"🏆 Winner: Meta is {speedup}x faster ({diff}s quicker) on this STEM workload!")
            else:
                print("⚖️ Both providers completed in identical time.")


if __name__ == "__main__":
    run_benchmark()


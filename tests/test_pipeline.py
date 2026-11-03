"""
End-to-End Pipeline Integration Test
Performs a headless test of the entire VisualScribe workflow:
1. Synthetic image generation (diagram + equation)
2. SAM / morphological region segmentation & cropping
3. Multimodal analysis & LaTeX transcription
4. High-yield summary synthesis
5. Active-recall quiz generation
6. Markdown and HTML export generation
"""

import os
import sys
from PIL import Image, ImageDraw

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import verify_api_connection
from core.segmenter import segment_visual_regions, draw_bounding_boxes
from core.analyzer import analyze_crop
from core.generator import (
    generate_cheatsheet_summary,
    generate_practice_quiz,
    generate_cheatsheet_and_quiz_unified
)
from core.exporter import export_study_guide_markdown, export_study_guide_html, export_study_guide_pdf


def create_test_slide() -> Image.Image:
    """Generates a synthetic technical diagram slide."""
    img = Image.new("RGB", (1000, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(0, 0), (1000, 60)], fill=(40, 50, 70))
    draw.text((30, 20), "Maxwell's Equations and Wave Solutions", fill=(255, 255, 255))

    # Box 1: Equation Block
    draw.rectangle([(50, 100), (450, 300)], outline=(147, 51, 234), width=3)
    draw.text((70, 120), "Ampere-Maxwell Law:", fill=(0, 0, 0))
    draw.text((70, 180), "curl(B) = mu_0 * J + mu_0 * eps_0 * dE/dt", fill=(0, 0, 0))
    draw.text((70, 240), "Describes displacement current & magnetic curl", fill=(100, 100, 100))

    # Box 2: Electromagnetic Wave Diagram
    draw.rectangle([(520, 100), (950, 480)], outline=(37, 99, 235), width=3)
    draw.text((540, 120), "Transverse EM Wave Propagation (k-vector || z)", fill=(0, 0, 0))
    draw.line([(550, 290), (920, 290)], fill=(120, 120, 120), width=2)
    draw.text((560, 250), "E-field (y-axis)", fill=(37, 99, 235))
    draw.text((560, 320), "B-field (x-axis)", fill=(225, 29, 72))

    return img


def run_pipeline_test():
    print("=" * 65)
    print("🚀 Running VisualScribe End-to-End Pipeline Test")
    print("=" * 65)

    # Step 0: Check API connectivity
    print("\n[Step 0] Testing API connectivity...")
    api_status = verify_api_connection()
    if api_status["status"]:
        print(f"  ✅ API Status: Connected ({api_status['message']})")
    else:
        print(f"  ⚠️ API Notice: {api_status['message']} (Pipeline will exercise resilient fallbacks)")

    # Step 1: Create synthetic test slide
    print("\n[Step 1] Creating synthetic technical slide...")
    test_slide = create_test_slide()
    print(f"  ✅ Slide generated: {test_slide.size[0]}x{test_slide.size[1]} px")

    # Step 2: Segment visual regions
    print("\n[Step 2] Segmenting visual regions (SAM / OpenCV Fallback)...")
    regions = segment_visual_regions(test_slide, max_regions=4, use_sam_api=False)
    assert len(regions) > 0, "No regions segmented!"
    print(f"  ✅ Successfully segmented {len(regions)} regions:")
    for reg in regions:
        print(f"     - Region #{reg['id']}: [{reg['category_hint']}] Box: {reg['box']}")

    # Step 3: Draw bounding boxes
    print("\n[Step 3] Rendering bounding box overlays...")
    annotated = draw_bounding_boxes(test_slide, regions)
    assert annotated.size == test_slide.size, "Annotated image size mismatch!"
    print("  ✅ Bounding overlays rendered.")

    # Step 4: Multimodal Analysis & Vision-to-LaTeX
    print("\n[Step 4] Running Multimodal Analysis (Muse Wonder)...")
    analyses = []
    for reg in regions:
        analysis = analyze_crop(reg["crop"], category_hint=reg["category_hint"], region_id=reg["id"])
        analyses.append(analysis)
        print(f"  ✅ Analysis for Region #{reg['id']}: '{analysis.get('title')}'")
        if analysis.get("latex"):
            print(f"     LaTeX: {analysis.get('latex')}")

    # Step 5: High-speed Unified Synthesis & Quiz Generation
    print("\n[Step 5] Synthesizing High-Yield Master Cheatsheet & Quiz (Unified Pass)...")
    summary, quiz = generate_cheatsheet_and_quiz_unified(analyses, document_title="Maxwell_Electromagnetism", num_questions=3)
    assert len(summary) > 50, "Summary generation was unexpectedly empty!"
    assert len(quiz) > 0, "Quiz questions were not generated!"
    print("  ✅ Master cheatsheet synthesized successfully.")
    print(f"  ✅ Generated {len(quiz)} active-recall drill questions:")
    for q in quiz:
        print(f"     Q{q['id']}: {q['question']}")

    # Step 6: Export Markdown, HTML & Printable PDF
    print("\n[Step 6] Testing Study Guide Exporter...")
    md_output = export_study_guide_markdown("Maxwell_Electromagnetism", summary, analyses, quiz)
    html_output = export_study_guide_html("Maxwell_Electromagnetism", summary, analyses, quiz)
    pdf_bytes = export_study_guide_pdf("Maxwell_Electromagnetism", summary, analyses, quiz, regions)

    assert len(md_output) > 100, "Markdown export failed!"
    assert len(html_output) > 100, "HTML export failed!"
    assert "MathJax" in html_output, "HTML export missing MathJax LaTeX renderer!"
    assert len(pdf_bytes) > 500, "PDF export failed!"
    assert pdf_bytes.startswith(b"%PDF-"), "Generated file is not a valid PDF binary!"
    print(f"  ✅ Markdown export: {len(md_output)} characters.")
    print(f"  ✅ HTML printable export: {len(html_output)} characters.")
    print(f"  ✅ Native PDF export: {len(pdf_bytes)} bytes.")

    # Step 7: Test Advanced Mathematical Formula Rendering in PDF
    print("\n[Step 7] Testing Mathematical Formula Typesetting in PDF...")
    math_summary = (
        "## 1. Executive Conceptual Overview\n"
        "where $a,b > 0$ are legs, $c > 0$ is the hypotenuse opposite $\\angle C$.\n\n"
        "## 2. Mathematical Formulations\n"
        "$$a = |BC|, \\quad b = |CA|, \\quad c = |AB|$$\n"
        "Theorem (Pythagoras):\n"
        "$$c^2 = a^2 + b^2 \\tag{1}$$\n"
        "Corollaries by algebraic rearrangement:\n"
        "$$c = \\sqrt{a^2 + b^2} \\tag{2}$$\n"
        "$$a^2 = c^2 - b^2, \\quad a = \\sqrt{c^2 - b^2} \\tag{3}$$\n"
        "$$b^2 = c^2 - a^2, \\quad b = \\sqrt{c^2 - a^2} \\tag{4}$$\n"
    )
    math_analyses = [
        {
            "id": 1,
            "title": "Pythagorean Metric Formulation",
            "category_hint": "equation",
            "latex": "c = \\sqrt{a^2 + b^2}",
            "explanation": "Fundamental Euclidean distance metric where $a^2 + b^2 = c^2$.",
            "key_takeaways": ["Valid for all Euclidean right triangles with $\\angle C = 90^\\circ$."],
            "is_valid_asset": True
        }
    ]
    math_quiz = [
        {
            "id": 1,
            "question": "What is the hypotenuse $c$ when $a = 3$ and $b = 4$ using $c = \\sqrt{a^2 + b^2}$?",
            "options": ["c = 5", "c = 7", "c = \\sqrt{7}", "c = 25"],
            "correct_index": 0,
            "explanation": "Evaluating $c = \\sqrt{3^2 + 4^2} = \\sqrt{9 + 16} = \\sqrt{25} = 5$."
        }
    ]
    math_pdf = export_study_guide_pdf("Pythagorean_Theorem", math_summary, math_analyses, math_quiz)
    assert len(math_pdf) > 1000, "Mathematical PDF compilation failed!"
    assert math_pdf.startswith(b"%PDF-"), "Mathematical PDF binary invalid!"
    print(f"  ✅ Mathematical Formula PDF compiled: {len(math_pdf)} bytes.")

    print("\n" + "=" * 65)
    print("🎉 ALL PIPELINE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline_test()

"""
VisualScribe Core Processing Package
"""

from core.pdf_loader import load_document, get_document_bytes
from core.segmenter import segment_visual_regions, draw_bounding_boxes
from core.analyzer import analyze_crop
from core.generator import (
    generate_cheatsheet_summary,
    generate_practice_quiz,
    generate_cheatsheet_and_quiz_unified
)
from core.exporter import export_study_guide_markdown, export_study_guide_html, export_study_guide_pdf

__all__ = [
    "load_document",
    "get_document_bytes",
    "segment_visual_regions",
    "draw_bounding_boxes",
    "analyze_crop",
    "generate_cheatsheet_summary",
    "generate_practice_quiz",
    "generate_cheatsheet_and_quiz_unified",
    "export_study_guide_markdown",
    "export_study_guide_html",
    "export_study_guide_pdf"
]

"""
PDF and Image Document Loader
Extracts individual pages and slides into crisp, high-resolution PIL Images in-memory.
"""

import io
import os
import logging
from typing import List, Union, Any
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from config import DEFAULT_RENDER_DPI

logger = logging.getLogger("VisualScribe.PDFLoader")


def get_document_bytes(file_source: Union[str, bytes, io.BytesIO, Any]) -> bytes:
    """Extracts raw bytes from file source safely."""
    if hasattr(file_source, "read"):
        file_bytes = file_source.read()
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        return file_bytes
    elif isinstance(file_source, str):
        with open(file_source, "rb") as f:
            return f.read()
    elif isinstance(file_source, bytes):
        return file_source
    elif isinstance(file_source, io.BytesIO):
        return file_source.getvalue()
    return b""


def load_document(file_source: Union[str, bytes, io.BytesIO, Any], dpi: int = DEFAULT_RENDER_DPI) -> List[Image.Image]:
    """
    Loads an uploaded file (PDF, PNG, JPG, JPEG, WEBP) and returns a list of RGB PIL Image objects.
    Each page of a multi-page PDF becomes a separate high-resolution image.
    
    Args:
        file_source: File path, bytes, or Streamlit UploadedFile object.
        dpi: Target DPI for rasterizing PDF pages (default 200 for crisp text and equations).
        
    Returns:
        List of PIL.Image.Image objects in RGB format.
    """
    filename = ""
    file_bytes = None

    # Handle Streamlit UploadedFile or file-like object
    if hasattr(file_source, "read") and hasattr(file_source, "name"):
        filename = getattr(file_source, "name", "").lower()
        file_bytes = file_source.read()
        # Reset file pointer if seekable
        if hasattr(file_source, "seek"):
            file_source.seek(0)
    elif isinstance(file_source, str):
        filename = file_source.lower()
        with open(file_source, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    elif isinstance(file_source, io.BytesIO):
        file_bytes = file_source.getvalue()
    else:
        raise ValueError(f"Unsupported file source type: {type(file_source)}")

    if not file_bytes:
        raise ValueError("File content is empty.")

    # Determine if PDF or image
    is_pdf = filename.endswith(".pdf") or file_bytes.startswith(b"%PDF-")

    if is_pdf:
        return _render_pdf_pages(file_bytes, dpi=dpi)
    else:
        return [_load_single_image(file_bytes)]


def _load_single_image(image_bytes: bytes) -> Image.Image:
    """Decodes raw image bytes into an RGB PIL Image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # Force load into memory
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ValueError(f"Could not load image format: {e}")


def _render_pdf_pages(pdf_bytes: bytes, dpi: int = DEFAULT_RENDER_DPI) -> List[Image.Image]:
    """
    Renders each page of a PDF into a high-resolution PIL Image using PyMuPDF (fitz).
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF (`fitz`) is required for PDF parsing. Install it with: pip install pymupdf"
        )

    images: List[Image.Image] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        logger.info(f"Rendering {total_pages} PDF pages at {dpi} DPI...")

        # Standard PDF base resolution is 72 DPI
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)

        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            # Render page to pixmap with anti-aliasing
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert pixmap bytes to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            logger.debug(f"Page {page_idx + 1}/{total_pages} rendered: {img.size}")

        doc.close()
        return images

    except Exception as e:
        logger.error(f"Error rendering PDF pages: {e}")
        raise RuntimeError(f"Failed to process PDF: {e}")

# Project Specification: VisualScribe

**Event:** Meta Global AI Developer Hackathon

**Track:** Indie Developer Track

**Theme:** Build for Your World

**Target Environment:** Python 3.10+ / Streamlit / Meta Model API / Modal Endpoints

## 1. Executive Summary & Objective

**VisualScribe** is an automated, multimodal study assistant and cheatsheet generator designed for students, researchers, and technical learners. It ingests dense academic lecture slides, textbook pages, and whiteboard snapshots, detects and segments visual elements (complex diagrams, circuits, flowcharts, and handwritten mathematical proofs) using **SAM 3.1**, and performs automated multimodal reasoning via **Muse Wonder** and **Muse Spark**.

The deliverable is a self-contained, interactive Python web application (built using Streamlit) that:

1. Accepts image or multi-page PDF uploads.

2. Extracts and segments visual diagrams and mathematical formulas cleanly.

3. Translates formulas into validated LaTeX syntax and generates conceptual step-by-step explanations.

4. Generates active-recall revision questions and flashcards.

5. Exports an assembled, formatted study guide (Markdown and PDF).

## 2. Technical Stack & Architecture

* **Language:** Python 3.10+

* **Frontend / UI:** Streamlit (no HTML/CSS overhead; reactive components, file uploader, side-by-side comparison columns)

* **Document & Image Processing:** `pdf2image`, `PyMuPDF` (`fitz`), `Pillow` (`PIL`), `numpy`, `opencv-python-headless`

* **Model APIs & Inference:**

  * **SAM 3.1 (`sam-3.1`):** Segmentation and automated bounding-box/mask extraction of visual elements from slide images.

  * **Muse Wonder (`muse-wonder-1.0`):** Multimodal analysis, diagram deconstruction, and vision-to-LaTeX transcription.

  * **Muse Spark (`muse-spark-1.3`):** High-speed conceptual synthesis, study summary compilation, and quiz generation.

  * **Meta Model API Endpoint:** Standard base URL: `https://api.meta.ai/v1` (OpenAI-compatible chat completions and vision schema).

* **Environment & Secrets:** `python-dotenv` reading `META_API_KEY`.

## 3. Verified API Model Catalog

From the verified API check, the key contains access to the following model slugs:

* `muse-wonder-1.0` (Multimodal analysis and vision reasoning)

* `sam-3.1` (Segmentation masks and visual region extraction)

* `muse-spark-1.3` (High-speed reasoning and text/LaTeX formatting)

* `muse-image-1.0` (Image generation/enhancement)

* `muse-voice-transcribe-1.0` (Audio/lecture transcription, optional expansion)

## 4. System Pipeline & Data Flow

```
[User Upload (PDF / PNG / JPG)]
              │
              ▼
   [Image / Page Extraction] (Pillow / PyMuPDF)
              │
              ▼
   [SAM 3.1 Segmentation]
   ├── Detects diagram/formula regions
   └── Crops individual bounding boxes -> [Cropped Image Assets]
              │
              ▼
   [Muse Wonder Multimodal Analysis]
   ├── Transcribes math/symbols to LaTeX ($...$, $$...$$)
   └── Explains diagram mechanics & relationships
              │
              ▼
   [Muse Spark Synthesis Engine]
   ├── Compiles summary cheatsheet
   └── Generates active-recall drill questions
              │
              ▼
   [Streamlit Interactive View]
   ├── Left: Original slide with bounding overlay
   ├── Right: Segmented visual asset + LaTeX + Concept Card
   └── Bottom: Full Markdown Export & Download

```

## 5. File Structure to Generate

```
VisualScribe/
├── .env                  # META_API_KEY="your-token"
├── .gitignore            # Ignores .venv, .env, __pycache__, temp_assets/
├── requirements.txt      # Core project dependencies
├── app.py                # Main Streamlit UI entry point
├── config.py             # Global configurations, endpoints, model slugs
├── core/
│   ├── __init__.py
│   ├── pdf_loader.py     # PDF to high-resolution image converter
│   ├── segmenter.py      # SAM 3.1 API wrapper and image cropping logic
│   ├── analyzer.py       # Muse Wonder multimodal transcription & LaTeX engine
│   ├── generator.py      # Muse Spark study guide & quiz synthesizer
│   └── exporter.py       # Markdown and printable PDF compiler
└── tests/
    └── test_pipeline.py  # End-to-end integration test on a sample image

```

## 6. Detailed Component Specifications

### 6.1 `requirements.txt`

```
streamlit>=1.35.0
requests>=2.31.0
pillow>=10.2.0
pymupdf>=1.24.0
pdf2image>=1.17.0
python-dotenv>=1.0.1
pydantic>=2.7.0

```

### 6.2 `config.py`

* Reads `META_API_KEY` via `python-dotenv`.

* Defines:

  ```
  API_BASE_URL = "https://api.meta.ai/v1"
  MODEL_MUSE_WONDER = "muse-wonder-1.0"
  MODEL_MUSE_SPARK = "muse-spark-1.3"
  MODEL_SAM = "sam-3.1"
  
  ```

* Sets request timeout parameters and default image resolutions (DPI 200–300 for slide rendering).

### 6.3 `core/pdf_loader.py`

* Implements `load_document(uploaded_file) -> list[PIL.Image]`:

  * Inspects file type. If image (`.png`, `.jpg`, `.jpeg`), converts to RGB `PIL.Image`.

  * If PDF (`.pdf`), utilizes `pymupdf` to render each page into high-resolution `PIL.Image` objects in-memory without saving unnecessary disk clutter.

### 6.4 `core/segmenter.py`

* Interacts with `sam-3.1` via the Meta Model API.

* Implements `segment_visual_regions(image: PIL.Image) -> list[dict]`:

  * Encodes the image to base64.

  * Sends to the SAM 3.1 endpoint to retrieve bounding boxes `[x1, y1, x2, y2]` or binary masks of high-entropy visual elements (diagrams, isolated equations, tables).

  * Handles response parsing and applies fallback thresholding/slicing if SAM returns coordinate bounding arrays.

  * Returns a list of dictionaries:

    ```
    {
        "id": int,
        "box": [x1, y1, x2, y2],
        "crop": PIL.Image,
        "category_hint": str # e.g. "diagram", "equation", "table"
    }
    
    ```

### 6.5 `core/analyzer.py`

* Interacts with `muse-wonder-1.0`.

* Implements `analyze_crop(crop_image: PIL.Image, category_hint: str) -> dict`:

  * Sends the cropped image as a base64 inline payload.

  * System prompt enforces structured output with strict formatting rules:

    * All mathematical formulas must be rendered in standard LaTeX enclosed in `$ ... $` for inline or `$$ ... $$` for display blocks.

    * Plain-English breakdown of the visual components (arrows, nodes, axis labels).

    * Conceptual takeaway: What does this visual prove or illustrate?

  * Returns a structured dictionary:

    ```
    {
        "title": str,
        "latex": str,
        "explanation": str,
        "key_takeaways": list[str]
    }
    
    ```

### 6.6 `core/generator.py`

* Interacts with `muse-spark-1.3`.

* Implements:

  1. `generate_cheatsheet_summary(analyses: list[dict]) -> str`:

     * Combines the individual crop explanations into a unified, high-yield study sheet.

  2. `generate_practice_quiz(analyses: list[dict]) -> list[dict]`:

     * Creates 3–5 active recall multiple choice or short-answer questions directly tied to the diagrams and equations parsed.

### 6.7 `core/exporter.py`

* Compiles Markdown output containing:

  * Header, source document metadata, date.

  * Extracted assets with transcribed LaTeX equations and breakdowns.

  * Practice quiz and answer key at the bottom.

* Provides a download button directly inside Streamlit for `.md` and HTML-printable format.

### 6.8 `app.py` (Streamlit UI)

* Page layout configured to `layout="wide"`.

* **Sidebar:** API status check, model selection confirmation, file uploader (PDF/PNG/JPG), and parameter controls (confidence threshold, quiz generation toggle).

* **Main View Tabs:**

  * **Tab 1: Visual Inspection & Segmentation:** Displays the slide image with interactive overlays of segmented regions alongside side-by-side cropped cards.

  * **Tab 2: Interactive Study Cheatsheet:** Formatted LaTeX equations (rendered natively via Streamlit's `st.latex`), explanations, and collapsible drill questions.

  * **Tab 3: Export & Raw Output:** Direct download triggers for Markdown and structured JSON logs.

## 7. Robust API Handling & Fallbacks

1. **Payload Extraction Resilience:**
   The client response parser must handle varied message structures:

   * Plain string `choice["message"]["content"]`

   * Content block arrays `[{"type": "text", "text": "..."}]`

   * Extended reasoning fields (`reasoning_content`) to prevent returning `None`.

2. **Rate-Limit & Retry Mechanics:**
   Implement an exponential backoff decorator for API calls using Python's standard `time` library to handle status codes `429` or transient `500` errors gracefully.

3. **Local Fallback for SAM Bounding Boxes:**
   If the experimental SAM 3.1 endpoint experiences rate limits during judging testing, include an OpenCV contour-based bounding-box fallback (`cv2.findContours` on high-contrast regions) to ensure the UI demo never fails or crashes.

## 8. Step-by-Step Task Instructions for Coding Agent

When implementing the project, proceed in the following order:

1. **Environment Setup:** Generate `requirements.txt`, `.env.example`, and `.gitignore`.

2. **Config & Client Helper:** Write `config.py` with verified response parsing logic to prevent `None` returns.

3. **Core Processing Modules:** Implement `core/pdf_loader.py`, `core/segmenter.py`, `core/analyzer.py`, `core/generator.py`, and `core/exporter.py`.

4. **Streamlit App Assembly:** Construct `app.py` with responsive column layouts and native LaTeX rendering.

5. **Self-Verification Test:** Create `tests/test_pipeline.py` to run a headless integration check on a sample image.
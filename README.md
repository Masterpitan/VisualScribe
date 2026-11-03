# 📐 VisualScribe

> **Automated Multimodal Study Assistant & Cheatsheet Generator**  
> *Built for the Meta Global AI Developer Hackathon — Indie Developer Track*

VisualScribe ingests dense academic lecture slides, textbook pages, and whiteboard snapshots, detects and segments visual diagrams and mathematical formulas using **SAM 3.1**, performs multimodal vision-to-LaTeX transcription with **Muse Wonder**, and synthesizes high-yield study cheatsheets and active-recall quizzes with **Muse Spark**.

---

## 🌟 Key Features

1. **Intelligent Visual Element Segmentation**:
   - Isolates complex flowcharts, circuit schematics, graphs, and mathematical formulas using **SAM 3.1** with high-contrast morphological OpenCV fallbacks.
2. **Vision-to-LaTeX & Conceptual Deconstruction**:
   - Uses **Muse Wonder (`muse-wonder-1.0`)** to transcribe handwriting and printed equations into validated standard LaTeX (`$...$`, `$$...$$`) and analyze diagram structural mechanics.
3. **High-Yield Synthesis & Active-Recall Quizzes**:
   - Uses **Muse Spark (`muse-spark-1.3`)** to generate executive study guides and interactive multiple-choice practice questions with instant grading and explanations.
4. **Interactive Multi-Tab Workspace (Streamlit)**:
   - **Tab 1:** Slide inspection with color-coded bounding overlays alongside cropped visual asset cards.
   - **Tab 2:** Interactive study cheatsheet with native LaTeX rendering and interactive drill quiz cards.
   - **Tab 3:** One-click exports to GitHub-flavored Markdown (`.md`) and publication-grade HTML printable to PDF via MathJax.

---

## 🛠️ Architecture & Data Pipeline

```
[User Upload (PDF / PNG / JPG)]
              │
              ▼
   [Document Loader] (PyMuPDF / Pillow)
              │
              ▼
   [SAM 3.1 Segmentation & Cropping]
   ├── Detects diagram & formula regions
   └── Auto OpenCV morphological contour fallback
              │
              ▼
   [Muse Wonder Multimodal Reasoning]
   ├── Vision-to-LaTeX transcription
   └── Diagram component & arrow mechanics
              │
              ▼
   [Muse Spark Synthesis Engine]
   ├── High-yield cheatsheet compilation
   └── Active-recall quiz generation
              │
              ▼
   [Interactive Streamlit Application]
   ├── Tab 1: Visual overlays & cropped cards
   ├── Tab 2: LaTeX formulas & interactive quiz
   └── Tab 3: Markdown & Printable PDF export
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure Environment
Ensure your `.env` contains your Meta API key:
```ini
META_API_KEY=your_meta_api_key_here
```

### 3. Run Pipeline Verification Test
Run the end-to-end integration test to verify segmentation, LaTeX transcription, synthesis, and export:
```powershell
python tests/test_pipeline.py
```

### 4. Launch the Streamlit App
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
*(Tip: If you don't have a PDF handy, click the **"Load Demo Slide"** button in the sidebar to test immediately with a Quantum Mechanics lecture slide!)*

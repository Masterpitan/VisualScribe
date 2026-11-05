"""
VisualScribe - Interactive Multimodal Study Assistant & Cheatsheet Generator
Built for the Meta Global AI Developer Hackathon.
Powered by SAM 3.1, Muse Wonder (muse-wonder-1.0), and Muse Spark (muse-spark-1.3).
"""

import os
import json
import io
from typing import List, Dict, Any
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Import core modules
from config import (
    META_API_KEY,
    BASETEN_API_KEY,
    BASETEN_MODEL,
    MODEL_MUSE_WONDER,
    MODEL_MUSE_SPARK,
    MODEL_SAM,
    verify_api_connection,
    verify_baseten_connection
)
from core.pdf_loader import load_document, get_document_bytes
from core.segmenter import segment_visual_regions, draw_bounding_boxes
from core.analyzer import analyze_crop
from core.generator import (
    generate_cheatsheet_summary,
    generate_practice_quiz,
    generate_cheatsheet_and_quiz_unified
)
from core.exporter import export_study_guide_markdown, export_study_guide_html, export_study_guide_pdf


# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VisualScribe | Multimodal AI Study Assistant",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, badges, and responsive cards
st.markdown("""
<style>
    /* Metric / Chip Styles */
    .chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .chip-blue { background: #dbeafe; color: #1e40af; }
    .chip-purple { background: #f3e8ff; color: #6b21a8; }
    .chip-green { background: #d1fae5; color: #065f46; }
    .chip-amber { background: #fef3c7; color: #92400e; }

    /* Visual Card Containers */
    .visual-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    .visual-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(49, 46, 129, 0.2);
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 32px;
        margin-bottom: 8px;
    }
    .hero-banner p {
        color: #e0e7ff;
        font-size: 16px;
        max-width: 800px;
        margin-bottom: 0;
    }

    /* Score feedback */
    .quiz-correct {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 12px;
        border-radius: 6px;
        margin-top: 8px;
        color: #065f46;
    }
    .quiz-incorrect {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 12px;
        border-radius: 6px;
        margin-top: 8px;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "pages" not in st.session_state:
    st.session_state.pages = []
if "selected_page_idx" not in st.session_state:
    st.session_state.selected_page_idx = 0
if "regions" not in st.session_state:
    st.session_state.regions = []
if "analyses" not in st.session_state:
    st.session_state.analyses = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = "Lecture_Slide"


# ---------------------------------------------------------------------------
# Helper: Create Synthetic Sample Slide
# ---------------------------------------------------------------------------
def create_sample_slide() -> Image.Image:
    """Generates a high-resolution sample technical slide for quick 1-click testing."""
    img = Image.new("RGB", (1200, 750), color=(250, 252, 255))
    draw = ImageDraw.Draw(img)

    # Slide Header
    draw.rectangle([(0, 0), (1200, 90)], fill=(30, 41, 59))
    draw.text((40, 28), "PHYS 401: Quantum Mechanics & Wave Equations", fill=(255, 255, 255))

    # Box 1: Time-Dependent Equation (Formula region)
    draw.rectangle([(60, 140), (550, 360)], outline=(147, 51, 234), width=3, fill=(255, 255, 255))
    draw.rectangle([(60, 140), (550, 180)], fill=(243, 232, 255))
    draw.text((80, 150), "1. Time-Dependent Schrodinger Equation", fill=(107, 33, 168))
    draw.text((100, 220), "i * h_bar * d(Psi)/dt = H * Psi", fill=(15, 23, 42))
    draw.text((100, 270), "Where H = - (h_bar^2 / 2m) * Del^2 + V(r, t)", fill=(71, 85, 105))

    # Box 2: Wavefunction Barrier Diagram (Visual Diagram region)
    draw.rectangle([(620, 140), (1140, 420)], outline=(37, 99, 235), width=3, fill=(255, 255, 255))
    draw.rectangle([(620, 140), (1140, 180)], fill=(219, 234, 254))
    draw.text((640, 150), "2. Quantum Tunneling Barrier Transmission", fill=(30, 64, 175))
    # Draw barrier diagram lines
    draw.line([(660, 360), (1100, 360)], fill=(148, 163, 184), width=3) # baseline
    draw.rectangle([(820, 230), (920, 360)], fill=(203, 213, 225), outline=(100, 116, 139), width=2) # barrier V_0
    draw.text((845, 270), "V_0", fill=(15, 23, 42))
    draw.text((700, 280), "Incident Wave (E < V_0)", fill=(37, 99, 235))
    draw.text((950, 280), "Tunneling Decay", fill=(225, 29, 72))

    # Box 3: Energy Eigenvalue Summary Table
    draw.rectangle([(60, 420), (550, 680)], outline=(16, 185, 129), width=3, fill=(255, 255, 255))
    draw.rectangle([(60, 420), (550, 460)], fill=(209, 250, 229))
    draw.text((80, 430), "3. Infinite Potential Well Eigenvalues", fill=(6, 95, 70))
    draw.text((100, 490), "E_n = (n^2 * pi^2 * h_bar^2) / (2 * m * L^2)", fill=(15, 23, 42))
    draw.text((100, 560), "- Ground state (n = 1): E_1 = (pi^2 * h_bar^2) / (2mL^2)", fill=(51, 65, 85))
    draw.text((100, 610), "- Spatial wavefunction: psi_n(x) = sqrt(2/L) * sin(n*pi*x / L)", fill=(51, 65, 85))

    # Box 4: Boundary Conditions Notice
    draw.rectangle([(620, 470), (1140, 680)], outline=(245, 158, 11), width=3, fill=(255, 255, 255))
    draw.rectangle([(620, 470), (1140, 510)], fill=(254, 243, 199))
    draw.text((640, 480), "4. Boundary Continuity Requirements", fill=(146, 64, 14))
    draw.text((640, 540), "1. Continuity: Psi(x) must be continuous everywhere.", fill=(51, 65, 85))
    draw.text((640, 590), "2. Smoothness: dPsi/dx must be continuous (for finite V).", fill=(51, 65, 85))
    draw.text((640, 640), "3. Normalization: Integral over all space of |Psi|^2 = 1.", fill=(51, 65, 85))

    return img


# ---------------------------------------------------------------------------
# Sidebar: Credentials, Controls & Document Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📐 VisualScribe")
    st.caption("Multimodal AI Study Assistant & Cheatsheet Generator")
    st.divider()

    # API Status Check
    st.subheader("System Status")
    if META_API_KEY:
        st.markdown(
            '<div style="background:#ecfdf5; border-left:4px solid #10b981; padding:8px 12px; border-radius:4px; font-size:13px; color:#065f46;">'
            '● <strong>Meta API: Connected</strong></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background:#fef2f2; border-left:4px solid #ef4444; padding:8px 12px; border-radius:4px; font-size:13px; color:#991b1b;">'
            '● <strong>META_API_KEY Missing</strong> in .env</div>',
            unsafe_allow_html=True
        )

    # Model catalog chips
    st.markdown("""
    <div style="margin-top:10px;">
        <span class="chip chip-blue">sam-3.1</span>
        <span class="chip chip-purple">muse-wonder-1.0</span>
        <span class="chip chip-green">muse-spark-1.3</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Document Uploader
    st.subheader("1. Ingest Document or Image")
    uploaded_file = st.file_uploader(
        "Upload PDF or Image",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        help="Upload multi-page PDFs, textbook photos, whiteboard snapshots, or diagram images."
    )

    with st.expander("📸 Or capture photo with device camera"):
        enable_camera = st.checkbox("Turn on Camera Hardware", value=False, help="Check this box only when you want to take a live photo of physical notes or a whiteboard.")
        if enable_camera:
            camera_img = st.camera_input("Take photo of your textbook, notes, or whiteboard")
            if camera_img is not None:
                uploaded_file = camera_img

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn2:
        if st.button("Load Demo Slide", help="Instantly load an academic Quantum Mechanics sample slide to test without uploading."):
            sample_img = create_sample_slide()
            st.session_state.pages = [sample_img]
            st.session_state.selected_page_idx = 0
            st.session_state.doc_name = "PHYS_401_Quantum_Lecture_Slide"
            st.session_state.pdf_bytes = None
            st.session_state.current_file_id = "demo_slide"
            st.session_state.processed = False
            st.session_state.regions = []
            st.session_state.analyses = []
            st.session_state.summary = ""
            st.session_state.quiz = []
            st.success("Loaded demo slide!")

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_id") != file_id:
            try:
                pages = load_document(uploaded_file)
                st.session_state.pages = pages
                st.session_state.selected_page_idx = 0
                st.session_state.doc_name = uploaded_file.name.rsplit(".", 1)[0]
                if uploaded_file.name.lower().endswith(".pdf"):
                    st.session_state.pdf_bytes = get_document_bytes(uploaded_file)
                else:
                    st.session_state.pdf_bytes = None
                st.session_state.current_file_id = file_id
                st.session_state.processed = False
                st.session_state.regions = []
                st.session_state.analyses = []
                st.session_state.summary = ""
                st.session_state.quiz = []
                st.success(f"Loaded {len(pages)} page(s)/image(s) successfully.")
            except Exception as e:
                st.error(f"Error loading document: {e}")

    # Multi-Page Navigation Directly in Sidebar
    if len(st.session_state.pages) > 1:
        st.divider()
        st.subheader("📄 Target Page Selection")
        page_options = [f"Page {i+1}" for i in range(len(st.session_state.pages))]
        chosen_page_str = st.selectbox(
            "Select Page to Extract & Summarize",
            page_options,
            index=st.session_state.selected_page_idx,
            help="Navigate directly to any page in your multi-page document."
        )
        chosen_idx = page_options.index(chosen_page_str)
        if chosen_idx != st.session_state.selected_page_idx:
            st.session_state.selected_page_idx = chosen_idx
            st.session_state.processed = False
            st.rerun()

    # User Custom Guidance Prompt
    st.divider()
    st.subheader("💬 Study Focus & Directives")
    st.caption("Instruct the AI on what to emphasize in the visual analysis, cheatsheet, and quiz:")

    if "prompt_input" not in st.session_state:
        st.session_state.prompt_input = ""

    # Quick preset chips for rapid directive selection
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("📐 Rigorous Proofs", use_container_width=True, help="Emphasize mathematical derivations, algebraic steps, and proofs"):
            st.session_state.prompt_input = "Focus on step-by-step mathematical derivations and formal proofs with clear variable definitions."
    with preset_col2:
        if st.button("📝 Exam-Style Quiz", use_container_width=True, help="Make the quiz challenging, testing numerical problems and edge cases"):
            st.session_state.prompt_input = "Make the active-recall quiz challenging and calculation-heavy with numerical problem-solving and boundary conditions."

    user_prompt = st.text_area(
        "Custom Instructions (Optional)",
        value=st.session_state.prompt_input,
        placeholder="e.g. 'Focus on the circuit diagram and explain Ohm's law step-by-step', 'Make the quiz calculation-heavy', 'Explain for an undergraduate exam prep'",
        help="Guide the AI to tailor its explanations, derivations, and quiz questions to your specific study needs."
    )
    st.session_state.prompt_input = user_prompt

    # Parameter Controls
    st.divider()
    st.subheader("2. Segmentation Settings")
    focus_selection = st.selectbox(
        "Extraction Focus",
        [
            "⚡ Balanced (Calculations & Diagrams)",
            "🎯 Mathematical Formulas & Calculations Only",
            "📊 Technical Diagrams & Plots Only"
        ],
        index=0,
        help="Choose whether to focus on mathematical equations/derivations, diagrams, or both."
    )
    focus_mode_map = {
        "⚡ Balanced (Calculations & Diagrams)": "balanced",
        "🎯 Mathematical Formulas & Calculations Only": "equations_only",
        "📊 Technical Diagrams & Plots Only": "diagrams_only"
    }
    focus_mode = focus_mode_map[focus_selection]

    max_regions = st.slider(
        "Target Number of Extracts",
        min_value=3,
        max_value=10,
        value=5,
        help="Specify how many diagram and calculation assets to isolate (default: 5)."
    )
    num_quiz_q = st.slider("Active-Recall Quiz Questions", min_value=2, max_value=6, value=4)
    use_sam_api = st.checkbox("Attempt SAM 3.1 / Vision AI", value=True, help="Calls Multimodal Vision models with anti-text constraints; automatically falls back to local OpenCV visual segmentation if unavailable.")

    # 3. Model & Inference Engine
    st.divider()
    st.subheader("3. Reasoning & Inference Engine")
    provider_options = [
        f"🚀 Baseten [{BASETEN_MODEL}] (2M TPM)",
        f"🌐 Meta Model API [{MODEL_MUSE_SPARK}]"
    ]
    raw_env_provider = os.getenv("LLM_PROVIDER", "baseten").split("#")[0].strip().lower()
    default_provider_idx = 0 if (raw_env_provider == "baseten" or (not META_API_KEY and BASETEN_API_KEY)) else 1
    selected_provider_label = st.selectbox(
        "Inference Engine",
        provider_options,
        index=default_provider_idx,
        help="Select which high-performance backend to route cheatsheet synthesis and active-recall quiz generation through."
    )
    active_provider = "baseten" if "Baseten" in selected_provider_label else "meta"

    # Action Trigger
    st.divider()
    process_btn = st.button("🚀 Process & Generate Study Guide", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main View: Hero Banner & Processing Pipeline
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>VisualScribe</h1>
    <p>Automated multimodal study assistant that isolates genuine technical diagrams, circuits, and formulas,
    transcribes them into validated LaTeX, and generates high-yield cheatsheets, active-recall quizzes, and printable PDFs.</p>
</div>
""", unsafe_allow_html=True)

# Prominent Call-To-Action if document/image is loaded but not yet processed
if st.session_state.pages and not st.session_state.processed:
    st.markdown(
        f'<div style="background:#eff6ff; border:1px solid #bfdbfe; border-left:5px solid #3b82f6; padding:12px 16px; border-radius:6px; margin: 12px 0;">'
        f'<strong>Document/Image Loaded:</strong> <code>{st.session_state.doc_name}</code> '
        f'({len(st.session_state.pages)} page/image). Ready to extract visual assets & synthesize study guide.</div>',
        unsafe_allow_html=True
    )
    if st.button("🚀 Process & Generate Study Guide Now", type="primary", use_container_width=True, key="hero_process_btn"):
        process_btn = True

# Process Document Action
if process_btn:
    if not st.session_state.pages:
        st.warning("Please upload a PDF/Image or click 'Load Demo Slide' in the sidebar first.")
    else:
        with st.status("Analyzing document with Meta Multimodal models...", expanded=True) as status:
            page_img = st.session_state.pages[st.session_state.selected_page_idx]
            pdf_bytes = st.session_state.get("pdf_bytes")
            
            # Step 1: Segment visual regions with anti-text filtering and calculation priority
            st.write(f"🔍 **Segmenting Page {st.session_state.selected_page_idx + 1}** (target: {max_regions} visual/formula assets)...")
            raw_regions = segment_visual_regions(
                page_img,
                pdf_bytes=pdf_bytes,
                page_idx=st.session_state.selected_page_idx,
                max_regions=max_regions,
                use_sam_api=use_sam_api,
                focus_mode=focus_mode,
                user_instruction=user_prompt
            )
            st.write(f"✅ Isolated **{len(raw_regions)}** candidate visual / diagram regions.")

            # Step 2: High-speed parallel multimodal analysis via ThreadPoolExecutor
            st.write(f"🔬 **Analyzing {len(raw_regions)} visual structures in parallel** via Muse Wonder...")
            analyses = []
            valid_regions = []
            progress_bar = st.progress(0)
            
            import concurrent.futures

            def _process_single_crop(reg):
                res = analyze_crop(
                    reg["crop"],
                    category_hint=reg["category_hint"],
                    region_id=reg["id"],
                    user_instruction=user_prompt
                )
                return reg, res

            completed_count = 0
            workers = min(5, max(1, len(raw_regions)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_reg = {executor.submit(_process_single_crop, r): r for r in raw_regions}
                for future in concurrent.futures.as_completed(future_to_reg):
                    completed_count += 1
                    try:
                        orig_reg, analysis = future.result()
                        # Keep genuine visual assets (diagrams, tables, flowcharts, circuits, screenshots, equations)
                        if analysis.get("is_valid_asset", True) and analysis.get("category_hint") not in ("plain_text", "text"):
                            analyses.append(analysis)
                            orig_reg["category_hint"] = analysis.get("category_hint", orig_reg["category_hint"])
                            valid_regions.append(orig_reg)
                        else:
                            st.write(f"ℹ️ *Filtered out plain text element #{orig_reg['id']}.*")
                    except Exception as e:
                        pass
                    progress_bar.progress(completed_count / float(max(1, len(raw_regions))))

            # Fallback guarantee: If all regions were filtered out, preserve raw regions so analysis never fails
            if not valid_regions:
                valid_regions = raw_regions
                if not analyses:
                    for r in raw_regions:
                        analyses.append(analyze_crop(r["crop"], r["category_hint"], r["id"], user_instruction=user_prompt))

            # Sort by original ID to preserve visual reading order
            valid_regions.sort(key=lambda r: r["id"])
            analyses.sort(key=lambda a: a.get("id", 0))

            st.session_state.regions = valid_regions
            st.session_state.analyses = analyses

            st.write(f"✅ Parallel analysis complete for **{len(st.session_state.regions)}** assets.")

            # Step 3: Fast Unified Synthesis & Quiz Generation in 1 pass
            import time
            provider_display = f"Baseten [{BASETEN_MODEL}] (2M TPM)" if active_provider == "baseten" else f"Meta Model API [{MODEL_MUSE_SPARK}]"
            st.write(f"🧠 **Synthesizing master cheatsheet & active-recall quiz (High-Speed Unified Pass)** via {provider_display}...")
            doc_context = f"{st.session_state.doc_name} (Page {st.session_state.selected_page_idx + 1})" if len(st.session_state.pages) > 1 else st.session_state.doc_name
            
            gen_start = time.time()
            summary, quiz = generate_cheatsheet_and_quiz_unified(
                st.session_state.analyses,
                document_title=doc_context,
                num_questions=num_quiz_q,
                user_instruction=user_prompt,
                provider=active_provider
            )
            gen_elapsed = round(time.time() - gen_start, 2)
            st.session_state.gen_latency = gen_elapsed
            st.session_state.gen_provider = provider_display
            st.session_state.summary = summary
            st.session_state.quiz = quiz

            st.session_state.processed = True
            status.update(label=f"Study Guide Generated in {gen_elapsed}s with {len(st.session_state.regions)} Extracts via {provider_display}!", state="complete", expanded=False)
            st.rerun()


# ---------------------------------------------------------------------------
# Tabs: Inspection, Cheatsheet, and Export
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Tab 1: Visual Inspection & Segmentation",
    "📚 Tab 2: Interactive Study Cheatsheet & Quiz",
    "💾 Tab 3: Export & PDF Download"
])

# TAB 1: Visual Inspection & Segmentation
with tab1:
    if not st.session_state.pages:
        st.info("👋 Upload a technical lecture PDF or click **'Load Demo Slide'** in the sidebar to begin.")
    else:
        # Multi-page selector if PDF has multiple pages
        if len(st.session_state.pages) > 1:
            page_options = [f"Page {i+1}" for i in range(len(st.session_state.pages))]
            selected_page_str = st.selectbox(
                "Select Page to Inspect",
                page_options,
                index=st.session_state.selected_page_idx,
                key="tab1_page_selector"
            )
            new_idx = page_options.index(selected_page_str)
            if new_idx != st.session_state.selected_page_idx:
                st.session_state.selected_page_idx = new_idx
                st.session_state.processed = False
                st.rerun()

        current_page = st.session_state.pages[st.session_state.selected_page_idx]

        col_left, col_right = st.columns([1.2, 1.0])

        with col_left:
            st.subheader("Slide Overview with Bounding Overlays")
            if st.session_state.regions:
                annotated_img = draw_bounding_boxes(current_page, st.session_state.regions)
                st.image(annotated_img, caption="Isolated Visual Diagrams & Formulas (Body text excluded)", use_container_width=True)
            else:
                st.image(current_page, caption="Original Slide", use_container_width=True)

        with col_right:
            st.subheader("Isolated Visual Asset Crops")
            if st.session_state.regions:
                for idx, reg in enumerate(st.session_state.regions):
                    with st.container():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            cat = reg['category_hint'].upper()
                            st.markdown(f"**Asset #{reg['id']}** — `<span class='chip chip-blue'>{cat}</span>`", unsafe_allow_html=True)
                        with c2:
                            if st.button("🗑️ Discard", key=f"del_crop_{reg['id']}", help="Remove this crop from study guide"):
                                st.session_state.regions.pop(idx)
                                st.session_state.analyses = [a for a in st.session_state.analyses if a.get("id") != reg["id"]]
                                st.rerun()

                        st.image(reg["crop"], caption=f"Crop #{reg['id']}: {reg.get('label', '')}", use_container_width=True)
                        st.divider()
            else:
                st.info("Click **'🚀 Process & Generate Study Guide'** to isolate and inspect diagram and formula assets.")


# TAB 2: Interactive Study Cheatsheet & Active-Recall Quiz
with tab2:
    if not st.session_state.processed:
        st.info("Generate the study guide first to unlock the interactive cheatsheet and active-recall drill cards.")
    else:
        # High-Yield Summary Section
        st.subheader("📖 High-Yield Master Cheatsheet")
        if "gen_latency" in st.session_state:
            st.caption(f"⚡ Synthesis Engine: **{st.session_state.get('gen_provider', 'Inference Backend')}** | Response Latency: **{st.session_state.get('gen_latency', 0)}s**")
        st.markdown(st.session_state.summary)
        st.divider()

        # Detailed Visual Element Breakdowns
        st.subheader("🔬 Extracted Visual Assets & Technical Deconstructions")
        valid_analyses = [
            a for a in st.session_state.analyses
            if a.get("is_valid_asset", True) and a.get("category_hint") not in ("plain_text", "text")
        ]

        if not valid_analyses:
            valid_analyses = st.session_state.analyses

        for item in valid_analyses:
            matching_crop = next((r["crop"] for r in st.session_state.regions if r["id"] == item["id"]), None)

            with st.container():
                col_crop, col_text = st.columns([1, 1.4])
                with col_crop:
                    if matching_crop:
                        st.image(matching_crop, caption=f"Visual Asset #{item['id']}", use_container_width=True)
                with col_text:
                    st.markdown(f"### #{item['id']} {item.get('title', 'Concept Card')}")
                    cat = item.get("category_hint", "diagram").upper()
                    st.markdown(f'<span class="chip chip-blue">{cat}</span>', unsafe_allow_html=True)

                    if item.get("latex"):
                        st.markdown("**Transcribed LaTeX Formulation:**")
                        clean_latex = item["latex"].strip().replace("$$", "").strip()
                        st.latex(clean_latex)

                    st.markdown(f"**Conceptual Explanation:** {item.get('explanation', '')}")

                    if item.get("key_takeaways"):
                        st.markdown("**Key Takeaways:**")
                        for t in item["key_takeaways"]:
                            st.markdown(f"- {t}")

                st.divider()

        # Active-Recall Practice Quiz
        if st.session_state.quiz:
            st.subheader("🎯 Active-Recall Revision Quiz (Test Your Understanding)")
            st.write("Answer these questions generated directly from the visual equations and diagrams.")

            for q in st.session_state.quiz:
                q_id = q["id"]
                st.markdown(f"#### Question {q_id}: {q['question']}")
                
                selected_opt = st.radio(
                    f"Options for Q{q_id}",
                    q["options"],
                    key=f"quiz_opt_{q_id}",
                    label_visibility="collapsed"
                )

                if st.button(f"Submit Answer for Q{q_id}", key=f"btn_check_{q_id}"):
                    user_idx = q["options"].index(selected_opt)
                    correct_idx = q["correct_index"]
                    if user_idx == correct_idx:
                        st.markdown(
                            f'<div class="quiz-correct">✅ <strong>Correct!</strong> {q["explanation"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        correct_text = q["options"][correct_idx]
                        st.markdown(
                            f'<div class="quiz-incorrect">❌ <strong>Incorrect.</strong> The correct answer is: '
                            f'<strong>{correct_text}</strong>.<br/>{q["explanation"]}</div>',
                            unsafe_allow_html=True
                        )
                st.write("")


# TAB 3: Export & PDF Download
with tab3:
    if not st.session_state.processed:
        st.info("Process a document first to export compiled study guides.")
    else:
        st.subheader("💾 Export Assembled Study Guide & Printable PDF")
        st.write("Download your completed study guide formatted with embedded diagrams, equations, and quiz.")

        # Compile PDF directly with PyMuPDF
        pdf_bytes = export_study_guide_pdf(
            document_title=st.session_state.doc_name,
            summary_markdown=st.session_state.summary,
            analyses=st.session_state.analyses,
            quiz=st.session_state.quiz,
            regions=st.session_state.regions
        )

        # Primary PDF Download Button
        st.download_button(
            label="📑 Download Study Guide (.pdf)",
            data=pdf_bytes,
            file_name=f"{st.session_state.doc_name}_StudyGuide.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        st.divider()
        st.write("**Additional Export Formats:**")

        md_content = export_study_guide_markdown(
            document_title=st.session_state.doc_name,
            summary_markdown=st.session_state.summary,
            analyses=st.session_state.analyses,
            quiz=st.session_state.quiz
        )

        html_content = export_study_guide_html(
            document_title=st.session_state.doc_name,
            summary_markdown=st.session_state.summary,
            analyses=st.session_state.analyses,
            quiz=st.session_state.quiz
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Download Study Guide (.md)",
                data=md_content,
                file_name=f"{st.session_state.doc_name}_StudyGuide.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                label="🌐 Download Web Edition (.html)",
                data=html_content,
                file_name=f"{st.session_state.doc_name}_WebGuide.html",
                mime="text/html",
                use_container_width=True
            )

        st.divider()
        st.subheader("Inspect Raw Pipeline Telemetry & JSON")
        with st.expander("Show Raw Structured Analyses & Model Outputs"):
            st.json({
                "document": st.session_state.doc_name,
                "engine_used": st.session_state.get("gen_provider", "Unknown"),
                "generation_latency_seconds": st.session_state.get("gen_latency", None),
                "regions_detected": len(st.session_state.regions),
                "analyses": st.session_state.analyses,
                "quiz": st.session_state.quiz
            })

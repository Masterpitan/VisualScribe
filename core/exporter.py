"""
Study Guide Exporter
Compiles assembled study sheets into GitHub-flavored Markdown and
print-ready HTML with KaTeX math rendering and print stylesheets for PDF saving.
"""

import re
import datetime
from typing import List, Dict, Any, Tuple


def format_latex_to_math_typography(text: str) -> str:
    """
    Translates raw LaTeX notation into clean, publication-grade mathematical typography.
    Converts radicals, exponents, subscripts, Greek variables, and operators so that
    raw dollar signs and backslashes never pollute the PDF layout.
    """
    if not text:
        return ""

    s = text

    # Remove enclosing $$ or $
    s = re.sub(r"^\$\$\s*", "", s)
    s = re.sub(r"\s*\$\$$", "", s)

    # 1. Equation tags: \tag{1} -> (1)
    s = re.sub(r"\\tag\{([^}]+)\}", r"    (\1)", s)

    # 2. Spacing commands
    s = s.replace(r"\qquad", "        ").replace(r"\quad", "    ")
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ")

    # 3. Text & Operator styles: \text{short side} -> short side
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit)\{([^}]+)\}", r"\1", s)

    # 4. Roots / Radicals: \sqrt{a^2 + b^2} -> √(a² + b²)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"√(\1)", s)
    s = s.replace(r"\sqrt", "√")

    # 5. Fractions: \frac{a}{b} -> (a / b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1 / \2)", s)

    # 6. Exponents / Superscripts: ^2 -> ², ^{10} -> ¹⁰
    SUPERSCRIPTS = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
        'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'y': 'ʸ', 'a': 'ᵃ', 'b': 'ᵇ'
    }
    def _rep_sup(match):
        content = match.group(1) or match.group(2)
        return "".join(SUPERSCRIPTS.get(c, c) for c in content)

    s = re.sub(r"\^\{([0-9a-zA-Z\+\-]+)\}|\^([0-9a-zA-Z\+\-])", _rep_sup, s)

    # 7. Subscripts: _0 -> ₀, _{n} -> ₙ
    SUBSCRIPTS = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
        'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ', 'k': 'ₖ',
        'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ', 'p': 'ₚ', 'r': 'ᵣ',
        's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ', 'v': 'ᵥ', 'x': 'ₓ'
    }
    def _rep_sub(match):
        content = match.group(1) or match.group(2)
        return "".join(SUBSCRIPTS.get(c, c) for c in content)

    s = re.sub(r"_\{([0-9a-zA-Z\+\-]+)\}|_([0-9a-zA-Z\+\-])", _rep_sub, s)

    # 8. Math Operators & Symbols
    REPLACEMENTS = [
        (r"\angle", "∠"),
        (r"\approx", "≈"),
        (r"\neq", "≠"),
        (r"\ne", "≠"),
        (r"\leq", "≤"),
        (r"\le", "≤"),
        (r"\geq", "≥"),
        (r"\ge", "≥"),
        (r"\pm", "±"),
        (r"\mp", "∓"),
        (r"\times", "×"),
        (r"\cdot", "·"),
        (r"\circ", "°"),
        (r"\infty", "∞"),
        (r"\partial", "∂"),
        (r"\nabla", "∇"),
        (r"\int", "∫"),
        (r"\sum", "∑"),
        (r"\prod", "∏"),
        (r"\in", "∈"),
        (r"\notin", "∉"),
        (r"\subset", "⊂"),
        (r"\rightarrow", "→"),
        (r"\to", "→"),
        (r"\Rightarrow", "⇒"),
        (r"\forall", "∀"),
        (r"\exists", "∃"),
        # Greek lowercase
        (r"\alpha", "α"), (r"\beta", "β"), (r"\gamma", "γ"), (r"\delta", "δ"),
        (r"\epsilon", "ε"), (r"\theta", "θ"), (r"\lambda", "λ"), (r"\mu", "μ"),
        (r"\pi", "π"), (r"\rho", "ρ"), (r"\sigma", "σ"), (r"\tau", "τ"),
        (r"\phi", "φ"), (r"\psi", "ψ"), (r"\omega", "ω"), (r"\xi", "ξ"),
        # Greek uppercase
        (r"\Gamma", "Γ"), (r"\Delta", "Δ"), (r"\Theta", "Θ"), (r"\Lambda", "Λ"),
        (r"\Sigma", "Σ"), (r"\Phi", "Φ"), (r"\Psi", "Ψ"), (r"\Omega", "Ω"),
    ]

    for k, v in REPLACEMENTS:
        s = s.replace(k, v)

    # 9. Strip any leftover dollar signs and stray braces
    s = s.replace("$", "")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"\{|\}", "", s)

    return s.strip()


def render_latex_math_image(latex_str: str, dpi: int = 250, fontsize: int = 12, max_width_pt: int = 480) -> Tuple[bytes, int, int]:
    """
    Renders a LaTeX equation into high-resolution transparent PNG bytes using Matplotlib MathText.
    Returns (png_bytes, rendered_width_pt, rendered_height_pt) or (None, 0, 0) if unavailable.
    """
    import io
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clean = latex_str.strip()
        # Strip display or inline math delimiters
        if clean.startswith("$$") and clean.endswith("$$"):
            clean = clean[2:-2].strip()
        elif clean.startswith("$") and clean.endswith("$"):
            clean = clean[1:-1].strip()

        # Handle tag: \tag{1} -> (1)
        tag_match = re.search(r"\\tag\{([^}]+)\}", clean)
        tag_str = ""
        if tag_match:
            tag_str = f"({tag_match.group(1)})"
            clean = re.sub(r"\\tag\{[^}]+\}", "", clean).strip()

        # MathText adaptations
        clean = re.sub(r"\\text\{([^}]+)\}", r"\\mathrm{\1}", clean)
        clean = clean.replace(r"\qquad", r"\quad\quad")
        clean = clean.replace(r"\;", " ").replace(r"\,", " ")
        clean = clean.replace("\r", "").strip()

        # Avoid multi-line crash in mathtext
        if "\n" in clean:
            clean = clean.split("\n")[0].strip()

        fig = plt.figure(figsize=(7.5, 1.1), dpi=dpi)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.patch.set_alpha(0.0)

        math_text = f"${clean}$"
        ax.text(0.5, 0.5, math_text, fontsize=fontsize, ha="center", va="center", color="#0f172a")
        if tag_str:
            ax.text(0.97, 0.5, tag_str, fontsize=fontsize - 2, ha="right", va="center", color="#475569")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.06, transparent=True)

        from PIL import Image
        buf.seek(0)
        img = Image.open(buf)
        iw, ih = img.size
        pt_w = int(iw * 72.0 / dpi)
        pt_h = int(ih * 72.0 / dpi)

        if pt_w > max_width_pt:
            scale = max_width_pt / float(pt_w)
            pt_w = max_width_pt
            pt_h = int(pt_h * scale)

        buf.seek(0)
        return buf.getvalue(), pt_w, max(22, pt_h)
    except Exception:
        return None, 0, 0
    finally:
        if fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass


def extract_summary_blocks(summary_text: str) -> List[Tuple[str, str]]:
    """
    Parses summary markdown text into typed blocks:
    - ('heading', text)
    - ('equation', latex_str)
    - ('text', prose_text)
    """
    if not summary_text:
        return []

    blocks = []
    lines = summary_text.replace("\r\n", "\n").split("\n")
    in_equation = False
    eq_buffer = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line == "$$":
            if in_equation:
                eq_text = "\n".join(eq_buffer).strip()
                if eq_text:
                    blocks.append(("equation", eq_text))
                eq_buffer = []
                in_equation = False
            else:
                in_equation = True
                eq_buffer = []
            continue

        if in_equation:
            if line.endswith("$$"):
                eq_buffer.append(line[:-2].strip())
                eq_text = "\n".join(eq_buffer).strip()
                if eq_text:
                    blocks.append(("equation", eq_text))
                eq_buffer = []
                in_equation = False
            else:
                eq_buffer.append(line)
            continue

        if line.startswith("$$") and line.endswith("$$") and len(line) >= 4:
            blocks.append(("equation", line[2:-2].strip()))
            continue

        # If a line contains $$...$$, extract equation and text surrounding it
        if "$$" in line:
            parts = re.split(r"(\$\$.*?\$\$)", line)
            for part in parts:
                p = part.strip()
                if not p:
                    continue
                if p.startswith("$$") and p.endswith("$$"):
                    blocks.append(("equation", p[2:-2].strip()))
                else:
                    blocks.append(("text", p))
            continue

        if line.startswith(("#", "1.", "2.", "3.", "4.", "Executive", "Mathematical", "Structural", "Theorem")):
            cleaned = re.sub(r"^#+\s*", "", line)
            blocks.append(("heading", cleaned))
            continue

        blocks.append(("text", line))

    if in_equation and eq_buffer:
        blocks.append(("equation", "\n".join(eq_buffer).strip()))

    return blocks


def export_study_guide_markdown(
    document_title: str,
    summary_markdown: str,
    analyses: List[Dict[str, Any]],
    quiz: List[Dict[str, Any]]
) -> str:
    """
    Assembles a comprehensive Markdown study guide including document metadata,
    high-yield summary, segmented visual details with LaTeX, and an active-recall quiz.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# VisualScribe Study Guide: {document_title}",
        f"**Generated:** {timestamp} | **Powered by:** Meta SAM 3.1, Muse Wonder & Muse Spark",
        "",
        "---",
        "",
        "## Part 1: High-Yield Cheatsheet Summary",
        "",
        summary_markdown,
        "",
        "---",
        "",
        "## Part 2: Segmented Visual Assets & LaTeX Formulations",
        ""
    ]

    for item in analyses:
        lines.append(f"### Region #{item.get('id', '?')}: {item.get('title', 'Visual Asset')}")
        lines.append(f"- **Classification:** `{item.get('category_hint', 'diagram')}`")
        if item.get("latex"):
            lines.append(f"- **Transcribed LaTeX Formulation:**")
            lines.append(f"  {item.get('latex')}")
        lines.append(f"- **Conceptual Explanation:** {item.get('explanation', '')}")
        if item.get("key_takeaways"):
            lines.append("- **Key Observations:**")
            for t in item.get("key_takeaways", []):
                lines.append(f"  * {t}")
        lines.append("")

    if quiz:
        lines.extend([
            "---",
            "",
            "## Part 3: Active-Recall Practice Quiz & Drill",
            ""
        ])
        for q in quiz:
            lines.append(f"**Q{q.get('id')}. {q.get('question')}**")
            for idx, opt in enumerate(q.get("options", [])):
                letter = chr(ord('A') + idx)
                lines.append(f"- [{letter}] {opt}")
            
            # Answer key in a collapsible spoiler block
            correct_letter = chr(ord('A') + q.get('correct_index', 0))
            lines.append("<details>")
            lines.append(f"<summary>Click to reveal Answer & Explanation</summary>")
            lines.append("")
            lines.append(f"> **Correct Answer:** **[{correct_letter}]**")
            lines.append(f"> **Explanation:** {q.get('explanation', '')}")
            lines.append("</details>")
            lines.append("")

    lines.extend([
        "---",
        "*Compiled automatically by VisualScribe — Meta Global AI Developer Hackathon Submission.*"
    ])

    return "\n".join(lines)


def export_study_guide_html(
    document_title: str,
    summary_markdown: str,
    analyses: List[Dict[str, Any]],
    quiz: List[Dict[str, Any]]
) -> str:
    """
    Creates a standalone, publication-quality HTML document with MathJax CDN
    and clean CSS with print rules (`@media print`) so users can 'Print to PDF'
    with flawless formatting.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Simple markdown to HTML converter for summary sections
    summary_html = summary_markdown.replace("\n", "<br/>")

    elements_html = ""
    for item in analyses:
        latex_block = f'<div class="latex-box">{item.get("latex")}</div>' if item.get("latex") else ""
        takeaways_html = "".join([f"<li>{t}</li>" for t in item.get("key_takeaways", [])])
        elements_html += f"""
        <div class="card">
            <div class="card-header">
                <h3>#{item.get('id')} — {item.get('title')}</h3>
                <span class="badge">{item.get('category_hint', 'diagram').upper()}</span>
            </div>
            {latex_block}
            <p><strong>Explanation:</strong> {item.get('explanation')}</p>
            <ul>{takeaways_html}</ul>
        </div>
        """

    quiz_html = ""
    for q in quiz:
        opts_html = "".join([
            f"<li><strong>[{chr(ord('A') + i)}]</strong> {opt}</li>"
            for i, opt in enumerate(q.get("options", []))
        ])
        correct_letter = chr(ord('A') + q.get('correct_index', 0))
        quiz_html += f"""
        <div class="quiz-box">
            <p class="quiz-question"><strong>Q{q.get('id')}:</strong> {q.get('question')}</p>
            <ul class="quiz-options">{opts_html}</ul>
            <div class="answer-banner">
                <strong>Answer: [{correct_letter}]</strong> — {q.get('explanation')}
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisualScribe Study Guide - {document_title}</title>
    <!-- MathJax for rendering standard LaTeX equations -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #1e293b;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 24px;
            background: #f8fafc;
        }}
        .header {{
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 16px;
            margin-bottom: 32px;
        }}
        .header h1 {{
            color: #0f172a;
            margin: 0 0 8px 0;
            font-size: 28px;
        }}
        .meta {{
            color: #64748b;
            font-size: 14px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .badge {{
            background: #e0f2fe;
            color: #0369a1;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .latex-box {{
            background: #f1f5f9;
            padding: 14px;
            border-radius: 6px;
            margin: 12px 0;
            overflow-x: auto;
            text-align: center;
        }}
        .quiz-box {{
            background: white;
            border-left: 4px solid #10b981;
            padding: 16px 20px;
            margin-bottom: 16px;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .quiz-options {{
            list-style: none;
            padding-left: 0;
        }}
        .quiz-options li {{
            padding: 4px 0;
        }}
        .answer-banner {{
            background: #ecfdf5;
            color: #065f46;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 14px;
        }}
        @media print {{
            body {{
                background: white;
                max-width: 100%;
                margin: 0;
                padding: 10mm;
            }}
            .card, .quiz-box {{
                box-shadow: none;
                border: 1px solid #ccc;
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>VisualScribe Study Guide: {document_title}</h1>
        <div class="meta">Compiled on {timestamp} | Meta Global AI Developer Hackathon</div>
    </div>

    <h2>1. High-Yield Synthesis</h2>
    <div class="card">
        <div>{summary_html}</div>
    </div>

    <h2>2. Visual Segmentations & Formulations</h2>
    {elements_html}

    <h2>3. Active-Recall Revision Quiz</h2>
    {quiz_html}
</body>
</html>
"""
    return html


def export_study_guide_pdf(
    document_title: str,
    summary_markdown: str,
    analyses: List[Dict[str, Any]],
    quiz: List[Dict[str, Any]],
    regions: List[Dict[str, Any]] = None
) -> bytes:
    """
    Directly compiles an assembled, publication-quality A4 PDF document using PyMuPDF (fitz).
    Embeds high-resolution cropped diagram images, LaTeX formulations, executive summaries,
    and active-recall quiz sections without needing external browser or OS tools.
    
    Returns:
        Raw PDF file bytes ready for direct st.download_button.
    """
    import io
    from PIL import Image

    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF (`fitz`) is required for PDF export. Run: pip install pymupdf")

    doc = fitz.open()
    page_w, page_h = 595, 842  # Standard A4 in points
    margin_x = 45
    content_w = page_w - (2 * margin_x)
    max_y = page_h - 55

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Helper to add new page with consistent running header
    def create_page():
        p = doc.new_page(width=page_w, height=page_h)
        # Running header rule
        if doc.page_count > 1:
            p.draw_line(fitz.Point(margin_x, 35), fitz.Point(page_w - margin_x, 35), color=(0.8, 0.85, 0.9), width=0.8)
            p.insert_text(fitz.Point(margin_x, 28), f"VisualScribe: {document_title[:45]}", fontsize=8, color=(0.5, 0.55, 0.6))
        return p

    current_page = create_page()
    curr_y = 45

    def check_space(needed: float):
        nonlocal current_page, curr_y
        if curr_y + needed > max_y:
            current_page = create_page()
            curr_y = 50

    # ---------------------------------------------------------
    # PAGE 1: TITLE BANNER
    # ---------------------------------------------------------
    banner_rect = fitz.Rect(margin_x, curr_y, page_w - margin_x, curr_y + 60)
    current_page.draw_rect(banner_rect, fill=(0.12, 0.15, 0.35), color=None)  # Deep Indigo
    current_page.insert_text(fitz.Point(margin_x + 16, curr_y + 26), f"VisualScribe Study Guide", fontsize=16, color=(1, 1, 1), fontname="helv", fontfile=None)
    current_page.insert_text(fitz.Point(margin_x + 16, curr_y + 44), f"Document: {document_title}  |  Compiled: {timestamp}", fontsize=9, color=(0.85, 0.9, 1))
    curr_y += 75

    # ---------------------------------------------------------
    # SECTION 1: HIGH-YIELD SUMMARY & CHEATSHEET
    # ---------------------------------------------------------
    check_space(40)
    current_page.draw_rect(fitz.Rect(margin_x, curr_y, margin_x + 4, curr_y + 16), fill=(0.15, 0.4, 0.9))
    current_page.insert_text(fitz.Point(margin_x + 12, curr_y + 14), "1. Executive Concept Synthesis & Cheatsheet", fontsize=12, color=(0.1, 0.15, 0.3))
    curr_y += 24

    summary_blocks = extract_summary_blocks(summary_markdown)

    for block_type, content in summary_blocks:
        if block_type == "heading":
            clean_head = format_latex_to_math_typography(content.replace("**", "").replace("*", ""))
            check_space(26)
            current_page.insert_text(fitz.Point(margin_x, curr_y + 14), clean_head, fontsize=10.5, fontname="helv", color=(0.1, 0.18, 0.35))
            curr_y += 22

        elif block_type == "equation":
            # 1. Attempt true mathematical rendering via Matplotlib MathText
            png_bytes, pt_w, pt_h = render_latex_math_image(content, dpi=250, fontsize=12, max_width_pt=int(content_w - 40))
            if png_bytes and pt_w > 0 and pt_h > 0:
                card_h = pt_h + 14
                check_space(card_h + 8)
                eq_rect = fitz.Rect(margin_x, curr_y, margin_x + content_w, curr_y + card_h)
                current_page.draw_rect(eq_rect, fill=(0.975, 0.985, 1.0), color=(0.85, 0.89, 0.96), width=0.8)
                img_x = margin_x + max(12, int((content_w - pt_w) / 2))
                img_y = curr_y + 7
                current_page.insert_image(fitz.Rect(img_x, img_y, img_x + pt_w, img_y + pt_h), stream=png_bytes)
                curr_y += card_h + 8
            else:
                # 2. Mathematical typography fallback
                fmt_eq = format_latex_to_math_typography(content)
                check_space(32)
                eq_rect = fitz.Rect(margin_x, curr_y, margin_x + content_w, curr_y + 24)
                current_page.draw_rect(eq_rect, fill=(0.975, 0.985, 1.0), color=(0.85, 0.89, 0.96), width=0.8)
                current_page.insert_text(fitz.Point(margin_x + 16, curr_y + 16), fmt_eq, fontsize=9.5, fontname="times-italic", color=(0.1, 0.18, 0.45))
                curr_y += 30

        elif block_type == "text":
            clean_text = format_latex_to_math_typography(content.replace("**", "").replace("*", ""))
            if not clean_text:
                continue
            est_height = max(16, int(len(clean_text) / 80 + 1) * 14)
            check_space(est_height + 4)
            t_rect = fitz.Rect(margin_x, curr_y, margin_x + content_w, curr_y + est_height)
            current_page.insert_textbox(t_rect, clean_text, fontsize=9, fontname="helv", color=(0.2, 0.25, 0.3))
            curr_y += est_height + 4

    curr_y += 15

    # ---------------------------------------------------------
    # SECTION 2: VISUAL DIAGRAMS & FORMULAS (WITH IMAGES!)
    # ---------------------------------------------------------
    check_space(40)
    current_page.draw_rect(fitz.Rect(margin_x, curr_y, margin_x + 4, curr_y + 16), fill=(0.15, 0.4, 0.9))
    current_page.insert_text(fitz.Point(margin_x + 12, curr_y + 14), "2. Segmented Visual Diagrams & LaTeX Formulations", fontsize=12, color=(0.1, 0.15, 0.3))
    curr_y += 24

    # Build map of crops from regions
    crop_map = {}
    if regions:
        for r in regions:
            if r.get("id") and r.get("crop"):
                crop_map[r["id"]] = r["crop"]

    # Filter analyses: keep only genuine visual/formula assets
    valid_analyses = [a for a in analyses if a.get("is_valid_asset", True) and a.get("category_hint") not in ("plain_text", "text")]

    if not valid_analyses:
        valid_analyses = analyses

    for item in valid_analyses:
        item_id = item.get("id", 1)
        title = item.get("title", f"Visual Asset #{item_id}")
        cat = item.get("category_hint", "diagram").upper()
        latex = item.get("latex", "")
        explanation = item.get("explanation", "")
        takeaways = item.get("key_takeaways", [])

        check_space(50)
        # Item Header
        current_page.insert_text(fitz.Point(margin_x, curr_y + 12), f"#{item_id} — {title}", fontsize=11, color=(0.1, 0.15, 0.35))
        current_page.insert_text(fitz.Point(page_w - margin_x - 70, curr_y + 12), f"[{cat}]", fontsize=9, color=(0.2, 0.5, 0.8))
        curr_y += 20

        # Embed cropped image if available
        crop_img = crop_map.get(item_id)
        if crop_img and isinstance(crop_img, Image.Image):
            cw, ch = crop_img.size
            max_img_w = min(420, content_w)
            max_img_h = 160
            scale = min(max_img_w / float(max(1, cw)), max_img_h / float(max(1, ch)), 1.0)
            render_w = int(cw * scale)
            render_h = int(ch * scale)

            check_space(render_h + 20)
            img_buf = io.BytesIO()
            if crop_img.mode != "RGB":
                crop_img = crop_img.convert("RGB")
            crop_img.save(img_buf, format="PNG")

            img_rect = fitz.Rect(margin_x, curr_y, margin_x + render_w, curr_y + render_h)
            # Light border card around image
            current_page.draw_rect(fitz.Rect(margin_x - 4, curr_y - 4, margin_x + render_w + 4, curr_y + render_h + 4), color=(0.85, 0.88, 0.92), width=1)
            current_page.insert_image(img_rect, stream=img_buf.getvalue())
            curr_y += render_h + 14

        # Transcribed LaTeX Formulation - True Visual / Mathematical Rendering
        if latex:
            png_bytes, pt_w, pt_h = render_latex_math_image(latex, dpi=250, fontsize=12, max_width_pt=int(content_w - 40))
            if png_bytes and pt_w > 0 and pt_h > 0:
                card_h = pt_h + 20
                check_space(card_h + 8)
                eq_rect = fitz.Rect(margin_x, curr_y, margin_x + content_w, curr_y + card_h)
                current_page.draw_rect(eq_rect, fill=(0.975, 0.985, 1.0), color=(0.85, 0.89, 0.96), width=0.8)
                current_page.insert_text(fitz.Point(margin_x + 10, curr_y + 11), "MATHEMATICAL FORMULATION", fontsize=7.5, color=(0.35, 0.4, 0.65))
                img_x = margin_x + max(12, int((content_w - pt_w) / 2))
                img_y = curr_y + 14
                current_page.insert_image(fitz.Rect(img_x, img_y, img_x + pt_w, img_y + pt_h), stream=png_bytes)
                curr_y += card_h + 10
            else:
                clean_lat = format_latex_to_math_typography(latex)
                check_space(32)
                lat_rect = fitz.Rect(margin_x, curr_y, margin_x + content_w, curr_y + 24)
                current_page.draw_rect(lat_rect, fill=(0.975, 0.985, 1.0), color=(0.85, 0.89, 0.96), width=0.8)
                current_page.insert_text(fitz.Point(margin_x + 10, curr_y + 16), f"Formula: {clean_lat[:85]}", fontsize=9.5, fontname="times-italic", color=(0.1, 0.18, 0.45))
                curr_y += 30

        # Explanation
        if explanation:
            clean_exp = format_latex_to_math_typography(explanation)
            exp_lines = max(18, int(len(clean_exp) / 85 + 1) * 14)
            check_space(exp_lines + 8)
            e_rect = fitz.Rect(margin_x, curr_y, margin_x + content_w, curr_y + exp_lines)
            current_page.insert_textbox(e_rect, f"Explanation: {clean_exp}", fontsize=8.5, color=(0.2, 0.25, 0.3))
            curr_y += exp_lines + 8

        # Bullet takeaways
        for tk in takeaways[:3]:
            clean_tk = format_latex_to_math_typography(tk)
            check_space(18)
            current_page.insert_text(fitz.Point(margin_x + 8, curr_y + 11), f"• {clean_tk}", fontsize=8, color=(0.3, 0.35, 0.4))
            curr_y += 15

        curr_y += 12

    # ---------------------------------------------------------
    # SECTION 3: ACTIVE-RECALL QUIZ
    # ---------------------------------------------------------
    if quiz:
        check_space(45)
        current_page.draw_rect(fitz.Rect(margin_x, curr_y, margin_x + 4, curr_y + 16), fill=(0.1, 0.65, 0.45))
        current_page.insert_text(fitz.Point(margin_x + 12, curr_y + 14), "3. Active-Recall Revision Quiz", fontsize=12, color=(0.1, 0.15, 0.3))
        curr_y += 24

        for q in quiz:
            q_num = q.get("id", 1)
            q_text = format_latex_to_math_typography(q.get("question", ""))
            opts = [format_latex_to_math_typography(opt) for opt in q.get("options", [])]
            c_idx = q.get("correct_index", 0)
            expl = format_latex_to_math_typography(q.get("explanation", ""))

            check_space(55 + len(opts) * 14)
            current_page.insert_text(fitz.Point(margin_x, curr_y + 12), f"Q{q_num}: {q_text}", fontsize=9.5, color=(0.1, 0.15, 0.35))
            curr_y += 18

            for o_idx, opt in enumerate(opts):
                letter = chr(ord('A') + o_idx)
                current_page.insert_text(fitz.Point(margin_x + 14, curr_y + 11), f"[{letter}] {opt}", fontsize=8.5, color=(0.25, 0.3, 0.35))
                curr_y += 14

            # Answer line
            curr_y += 4
            correct_letter = chr(ord('A') + c_idx)
            ans_box = fitz.Rect(margin_x, curr_y, margin_x + content_w, curr_y + 20)
            current_page.draw_rect(ans_box, fill=(0.92, 0.98, 0.95), color=(0.6, 0.85, 0.7), width=0.8)
            current_page.insert_text(fitz.Point(margin_x + 8, curr_y + 14), f"Correct Answer: [{correct_letter}] — {expl[:100]}", fontsize=8, color=(0.05, 0.45, 0.3))
            curr_y += 28

    # ---------------------------------------------------------
    # FOOTERS ON ALL PAGES
    # ---------------------------------------------------------
    total_pages = doc.page_count
    for p_num in range(total_pages):
        p = doc[p_num]
        p.draw_line(fitz.Point(margin_x, page_h - 40), fitz.Point(page_w - margin_x, page_h - 40), color=(0.85, 0.88, 0.9), width=0.8)
        p.insert_text(fitz.Point(margin_x, page_h - 26), "VisualScribe | Powered by Meta SAM 3.1 & Muse Models", fontsize=8, color=(0.55, 0.6, 0.65))
        p.insert_text(fitz.Point(page_w - margin_x - 60, page_h - 26), f"Page {p_num + 1} of {total_pages}", fontsize=8, color=(0.55, 0.6, 0.65))

    pdf_bytes = doc.tobytes(deflate=True)
    doc.close()
    return pdf_bytes


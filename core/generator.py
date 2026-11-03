"""
Study Synthesis & Active-Recall Quiz Generator
Powered by Muse Spark (muse-spark-1.3) for high-speed conceptual synthesis,
formula derivation summaries, and targeted revision flashcards.
"""

import json
import logging
import re
from typing import List, Dict, Any, Tuple

from config import (
    MODEL_MUSE_SPARK,
    call_chat_completion,
    DEFAULT_MAX_TOKENS
)

logger = logging.getLogger("VisualScribe.Generator")


def generate_cheatsheet_and_quiz_unified(
    analyses: List[Dict[str, Any]],
    document_title: str = "Technical Lecture Notes",
    num_questions: int = 4,
    user_instruction: str = ""
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    High-performance unified generator: compiles BOTH the master cheatsheet
    and the active-recall quiz in a single Muse Spark round-trip.
    Saves an entire ~30-40 second LLM round trip!
    """
    if not analyses:
        return "No visual elements were segmented to generate a summary.", []

    elements_context = []
    for item in analyses:
        elem_text = f"### Element #{item.get('id', '?')}: {item.get('title', 'Visual Element')}\n"
        elem_text += f"- Type: {item.get('category_hint', 'diagram')}\n"
        if item.get("latex"):
            elem_text += f"- LaTeX Formula: {item.get('latex')}\n"
        elem_text += f"- Details: {item.get('explanation', '')}\n"
        if item.get("key_takeaways"):
            elem_text += f"- Takeaways: {', '.join(item.get('key_takeaways', []))}\n"
        elements_context.append(elem_text)

    prompt = (
        f"You are a distinguished technical editor and STEM professor compiling an authoritative study guide.\n"
        f"Synthesize the extracted mathematical formulas, derivations, and diagrams from '{document_title}'.\n\n"
        f"OUTPUT FORMAT REQUIREMENT (CRITICAL):\n"
        f"You must deliver TWO sections separated by explicit delimiter banners:\n\n"
        f"===CHEATSHEET_START===\n"
        f"# {document_title}: Technical Reference Guide & Formulation Summary\n\n"
        f"## 1. Executive Conceptual Overview\n"
        f"(Concise technical synthesis of governing physical/mathematical laws and framework)\n\n"
        f"## 2. Formal Mathematical Formulations & Derivations\n"
        f"(Rigorous LaTeX equation blocks with formal variable specifications and step-by-step calculus/algebra)\n\n"
        f"## 3. Structural Schematics & Diagram Mechanics\n"
        f"(Precise deconstruction of diagram relationships, vectors, coordinate systems, and input-output flows)\n\n"
        f"## 4. Analytical Constraints, Boundary Conditions & Edge Cases\n"
        f"(Exact mathematical validity domains, boundary conditions, and analytical pitfalls)\n"
        f"===CHEATSHEET_END===\n\n"
        f"===QUIZ_JSON_START===\n"
        f"[\n"
        f"  {{\n"
        f'    "id": 1,\n'
        f'    "question": "Question text with LaTeX if applicable?",\n'
        f'    "options": ["Option A", "Option B", "Option C", "Option D"],\n'
        f'    "correct_index": 0,\n'
        f'    "explanation": "Thorough technical explanation."\n'
        f"  }}\n"
        f"]\n"
        f"===QUIZ_JSON_END===\n\n"
        f"TONE & STYLE:\n"
        f"- Strict academic rigor. Zero colloquial filler or childish verbalized arithmetic.\n"
        f"- Generate {num_questions} conceptual active-recall questions.\n\n"
    )

    if user_instruction:
        prompt += (
            f"USER DIRECTIVE & SPECIAL FOCUS REQUIREMENT:\n"
            f"\"{user_instruction}\"\n"
            f"Address and prioritize this directive thoroughly in both the summary and the quiz questions.\n\n"
        )

    prompt += "Extracted Elements:\n" + "\n".join(elements_context)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an elite academic reference guide compiler. Produce rigorous, publication-grade "
                "STEM documentation and active-recall questions using the exact delimiter blocks requested."
            )
        },
        {"role": "user", "content": prompt}
    ]

    try:
        raw_output = call_chat_completion(
            model=MODEL_MUSE_SPARK,
            messages=messages,
            max_tokens=3200,
            temperature=0.3
        )

        summary_md = ""
        quiz_list = []

        # Parse cheatsheet
        if "===CHEATSHEET_START===" in raw_output:
            part = raw_output.split("===CHEATSHEET_START===")[1]
            if "===CHEATSHEET_END===" in part:
                summary_md = part.split("===CHEATSHEET_END===")[0].strip()
            else:
                summary_md = part.split("===QUIZ_JSON_START===")[0].strip()

        # Parse quiz
        if "===QUIZ_JSON_START===" in raw_output:
            q_part = raw_output.split("===QUIZ_JSON_START===")[1]
            if "===QUIZ_JSON_END===" in q_part:
                q_raw = q_part.split("===QUIZ_JSON_END===")[0].strip()
            else:
                q_raw = q_part.strip()
            quiz_list = _parse_quiz_json(q_raw)

        # Fallback if delimiters were omitted by the model
        if not summary_md:
            summary_md = raw_output.split("[")[0].strip() if "[" in raw_output else raw_output
        if not quiz_list:
            quiz_list = _parse_quiz_json(raw_output)

        if not quiz_list:
            quiz_list = _generate_fallback_quiz(analyses, num_questions)

        return summary_md, quiz_list

    except Exception as e:
        logger.error(f"Error in unified cheatsheet & quiz generation: {e}")
        return _generate_fallback_summary(analyses, document_title), _generate_fallback_quiz(analyses, num_questions)


def generate_cheatsheet_summary(
    analyses: List[Dict[str, Any]],
    document_title: str = "Technical Lecture Notes",
    user_instruction: str = ""
) -> str:
    """
    Synthesizes multiple segmented visual element analyses into a cohesive,
    high-yield study cheatsheet with LaTeX expressions and conceptual summaries.
    """
    if not analyses:
        return "No visual elements were segmented to generate a summary."

    # Build context from individual crop analyses
    elements_context = []
    for item in analyses:
        elem_text = f"### Element #{item.get('id', '?')}: {item.get('title', 'Visual Element')}\n"
        elem_text += f"- Type: {item.get('category_hint', 'diagram')}\n"
        if item.get("latex"):
            elem_text += f"- LaTeX Formula: {item.get('latex')}\n"
        elem_text += f"- Details: {item.get('explanation', '')}\n"
        if item.get("key_takeaways"):
            elem_text += f"- Takeaways: {', '.join(item.get('key_takeaways', []))}\n"
        elements_context.append(elem_text)

    prompt = (
        f"You are a distinguished technical editor and STEM professor compiling an authoritative reference cheatsheet.\n"
        f"Synthesize the extracted mathematical formulas, derivations, and diagrams from '{document_title}' into an elite, publication-grade technical study guide.\n\n"
        f"TONE & STYLE GUIDELINES (CRITICAL):\n"
        f"1. STRICTLY FORMAL & ACADEMIC: Write in the rigorous, objective style of an advanced university textbook or peer-reviewed reference handbook (e.g., Springer, MIT Press, IEEE).\n"
        f"2. ZERO COLLOQUIALISMS / NO BABY-TALK:\n"
        f"   - DO NOT use informal tutor phrasing such as 'The Master Equation', 'The Golden Rule', 'Pro-tip', or 'In words: (short side)^2 + (other side)^2...'.\n"
        f"   - DO NOT write childish verbalized arithmetic like 'In words: (leg)^2 + ...'.\n"
        f"3. MATHEMATICAL RIGOR:\n"
        f"   - Present all equations in clean, standardized LaTeX: `$$ ... $$` for display equations and `$ ... $` for inline variables.\n"
        f"   - Immediately beneath each equation, specify all variables, units, and domain definitions formally:\n"
        f"     *Where $x$ is [technical parameter], $y$ is [parameter], and constraints are [domain conditions].*\n"
        f"4. DERIVATIONS & STEP-BY-STEP PROOFS:\n"
        f"   - Clearly outline theoretical derivations, algebraic transformations, or physical assumptions.\n\n"
    )

    if user_instruction:
        prompt += (
            f"CRITICAL USER DIRECTIVE & SPECIAL FOCUS REQUIREMENT:\n"
            f"\"{user_instruction}\"\n"
            f"Thoroughly emphasize and tailor the summary, formulations, and derivations to fulfill this directive.\n\n"
        )

    prompt += (
        f"REQUIRED MARKDOWN STRUCTURE:\n"
        f"# {document_title}: Technical Reference Guide & Formulation Summary\n\n"
        f"## 1. Executive Conceptual Overview\n"
        f"(Concise technical synthesis of governing physical/mathematical laws and framework)\n\n"
        f"## 2. Formal Mathematical Formulations & Derivations\n"
        f"(Rigorous LaTeX equation blocks with formal variable specifications and step-by-step calculus/algebra)\n\n"
        f"## 3. Structural Schematics & Diagram Mechanics\n"
        f"(Precise deconstruction of diagram relationships, vectors, coordinate systems, and input-output flows)\n\n"
        f"## 4. Analytical Constraints, Boundary Conditions & Edge Cases\n"
        f"(Exact mathematical validity domains, boundary conditions, and analytical pitfalls)\n\n"
        f"Extracted Elements:\n"
        + "\n".join(elements_context)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an elite academic reference guide compiler. Produce rigorous, publication-grade "
                "STEM documentation. Strictly avoid conversational filler, colloquial phrases ('The Master Equation'), "
                "and childish verbalized equation text ('In words: ...'). Use formal technical notation."
            )
        },
        {"role": "user", "content": prompt}
    ]

    try:
        # Generous max_tokens to accommodate Muse Spark's reasoning tokens + full markdown
        summary_markdown = call_chat_completion(
            model=MODEL_MUSE_SPARK,
            messages=messages,
            max_tokens=3000,
            temperature=0.3
        )
        if summary_markdown:
            return summary_markdown
    except Exception as e:
        logger.error(f"Error calling Muse Spark for cheatsheet: {e}")

    # Fallback compilation if API is unavailable
    return _generate_fallback_summary(analyses, document_title)


def generate_practice_quiz(
    analyses: List[Dict[str, Any]],
    num_questions: int = 4,
    user_instruction: str = ""
) -> List[Dict[str, Any]]:
    """
    Generates active-recall multiple choice questions directly tied to the visual
    diagrams and mathematical formulas extracted from the document.
    """
    if not analyses:
        return []

    elements_summary = []
    for item in analyses:
        summary_str = f"Element {item.get('id')}: {item.get('title')} ({item.get('category_hint')}) - {item.get('explanation')}"
        if item.get("latex"):
            summary_str += f" | LaTeX: {item.get('latex')}"
        elements_summary.append(summary_str)

    prompt = (
        f"You are Muse Spark, creating active-recall revision questions for technical students.\n"
        f"Based on these extracted slide elements:\n"
        + "\n".join(elements_summary) + "\n\n"
    )
    if user_instruction:
        prompt += f"USER GUIDANCE / FOCUS GOAL:\n\"{user_instruction}\"\nFocus questions on concepts relevant to this directive.\n\n"

    prompt += (
        f"Generate {num_questions} high-quality, conceptual multiple-choice questions.\n"
        f"Include equations in LaTeX where appropriate ($...$).\n\n"
        f"You MUST output ONLY a valid JSON array of objects with the exact schema:\n"
        f"[\n"
        f"  {{\n"
        f'    "id": 1,\n'
        f'    "question": "Question text here?",\n'
        f'    "options": ["Option A", "Option B", "Option C", "Option D"],\n'
        f'    "correct_index": 0,\n'
        f'    "explanation": "Detailed explanation of why this answer is correct."\n'
        f"  }}\n"
        f"]"
    )

    messages = [
        {"role": "system", "content": "You are an expert STEM quiz and test designer. Output valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    try:
        raw_output = call_chat_completion(
            model=MODEL_MUSE_SPARK,
            messages=messages,
            max_tokens=3000,
            temperature=0.3
        )
        quiz = _parse_quiz_json(raw_output)
        if quiz:
            return quiz
    except Exception as e:
        logger.error(f"Error calling Muse Spark for quiz: {e}")

    # Fallback questions
    return _generate_fallback_quiz(analyses, num_questions)


def _parse_quiz_json(raw_text: str) -> List[Dict[str, Any]]:
    """Safely extracts quiz JSON array from model output."""
    cleaned = raw_text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()

    try:
        items = json.loads(cleaned)
        if isinstance(items, list):
            valid_quiz = []
            for idx, item in enumerate(items, start=1):
                if "question" in item and "options" in item:
                    valid_quiz.append({
                        "id": idx,
                        "question": str(item["question"]),
                        "options": [str(opt) for opt in item.get("options", [])],
                        "correct_index": int(item.get("correct_index", 0)) % max(1, len(item.get("options", [1]))),
                        "explanation": str(item.get("explanation", "Review the visual diagram for confirmation."))
                    })
            if valid_quiz:
                return valid_quiz
    except Exception as e:
        logger.warning(f"Could not parse quiz JSON: {e}")

    # Bracket search fallback
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            items = json.loads(cleaned[start:end + 1])
            if isinstance(items, list):
                return [
                    {
                        "id": idx + 1,
                        "question": item.get("question", f"Question {idx + 1}"),
                        "options": item.get("options", ["A", "B", "C", "D"]),
                        "correct_index": int(item.get("correct_index", 0)),
                        "explanation": item.get("explanation", "")
                    }
                    for idx, item in enumerate(items)
                ]
        except Exception:
            pass

    return []


def _generate_fallback_summary(analyses: List[Dict[str, Any]], title: str) -> str:
    """Fallback study sheet generator."""
    lines = [
        f"# {title}: Technical Reference Guide & Formulation Summary",
        "",
        "## 1. Executive Conceptual Overview",
        "This reference summary synthesizes the governing mathematical laws, theoretical principles, and visual models extracted from the primary source document.",
        "",
        "## 2. Formal Mathematical Formulations & Derivations",
    ]
    for item in analyses:
        if item.get("latex"):
            lines.append(f"### {item.get('title')}")
            lines.append(f"{item.get('latex')}")
            lines.append(f"**Variable Specification & Physical Context:**")
            lines.append(f"{item.get('explanation')}")
            lines.append("")

    lines.extend([
        "## 3. Structural Schematics & Diagram Mechanics",
        "- Extracted visual architectures isolate fundamental system components, state transitions, and vector relations.",
        "- Diagram elements have been aligned with their corresponding mathematical proofs and governing operators.",
        "",
        "## 4. Analytical Constraints, Boundary Conditions & Edge Cases",
        "- Parameter validity is restricted to specified physical and domain conditions.",
        "- Ensure boundary assumptions are satisfied when applying these formulations to real-world datasets."
    ])
    return "\n".join(lines)


def _generate_fallback_quiz(analyses: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """Fallback questions when generator API is unavailable."""
    quiz = []
    for idx, item in enumerate(analyses[:count], start=1):
        quiz.append({
            "id": idx,
            "question": f"What is the primary role of '{item.get('title', 'the segmented element')}' in this system?",
            "options": [
                f"It governs the primary operational behavior of the system as described.",
                "It serves only as a decorative visual boundary.",
                "It represents an inactive parameter with negligible influence.",
                "It contradicts the governing theoretical equations."
            ],
            "correct_index": 0,
            "explanation": f"Based on the analysis: {item.get('explanation', 'The visual element is central to the core workflow.')}"
        })
    return quiz

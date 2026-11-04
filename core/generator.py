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
    user_instruction: str = "",
    provider: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    High-performance unified generator: compiles BOTH the master cheatsheet
    and the active-recall quiz in a single LLM round-trip.
    Supports routing via Baseten (2M TPM) or Meta Model API.
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
        f"You are an expert technical editor, system architect, and professor compiling an authoritative study guide.\n"
        f"Synthesize the extracted visual elements, diagrams, data tables, or formulations from '{document_title}'.\n\n"
        f"CONTENT DOMAIN ADAPTATION & ACCURACY (CRITICAL):\n"
        f"1. Detect the specific domain of the source material dynamically (e.g., Computer Networking, Software Architecture, Biology/Medicine, Physics, Mathematics, Electrical Engineering, Business Analytics, etc.).\n"
        f"2. STRICT FIDELITY: Base your analysis exclusively on what is actually shown in the visual elements.\n"
        f"3. MATHEMATICAL POLICY: IF the source contains genuine mathematical equations or physical laws, document them rigorously in LaTeX ($$ ... $$). BUT IF the source is a network trace (e.g. Wireshark), a software flowchart, a system UI, or a biological diagram with NO mathematical equations, DO NOT invent or hallucinate artificial math or set-theory formulas! Instead, explain the actual protocols, packet headers, flags, data fields, components, or workflows.\n\n"
        f"OUTPUT FORMAT REQUIREMENT (CRITICAL):\n"
        f"You must deliver TWO sections. Deliver the Quiz JSON first to ensure it is never truncated, followed by the Cheatsheet:\n\n"
        f"===QUIZ_JSON_START===\n"
        f"[\n"
        f"  {{\n"
        f'    "id": 1,\n'
        f'    "question": "A technical question testing understanding of what is shown in the visual (e.g., Wireshark filter syntax, port numbers, protocols, mathematical formulas, or biology pathways)?",\n'
        f'    "options": ["Accurate Option A", "Plausible Distractor B", "Plausible Distractor C", "Plausible Distractor D"],\n'
        f'    "correct_index": 0,\n'
        f'    "explanation": "Precise technical explanation referencing the source visual."\n'
        f"  }}\n"
        f"]\n"
        f"===QUIZ_JSON_END===\n\n"
        f"===CHEATSHEET_START===\n"
        f"# {document_title}: Technical Reference Guide & System Synthesis\n\n"
        f"## 1. Executive Conceptual Overview\n"
        f"(High-yield technical synthesis: What is this system/document? What is its core function, protocol, biological mechanism, or governing principle? Be concise and direct.)\n\n"
        f"## 2. Core Technical Mechanics, Formulations & Workflows\n"
        f"(Rigorous breakdown: If equations exist, provide formal LaTeX derivations. If network/software, detail packet structures, protocols, flags, IPs, and filter expressions. If biology/engineering, detail component interactions and functional sequences.)\n\n"
        f"## 3. Structural Schematics, Tables & Visual Layout\n"
        f"(Deconstruction of what the extracted diagrams, tables, figures, or screenshots show, explaining directional arrows, columns, nodes, and parameters.)\n\n"
        f"## 4. Practical Constraints, Boundary Conditions & Operational Notes\n"
        f"(Domain limitations, failure modes, error codes, edge cases, or analytical constraints.)\n"
        f"===CHEATSHEET_END===\n\n"
        f"TONE & STYLE:\n"
        f"- Strict technical rigor. Keep explanations crisp and high-yield.\n"
        f"- Generate {num_questions} conceptual active-recall questions grounded in the actual content of the visual.\n\n"
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
                "documentation and active-recall questions using the exact delimiter blocks requested."
            )
        },
        {"role": "user", "content": prompt}
    ]

    try:
        raw_output = call_chat_completion(
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
            provider=provider
        )

        summary_md = ""
        quiz_list = []

        # Parse quiz (handles quiz delivered first or second)
        if "===QUIZ_JSON_START===" in raw_output:
            q_part = raw_output.split("===QUIZ_JSON_START===")[1]
            if "===QUIZ_JSON_END===" in q_part:
                q_raw = q_part.split("===QUIZ_JSON_END===")[0].strip()
            else:
                q_raw = q_part.split("===CHEATSHEET_START===")[0].strip() if "===CHEATSHEET_START===" in q_part else q_part.strip()
            quiz_list = _parse_quiz_json(q_raw)

        # Parse cheatsheet
        if "===CHEATSHEET_START===" in raw_output:
            part = raw_output.split("===CHEATSHEET_START===")[1]
            if "===CHEATSHEET_END===" in part:
                summary_md = part.split("===CHEATSHEET_END===")[0].strip()
            else:
                summary_md = part.split("===QUIZ_JSON_START===")[0].strip() if "===QUIZ_JSON_START===" in part else part.strip()

        # Fallback if delimiters were omitted by the model
        if not summary_md:
            summary_md = raw_output.split("[")[0].strip() if "[" in raw_output else raw_output
        if not quiz_list:
            quiz_list = _parse_quiz_json(raw_output)

        if not quiz_list:
            quiz_list = _generate_fallback_quiz(analyses, num_questions, summary_text=summary_md)

        return summary_md, quiz_list

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
        f"You are an expert technical editor, system architect, and professor compiling an authoritative reference guide.\n"
        f"Synthesize the extracted visual elements, diagrams, data tables, or formulations from '{document_title}'.\n\n"
        f"CONTENT DOMAIN ADAPTATION & ACCURACY (CRITICAL):\n"
        f"1. Detect the specific domain of the source material dynamically (e.g., Computer Networking, Software Architecture, Biology/Medicine, Physics, Mathematics, Electrical Engineering, Business Analytics, etc.).\n"
        f"2. STRICT FIDELITY: Base your analysis exclusively on what is actually shown in the visual elements.\n"
        f"3. MATHEMATICAL POLICY: IF the source contains genuine mathematical equations or physical laws, document them rigorously in LaTeX ($$ ... $$). BUT IF the source is a network trace (e.g. Wireshark), a software flowchart, a system UI, or a biological diagram with NO mathematical equations, DO NOT invent or hallucinate artificial math or set-theory formulas! Instead, explain the actual protocols, packet headers, flags, data fields, components, or workflows.\n\n"
        f"REQUIRED MARKDOWN STRUCTURE:\n"
        f"# {document_title}: Technical Reference Guide & System Synthesis\n\n"
        f"## 1. Executive Conceptual Overview\n"
        f"(High-yield technical synthesis: What is this system/document? What is its core function, protocol, biological mechanism, or governing principle?)\n\n"
        f"## 2. Core Technical Mechanics, Formulations & Workflows\n"
        f"(Rigorous breakdown: If equations exist, provide formal LaTeX derivations. If network/software, detail packet structures, protocols, flags, IPs, and filter expressions. If biology/engineering, detail component interactions and functional sequences.)\n\n"
        f"## 3. Structural Schematics, Tables & Visual Layout\n"
        f"(Deconstruction of what the extracted diagrams, tables, figures, or screenshots show, explaining directional arrows, columns, nodes, and parameters.)\n\n"
        f"## 4. Practical Constraints, Boundary Conditions & Operational Notes\n"
        f"(Domain limitations, failure modes, error codes, edge cases, or analytical constraints.)\n\n"
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
        # Generous max_tokens to accommodate reasoning tokens + full markdown
        summary_markdown = call_chat_completion(
            messages=messages,
            max_tokens=3000,
            temperature=0.3,
            provider=provider
        )
        if summary_markdown:
            return summary_markdown
    except Exception as e:
        logger.error(f"Error calling LLM for cheatsheet: {e}")

    # Fallback compilation if API is unavailable
    return _generate_fallback_summary(analyses, document_title)


def generate_practice_quiz(
    analyses: List[Dict[str, Any]],
    num_questions: int = 4,
    user_instruction: str = "",
    provider: Optional[str] = None
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
        f"You are an expert technical instructor and exam designer creating active-recall revision questions based strictly on the provided visual elements.\n"
        f"Based on these extracted visual elements:\n"
        + "\n".join(elements_summary) + "\n\n"
    )
    if user_instruction:
        prompt += f"USER GUIDANCE / FOCUS GOAL:\n\"{user_instruction}\"\nFocus questions on concepts relevant to this directive.\n\n"

    prompt += (
        f"Generate {num_questions} high-quality, conceptual multiple-choice questions grounded strictly in the extracted visual content.\n"
        f"If the content is network traces/software, ask about protocols, packet fields, and filters. If it is math/physics, ask about equations.\n\n"
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
            messages=messages,
            max_tokens=3000,
            temperature=0.3,
            provider=provider
        )
        quiz = _parse_quiz_json(raw_output)
        if quiz:
            return quiz
    except Exception as e:
        logger.error(f"Error calling LLM for quiz: {e}")

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
        f"# {title}: Technical Reference Guide & System Synthesis",
        "",
        "## 1. Executive Conceptual Overview",
        f"This technical reference guide synthesizes the operational, architectural, and visual elements identified in '{title}'.",
        "",
        "## 2. Core Technical Mechanics, Formulations & Workflows",
    ]
    for item in analyses:
        item_title = item.get('title', 'Visual Asset')
        lines.append(f"### {item_title}")
        if item.get("latex"):
            lines.append(f"{item.get('latex')}")
        if item.get("explanation"):
            lines.append(f"{item.get('explanation')}")
        if item.get("key_takeaways"):
            lines.append("**Key Takeaways:**")
            for tk in item.get("key_takeaways", []):
                lines.append(f"- {tk}")
        lines.append("")

    lines.extend([
        "## 3. Structural Schematics, Tables & Visual Layout",
        "- Extracted visual regions isolate key components, parameters, data flows, and structural relationships.",
        "",
        "## 4. Practical Constraints, Boundary Conditions & Operational Notes",
        "- Verify domain conditions, parameter configurations, and protocol rules when interpreting this visual data."
    ])
    return "\n".join(lines)


def _generate_fallback_quiz(analyses: List[Dict[str, Any]], count: int, summary_text: str = "") -> List[Dict[str, Any]]:
    """Fallback questions when generator API is unavailable."""
    quiz = []
    
    # Extract candidate facts from summary_text if available
    summary_sentences = [
        s.strip() for s in re.split(r"[.\n]+", summary_text)
        if 25 < len(s.strip()) < 140 and not s.strip().startswith(("#", "-", "="))
    ]

    for idx, item in enumerate(analyses[:count], start=1):
        title = item.get('title', f'Visual Element #{idx}')
        explanation = item.get('explanation', '')
        takeaways = item.get('key_takeaways', [])
        
        # Pick best technical statement
        meaningful_takeaways = [t for t in takeaways if len(t) > 15 and "isolated" not in t.lower() and "preserved" not in t.lower()]
        if meaningful_takeaways:
            correct_ans = meaningful_takeaways[0]
        elif summary_sentences:
            correct_ans = summary_sentences[idx % len(summary_sentences)]
        else:
            correct_ans = f"Provides core operational data and protocol parameters for {title}."

        quiz.append({
            "id": idx,
            "question": f"Based on the analysis of '{title}', what is its primary technical significance?",
            "options": [
                correct_ans,
                "It serves only as a decorative visual boundary with no functional relevance.",
                "It indicates a deprecated legacy protocol replaced by modern standards.",
                "It represents an inactive parameter with negligible influence."
            ],
            "correct_index": 0,
            "explanation": explanation or f"The visual asset provides primary technical data for {title}."
        })
    return quiz

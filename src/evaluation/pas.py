# ══════════════════════════════════════════════════════════════════
# src/evaluation/pas.py
# Persona Adherence Score (PAS) + 4 sub-dimensions.
#
# Sub-dimensions (same names, different rubrics per persona):
#   tone_appropriateness  : technical depth (ENG) vs plain language (MKT)
#   structural_compliance : format and length constraints
#   audience_fit          : practical utility for the target reader
#   constraint_adherence  : absence of prohibited elements
#
# PAS weight in composite:
#   engineering 15%  marketing 25%
# ══════════════════════════════════════════════════════════════════

import json
import re
import logging
from langchain_huggingface import HuggingFacePipeline
from configs.personas import get_pas_sub_dims

logger = logging.getLogger(__name__)

_RUBRICS: dict[str, dict[str, str]] = {
    "engineering": {
        "tone_appropriateness":  "Does the answer use technical language, explain mechanisms (WHY/HOW), and include specific proper nouns and numerical values?",
        "structural_compliance": "Is the answer a single unbroken paragraph of 3–5 sentences with no bullets, headers, or paragraph breaks?",
        "audience_fit":          "Would a software engineer find this answer precise and directly actionable for a technical decision?",
        "constraint_adherence":  "Does the answer avoid citations, author names, parenthetical source echoes, and raw quoted passages?",
    },
    "marketing": {
        "tone_appropriateness":  "Does the answer use plain, jargon-free language and frame the topic in terms of business value or user benefit?",
        "structural_compliance": "Is the answer limited to 1–3 complete sentences with no bullets, headers, formatting, or uncertainty disclaimers?",
        "audience_fit":          "Would a marketing professional find this answer clear, credible, and immediately usable in content or communications?",
        "constraint_adherence":  "Does the answer avoid citations, technical acronyms, uncertainty qualifiers ('may', 'might', 'according to'), and parenthetical echoes?",
    },
}

_PAS_PROMPT = """\
You are a strict persona compliance judge.
Score the answer for adherence to the {persona} persona on a scale of 0.0 to 1.0 per dimension.
Return ONLY valid JSON with no explanation, no preamble, no markdown.

Answer to evaluate:
{answer}

Scoring rubric:
{rubric}

Return exactly this JSON:
{{
{json_schema}
}}

JSON:"""


def run_pas_judge(
    answer:  str,
    persona: str,
    llm:     HuggingFacePipeline,
) -> dict[str, float]:
    """
    Score an answer for persona adherence.
    Returns one float per sub-dimension + "persona_adherence_score" (mean).
    All values are NaN on failure.
    """
    sub_dims    = get_pas_sub_dims(persona)
    rubric_text = "\n".join(f"  {d}: {_RUBRICS[persona][d]}" for d in sub_dims)
    json_schema = "\n".join(f'  "{d}": <float 0.0–1.0>,' for d in sub_dims).rstrip(",")

    prompt = _PAS_PROMPT.format(
        persona     = persona,
        answer      = answer,
        rubric      = rubric_text,
        json_schema = json_schema,
    )

    fallback = {d: float("nan") for d in sub_dims}
    fallback["persona_adherence_score"] = float("nan")

    try:
        raw    = llm.invoke(prompt)
        output = raw if isinstance(raw, str) else str(raw)

        match = re.search(r"\{[^{}]+\}", output, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in PAS output: {output[:300]}")

        parsed = json.loads(match.group())
        result = {}
        valid  = []
        for dim in sub_dims:
            val = parsed.get(dim)
            if val is None:
                result[dim] = float("nan")
            else:
                clipped     = float(max(0.0, min(1.0, float(val))))
                result[dim] = clipped
                valid.append(clipped)

        result["persona_adherence_score"] = (
            sum(valid) / len(valid) if valid else float("nan")
        )
        return result

    except Exception as e:
        logger.warning(f"PAS judge failed (persona={persona}): {e}")
        return fallback

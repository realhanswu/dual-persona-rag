# ══════════════════════════════════════════════════════════════════
# src/evaluation/llm_judges.py
# On-device Mistral-7B judge for structured scoring.
#
#   judge_faithfulness       — primary faithfulness gate source
#   judge_answer_correctness — selected for composite
#   judge_context_relevance  — diagnostic
#   judge_answer_relevance   — selected for composite
#
# On parse/inference failure all scores return NaN;
# core_metrics.py falls back to semantic approximations.
# ══════════════════════════════════════════════════════════════════

import json
import re
import math
import logging
from langchain_huggingface import HuggingFacePipeline

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are a strict evaluation judge. Score the answer on a scale of 0.0 to 1.0 for each criterion.
Return ONLY valid JSON with no explanation, no preamble, no markdown.

Question:
{question}

Retrieved Context:
{context}

Gold Reference Answer:
{gold}

Generated Answer:
{answer}

Scoring criteria:
- judge_faithfulness: Is every claim in the answer supported by the retrieved context? (1.0 = fully grounded, 0.0 = hallucinated)
- judge_answer_correctness: Does the answer match the factual content of the gold reference? (1.0 = exact match, 0.0 = wrong)
- judge_context_relevance: Does the retrieved context contain information relevant to the question? (1.0 = highly relevant, 0.0 = irrelevant)
- judge_answer_relevance: Does the answer directly and completely address the question? (1.0 = fully answers it, 0.0 = off-topic)

Return exactly this JSON:
{{
  "judge_faithfulness": <float>,
  "judge_answer_correctness": <float>,
  "judge_context_relevance": <float>,
  "judge_answer_relevance": <float>
}}

JSON:"""

_FALLBACK: dict[str, float] = {
    "judge_faithfulness":       float("nan"),
    "judge_answer_correctness": float("nan"),
    "judge_context_relevance":  float("nan"),
    "judge_answer_relevance":   float("nan"),
}

_CONTEXT_CHAR_LIMIT = 3000


def run_llm_judge(
    question: str,
    context:  str,
    answer:   str,
    gold:     str,
    llm:      HuggingFacePipeline,
) -> dict[str, float]:
    """
    Run the Mistral-7B judge on a single question/answer instance.
    Returns a dict with four float scores, falling back to NaN on any failure.
    """
    prompt = _JUDGE_PROMPT.format(
        question = question,
        context  = context[:_CONTEXT_CHAR_LIMIT],
        gold     = gold,
        answer   = answer,
    )

    try:
        raw    = llm.invoke(prompt)
        output = raw if isinstance(raw, str) else str(raw)

        match = re.search(r"\{[^{}]+\}", output, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in judge output: {output[:300]}")

        parsed = json.loads(match.group())
        result = {}
        for key in _FALLBACK:
            val = parsed.get(key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                result[key] = float("nan")
            else:
                result[key] = float(max(0.0, min(1.0, float(val))))
        return result

    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return _FALLBACK.copy()

# ══════════════════════════════════════════════════════════════════
# configs/personas.py
# Per-persona thresholds, composite weights, PAS sub-dimensions,
# and cross-persona combination weights.
#
# Design rationale:
#   Engineering  — retrieval precision + factual depth paramount.
#                  Context Hit Rate and BERTScore F1 carry 20% each.
#   Marketing    — tonal appropriateness operationally critical.
#                  Persona Adherence (PAS) carries 25%.
#
#   Cross-persona weights: 75% ENG / 25% MKT, reflecting ~300
#   engineers vs. ~40 marketing staff (actual deployment ratio).
#
# Faithfulness gate thresholds are set HIGHER for marketing (0.80)
# because a factually unreliable response published in customer-
# facing content carries greater organisational risk than an
# imprecise internal technical answer.
# ══════════════════════════════════════════════════════════════════

from typing import Literal

Persona = Literal["engineering", "marketing"]


# ──────────────────────────────────────────────────────────────────
# CROSS-PERSONA COMBINATION WEIGHTS
# Applied in compute_combined_persona_score().
# Must sum to 1.0.
# ──────────────────────────────────────────────────────────────────
PERSONA_COMBINATION_WEIGHTS: dict[Persona, float] = {
    "engineering": 0.75,
    "marketing":   0.25,
}

# Combined GO threshold — applied to the weighted cross-persona score.
COMBINED_GO_THRESHOLD: float = 0.65


# ──────────────────────────────────────────────────────────────────
# PER-PERSONA METRIC THRESHOLDS
# Each metric is checked independently against its threshold.
# FP6 (specificity gap) is a lower-is-better metric — flagged ❌
# when value EXCEEDS the threshold.
# ──────────────────────────────────────────────────────────────────
PERSONA_THRESHOLDS: dict[Persona, dict[str, float]] = {
    "engineering": {
        # Retrieval
        "FP3_context_utilization":   0.80,
        "context_recall":            0.75,
        "context_precision_at_k":    0.70,
        "MRR":                       0.70,
        "NDCG_at_k":                 0.75,
        # Generation — semantic
        "FP4_faithfulness":          0.75,  # faithfulness gate threshold ★
        "FP7_completeness_recall":   0.78,
        "FP6_specificity_gap":       0.12,  # lower is better
        "bertscore_f1":              0.82,
        "answer_correctness":        0.75,
        # LLM Judge
        "judge_faithfulness":        0.70,
        "judge_answer_correctness":  0.70,
        "judge_context_relevance":   0.65,
        "judge_answer_relevance":    0.70,
        # Persona
        "persona_adherence_score":   0.70,
        # Composite
        "composite":                 0.72,
    },
    "marketing": {
        # Retrieval
        "FP3_context_utilization":   0.75,
        "context_recall":            0.70,
        "context_precision_at_k":    0.65,
        "MRR":                       0.65,
        "NDCG_at_k":                 0.70,
        # Generation — semantic
        "FP4_faithfulness":          0.80,  # faithfulness gate threshold ★ (higher — customer-facing risk)
        "FP7_completeness_recall":   0.72,
        "FP6_specificity_gap":       0.10,  # lower is better (tighter — marketing copy must be precise)
        "bertscore_f1":              0.78,
        "answer_correctness":        0.70,
        # LLM Judge
        "judge_faithfulness":        0.75,
        "judge_answer_correctness":  0.70,
        "judge_context_relevance":   0.65,
        "judge_answer_relevance":    0.72,
        # Persona
        "persona_adherence_score":   0.72,
        # Composite
        "composite":                 0.70,
    },
}


# ──────────────────────────────────────────────────────────────────
# PER-PERSONA COMPOSITE WEIGHTS  (6 components, must sum to 1.0)
# Applied in compute_core_metrics() and compute_combined_persona_score().
# Keys must exactly match compute_core_metrics() output field names.
# ──────────────────────────────────────────────────────────────────
PERSONA_COMPOSITE_WEIGHTS: dict[Persona, dict[str, float]] = {
    "engineering": {
        "FP3_context_utilization":   0.20,  # retrieval coverage ★★
        "NDCG_at_k":                 0.15,  # retrieval ranking
        "bertscore_f1":              0.20,  # generation quality ★★
        "judge_answer_relevance":    0.15,  # judge: on-topic?
        "judge_answer_correctness":  0.15,  # judge: factually correct?
        "persona_adherence_score":   0.15,  # persona fit
    },
    "marketing": {
        "FP3_context_utilization":   0.15,  # retrieval coverage
        "NDCG_at_k":                 0.10,  # retrieval ranking
        "bertscore_f1":              0.15,  # generation quality
        "judge_answer_relevance":    0.20,  # judge: on-topic? ★★ (audience judges relevance first)
        "judge_answer_correctness":  0.15,  # judge: factually correct?
        "persona_adherence_score":   0.25,  # persona fit ★★ (tonal appropriateness is critical)
    },
}

# Validate at import time
for _persona, _weights in PERSONA_COMPOSITE_WEIGHTS.items():
    _total = sum(_weights.values())
    assert abs(_total - 1.0) < 1e-6, (
        f"PERSONA_COMPOSITE_WEIGHTS['{_persona}'] sums to {_total:.6f}, expected 1.0"
    )

_comb_total = sum(PERSONA_COMBINATION_WEIGHTS.values())
assert abs(_comb_total - 1.0) < 1e-6, (
    f"PERSONA_COMBINATION_WEIGHTS sums to {_comb_total:.6f}, expected 1.0"
)


# ──────────────────────────────────────────────────────────────────
# PERSONA ADHERENCE SCORE (PAS) SUB-DIMENSIONS
# Each sub-dimension is scored 0.0–1.0 by the on-device Mistral-7B
# judge and averaged into the PAS overall score.
#
# Dimension definitions:
#   tone_appropriateness  — Technical depth (ENG) vs. plain language (MKT)
#   structural_compliance — Sentence count, paragraph format, no bullets/headers
#   audience_fit          — Relevance and practical utility to the target reader
#   constraint_adherence  — No citations, raw quotes, parenthetical echoes, jargon (MKT)
# ──────────────────────────────────────────────────────────────────
PAS_SUB_DIMS: dict[Persona, list[str]] = {
    "engineering": [
        "tone_appropriateness",   # Technical depth, mechanism explanation, exact proper nouns
        "structural_compliance",  # Single paragraph, 3–5 sentences, no bullets/headers
        "audience_fit",           # Useful to an engineer acting on technical detail
        "constraint_adherence",   # No citations, no parenthetical source echoes
    ],
    "marketing": [
        "tone_appropriateness",   # Plain language, no jargon, business-relevant framing
        "structural_compliance",  # 1–3 sentences, no bullets/headers/disclaimers
        "audience_fit",           # Useful to a content producer under deadline
        "constraint_adherence",   # No citations, no uncertainty disclaimers, no technical register
    ],
}

# PAS sub-dimension display labels (used in console reports)
PAS_SUB_DIM_LABELS: dict[str, str] = {
    "tone_appropriateness":  "Tone Appropriateness",
    "structural_compliance": "Structural Compliance",
    "audience_fit":          "Audience Fit",
    "constraint_adherence":  "Constraint Adherence",
}


# ──────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────
def get_thresholds(persona: Persona) -> dict[str, float]:
    if persona not in PERSONA_THRESHOLDS:
        raise KeyError(f"Unknown persona '{persona}'. Choose: {list(PERSONA_THRESHOLDS)}")
    return PERSONA_THRESHOLDS[persona]


def get_composite_weights(persona: Persona) -> dict[str, float]:
    if persona not in PERSONA_COMPOSITE_WEIGHTS:
        raise KeyError(f"Unknown persona '{persona}'. Choose: {list(PERSONA_COMPOSITE_WEIGHTS)}")
    return PERSONA_COMPOSITE_WEIGHTS[persona]


def get_pas_sub_dims(persona: Persona) -> list[str]:
    if persona not in PAS_SUB_DIMS:
        raise KeyError(f"Unknown persona '{persona}'. Choose: {list(PAS_SUB_DIMS)}")
    return PAS_SUB_DIMS[persona]


def print_persona_summary() -> None:
    """Print a human-readable summary of all persona config."""
    W = 68
    for persona in ("engineering", "marketing"):
        print(f"\n{'═' * W}")
        print(f"  PERSONA: {persona.upper()}")
        print(f"{'─' * W}")
        print(f"  {'COMPOSITE WEIGHTS':}")
        for k, v in PERSONA_COMPOSITE_WEIGHTS[persona].items():
            bar = "█" * int(v * 40)
            print(f"    {k:<36}  {v:.2f}  {bar}")
        print(f"{'─' * W}")
        print(f"  {'KEY THRESHOLDS':}")
        key_thresh = [
            "FP3_context_utilization", "NDCG_at_k", "bertscore_f1",
            "FP4_faithfulness", "judge_answer_correctness",
            "persona_adherence_score", "composite",
        ]
        for k in key_thresh:
            v = PERSONA_THRESHOLDS[persona][k]
            direction = "↓ <=" if k == "FP6_specificity_gap" else "↑ >="
            print(f"    {k:<36}  {direction} {v}")
        print(f"{'─' * W}")
        print(f"  PAS SUB-DIMENSIONS:")
        for dim in PAS_SUB_DIMS[persona]:
            print(f"    • {PAS_SUB_DIM_LABELS[dim]}")
    print(f"\n  Cross-persona weights: "
          f"ENG={PERSONA_COMBINATION_WEIGHTS['engineering']:.0%}  "
          f"MKT={PERSONA_COMBINATION_WEIGHTS['marketing']:.0%}  "
          f"(GO threshold >= {COMBINED_GO_THRESHOLD})")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    print_persona_summary()

# ══════════════════════════════════════════════════════════════════
# src/evaluation/combined_score.py
# Combines per-persona composites into a cross-persona score
# and issues the final GO / NO-GO verdict.
#
# Decision rule (BOTH conditions must pass):
#   1. combined_score >= COMBINED_GO_THRESHOLD (0.75)
#   2. Both persona faithfulness gates pass independently
#
# Cross-persona weights: 75% ENG / 25% MKT
# ══════════════════════════════════════════════════════════════════

import math
import logging
from configs.personas import (
    PERSONA_COMBINATION_WEIGHTS,
    COMBINED_GO_THRESHOLD,
    get_thresholds,
)

logger = logging.getLogger(__name__)


def _nanmean(values: list) -> float:
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(valid) / len(valid) if valid else float("nan")


def compute_combined_persona_score(
    eng_records: list[dict],
    mkt_records: list[dict],
) -> dict:
    """
    Aggregate per-question records → persona composites → GO/NO-GO.

    Args:
        eng_records : list of per-question dicts from compute_core_metrics (engineering)
        mkt_records : list of per-question dicts from compute_core_metrics (marketing)

    Returns:
        Flat dict with verdict, scores, gate pass/fail flags, and counts.
    """
    eng_composite = _nanmean([r["composite_score"]         for r in eng_records])
    mkt_composite = _nanmean([r["composite_score"]         for r in mkt_records])
    eng_faith     = _nanmean([r["faithfulness_gate_score"] for r in eng_records])
    mkt_faith     = _nanmean([r["faithfulness_gate_score"] for r in mkt_records])

    eng_thresh = get_thresholds("engineering")
    mkt_thresh = get_thresholds("marketing")

    eng_gate_pass = not math.isnan(eng_faith) and eng_faith >= eng_thresh["FP4_faithfulness"]
    mkt_gate_pass = not math.isnan(mkt_faith) and mkt_faith >= mkt_thresh["FP4_faithfulness"]

    combined_score = (
        PERSONA_COMBINATION_WEIGHTS["engineering"] * eng_composite
        + PERSONA_COMBINATION_WEIGHTS["marketing"]  * mkt_composite
    )

    score_pass = not math.isnan(combined_score) and combined_score >= COMBINED_GO_THRESHOLD
    verdict    = "GO" if (score_pass and eng_gate_pass and mkt_gate_pass) else "NO-GO"

    result = {
        "verdict":             verdict,
        "combined_score":      round(combined_score, 4),
        "combined_threshold":  COMBINED_GO_THRESHOLD,
        "score_pass":          score_pass,
        "eng_composite":       round(eng_composite, 4),
        "mkt_composite":       round(mkt_composite, 4),
        "eng_faithfulness":    round(eng_faith,     4),
        "mkt_faithfulness":    round(mkt_faith,     4),
        "eng_faith_threshold": eng_thresh["FP4_faithfulness"],
        "mkt_faith_threshold": mkt_thresh["FP4_faithfulness"],
        "eng_gate_pass":       eng_gate_pass,
        "mkt_gate_pass":       mkt_gate_pass,
        "n_eng_questions":     len(eng_records),
        "n_mkt_questions":     len(mkt_records),
    }

    W = 54
    logger.info(
        f"\n{'═' * W}\n"
        f"  VERDICT  :  {verdict}\n"
        f"{'─' * W}\n"
        f"  Combined Score   : {combined_score:.4f}  (threshold >= {COMBINED_GO_THRESHOLD})  {'✅' if score_pass else '❌'}\n"
        f"  ENG composite    : {eng_composite:.4f}  | Faith gate {'✅ PASS' if eng_gate_pass else '❌ FAIL'}  ({eng_faith:.4f} vs {eng_thresh['FP4_faithfulness']})\n"
        f"  MKT composite    : {mkt_composite:.4f}  | Faith gate {'✅ PASS' if mkt_gate_pass else '❌ FAIL'}  ({mkt_faith:.4f} vs {mkt_thresh['FP4_faithfulness']})\n"
        f"{'═' * W}"
    )

    return result

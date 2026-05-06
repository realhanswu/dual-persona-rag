# ══════════════════════════════════════════════════════════════════
# src/evaluation/combined_score.py
# Combines per-persona composites into a single cross-persona score
# and issues the final GO / NO-GO verdict.
# ENG 75% / MKT 25% — reflects ~300 engineers vs ~40 marketing staff.
# ══════════════════════════════════════════════════════════════════

import math
import logging
from configs.personas import (
    PERSONA_COMBINATION_WEIGHTS,
    COMBINED_GO_THRESHOLD,
    get_thresholds,
)

logger = logging.getLogger(__name__)


def compute_combined_persona_score(
    eng_records: list[dict],
    mkt_records: list[dict],
) -> dict:
    """
    Aggregate per-question records into persona composites,
    apply faithfulness gates, and issue GO/NO-GO verdict.
    """
    def _nanmean(values):
        valid = [v for v in values if v is not None and not math.isnan(v)]
        return sum(valid) / len(valid) if valid else float("nan")

    eng_composite  = _nanmean([r["composite_score"]        for r in eng_records])
    mkt_composite  = _nanmean([r["composite_score"]        for r in mkt_records])
    eng_faith_gate = _nanmean([r["faithfulness_gate_score"] for r in eng_records])
    mkt_faith_gate = _nanmean([r["faithfulness_gate_score"] for r in mkt_records])

    eng_threshold = get_thresholds("engineering")
    mkt_threshold = get_thresholds("marketing")

    eng_gate_pass = (
        not math.isnan(eng_faith_gate)
        and eng_faith_gate >= eng_threshold["FP4_faithfulness"]
    )
    mkt_gate_pass = (
        not math.isnan(mkt_faith_gate)
        and mkt_faith_gate >= mkt_threshold["FP4_faithfulness"]
    )

    combined_score = (
        PERSONA_COMBINATION_WEIGHTS["engineering"] * eng_composite
        + PERSONA_COMBINATION_WEIGHTS["marketing"]  * mkt_composite
    )

    score_pass = (
        not math.isnan(combined_score)
        and combined_score >= COMBINED_GO_THRESHOLD
    )

    verdict = "GO" if (score_pass and eng_gate_pass and mkt_gate_pass) else "NO-GO"

    result = {
        "verdict":             verdict,
        "combined_score":      round(combined_score, 4),
        "combined_threshold":  COMBINED_GO_THRESHOLD,
        "score_pass":          score_pass,
        "eng_composite":       round(eng_composite, 4),
        "mkt_composite":       round(mkt_composite, 4),
        "eng_faithfulness":    round(eng_faith_gate, 4),
        "mkt_faithfulness":    round(mkt_faith_gate, 4),
        "eng_faith_threshold": eng_threshold["FP4_faithfulness"],
        "mkt_faith_threshold": mkt_threshold["FP4_faithfulness"],
        "eng_gate_pass":       eng_gate_pass,
        "mkt_gate_pass":       mkt_gate_pass,
        "n_eng_questions":     len(eng_records),
        "n_mkt_questions":     len(mkt_records),
    }

    logger.info(
        f"\n{'═'*52}\n"
        f"  VERDICT: {verdict}\n"
        f"  Combined Score : {combined_score:.4f}  (threshold >= {COMBINED_GO_THRESHOLD})\n"
        f"  ENG composite  : {eng_composite:.4f}  | Faith gate: {'✅ PASS' if eng_gate_pass else '❌ FAIL'} ({eng_faith_gate:.4f})\n"
        f"  MKT composite  : {mkt_composite:.4f}  | Faith gate: {'✅ PASS' if mkt_gate_pass else '❌ FAIL'} ({mkt_faith_gate:.4f})\n"
        f"{'═'*52}"
    )
    return result

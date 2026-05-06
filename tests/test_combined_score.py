import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.combined_score import compute_combined_persona_score


def _make_records(composite: float, faithfulness: float, n: int = 5) -> list[dict]:
    return [
        {"composite_score": composite, "faithfulness_gate_score": faithfulness}
        for _ in range(n)
    ]


def test_go_verdict():
    eng = _make_records(0.92, 0.98)
    mkt = _make_records(0.88, 0.95)
    result = compute_combined_persona_score(eng, mkt)
    assert result["verdict"] == "GO"
    assert result["combined_score"] >= 0.65


def test_nogo_on_low_composite():
    eng = _make_records(0.40, 0.90)
    mkt = _make_records(0.35, 0.90)
    result = compute_combined_persona_score(eng, mkt)
    assert result["verdict"] == "NO-GO"
    assert not result["score_pass"]


def test_nogo_on_failed_faithfulness_gate():
    eng = _make_records(0.95, 0.50)   # faith gate fails for ENG (threshold 0.75)
    mkt = _make_records(0.90, 0.95)
    result = compute_combined_persona_score(eng, mkt)
    assert result["verdict"] == "NO-GO"
    assert not result["eng_gate_pass"]


def test_combined_score_weighting():
    eng = _make_records(1.0, 1.0)
    mkt = _make_records(0.0, 1.0)
    result = compute_combined_persona_score(eng, mkt)
    assert abs(result["combined_score"] - 0.75) < 0.01   # 75% ENG * 1.0 + 25% MKT * 0.0

import pytest
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentence_transformers import SentenceTransformer
from src.evaluation.generation_metrics import (
    fp4_semantic_faithfulness,
    bertscore_metrics,
    semantic_answer_correctness,
)

@pytest.fixture(scope="module")
def model():
    return SentenceTransformer("multi-qa-mpnet-base-dot-v1")

ANSWER  = "RAG improves factual accuracy by grounding responses in retrieved documents."
CONTEXT = ["RAG uses a retriever to surface relevant passages before generation."]
GOLD    = "Retrieval-augmented generation grounds model outputs in retrieved source material."

def test_fp4_grounded(model):
    score = fp4_semantic_faithfulness(ANSWER, CONTEXT, model)
    assert not math.isnan(score)
    assert score >= 0.0

def test_fp4_empty_context(model):
    score = fp4_semantic_faithfulness(ANSWER, [], model)
    assert math.isnan(score)

def test_bertscore_f1_range():
    result = bertscore_metrics(ANSWER, GOLD)
    assert 0.0 <= result["bertscore_f1"] <= 1.0
    assert result["FP6_specificity_gap"] >= 0.0

def test_semantic_correctness_self(model):
    score = semantic_answer_correctness(ANSWER, ANSWER, model)
    assert score > 0.99

def test_semantic_correctness_empty(model):
    score = semantic_answer_correctness("", GOLD, model)
    assert math.isnan(score)

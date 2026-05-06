import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentence_transformers import SentenceTransformer
from src.evaluation.retrieval_metrics import (
    fp3_context_hit_rate,
    context_recall_semantic,
    context_precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)

@pytest.fixture(scope="module")
def model():
    return SentenceTransformer("multi-qa-mpnet-base-dot-v1")

GOLD    = "RAG combines a retrieval step with a generative language model."
MATCH   = "Retrieval-Augmented Generation (RAG) integrates a retriever with a generator."
NOMATCH = "The weather in San Francisco is often foggy in summer."

def test_fp3_hit(model):
    assert fp3_context_hit_rate(GOLD, [MATCH], model) == 1.0

def test_fp3_miss(model):
    assert fp3_context_hit_rate(GOLD, [NOMATCH], model) == 0.0

def test_fp3_empty_chunks(model):
    import math
    assert math.isnan(fp3_context_hit_rate(GOLD, [], model))

def test_context_recall_full(model):
    score = context_recall_semantic(GOLD, [MATCH, MATCH], model)
    assert score >= 0.5

def test_mrr_first_position(model):
    score = mean_reciprocal_rank(GOLD, [MATCH, NOMATCH], model)
    assert score == 1.0

def test_ndcg_perfect(model):
    score = ndcg_at_k(GOLD, [MATCH, MATCH, MATCH], model, k=3)
    assert score > 0.8

# ══════════════════════════════════════════════════════════════════
# src/evaluation/retrieval_metrics.py
# Stage 1 retrieval metrics — five metrics computed per question.
# All metrics reuse the production embedding model to ensure
# similarity thresholds are calibrated to the same vector space.
#
#   FP3  — Context Hit Rate     selected for composite
#   CR   — Context Recall       diagnostic
#   CP   — Context Precision@K  diagnostic
#   MRR  — Mean Reciprocal Rank diagnostic
#   NDCG — NDCG@K               selected for composite
# ══════════════════════════════════════════════════════════════════

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_HIT_THRESHOLD = 0.60   # FP3 cosine similarity threshold (empirically calibrated)
_TOP_K         = 3      # position cutoff for precision and NDCG


def _embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True)


def fp3_context_hit_rate(
    gold:      str,
    chunks:    list[str],
    model:     SentenceTransformer,
    threshold: float = _HIT_THRESHOLD,
) -> float:
    """
    1.0 if any retrieved chunk exceeds the cosine similarity threshold
    with the gold answer, else 0.0.
    Selected for composite — most direct indicator of retrieval success.
    """
    if not chunks:
        return float("nan")
    gold_emb  = _embed(model, [gold])
    chunk_emb = _embed(model, chunks)
    max_sim   = float(cosine_similarity(gold_emb, chunk_emb).max())
    return 1.0 if max_sim >= threshold else 0.0


def context_recall_semantic(
    gold:   str,
    chunks: list[str],
    model:  SentenceTransformer,
) -> float:
    """
    Fraction of gold answer sentences covered semantically by
    at least one retrieved chunk (cosine sim >= threshold).
    """
    if not chunks:
        return float("nan")
    sents = [s.strip() for s in gold.split(".") if s.strip()]
    if not sents:
        return float("nan")
    gold_emb  = _embed(model, sents)
    chunk_emb = _embed(model, chunks)
    sims      = cosine_similarity(gold_emb, chunk_emb)
    covered   = (sims.max(axis=1) >= _HIT_THRESHOLD).sum()
    return float(covered / len(sents))


def context_precision_at_k(
    gold:   str,
    chunks: list[str],
    model:  SentenceTransformer,
    k:      int = _TOP_K,
) -> float:
    """
    Proportion of the top-k retrieved chunks that are relevant
    to the gold answer.
    """
    if not chunks:
        return float("nan")
    top_k     = chunks[:k]
    gold_emb  = _embed(model, [gold])
    chunk_emb = _embed(model, top_k)
    sims      = cosine_similarity(gold_emb, chunk_emb).flatten()
    relevant  = (sims >= _HIT_THRESHOLD).sum()
    return float(relevant / len(top_k))


def mean_reciprocal_rank(
    gold:   str,
    chunks: list[str],
    model:  SentenceTransformer,
) -> float:
    """
    Reciprocal rank of the first relevant chunk in the ranked list.
    Returns 0.0 if no relevant chunk is found.
    """
    if not chunks:
        return float("nan")
    

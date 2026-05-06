# ══════════════════════════════════════════════════════════════════
# src/retrieval/reranker.py
# Cross-encoder reranker.
# Modest but non-negative gains at POC corpus scale.
# Adds 40–60% to inference runtime per question.
# ══════════════════════════════════════════════════════════════════

import logging
from langchain.schema import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_cross_encoder(
    model_name: str = _DEFAULT_RERANKER_MODEL,
) -> CrossEncoder:
    """Load and return the cross-encoder model."""
    logger.info(f"Loading cross-encoder: {model_name}")
    return CrossEncoder(model_name)


def rerank_chunks(
    query:         str,
    chunks:        list[Document],
    cross_encoder: CrossEncoder,
    top_n:         int,
) -> list[Document]:
    """Re-score candidate chunks against the query and return top_n."""
    if not chunks:
        return []

    pairs  = [(query, chunk.page_content) for chunk in chunks]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(scores, chunks),
        key     = lambda x: x[0],
        reverse = True,
    )

    reranked = [chunk for _, chunk in ranked[:top_n]]
    logger.debug(
        f"Reranked {len(chunks)} → {len(reranked)} chunks "
        f"(top score: {ranked[0][0]:.4f}, bottom kept: {ranked[top_n - 1][0]:.4f})"
    )
    return reranked

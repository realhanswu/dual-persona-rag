# ══════════════════════════════════════════════════════════════════
# src/retrieval/retriever.py
# MMR retriever factory.
# Parameters:
#   fetch_k     : initial candidate pool before diversity filtering
#   retriever_k : diverse candidates forwarded to the reranker
# ══════════════════════════════════════════════════════════════════

import logging
from langchain_community.vectorstores import Qdrant
from langchain.schema.retriever import BaseRetriever
from src.utils.config import RAGExperimentConfig

logger = logging.getLogger(__name__)


def build_retriever(
    vector_store: Qdrant,
    config:       RAGExperimentConfig,
) -> BaseRetriever:
    """
    Build a retriever from the Qdrant vector store.
    Routes to hybrid.py when search_type="hybrid".
    """
    if config.search_type == "hybrid":
        from src.retrieval.hybrid import build_hybrid_retriever
        logger.info("Building hybrid BM25 + dense MMR retriever")
        return build_hybrid_retriever(vector_store, config)

    search_kwargs: dict = {"k": config.retriever_k}

    if config.search_type == "mmr":
        search_kwargs["fetch_k"] = config.fetch_k
        logger.info(
            f"Building MMR retriever "
            f"(fetch_k={config.fetch_k}, k={config.retriever_k})"
        )
    else:
        logger.info(f"Building similarity retriever (k={config.retriever_k})")

    return vector_store.as_retriever(
        search_type   = config.search_type,
        search_kwargs = search_kwargs,
    )

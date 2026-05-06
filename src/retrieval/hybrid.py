# ══════════════════════════════════════════════════════════════════
# src/retrieval/hybrid.py
# BM25 + dense MMR hybrid retriever (experimental).
# No meaningful improvement over dense-only MMR at POC corpus scale.
# Retained for evaluation on larger or proprietary corpora.
# ══════════════════════════════════════════════════════════════════

import logging
from typing import Any
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain_community.vectorstores import Qdrant
from rank_bm25 import BM25Okapi
from pydantic import Field
from src.utils.config import RAGExperimentConfig

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Combines BM25 sparse retrieval with dense MMR vector retrieval."""

    vector_store: Any            = Field(description="Qdrant vector store instance")
    bm25_index:   Any            = Field(description="BM25Okapi index over corpus chunks")
    corpus_docs:  list[Document] = Field(description="All documents in BM25 index")
    config:       Any            = Field(description="RAGExperimentConfig")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        dense_results = self.vector_store.max_marginal_relevance_search(
            query   = query,
            k       = self.config.retriever_k,
            fetch_k = self.config.fetch_k,
        )

        tokenized_query = query.lower().split()
        bm25_scores     = self.bm25_index.get_scores(tokenized_query)
        top_bm25_idx    = sorted(
            range(len(bm25_scores)),
            key     = lambda i: bm25_scores[i],
            reverse = True,
        )[:self.config.retriever_k]
        bm25_results = [self.corpus_docs[i] for i in top_bm25_idx]

        seen, merged = set(), []
        for doc in dense_results + bm25_results:
            key = hash(doc.page_content[:200])
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        logger.debug(
            f"Hybrid: dense={len(dense_results)}, "
            f"bm25={len(bm25_results)}, merged={len(merged)}"
        )
        return merged[:self.config.retriever_k]


def build_hybrid_retriever(
    vector_store: Qdrant,
    config:       RAGExperimentConfig,
) -> HybridRetriever:
    all_docs   = vector_store.similarity_search("", k=10_000)
    corpus     = [doc.page_content.lower().split() for doc in all_docs]
    bm25_index = BM25Okapi(corpus)
    logger.info(f"BM25 index built over {len(all_docs)} chunks")
    return HybridRetriever(
        vector_store = vector_store,
        bm25_index   = bm25_index,
        corpus_docs  = all_docs,
        config       = config,
    )

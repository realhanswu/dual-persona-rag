# ══════════════════════════════════════════════════════════════════
# src/utils/config.py
# RAGExperimentConfig dataclass — shared by presets.py and all
# pipeline modules. Single source of truth for experiment params.
# ══════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RAGExperimentConfig:
    """
    Full configuration for one RAG experiment run.
    All fields have sensible defaults matching the BEST preset.
    """

    # ── Identity ──────────────────────────────────────────────────
    experiment_id: str = "exp_default"

    # ── Embedding ─────────────────────────────────────────────────
    embedding_model: str = "multi-qa-mpnet-base-dot-v1"
    qdrant_distance: Literal["Dot", "Cosine", "Euclid"] = "Dot"

    # ── Chunking ──────────────────────────────────────────────────
    chunk_size:    int = 768
    chunk_overlap: int = 128

    # ── Retrieval ─────────────────────────────────────────────────
    search_type:  Literal["mmr", "similarity", "hybrid"] = "mmr"
    fetch_k:      int = 25    # MMR initial candidate pool
    retriever_k:  int = 7     # chunks forwarded to reranker

    # ── Reranker ──────────────────────────────────────────────────
    use_reranker:   bool = True
    reranker_top_n: int  = 3  # chunks passed to LLM prompt

    # ── Generation ────────────────────────────────────────────────
    temperature_eng: float = 0.0   # deterministic — factual precision
    temperature_mkt: float = 0.2   # slight variation — natural tone

    # ── Evaluation ────────────────────────────────────────────────
    n_eval_questions: int  = 80
    use_llm_judge:    bool = True

    def __post_init__(self):
        assert self.chunk_overlap < self.chunk_size, (
            f"chunk_overlap ({self.chunk_overlap}) must be < "
            f"chunk_size ({self.chunk_size})"
        )
        assert self.reranker_top_n <= self.retriever_k, (
            f"reranker_top_n ({self.reranker_top_n}) must be <= "
            f"retriever_k ({self.retriever_k})"
        )
        assert 0.0 <= self.temperature_eng <= 1.0
        assert 0.0 <= self.temperature_mkt <= 1.0

    @property
    def overlap_ratio(self) -> float:
        return round(self.chunk_overlap / self.chunk_size, 3)

    def summary(self) -> str:
        return (
            f"[{self.experiment_id}]  "
            f"embed={self.embedding_model}  "
            f"chunk={self.chunk_size}/{self.chunk_overlap} "
            f"({self.overlap_ratio:.0%} overlap)  "
            f"search={self.search_type}  "
            f"fetch_k={self.fetch_k}  k={self.retriever_k}  "
            f"rerank={'yes top' + str(self.reranker_top_n) if self.use_reranker else 'no'}  "
            f"temp=eng{self.temperature_eng}/mkt{self.temperature_mkt}  "
            f"n_eval={self.n_eval_questions}"
        )

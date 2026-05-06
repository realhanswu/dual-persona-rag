# ══════════════════════════════════════════════════════════════════
# configs/presets.py
# Named RAGExperimentConfig presets.
# Import: from configs.presets import PRESETS
# Usage:  config = PRESETS["BEST"]
# ══════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import RAGExperimentConfig

PRESETS: dict[str, RAGExperimentConfig] = {}


# ──────────────────────────────────────────────────────────────────
# BEST  —  exp_qa_mpnet_0406
# Final POC configuration. Combined score: 0.9117  ✅ GO
# ENG composite: 0.9161  |  MKT composite: 0.8983
# ──────────────────────────────────────────────────────────────────
PRESETS["BEST"] = RAGExperimentConfig(
    experiment_id    = "exp_qa_mpnet_0406",
    embedding_model  = "multi-qa-mpnet-base-dot-v1",
    qdrant_distance  = "Dot",
    chunk_size       = 768,
    chunk_overlap    = 128,
    search_type      = "mmr",
    fetch_k          = 25,
    retriever_k      = 7,
    use_reranker     = True,
    reranker_top_n   = 3,
    temperature_eng  = 0.0,
    temperature_mkt  = 0.2,
    n_eval_questions = 80,
)


# ──────────────────────────────────────────────────────────────────
# BASELINE  —  exp_baseline_0401
# General-purpose embedding, no reranker, cosine similarity.
# Establishes a lower-bound reference for improvement tracking.
# ──────────────────────────────────────────────────────────────────
PRESETS["BASELINE"] = RAGExperimentConfig(
    experiment_id    = "exp_baseline_0401",
    embedding_model  = "all-MiniLM-L6-v2",
    qdrant_distance  = "Cosine",
    chunk_size       = 512,
    chunk_overlap    = 64,
    search_type      = "similarity",
    fetch_k          = 10,
    retriever_k      = 5,
    use_reranker     = False,
    reranker_top_n   = 5,     # ignored when use_reranker=False
    temperature_eng  = 0.0,
    temperature_mkt  = 0.3,
    n_eval_questions = 80,
)


# ──────────────────────────────────────────────────────────────────
# NO_RERANK  —  exp_no_rerank_0404
# Best embedding + MMR but no cross-encoder reranker.
# Isolates the reranker's contribution to the composite score.
# ──────────────────────────────────────────────────────────────────
PRESETS["NO_RERANK"] = RAGExperimentConfig(
    experiment_id    = "exp_no_rerank_0404",
    embedding_model  = "multi-qa-mpnet-base-dot-v1",
    qdrant_distance  = "Dot",
    chunk_size       = 768,
    chunk_overlap    = 128,
    search_type      = "mmr",
    fetch_k          = 25,
    retriever_k      = 3,     # top-3 passed directly, no reranker
    use_reranker     = False,
    reranker_top_n   = 3,
    temperature_eng  = 0.0,
    temperature_mkt  = 0.2,
    n_eval_questions = 80,
)


# ──────────────────────────────────────────────────────────────────
# SMALL_CHUNK  —  exp_chunk256_0402
# Small chunks to study named-entity fragmentation failure mode.
# Expected: low FP3 + NDCG on name-lookup questions.
# ──────────────────────────────────────────────────────────────────
PRESETS["SMALL_CHUNK"] = RAGExperimentConfig(
    experiment_id    = "exp_chunk256_0402",
    embedding_model  = "multi-qa-mpnet-base-dot-v1",
    qdrant_distance  = "Dot",
    chunk_size       = 256,
    chunk_overlap    = 32,
    search_type      = "mmr",
    fetch_k          = 25,
    retriever_k      = 7,
    use_reranker     = True,
    reranker_top_n   = 3,
    temperature_eng  = 0.0,
    temperature_mkt  = 0.2,
    n_eval_questions = 80,
)


# ──────────────────────────────────────────────────────────────────
# LARGE_CHUNK  —  exp_chunk1024_0402
# Large chunks to study topical dilution + specificity gap inflation.
# Expected: low context precision, high FP6 specificity gap.
# ──────────────────────────────────────────────────────────────────
PRESETS["LARGE_CHUNK"] = RAGExperimentConfig(
    experiment_id    = "exp_chunk1024_0402",
    embedding_model  = "multi-qa-mpnet-base-dot-v1",
    qdrant_distance  = "Dot",
    chunk_size       = 1024,
    chunk_overlap    = 128,
    search_type      = "mmr",
    fetch_k          = 25,
    retriever_k      = 7,
    use_reranker     = True,
    reranker_top_n   = 3,
    temperature_eng  = 0.0,
    temperature_mkt  = 0.2,
    n_eval_questions = 80,
)


# ──────────────────────────────────────────────────────────────────
# HYBRID  —  exp_hybrid_bm25_0403
# BM25 + dense MMR hybrid. Tested whether sparse signals improve
# recall on precise technical terms. Result: marginal regression.
# ──────────────────────────────────────────────────────────────────
PRESETS["HYBRID"] = RAGExperimentConfig(
    experiment_id    = "exp_hybrid_bm25_0403",
    embedding_model  = "multi-qa-mpnet-base-dot-v1",
    qdrant_distance  = "Dot",
    chunk_size       = 768,
    chunk_overlap    = 128,
    search_type      = "hybrid",
    fetch_k          = 25,
    retriever_k      = 7,
    use_reranker     = True,
    reranker_top_n   = 3,
    temperature_eng  = 0.0,
    temperature_mkt  = 0.2,
    n_eval_questions = 80,
)


# ──────────────────────────────────────────────────────────────────
# FAST_EVAL  —  exp_fast_0406
# Reduced question count for rapid iteration during prompt tuning.
# Not suitable for final benchmarking.
# ──────────────────────────────────────────────────────────────────
PRESETS["FAST_EVAL"] = RAGExperimentConfig(
    experiment_id    = "exp_fast_0406",
    embedding_model  = "multi-qa-mpnet-base-dot-v1",
    qdrant_distance  = "Dot",
    chunk_size       = 768,
    chunk_overlap    = 128,
    search_type      = "mmr",
    fetch_k          = 25,
    retriever_k      = 7,
    use_reranker     = True,
    reranker_top_n   = 3,
    temperature_eng  = 0.0,
    temperature_mkt  = 0.2,
    n_eval_questions = 25,    # quick iteration only
    use_llm_judge    = False, # skip judge for speed
)


# ──────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────
def list_presets() -> None:
    """Print all available preset names and their summaries."""
    print(f"\n{'─' * 72}")
    print(f"  {'PRESET':<16}  SUMMARY")
    print(f"{'─' * 72}")
    for name, cfg in PRESETS.items():
        print(f"  {name:<16}  {cfg.summary()}")
    print(f"{'─' * 72}\n")


def get_preset(name: str) -> RAGExperimentConfig:
    """Retrieve a preset by name (case-insensitive). Raises KeyError if not found."""
    key = name.upper()
    if key not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Preset '{name}' not found. Available: {available}")
    return PRESETS[key]


if __name__ == "__main__":
    list_presets()

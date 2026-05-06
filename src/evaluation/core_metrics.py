# ══════════════════════════════════════════════════════════════════
# src/evaluation/core_metrics.py
# Orchestrates the full three-stage evaluation pipeline for a
# single Q/A instance.
#
# Stage 1: Retrieval  — FP3, Context Recall, Precision@K, MRR, NDCG@K
# Stage 2: Generation — FP4 (gate), FP7, FP6, BERTScore F1, SAC
# Stage 3: Judge      — Faithfulness (gate), Correctness, Relevance x2
# Stage 4: Persona    — PAS + 4 sub-dimensions
# ══════════════════════════════════════════════════════════════════

import math
import logging
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFacePipeline

from src.evaluation.retrieval_metrics import (
    fp3_context_hit_rate,
    context_recall_semantic,
    context_precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)
from src.evaluation.generation_metrics import (
    fp4_semantic_faithfulness,
    bertscore_metrics,
    semantic_answer_correctness,
)
from src.evaluation.llm_judges import run_llm_judge
from src.evaluation.pas import run_pas_judge
from configs.personas import get_composite_weights

logger = logging.getLogger(__name__)


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except Exception:
        return True


def compute_core_metrics(
    question:         str,
    gold_answer:      str,
    rag_answer:       str,
    retrieved_chunks: list[str],
    context_str:      str,
    persona:          str,
    embed_model:      SentenceTransformer,
    judge_llm:        HuggingFacePipeline,
    use_llm_judge:    bool = True,
) -> dict:
    """
    Compute the full metric suite for one Q/A pair.

    Returns:
        Flat dict with all metrics + composite_score + faithfulness_gate_score.
    """
    record: dict = {
        "question": question,
        "gold":     gold_answer,
        "answer":   rag_answer,
        "persona":  persona,
        "n_chunks": len(retrieved_chunks),
    }

    # ── Stage 1: Retrieval ────────────────────────────────────────
    record["FP3_context_utilization"] = fp3_context_hit_rate(
        gold_answer, retrieved_chunks, embed_model
    )
    record["context_recall"] = context_recall_semantic(
        gold_answer, retrieved_chunks, embed_model
    )
    record["context_precision_at_k"] = context_precision_at_k(
        gold_answer, retrieved_chunks, embed_model
    )
    record["MRR"]      = mean_reciprocal_rank(gold_answer, retrieved_chunks, embed_model)
    record["NDCG_at_k"] = ndcg_at_k(gold_answer, retrieved_chunks, embed_model)

    # ── Stage 2: Generation ───────────────────────────────────────
    record["FP4_faithfulness"] = fp4_semantic_faithfulness(
        rag_answer, retrieved_chunks, embed_model
    )
    record.update(bertscore_metrics(rag_answer, gold_answer))
    record["answer_correctness"] = semantic_answer_correctness(
        rag_answer, gold_answer, embed_model
    )

    # ── Stage 3: LLM Judge ────────────────────────────────────────
    if use_llm_judge:
        judge_scores = run_llm_judge(
            question, context_str, rag_answer, gold_answer, judge_llm
        )
    else:
        judge_scores = {
            "judge_faithfulness":       record["FP4_faithfulness"],
            "judge_answer_correctness": record["answer_correctness"],
            "judge_context_relevance":  record.get("FP3_context_utilization", float("nan")),
            "judge_answer_relevance":   record.get("context_recall",          float("nan")),
        }
    record.update(judge_scores)

    # Faithfulness gate: prefer judge score, fall back to semantic
    record["faithfulness_gate_score"] = (
        judge_scores["judge_faithfulness"]
        if not _is_nan(judge_scores.get("judge_faithfulness", float("nan")))
        else record["FP4_faithfulness"]
    )

    # ── Stage 4: Persona Adherence ────────────────────────────────
    pas_scores = run_pas_judge(rag_answer, persona, judge_llm)
    record.update(pas_scores)

    # ── Composite Score ───────────────────────────────────────────
    weights      = get_composite_weights(persona)
    weighted_sum = 0.0
    weight_total = 0.0
    for metric, weight in weights.items():
        val = record.get(metric, float("nan"))
        if not _is_nan(val):
            weighted_sum += float(val) * weight
            weight_total += weight

    record["composite_score"] = (
        weighted_sum / weight_total if weight_total > 0 else float("nan")
    )

    logger.debug(
        f"[{persona}] composite={record['composite_score']:.4f} | "
        f"faith_gate={record['faithfulness_gate_score']:.4f} | "
        f"Q: {question[:60]}..."
    )
    return record

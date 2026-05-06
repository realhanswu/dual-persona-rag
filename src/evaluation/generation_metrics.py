# ══════════════════════════════════════════════════════════════════
# src/evaluation/generation_metrics.py
# Stage 2 generation metrics — five metrics computed per question.
#
#   FP4  — Semantic Faithfulness     HARD GATE only (never in composite)
#   FP7  — Completeness Recall       BERTScore-R (diagnostic)
#   FP6  — Specificity Gap           |P - R|, lower is better (diagnostic)
#   F1   — BERTScore F1              selected for composite
#   SAC  — Semantic Answer Correct.  diagnostic (cosine sim vs gold)
# ══════════════════════════════════════════════════════════════════

import logging
import numpy as np
from bert_score import score as bert_score_fn
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_GROUNDING_THRESHOLD = 0.50  # lower than FP3 — single sentence vs full chunk


def fp4_semantic_faithfulness(
    answer: str,
    chunks: list[str],
    model:  SentenceTransformer,
) -> float:
    """
    Proportion of answer sentences grounded in retrieved context.
    A sentence is grounded if its max cosine similarity to any chunk
    exceeds _GROUNDING_THRESHOLD.
    Used exclusively as a hard gate — NOT in the composite score.
    Gate thresholds: ENG >= 0.75, MKT >= 0.80 (configs/personas.py).
    """
    if not answer or not chunks:
        return float("nan")
    sents = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sents:
        return float("nan")
    sent_emb  = model.encode(sents,  convert_to_numpy=True)
    chunk_emb = model.encode(chunks, convert_to_numpy=True)
    sims      = cosine_similarity(sent_emb, chunk_emb)
    grounded  = (sims.max(axis=1) >= _GROUNDING_THRESHOLD).sum()
    return float(grounded / len(sents))


def bertscore_metrics(
    answer:    str,
    reference: str,
    lang:      str = "en",
) -> dict[str, float]:
    """
    Compute BERTScore Precision, Recall, and F1 against the gold reference.

    Returns:
        bertscore_precision  : specificity proxy
        bertscore_recall     : FP7 completeness
        bertscore_f1         : selected for composite
        FP6_specificity_gap  : |P - R|
    """
    try:
        P, R, F1 = bert_score_fn(
            cands   = [answer],
            refs    = [reference],
            lang    = lang,
            verbose = False,
        )
        p_val = float(P[0])
        r_val = float(R[0])
        f_val = float(F1[0])
        return {
            "bertscore_precision": p_val,
            "bertscore_recall":    r_val,
            "bertscore_f1":        f_val,
            "FP6_specificity_gap": abs(p_val - r_val),
        }
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return {
            "bertscore_precision": float("nan"),
            "bertscore_recall":    float("nan"),
            "bertscore_f1":        float("nan"),
            "FP6_specificity_gap": float("nan"),
        }


def semantic_answer_correctness(
    answer:    str,
    reference: str,
    model:     SentenceTransformer,
) -> float:
    """
    Cosine similarity between answer and gold reference embeddings.
    Supplementary diagnostic metric — not selected for composite.
    """
    if not answer or not reference:
        return float("nan")
    embs = model.encode([answer, reference], convert_to_numpy=True)
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])

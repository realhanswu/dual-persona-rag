# Dual-Persona RAG

> **Can a single RAG pipeline serve a software engineer and a marketing manager — from the same corpus, with the same retriever — and be rigorously proven to do so?**

This project answers that question affirmatively. It is a proof-of-concept demonstrating that a single Retrieval-Augmented Generation pipeline, equipped with dual-persona prompt chains and evaluated through a three-stage, LLM-as-judge framework, can simultaneously serve the distinct information needs of a technical engineering audience and a plain-language marketing audience — clearing a combined cross-persona score of 0.9117 with both persona faithfulness gates passing independently.

***

## Why This Problem Is Harder Than It Looks

Most RAG systems are built for a single audience. A document Q&A system for engineers returns dense, mechanism-heavy answers. A marketing knowledge base returns concise, business-framed summaries. Building one pipeline that serves both audiences means solving three compounding problems at once.

**First, the audiences impose irreconcilable output constraints.** An engineering answer must lead with exact proper nouns, explain the causal mechanism (the *why* and *how*, not just the *what*), cite specific numerical thresholds, and deliver this in a single precise paragraph of 3–5 sentences at temperature 0.0 — deterministic by design. A marketing answer must strip every trace of jargon, reframe the same technical fact in terms of business value, and deliver it in no more than 1–3 sentences at temperature 0.2, with enough natural variation to feel human. These constraints are not just different — they actively pull the generation process in opposite directions.

**Second, standard evaluation metrics are blind to this distinction.** BERTScore, ROUGE, and cosine similarity reward fluent paraphrases of the reference answer regardless of audience fit. A technically accurate five-paragraph response would score well against an engineering reference but catastrophically violates the marketing persona's length constraint. A project that uses a single undifferentiated metric would declare both configurations equivalent and miss the failure entirely.

**Third, faithfulness cannot be approximated — it must be verified.** Embedding similarity between an answer and its retrieved context can be gamed by fluent but ungrounded generation. The only reliable faithfulness signal comes from a model that jointly reads the answer and the context and judges whether every claim is actually supported. This requires an LLM judge — and an LLM judge introduces its own calibration challenges.

***

## The Dual-Persona Architecture

The solution is a shared retrieval backbone with a clean bifurcation at generation time. A single MMR retriever fetches candidates from the same Qdrant index for every question. A cross-encoder reranker narrows the candidate pool to the three most relevant chunks. At that point, the pipeline forks: the same three chunks are passed to two completely separate LangChain chains, each carrying its own prompt template, its own temperature setting, and its own evaluation pipeline.
***

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [System Architecture](#system-architecture)
- [Persona Specifications](#persona-specifications)
- [Prompt Engineering: What Actually Moved the Metrics](#prompt-engineering-what-actually-moved-the-metrics)
- [Evaluation at Every Stage of the Pipeline](#evaluation-at-every-stage-of-the-pipeline)
  - [Stage 1 — Retrieval Evaluation](#stage-1--retrieval-evaluation)
  - [Stage 2 — Generation Evaluation](#stage-2--generation-evaluation)
  - [Stage 3 — LLM-as-Judge Evaluation](#stage-3--llm-as-judge-evaluation)
  - [Stage 4 — Persona Adherence Score (PAS)](#stage-4--persona-adherence-score-pas)
  - [The Faithfulness Gate](#the-faithfulness-gate)
  - [Cross-Persona Composite](#cross-persona-composite)
- [Key Results](#key-results)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Notebooks](#notebooks)
- [License](#license)
***

## Project Overview

| Attribute | Detail |
|---|---|
| **Model** | Mistral-7B-Instruct-v0.3 (4-bit quantised via BitsAndBytes) |
| **Embedding** | `multi-qa-mpnet-base-dot-v1` (dot product distance) |
| **Vector Store** | Qdrant (in-memory) |
| **Orchestration** | LangChain |
| **Chunk Size / Overlap** | 768 tokens / 128 tokens |
| **Retrieval** | MMR (`fetch_k=25`, `retriever_k=7`) + Cross-Encoder Reranker (`top_n=3`) |
| **Personas** | Engineering (`temp=0.0`) · Marketing (`temp=0.2`) |
| **Eval Questions** | 80 per persona |
| **Combined Score** | 0.9117 ✅ GO |

### Use Cases

- **Engineering** — Query technical documentation on GenAI concepts, model architectures, system design, and implementation details. Responses emphasise mechanism explanation, exact proper nouns, and numerical precision.
- **Marketing** — Retrieve accurate, approved messaging on product capabilities and competitive positioning. Responses emphasise plain language, business relevance, and strict conciseness (1–3 sentences).

***

## Repository Structure

```
dual-persona-rag/
│
├── README.md                      
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
│
├── configs/
│   ├── presets.py                   # RAGExperimentConfig presets (BEST, BASELINE, etc.)
│   └── personas.py                  # Persona thresholds, composite weights, PAS sub-dimensions
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                # PDF, web, Wikipedia, ArXiv document loaders
│   │   ├── chunker.py               # RecursiveCharacterTextSplitter wrapper
│   │   └── indexer.py               # Qdrant vector store builder + embedding
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py             # MMR retriever factory (fetch_k, retriever_k)
│   │   ├── reranker.py              # Cross-encoder reranker (top_n)
│   │   └── hybrid.py                # BM25 hybrid search (experimental)
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── chains.py                # Dual-persona LangChain RAG chains
│   │   ├── prompts.py               # Engineering + Marketing prompt templates
│   │   └── llm.py                   # Mistral-7B loader (4-bit BitsAndBytes config)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── core_metrics.py          # compute_core_metrics() — full 3-stage metric suite
│   │   ├── combined_score.py        # compute_combined_persona_score() — cross-persona GO/NO-GO
│   │   ├── retrieval_metrics.py     # FP3, Context Recall, Precision@K, MRR, NDCG@K
│   │   ├── generation_metrics.py    # FP4 (semantic gate), FP7, FP6, BERTScore, Answer Correctness
│   │   ├── llm_judges.py            # Mistral-7B judge: Faithfulness, Correctness, Relevance
│   │   ├── pas.py                   # Persona Adherence Score (PAS) + 4 sub-dimensions
│   │   └── experiment_store.py      # ExperimentStore — save/load JSON results per run
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # RAGExperimentConfig dataclass
│       └── helpers.py               # Shared utilities (safe float, nanmean, etc.)
│
├── data/
│   ├── corpus/                      # Raw source documents (PDFs, HTML, text)
│   │   └── .gitkeep
│   ├── gold_dataset/
│   │   ├── engineering_gold_qa.json # 80 validated engineering Q&A pairs
│   │   └── marketing_gold_qa.json   # 80 validated marketing Q&A pairs
│   └── experiments/                 # Per-experiment saved outputs (auto-generated)
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_corpus_ingestion.ipynb    # Load, chunk, embed, and index the corpus
│   ├── 02_retrieval_exploration.ipynb  # MMR vs cosine vs hybrid retrieval comparison
│   ├── 03_prompt_engineering.ipynb  # Prompt iteration and BERTScore/PAS impact analysis
│   ├── 04_hyperparameter_sweep.ipynb   # Chunk size, overlap, fetch_k, temperature grid search
│   ├── 05_evaluation_run.ipynb      # Full evaluation pipeline for a single config
│   ├── 06_combined_score.ipynb      # Cross-persona scoring and GO/NO-GO verdict
│   └── 07_results_analysis.ipynb    # Per-question inspection, failure analysis, visualisations
│
├── scripts/
│   ├── run_eval.py                  # CLI: run full evaluation for a named preset
│   └── build_gold_dataset.py        # Utility: validate and stratify gold Q&A pool
│
├── tests/
│   ├── test_retrieval_metrics.py
│   ├── test_generation_metrics.py
│   ├── test_combined_score.py
│   └── test_pas.py
│
├── outputs/
│   ├── experiments/                 # JSON outputs: core_metrics, question_records, combined_score
│   └── reports/                     # Evaluation summary reports (auto-generated)
│
└── docs/
    ├── evaluation_framework.md      # Detailed metric definitions, weights, and thresholds
    ├── prompt_design.md             # Engineering + Marketing prompt spec and constraint rationale
    ├── architecture.md              # System architecture and component interaction diagram
    └── experiment_log.md            # Chronological log of configurations and findings
```

***

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│  ArXiv PDFs · Web Articles · Wikipedia  →  Chunker  →  Qdrant   │
│  chunk_size=768 · overlap=128 · multi-qa-mpnet-base-dot-v1       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  MMR Search  │  fetch_k=25 → retriever_k=7
                    └──────┬──────┘
                           │
                  ┌─────────▼─────────┐
                  │  Cross-Encoder     │  top_n=3
                  │  Reranker          │
                  └────────┬──────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
   ┌──────▼──────┐                  ┌───────▼──────┐
   │ Engineering  │                  │  Marketing   │
   │    Chain     │                  │    Chain     │
   │  temp=0.0    │                  │  temp=0.2    │
   │  5 sentences │                  │  1-3 sents   │
   └──────┬───────┘                  └───────┬──────┘
          │                                  │
          └────────────────┬─────────────────┘
                           │
         ┌─────────────────▼──────────────────────────┐
         │              EVALUATION PIPELINE            │
         │                                             │
         │  Stage 1: Retrieval                         │
         │    FP3 · Recall · Precision@K · MRR · NDCG  │
         │                                             │
         │  Stage 2: Generation                        │
         │    FP4 (gate) · FP7 · FP6 · BERTScore-F1    │
         │    LLM Judge (Faith · Corr · Rel · CtxRel)  │
         │                                             │
         │  Stage 3: Persona Adherence (PAS)           │
         │    Tone · Structure · Audience · Constraints│
         │                                             │
         │  Composite (6-component) → Faithfulness Gate│
         │  Combined Score (75% ENG + 25% MKT)         │
         │  Verdict: GO / NO-GO                        │
         └─────────────────────────────────────────────┘
```

***

### Persona Specifications

| Dimension | Engineering | Marketing |
|---|---|---|
| **Primary goal** | Mechanism depth + factual precision | Plain-language business framing |
| **Sentence count** | Exactly 3–5 sentences | Exactly 1–3 sentences |
| **Temperature** | 0.0 (deterministic) | 0.2 (controlled variation) |
| **Must include** | Exact proper nouns, model names, numerical values | Business value, user benefit |
| **Strictly prohibited** | Bullets, headers, citations, author references | Jargon, acronyms, uncertainty qualifiers, parenthetical echoes |
| **Hallucination control** | `[NO CONTEXT RETRIEVED]` fallback | `[NO CONTEXT RETRIEVED]` fallback |

The `[NO CONTEXT RETRIEVED]` fallback was the single highest-impact prompt intervention in the entire project. Without it, Mistral-7B answered out-of-domain queries from parametric memory — producing fluent, confident, entirely ungrounded responses that passed BERTScore thresholds while failing the faithfulness gate. Adding the fallback marker, and instructing the model to detect it and return a fixed refusal, eliminated this failure mode entirely.

### Prompt Engineering: What Actually Moved the Metrics

Prompt design was not cosmetic. Each constraint in the final templates was added in response to a specific, measured failure mode observed during evaluation runs:

- **"Lead with the exact proper noun"** — without this instruction, the engineering chain occasionally front-loaded reasoning context before naming the concept, causing proper-noun fragmentation in the output that reduced FP3 retrieval alignment scores.
- **"Do NOT begin sentences with 'According to...'"** — early marketing answers consistently echoed source attribution phrases from the retrieved context, a hallmark of over-faithful extraction that sounded unnatural to a marketing reader and inflated PAS constraint-adherence penalties.
- **"Explain the mechanism: WHY it works and HOW it operates"** — without this, engineering answers answered the *what* but omitted the mechanistic detail that defines the engineering persona, depressing judge_answer_correctness scores on technical questions.
- **Sentence count caps** — the marketing chain at temperature 0.5 produced verbose outputs that violated the 1–3 sentence constraint in 23% of questions. Reducing temperature to 0.2 and encoding the cap as a hard rule in the prompt brought compliance to >97%.

***

## Evaluation at Every Stage of the Pipeline

The evaluation framework was designed on one principle: **every component that can fail should be measured independently**, so failures are localized rather than absorbed into a composite that hides their source.

### Stage 1 — Retrieval Evaluation

Before a single token is generated, the retrieval stage is evaluated against gold answers using five metrics:

| Metric | Symbol | Role |
|---|---|---|
| Context Hit Rate | FP3 | Binary: did any retrieved chunk exceed the 0.60 cosine similarity threshold with the gold answer? |
| Context Recall | CR | What fraction of the gold answer's sentences were semantically covered by the retrieved chunks? |
| Context Precision@K | CP | Of the top-K chunks retrieved, what proportion were actually relevant? |
| Mean Reciprocal Rank | MRR | How high in the ranking list did the first relevant chunk appear? |
| NDCG@K | NDCG | Combined relevance + ranking quality across the top-K position window |

FP3 and NDCG@K are selected for the composite score. The others are diagnostic: they surface whether a low FP3 reflects a ranking failure (low MRR), a coverage failure (low recall), or a precision failure (low CP), enabling targeted iteration.

### Stage 2 — Generation Evaluation

After generation, five metrics are computed against the gold reference:

| Metric | Symbol | Role |
|---|---|---|
| Semantic Faithfulness | FP4 | **Hard gate only** — proportion of answer sentences grounded in context (never in composite) |
| Completeness Recall | FP7 | BERTScore Recall — does the answer cover the gold content? |
| Specificity Gap | FP6 | \|BERTScore-P − BERTScore-R\| — penalises over-generation and under-generation equally |
| BERTScore F1 | F1 | Selected for composite — balances precision and recall simultaneously |
| Semantic Answer Correctness | SAC | Cosine similarity between answer and gold embeddings (diagnostic fallback) |

FP4 is deliberately excluded from the composite. It is used only as a gate — a binary pass/fail threshold that must be cleared independently of score optimisation. This prevents a pipeline from achieving a high composite score by excelling on retrieval and generation metrics while quietly producing ungrounded outputs.

### Stage 3 — LLM-as-Judge Evaluation

The third stage replaces embedding similarity with judgment. The same Mistral-7B model used for generation is repurposed as an evaluator, receiving a structured prompt that presents the question, the retrieved context, the gold reference answer, and the generated answer simultaneously, then returns four scores in JSON format:

| Judge Dimension | Role in Framework |
|---|---|
| `judge_faithfulness` | **Primary faithfulness gate** — preferred over FP4 when available; detects claim-level hallucination |
| `judge_answer_correctness` | Selected for composite — factual alignment with gold reference |
| `judge_context_relevance` | Diagnostic — was the retrieved context appropriate for the question? |
| `judge_answer_relevance` | Selected for composite — does the answer directly address what was asked? |

Using the generation model as its own judge introduces potential self-preference bias. This was accepted as an explicit limitation of the on-device POC constraint. The judge prompt is structured to minimise it: the model is never asked to compare its own output against another model's, only to evaluate a single answer against an explicit rubric with a gold reference anchor.

On any parse or inference failure, all judge scores fall back to `NaN` and `core_metrics.py` substitutes the corresponding semantic approximations (FP4 for faithfulness, SAC for correctness).

### Stage 4 — Persona Adherence Score (PAS)

The PAS measures something no standard NLP metric captures: *did the answer actually sound like it was written for this audience?* Four sub-dimensions are scored per persona by the same LLM judge:

| Sub-dimension | Engineering Rubric | Marketing Rubric |
|---|---|---|
| Tone Appropriateness | Technical depth, mechanism explanation, specific named values | Plain language, no jargon, business-value framing |
| Structural Compliance | Single unbroken paragraph, 3–5 sentences, no formatting | 1–3 sentences, no formatting, no uncertainty disclaimers |
| Audience Fit | Practically actionable for a technical decision | Immediately usable in external communications |
| Constraint Adherence | No citations, author names, or parenthetical echoes | No acronyms, qualifiers ('may', 'might'), or source phrases |

PAS carries 15% of the engineering composite and 25% of the marketing composite. The asymmetry is deliberate: tonal correctness is operationally critical for marketing — an answer that is factually accurate but sounds like a whitepaper cannot be published externally.

### The Faithfulness Gate

Before any composite score is computed, each persona's mean `faithfulness_gate_score` is evaluated against an independent threshold:

- **Engineering gate:** mean faithfulness ≥ 0.75
- **Marketing gate:** mean faithfulness ≥ 0.80 (stricter — public-facing content carries higher compliance risk)

Both gates must pass independently. A configuration that clears the combined score threshold but fails either gate receives an unconditional **NO-GO** verdict. This hard gate exists because a composite score can be optimised upward by improvements in retrieval and PAS metrics while faithfulness quietly degrades — a scenario that would result in a highly-scored system that produces confident, factually ungrounded answers.

### Cross-Persona Composite

The final combined score is a weighted average across both personas:

$$ \text{Combined Score} = 0.75 \times \text{ENG Composite} + 0.25 \times \text{MKT Composite} $$

The 75/25 weighting reflects the actual audience ratio of the target deployment: approximately 300 engineers versus 40 marketing staff. A single pooled composite at 50/50 would have over-weighted marketing metric variance and obscured engineering-specific failure modes during the hyperparameter sweep.

***

## Key Results

**Best configuration: `exp_qa_mpnet_0406`**

| Metric | Engineering | Marketing |
|---|---|---|
| **Composite Score** | **0.9161 ✅ GO** | **0.8983 ✅ GO** |
| **Faithfulness Gate** | **0.9840 ✅ PASS** | **0.9787 ✅ PASS** |
| Context Hit Rate (FP3) | 0.9437 | 0.8592 |
| Ranking Quality (NDCG@K) | 0.9860 | 0.9863 |
| BERTScore F1 | 0.8864 | 0.8963 |
| Answer Relevance (Judge) | 0.9760 | 0.9787 |
| Answer Correctness (Judge) | 0.8507 | 0.8560 |
| Persona Adherence (PAS) | 0.8545 | 0.8487 |
| **Combined Score** | **0.9117 ✅ GO** (threshold: 0.75) | |

### Top findings

1. **Embedding model selection** is the dominant performance driver — `multi-qa-mpnet-base-dot-v1` produced the largest single improvement of the POC.
2. **Chunk size requires empirical tuning** — 768/128 balanced named-entity containment against topical coherence; smaller sizes caused proper-noun fragmentation.
3. **Prompt engineering was the highest-leverage intervention** — sentence-count caps, structural constraints, and the `[NO CONTEXT RETRIEVED]` fallback drove the biggest metric lifts.
4. **Cross-encoder reranking yields diminishing returns** at this corpus scale when the embedding model is domain-matched.
5. **Dual-persona weighting is essential** — a single undifferentiated composite metric would have produced misleading conclusions and rewarded configurations that failed one audience.

***

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_corpus_ingestion.ipynb` | Load sources, chunk, embed, build Qdrant index |
| `02_retrieval_exploration.ipynb` | Compare MMR / cosine / hybrid; visualise chunk boundaries |
| `03_prompt_engineering.ipynb` | Iterate prompts; track BERTScore F1 and PAS across versions |
| `04_hyperparameter_sweep.ipynb` | Grid search over chunk size, overlap, fetch_k, temperature |
| `05_evaluation_run.ipynb` | Run the full 3-stage evaluation pipeline for one config |
| `06_combined_score.ipynb` | Compute cross-persona score; issue GO/NO-GO verdict |
| `07_results_analysis.ipynb` | Per-question inspection, failure mode analysis, visualisations |

***


***
## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-org/dual-persona-rag.git
cd dual-persona-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN for gated Mistral model access
```

### 3. Ingest the corpus

```bash
python scripts/run_eval.py --step ingest --preset BEST
```

### 4. Run evaluation

```bash
python scripts/run_eval.py --step eval --preset BEST --persona both
```

### 5. View results

```bash
# Outputs saved to outputs/experiments/exp_qa_mpnet_0406/
cat outputs/experiments/exp_qa_mpnet_0406/combined_score.json
```

Or open `notebooks/07_results_analysis.ipynb` for interactive per-question inspection.

***

## Configuration

All experiment configurations are defined in `configs/presets.py` as `RAGExperimentConfig` dataclasses.

### Best Configuration (`BEST`)

```python
PRESETS["BEST"] = RAGExperimentConfig(
    experiment_id     = "exp_qa_mpnet_0406",
    embedding_model   = "multi-qa-mpnet-base-dot-v1",
    qdrant_distance   = "Dot",
    chunk_size        = 768,
    chunk_overlap     = 128,
    search_type       = "mmr",
    fetch_k           = 25,
    retriever_k       = 7,
    use_reranker      = True,
    reranker_top_n    = 3,
    temperature_eng   = 0.0,
    temperature_mkt   = 0.2,
    n_eval_questions  = 80,
)
```

### Persona thresholds and composite weights

Defined in `configs/personas.py`. Key design decisions:

| Metric | ENG Weight | MKT Weight | Rationale |
|---|---|---|---|
| Context Hit Rate (FP3) | 20% | 15% | Retrieval precision critical for technical accuracy |
| NDCG@K | 15% | 10% | Ranking matters more when content is dense |
| BERTScore F1 | 20% | 15% | Factual completeness + conciseness balance |
| Answer Relevance | 15% | 20% | Marketing audience judges by relevance first |
| Answer Correctness | 15% | 15% | Equal weight — factual correctness is non-negotiable |
| Persona Adherence (PAS) | 15% | 25% | Tonal fit is operationally critical for marketing |

Faithfulness gate thresholds: **0.75 (ENG)** · **0.80 (MKT)**
Cross-persona weights: **75% ENG · 25% MKT** (reflects ~300 engineers vs. ~40 marketing staff)

***

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_corpus_ingestion.ipynb` | Load sources, chunk, embed, build Qdrant index |
| `02_retrieval_exploration.ipynb` | Compare MMR / cosine / hybrid; visualise chunk boundaries |
| `03_prompt_engineering.ipynb` | Iterate prompts; track BERTScore F1 and PAS across versions |
| `04_hyperparameter_sweep.ipynb` | Grid search over chunk size, overlap, fetch_k, temperature |
| `05_evaluation_run.ipynb` | Run the full 3-stage evaluation pipeline for one config |
| `06_combined_score.ipynb` | Compute cross-persona score; issue GO/NO-GO verdict |
| `07_results_analysis.ipynb` | Per-question inspection, failure mode analysis, visualisations |

***



## License

MIT License. See `LICENSE` for details.


# Dual-Persona RAG

> **Evaluating a Retrieval-Augmented Generation system for simultaneous Engineering and Marketing knowledge access**

A proof-of-concept demonstrating that a single RAG pipeline, with dual-persona prompt chains and a rigorous three-stage evaluation framework, can serve the distinct information needs of both a technical engineering audience and a plain-language marketing audience — clearing a combined cross-persona score of **0.9117** with both faithfulness gates passing independently.

***

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [System Architecture](#system-architecture)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Evaluation Framework](#evaluation-framework)
- [Key Results](#key-results)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
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

## Evaluation Framework

The evaluation pipeline operates across three progressive stages, computing an exhaustive diagnostic suite before six metrics are selected for the composite score:

| Stage | Metrics Computed | Selected for Composite |
|---|---|---|
| **Retrieval** | FP3, Context Recall, Precision@K, MRR, NDCG@K | FP3, NDCG@K |
| **Generation** | FP4 (gate), FP7, FP6, BERTScore F1, Semantic Correctness | BERTScore F1 |
| **LLM Judge** | Faithfulness (gate), Answer Correctness, Context Relevance, Answer Relevance | Answer Correctness, Answer Relevance |
| **Persona** | PAS (Tone + Structure + Audience Fit + Constraint Adherence) | PAS |

A **faithfulness hard gate** is enforced independently of the composite score. Both persona gates must pass for a GO verdict.

See `docs/evaluation_framework.md` for full metric definitions, threshold rationale, and per-question record schema.

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
| **Combined Score** | **0.9117 ✅ GO** (threshold: 0.65) | |

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

## License

MIT License. See `LICENSE` for details.

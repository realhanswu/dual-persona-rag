#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════
# scripts/run_eval.py
# CLI entry point for full pipeline execution.
# Usage:
#   python scripts/run_eval.py --preset BEST --step eval --persona both
#   python scripts/run_eval.py --preset FAST_EVAL --step ingest
# ══════════════════════════════════════════════════════════════════

import argparse
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.helpers import setup_logging
from configs.presets import get_preset, list_presets


def parse_args():
    parser = argparse.ArgumentParser(description="persona-lens evaluation pipeline")
    parser.add_argument("--preset",   default="BEST",  help="Config preset name")
    parser.add_argument("--step",     choices=["ingest", "eval", "score", "all"], default="all")
    parser.add_argument("--persona",  choices=["engineering", "marketing", "both"], default="both")
    parser.add_argument("--list",     action="store_true", help="List all presets and exit")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    if args.list:
        list_presets()
        return

    config = get_preset(args.preset)
    logging.info(f"Loaded preset: {config.summary()}")

    if args.step in ("ingest", "all"):
        logging.info("── Step: Ingest ──────────────────────────────────")
        from src.ingestion.loader import load_all_corpus
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.indexer import build_vector_store
        # Corpus sources configured here — replace with your internal sources
        docs   = load_all_corpus(pdf_dir="data/corpus")
        chunks = chunk_documents(docs, config)
        vs     = build_vector_store(chunks, config)
        logging.info(f"Ingestion complete: {len(chunks)} chunks indexed")

    if args.step in ("eval", "all"):
        logging.info("── Step: Evaluate ────────────────────────────────")
        logging.info(f"Persona(s): {args.persona} | Questions: {config.n_eval_questions}")
        logging.info("Load gold dataset, run chains, compute metrics per question.")
        logging.info("See notebooks/05_evaluation_run.ipynb for interactive version.")

    if args.step in ("score", "all"):
        logging.info("── Step: Score ───────────────────────────────────")
        logging.info("Aggregate per-question records → GO/NO-GO verdict.")
        logging.info("See notebooks/06_combined_score.ipynb for interactive version.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════
# scripts/build_gold_dataset.py
# Validates and stratifies the gold Q&A pool.
# Reads raw JSON files from data/gold_dataset/ and outputs
# stratified splits balanced by topic category.
#
# Expected input schema per record:
#   {
#     "id":       "eng_001",
#     "question": "What is RAG?",
#     "answer":   "RAG is ...",
#     "category": "architecture",
#     "persona":  "engineering"
#   }
# ══════════════════════════════════════════════════════════════════

import json
import sys
import os
import random
import logging
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.helpers import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

GOLD_DIR   = Path("data/gold_dataset")
N_EVAL     = 80
RANDOM_SEED = 42


def load_pool(persona: str) -> list[dict]:
    path = GOLD_DIR / f"{persona}_gold_qa.json"
    if not path.exists():
        raise FileNotFoundError(f"Gold dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def stratified_sample(records: list[dict], n: int, seed: int = RANDOM_SEED) -> list[dict]:
    """Sample n records with balanced topic category coverage."""
    by_category = defaultdict(list)
    for r in records:
        by_category[r.get("category", "uncategorised")].append(r)

    categories = sorted(by_category)
    per_cat    = max(1, n // len(categories))
    sampled    = []

    random.seed(seed)
    for cat in categories:
        pool   = by_category[cat]
        take   = min(per_cat, len(pool))
        sampled.extend(random.sample(pool, take))

    # Top up to n if rounding left us short
    remaining = [r for r in records if r not in sampled]
    random.shuffle(remaining)
    sampled.extend(remaining[:max(0, n - len(sampled))])

    return sampled[:n]


def validate_record(r: dict, idx: int) -> list[str]:
    errors = []
    for field in ("id", "question", "answer", "category", "persona"):
        if not r.get(field):
            errors.append(f"Record {idx}: missing '{field}'")
    if len(r.get("answer", "")) < 10:
        errors.append(f"Record {idx}: answer too short (<10 chars)")
    return errors


def main():
    for persona in ("engineering", "marketing"):
        logger.info(f"\nProcessing: {persona}")
        records = load_pool(persona)
        logger.info(f"  Pool size: {len(records)}")

        all_errors = []
        for i, r in enumerate(records):
            all_errors.extend(validate_record(r, i))
        if all_errors:
            for e in all_errors:
                logger.warning(f"  ⚠ {e}")
        else:
            logger.info("  Validation: ✅ all records valid")

        sample = stratified_sample(records, N_EVAL)
        out_path = GOLD_DIR / f"{persona}_eval_sample.json"
        with open(out_path, "w") as f:
            json.dump(sample, f, indent=2)
        logger.info(f"  Saved {len(sample)}-question eval sample → {out_path}")

        cats = defaultdict(int)
        for r in sample:
            cats[r.get("category", "uncategorised")] += 1
        for cat, count in sorted(cats.items()):
            logger.info(f"    {cat:<30} {count}")


if __name__ == "__main__":
    main()

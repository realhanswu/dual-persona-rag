# ══════════════════════════════════════════════════════════════════
# src/utils/helpers.py
# Shared utility functions across the pipeline.
# ══════════════════════════════════════════════════════════════════

import math
import logging
import os
from dotenv import load_dotenv

load_dotenv()


def setup_logging(level: str = None) -> None:
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level   = getattr(logging, log_level.upper(), logging.INFO),
        format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )


def safe_float(val, fallback: float = float("nan")) -> float:
    try:
        result = float(val)
        return result if not math.isnan(result) else fallback
    except (TypeError, ValueError):
        return fallback


def nanmean(values: list) -> float:
    valid = [safe_float(v) for v in values if not math.isnan(safe_float(v))]
    return sum(valid) / len(valid) if valid else float("nan")


def format_score(val: float, decimals: int = 4) -> str:
    if math.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def print_metric_table(metrics: dict, title: str = "Metrics") -> None:
    W = 52
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'─' * W}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<36}  {format_score(v)}")
        elif isinstance(v, bool):
            print(f"  {k:<36}  {'✅' if v else '❌'}")
        else:
            print(f"  {k:<36}  {v}")
    print(f"{'═' * W}\n")

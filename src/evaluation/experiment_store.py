# ══════════════════════════════════════════════════════════════════
# src/evaluation/experiment_store.py
# Saves and loads JSON outputs for a single experiment run.
# Directory layout: outputs/experiments/{experiment_id}/
#
# Standard files saved per run:
#   metadata.json     — config + timestamp
#   eng_records.json  — per-question metrics (engineering)
#   mkt_records.json  — per-question metrics (marketing)
#   core_metrics.json — aggregated per-persona metric means
#   combined_score.json — cross-persona composite + GO/NO-GO
# ══════════════════════════════════════════════════════════════════

import json
import os
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentStore:
    """
    Persists and retrieves evaluation artefacts for one experiment run.

    Usage:
        store = ExperimentStore("exp_qa_mpnet_0406")
        store.save_metadata(config)
        store.save("eng_records.json", eng_records)
        store.save("combined_score.json", result)
        result = store.load("combined_score.json")
    """

    def __init__(self, experiment_id: str, base_dir: str = None):
        self.experiment_id = experiment_id
        self.base_dir      = Path(
            base_dir or os.getenv("EXPERIMENT_OUTPUT_DIR", "outputs/experiments")
        )
        self.run_dir = self.base_dir / experiment_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ExperimentStore initialised at: {self.run_dir}")

    def save(self, filename: str, data: dict | list) -> Path:
        """Serialise data to JSON in the run directory."""
        path = self.run_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"  Saved: {path.relative_to(self.base_dir)}")
        return path

    def load(self, filename: str) -> dict | list:
        """Load a previously saved JSON file."""
        path = self.run_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file not found: {path}\n"
                f"Available: {self.list_files()}"
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_metadata(self, config) -> None:
        """Save experiment config and run timestamp as metadata.json."""
        meta = {
            "experiment_id": self.experiment_id,
            "timestamp":     datetime.now().isoformat(),
            "config":        config.__dict__ if hasattr(config, "__dict__") else str(config),
        }
        self.save("metadata.json", meta)

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self.run_dir.iterdir() if p.is_file())

    def exists(self, filename: str) -> bool:
        return (self.run_dir / filename).exists()

    def __repr__(self) -> str:
        return (
            f"ExperimentStore("
            f"id={self.experiment_id!r}, "
            f"files={self.list_files()})"
        )

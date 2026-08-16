"""Experiment tracking wrapper (Section 23). Uses Weights & Biases when
`logging.use_wandb: true` AND the wandb package + API key are available;
otherwise falls back to an append-only CSV logger so CPU smoke tests and
offline environments never fail because W&B isn't configured."""
from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, Optional


class ExperimentLogger:
    def __init__(self, run_name: str, results_dir: str, use_wandb: bool = False,
                 wandb_project: str = "geosar-vlm", config: Optional[Dict[str, Any]] = None):
        self.run_name = run_name
        self.results_dir = results_dir
        os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
        self.csv_path = os.path.join(results_dir, "metrics", f"{run_name}.csv")
        self._csv_fields_written = False
        self._wandb = None

        if use_wandb:
            try:
                import wandb

                self._wandb = wandb.init(project=wandb_project, name=run_name, config=config or {})
            except Exception as e:  # noqa: BLE001 — deliberately broad: logging must never crash training
                print(f"[ExperimentLogger] wandb unavailable ({e}); falling back to CSV at {self.csv_path}")
                self._wandb = None

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        metrics = {**metrics, "step": step, "timestamp": time.time()}
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        write_header = not self._csv_fields_written and not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
                self._csv_fields_written = True
            writer.writerow(metrics)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

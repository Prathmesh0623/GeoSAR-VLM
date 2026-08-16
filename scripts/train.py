#!/usr/bin/env python3
"""Unified training CLI (Section 27, Kaggle notebooks import this rather than
duplicating logic). Dispatches to train_vqa or train_retrieval based on --task.

VS Code / local CPU smoke test:
    python scripts/train.py --config configs/concat_fusion.yaml --task vqa --cpu-smoke-test

Kaggle (real GPU training):
    python scripts/train.py --config configs/cross_attention.yaml --task vqa
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="vqa", choices=["vqa", "retrieval"])
    parser.add_argument("--cpu-smoke-test", action="store_true",
                         help="Run 1 epoch on a few batches to verify shapes/wiring, no real training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config: {cfg['experiment_name']} (fusion_type={cfg['model']['fusion_type']})")

    if args.task == "vqa":
        from src.training.train_vqa import run_vqa_training

        run_vqa_training(cfg, cpu_smoke_test=args.cpu_smoke_test)
    else:
        from src.training.train_retrieval import run_retrieval_training

        run_retrieval_training(cfg, cpu_smoke_test=args.cpu_smoke_test)


if __name__ == "__main__":
    main()

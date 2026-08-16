#!/usr/bin/env python3
"""Unified evaluation CLI.

VS Code / local CPU smoke test:
    python scripts/evaluate.py --config configs/concat_fusion.yaml --task vqa --split val --cpu-smoke-test

Kaggle (real GPU, trained checkpoint):
    python scripts/evaluate.py --config configs/cross_attention.yaml --task retrieval --split test
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
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--cpu-smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.task == "vqa":
        from src.evaluation.evaluate_vqa import run_vqa_evaluation

        results = run_vqa_evaluation(cfg, split=args.split, cpu_smoke_test=args.cpu_smoke_test)
    else:
        from src.evaluation.evaluate_retrieval import run_retrieval_evaluation

        results = run_retrieval_evaluation(cfg, split=args.split, cpu_smoke_test=args.cpu_smoke_test)

    print(results)


if __name__ == "__main__":
    main()

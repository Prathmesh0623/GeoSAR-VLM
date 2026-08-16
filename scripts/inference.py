#!/usr/bin/env python3
"""Single-scene inference CLI: given a SAR .npy, EO .npy, and a question, print the
model's answer. Useful for manual qualitative inspection (Section 22).

Usage:
    python scripts/inference.py --config configs/concat_fusion.yaml \
        --sar data/processed/sar/scene_0000.npy --eo data/processed/eo/scene_0000.npy \
        --question "What land cover dominates this scene?" \
        --checkpoint experiments/fusion/concat/checkpoint.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.evaluation.evaluate_vqa import greedy_generate
from src.models.geosar_vlm import GeoSARVLM
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.tokenizer import SimpleTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sar", type=str, required=True)
    parser.add_argument("--eo", type=str, required=True)
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sar = torch.from_numpy(np.load(args.sar).astype(np.float32)).unsqueeze(0)
    eo = torch.from_numpy(np.load(args.eo).astype(np.float32)).unsqueeze(0)

    # NOTE: a real deployment reuses the exact tokenizer vocab saved at training time
    # (see docs/limitations.md) — this script rebuilds a placeholder vocab only for
    # the `tiny` CPU-test backend and is not meant for production inference.
    tokenizer = SimpleTokenizer.build_from_texts([args.question] if args.question else ["a"])
    model = GeoSARVLM(cfg, vocab_size=len(tokenizer))
    if args.checkpoint:
        model, _ = load_checkpoint(model, args.checkpoint)
    model.eval()

    batch = {"sar": sar, "eo": eo, "question": [args.question], "answer": [""]}
    outputs = greedy_generate(model, batch, tokenizer)
    print(f"Question: {args.question}")
    print(f"Answer:   {outputs[0]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare data from the 'sen12ms-asia' .pt shard format (Kaggle dataset:
krishnaanchal/sen12ms-asia). Each shard_XXX.pt is a dict:
    {"sar": FloatTensor[N,3,H,W], "opt": FloatTensor[N,3,H,W], "label": ByteTensor[N]}

This is NOT the standard SEN12MS GeoTIFF layout, so it needs its own loader
(scripts/prepare_dataset.py remains for the standard GeoTIFF layout).

Because the true IGBP class-name mapping for these numeric labels is not
documented in this dataset upload, labels are kept as raw integers
("class_<id>") rather than guessed at - see docs/dataset.md.

Real patches in this dataset are 256x256 - this script resizes them to
--image-size (default 224) to match configs/base.yaml's data.image_size.

Usage on Kaggle:
    python scripts/prepare_sen12ms_shards.py --shards-dir /kaggle/input/datasets/krishnaanchal/sen12ms-asia/asian_sen12ms_shards --out data/processed --max-shards 2 --samples-per-shard 300
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import resize_chw


def normalize_percentile_chw(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """arr: (C,H,W) float32. Per-channel percentile clip + min-max scale to [0,1].
    Used instead of fixed dB/reflectance constants because this shard format's
    raw value range is undocumented."""
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        band = arr[c]
        lo, hi = np.percentile(band, [lo_pct, hi_pct])
        if hi <= lo:
            hi = lo + 1e-6
        band = np.clip(band, lo, hi)
        out[c] = (band - lo) / (hi - lo)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--image-size", type=int, default=224,
                         help="Resize every patch to this size (must match configs/base.yaml data.image_size)")
    parser.add_argument("--max-shards", type=int, default=None,
                         help="Only process this many shard files (each is ~1.5GB / 2000 samples). "
                              "Start small (e.g. 2) to validate before doing the full dataset.")
    parser.add_argument("--samples-per-shard", type=int, default=None,
                         help="Randomly subsample this many scenes per shard instead of all 2000.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    shard_paths = sorted(glob.glob(os.path.join(args.shards_dir, "shard_*.pt")))
    if args.max_shards:
        shard_paths = shard_paths[: args.max_shards]
    print(f"Found {len(shard_paths)} shard file(s) to process")

    os.makedirs(os.path.join(args.out, "sar"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "eo"), exist_ok=True)

    rng = random.Random(args.seed)
    manifest = []
    label_counts = {}

    for shard_idx, shard_path in enumerate(shard_paths):
        print(f"Loading {shard_path} ...")
        data = torch.load(shard_path, map_location="cpu")
        sar_all, opt_all, label_all = data["sar"], data["opt"], data["label"]
        n = sar_all.shape[0]

        indices = list(range(n))
        if args.samples_per_shard and args.samples_per_shard < n:
            indices = rng.sample(indices, args.samples_per_shard)

        for i in indices:
            sar = sar_all[i].to(torch.float32).numpy()
            opt = opt_all[i].to(torch.float32).numpy()
            label_id = int(label_all[i].item())
            label_counts[label_id] = label_counts.get(label_id, 0) + 1

            sar_norm = normalize_percentile_chw(sar)
            opt_norm = normalize_percentile_chw(opt)

            if sar_norm.shape[1] != args.image_size or sar_norm.shape[2] != args.image_size:
                sar_norm = resize_chw(sar_norm, args.image_size)
                opt_norm = resize_chw(opt_norm, args.image_size)

            scene_id = f"shard{shard_idx:03d}_sample{i:04d}"
            sar_rel, eo_rel = f"sar/{scene_id}.npy", f"eo/{scene_id}.npy"
            np.save(os.path.join(args.out, sar_rel), sar_norm)
            np.save(os.path.join(args.out, eo_rel), opt_norm)
            manifest.append({"scene_id": scene_id, "sar": sar_rel, "eo": eo_rel, "label_id": label_id})

        del data, sar_all, opt_all, label_all
        print(f"  -> processed {len(indices)} sample(s) from shard {shard_idx}")

    manifest_path = os.path.join(args.out, "scene_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(manifest)} processed scene(s) to {args.out}")
    print(f"Manifest: {manifest_path}")
    print("Label distribution (raw numeric IDs, real class names unknown -- see docs/dataset.md):")
    for label_id, count in sorted(label_counts.items()):
        print(f"  class_{label_id}: {count}")
    print("\nNext: python scripts/create_annotations.py --from-manifest data/processed/scene_manifest.json --out data/processed")


if __name__ == "__main__":
    main()
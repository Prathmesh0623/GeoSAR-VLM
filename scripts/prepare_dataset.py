#!/usr/bin/env python3
"""Download/link SEN12MS and produce processed SAR/EO .npy patches (Kaggle-side,
Section 5, 26-29). This is the ONLY script that needs rasterio/GDAL + real disk
space for SEN12MS (~50GB) and is not expected to run on a laptop.

Usage on Kaggle (after adding the SEN12MS dataset to the notebook's Data sources):
    python scripts/prepare_dataset.py \
        --sen12ms-root /kaggle/input/sen12ms \
        --out data/processed \
        --image-size 224 \
        --sar-bands VV VH \
        --eo-bands B04 B03 B02 B08

For local CPU development, use scripts/create_annotations.py --synthetic instead.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import preprocess_eo, preprocess_sar, resize_chw


SEN12MS_ALL_S1_BANDS = ["VV", "VH"]
SEN12MS_ALL_S2_BANDS = [f"B{i:02d}" for i in range(1, 14) if i != 10]  # SEN12MS omits B10


def find_scene_pairs(sen12ms_root: str):
    """SEN12MS layout: <root>/<season>/s1_<scene>/..., s2_<scene>/... Each scene has
    many patches; here we treat each patch file as one "scene_id" for simplicity —
    adjust to your local SEN12MS folder layout if it differs."""
    s1_files = sorted(glob.glob(os.path.join(sen12ms_root, "**", "s1_*", "*.tif"), recursive=True))
    pairs = []
    for s1_path in s1_files:
        s2_path = s1_path.replace("s1_", "s2_").replace("_s1_", "_s2_").replace("_s1", "_s2")
        if os.path.exists(s2_path):
            scene_id = os.path.splitext(os.path.basename(s1_path))[0]
            pairs.append((scene_id, s1_path, s2_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sen12ms-root", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sar-bands", nargs="+", default=["VV", "VH"])
    parser.add_argument("--eo-bands", nargs="+", default=["B04", "B03", "B02", "B08"])
    parser.add_argument("--limit", type=int, default=None, help="cap number of scenes (debugging)")
    args = parser.parse_args()

    from src.data.preprocessing import load_geotiff  # imported lazily: needs rasterio

    pairs = find_scene_pairs(args.sen12ms_root)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Found {len(pairs)} SAR/EO scene pairs under {args.sen12ms_root}")

    os.makedirs(os.path.join(args.out, "sar"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "eo"), exist_ok=True)

    manifest = []
    for scene_id, s1_path, s2_path in pairs:
        sar_raw = load_geotiff(s1_path)
        eo_raw = load_geotiff(s2_path)

        sar = preprocess_sar(sar_raw, bands=args.sar_bands, all_bands=SEN12MS_ALL_S1_BANDS)
        eo = preprocess_eo(eo_raw, bands=args.eo_bands, all_bands=SEN12MS_ALL_S2_BANDS)
        sar = resize_chw(sar, args.image_size)
        eo = resize_chw(eo, args.image_size)

        import numpy as np

        sar_rel, eo_rel = f"sar/{scene_id}.npy", f"eo/{scene_id}.npy"
        np.save(os.path.join(args.out, sar_rel), sar)
        np.save(os.path.join(args.out, eo_rel), eo)
        manifest.append({"scene_id": scene_id, "sar": sar_rel, "eo": eo_rel})

    manifest_path = os.path.join(args.out, "scene_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} processed scene(s) to {args.out}, manifest at {manifest_path}")
    print("Next: run scripts/create_annotations.py (non-synthetic mode) to attach "
          "template captions/VQA/labels and produce the leakage-safe split.")


if __name__ == "__main__":
    main()

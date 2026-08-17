#!/usr/bin/env python3
"""Build the annotation layer (Section 6) on top of processed SAR/EO patches.

Modes:
  --synthetic                    Generate fake-but-correctly-shaped SAR/EO .npy
                                  patches + template captions/VQA/labels, entirely
                                  offline. Used for CPU smoke tests (Section 30).
  --from-manifest <path>         Build annotations from real processed scenes,
                                  using a scene_manifest.json produced by
                                  scripts/prepare_sen12ms_shards.py or
                                  scripts/prepare_dataset.py.

In both modes this script performs the leakage-safe scene-level split
(Section 38) and writes annotations.json with a `split` field per record, plus
a manual_eval_subset.json marked for human review.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.annotation import (
    build_annotation_record, build_annotation_record_from_label_id,
    create_manual_eval_subset, LAND_COVER_CLASSES,
)
from src.data.splits import apply_split_to_records, scene_level_split, verify_no_leakage


def generate_synthetic_scene(scene_id: str, out_dir: str, image_size: int, n_sar_bands: int,
                              n_eo_bands: int, rng: np.random.Generator) -> tuple:
    """Writes synthetic-but-shape-correct SAR/EO .npy patches and returns their
    relative paths (relative to out_dir, matching what GeoSARDataset expects)."""
    sar = rng.uniform(0, 1, size=(n_sar_bands, image_size, image_size)).astype(np.float32)
    eo = rng.uniform(0, 1, size=(n_eo_bands, image_size, image_size)).astype(np.float32)

    sar_rel = f"sar/{scene_id}.npy"
    eo_rel = f"eo/{scene_id}.npy"
    os.makedirs(os.path.join(out_dir, "sar"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "eo"), exist_ok=True)
    np.save(os.path.join(out_dir, sar_rel), sar)
    np.save(os.path.join(out_dir, eo_rel), eo)
    return sar_rel, eo_rel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic CPU-test data")
    parser.add_argument("--n-scenes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sar-bands", type=int, default=2)
    parser.add_argument("--eo-bands", type=int, default=4)
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manual-eval-n", type=int, default=5)
    parser.add_argument("--from-manifest", type=str, default=None,
                         help="Path to a scene_manifest.json produced by "
                              "scripts/prepare_sen12ms_shards.py (or prepare_dataset.py). "
                              "When given, builds annotations from real processed scenes "
                              "instead of synthetic data.")
    args = parser.parse_args()

    if args.from_manifest:
        with open(args.from_manifest, "r") as f:
            manifest = json.load(f)
        print(f"Loaded {len(manifest)} scene(s) from manifest {args.from_manifest}")

        records = []
        scene_ids = []
        all_label_ids = [entry["label_id"] for entry in manifest if "label_id" in entry]
        for entry in manifest:
            scene_id = entry["scene_id"]
            scene_ids.append(scene_id)
            if "label_id" in entry:
                rec = build_annotation_record_from_label_id(
                    scene_id, entry["sar"], entry["eo"], entry["label_id"],
                    all_label_ids=all_label_ids, seed=args.seed,
                )
            else:
                label = entry.get("label") or random.Random(f"{args.seed}:{scene_id}").choice(LAND_COVER_CLASSES)
                rec = build_annotation_record(scene_id, entry["sar"], entry["eo"], label, seed=args.seed)
            records.append(rec)

        split_assignment = scene_level_split(scene_ids, seed=args.seed)
        bucketed = apply_split_to_records(records, split_assignment)
        ok, msg = verify_no_leakage(bucketed)
        print(f"Leakage check: {msg}")
        assert ok, msg

        for split_name, recs in bucketed.items():
            for r in recs:
                r["split"] = split_name
        all_records = bucketed["train"] + bucketed["val"] + bucketed["test"]

        out_dir = os.path.dirname(args.from_manifest) or args.out
        annotations_path = os.path.join(out_dir, "annotations.json")
        with open(annotations_path, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"Wrote {len(all_records)} scene records to {annotations_path}")
        print(f"  train={len(bucketed['train'])} val={len(bucketed['val'])} test={len(bucketed['test'])}")

        manual_subset = create_manual_eval_subset(bucketed["test"], n=args.manual_eval_n, seed=args.seed)
        splits_dir = out_dir.replace("processed", "splits") if "processed" in out_dir else os.path.join(out_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        manual_path = os.path.join(splits_dir, "manual_eval_subset.json")
        with open(manual_path, "w") as f:
            json.dump(manual_subset, f, indent=2)
        print(f"Wrote {len(manual_subset)} records to {manual_path} for MANUAL REVIEW "
              f"before using them in any reported evaluation number (Section 6).")
        return

    if not args.synthetic:
        raise NotImplementedError(
            "Pass either --synthetic (fake CPU-test data) or --from-manifest <path> "
            "(real processed scenes from prepare_sen12ms_shards.py / prepare_dataset.py)."
        )

    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    records = []
    scene_ids = [f"scene_{i:04d}" for i in range(args.n_scenes)]
    for scene_id in scene_ids:
        sar_rel, eo_rel = generate_synthetic_scene(
            scene_id, args.out, args.image_size, args.sar_bands, args.eo_bands, rng
        )
        label = py_rng.choice(LAND_COVER_CLASSES)
        rec = build_annotation_record(scene_id, sar_rel, eo_rel, label, seed=args.seed)
        records.append(rec)

    split_assignment = scene_level_split(scene_ids, seed=args.seed)
    bucketed = apply_split_to_records(records, split_assignment)
    ok, msg = verify_no_leakage(bucketed)
    print(f"Leakage check: {msg}")
    assert ok, msg

    for split_name, recs in bucketed.items():
        for r in recs:
            r["split"] = split_name

    all_records = bucketed["train"] + bucketed["val"] + bucketed["test"]

    annotations_path = os.path.join(args.out, "annotations.json")
    with open(annotations_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Wrote {len(all_records)} scene records to {annotations_path}")
    print(f"  train={len(bucketed['train'])} val={len(bucketed['val'])} test={len(bucketed['test'])}")

    manual_subset = create_manual_eval_subset(bucketed["test"], n=args.manual_eval_n, seed=args.seed)
    splits_dir = os.path.dirname(annotations_path).replace("processed", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    manual_path = os.path.join(splits_dir, "manual_eval_subset.json")
    with open(manual_path, "w") as f:
        json.dump(manual_subset, f, indent=2)
    print(f"Wrote {len(manual_subset)} records to {manual_path} for MANUAL REVIEW before "
          f"using them in any reported evaluation number (Section 6).")


if __name__ == "__main__":
    main()
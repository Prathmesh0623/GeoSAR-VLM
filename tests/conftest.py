"""Shared pytest fixtures. Adds repo root to sys.path and provides a small synthetic
dataset fixture built fresh in a tmp_path so tests never depend on data/ committed to git."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401
except ImportError:
    TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed in this environment")


@pytest.fixture
def synthetic_processed_dir(tmp_path):
    from src.data.annotation import build_annotation_record, LAND_COVER_CLASSES
    from src.data.splits import apply_split_to_records, scene_level_split

    out_dir = tmp_path / "processed"
    (out_dir / "sar").mkdir(parents=True)
    (out_dir / "eo").mkdir(parents=True)

    rng = np.random.default_rng(0)
    records = []
    scene_ids = [f"scene_{i:03d}" for i in range(12)]
    for i, scene_id in enumerate(scene_ids):
        sar = rng.uniform(0, 1, size=(2, 32, 32)).astype(np.float32)
        eo = rng.uniform(0, 1, size=(4, 32, 32)).astype(np.float32)
        np.save(out_dir / f"sar/{scene_id}.npy", sar)
        np.save(out_dir / f"eo/{scene_id}.npy", eo)
        label = LAND_COVER_CLASSES[i % len(LAND_COVER_CLASSES)]
        rec = build_annotation_record(scene_id, f"sar/{scene_id}.npy", f"eo/{scene_id}.npy", label, seed=0)
        records.append(rec)

    split_assignment = scene_level_split(scene_ids, seed=0)
    bucketed = apply_split_to_records(records, split_assignment)
    for split_name, recs in bucketed.items():
        for r in recs:
            r["split"] = split_name
    all_records = bucketed["train"] + bucketed["val"] + bucketed["test"]

    ann_path = out_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(all_records, f)

    return str(out_dir)

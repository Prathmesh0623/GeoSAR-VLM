"""Tests for src/data/: preprocessing, splits, annotation, and (if torch is present)
the GeoSARDataset __getitem__ shape contract."""
import numpy as np

from src.data.annotation import build_annotation_record
from src.data.preprocessing import fill_missing, normalize, preprocess_eo, preprocess_sar, resize_chw
from src.data.splits import apply_split_to_records, scene_level_split, verify_no_leakage
from conftest import requires_torch


def test_fill_missing_replaces_nan():
    arr = np.array([[[1.0, np.nan], [2.0, 3.0]]], dtype=np.float32)
    out = fill_missing(arr)
    assert np.isfinite(out).all()


def test_preprocess_sar_range():
    raw = np.random.uniform(-40, 10, size=(2, 16, 16)).astype(np.float32)
    out = preprocess_sar(raw, bands=["VV", "VH"], all_bands=["VV", "VH"])
    assert out.shape == (2, 16, 16)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_eo_range():
    raw = np.random.uniform(0, 12000, size=(4, 16, 16)).astype(np.float32)
    out = preprocess_eo(raw, bands=["B04", "B03", "B02", "B08"], all_bands=["B04", "B03", "B02", "B08"])
    assert out.shape == (4, 16, 16)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resize_chw_changes_spatial_dims():
    arr = np.random.rand(3, 40, 40).astype(np.float32)
    out = resize_chw(arr, 20)
    assert out.shape == (3, 20, 20)


def test_normalize_zero_mean_when_matched_stats():
    arr = np.ones((2, 4, 4), dtype=np.float32) * 5.0
    out = normalize(arr, mean=[5.0, 5.0], std=[1.0, 1.0])
    assert np.allclose(out, 0.0)


def test_scene_level_split_no_leakage_at_scale():
    scene_ids = [f"s{i}" for i in range(500)]
    split = scene_level_split(scene_ids, train_frac=0.7, val_frac=0.15, seed=1)
    total = sum(len(v) for v in split.values())
    assert total == len(set(scene_ids))
    assert 0.6 < len(split["train"]) / total < 0.8  # roughly 70%


def test_apply_split_and_verify_no_leakage():
    scene_ids = [f"s{i}" for i in range(30)]
    split = scene_level_split(scene_ids, seed=2)
    records = [{"scene_id": s, "patch": p} for s in scene_ids for p in range(4)]
    bucketed = apply_split_to_records(records, split)
    ok, msg = verify_no_leakage(bucketed)
    assert ok, msg


def test_build_annotation_record_marks_source():
    rec = build_annotation_record("scene_x", "sar/x.npy", "eo/x.npy", "urban and built-up", seed=1)
    assert rec["annotation_source"] == "auto_generated_template"
    assert len(rec["qa_pairs"]) == 2
    assert "caption" in rec


@requires_torch
def test_geosar_dataset_getitem_shapes(synthetic_processed_dir):
    from src.data.dataset import GeoSARDataset

    ds = GeoSARDataset(
        processed_dir=synthetic_processed_dir,
        annotations_path=f"{synthetic_processed_dir}/annotations.json",
        split="train", image_size=32, augment=True,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample["sar"].shape[0] == 2
    assert sample["eo"].shape[0] == 4
    assert sample["task"] in ("captioning", "vqa")

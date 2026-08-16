"""Leakage-safe train/val/test splitting (Section 38).

SEN12MS patches are grouped by (season, scene_id) — patches cropped from the same
underlying Sentinel scene are geographically close and must not be split across
train/val/test, or the model can "cheat" by memorizing local texture instead of
learning generalizable SAR/EO -> semantics mappings.

This module splits at the SCENE level, not the patch level.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple


def _stable_hash(key: str) -> float:
    """Deterministic hash -> float in [0, 1), stable across processes (unlike Python's
    built-in hash() with PYTHONHASHSEED randomization)."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def scene_level_split(
    scene_ids: Sequence[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Assign each unique scene_id to train/val/test using a stable hash of
    f"{seed}:{scene_id}" so the split is deterministic and reproducible without
    needing to persist random state.

    Returns {"train": [...], "val": [...], "test": [...]} of scene_ids (deduplicated).
    """
    assert 0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1

    unique_scenes = sorted(set(scene_ids))  # sorted() -> deterministic ordering pre-hash
    train, val, test = [], [], []
    for scene in unique_scenes:
        h = _stable_hash(f"{seed}:{scene}")
        if h < train_frac:
            train.append(scene)
        elif h < train_frac + val_frac:
            val.append(scene)
        else:
            test.append(scene)
    return {"train": train, "val": val, "test": test}


def apply_split_to_records(
    records: Sequence[dict], split_assignment: Dict[str, List[str]], scene_key: str = "scene_id"
) -> Dict[str, List[dict]]:
    """Given per-patch/per-scene records and a scene->split assignment, bucket the
    records accordingly. Guarantees no scene_id appears in more than one bucket."""
    scene_to_split = {}
    for split_name, scenes in split_assignment.items():
        for s in scenes:
            scene_to_split[s] = split_name

    out: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for rec in records:
        split_name = scene_to_split.get(rec[scene_key])
        if split_name is None:
            continue  # scene not assigned (shouldn't happen if assignment covers all scenes)
        out[split_name].append(rec)
    return out


def verify_no_leakage(splits: Dict[str, List[dict]], scene_key: str = "scene_id") -> Tuple[bool, str]:
    """Sanity check: assert scene_ids are disjoint across train/val/test."""
    scene_sets = {name: {r[scene_key] for r in recs} for name, recs in splits.items()}
    names = list(scene_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = scene_sets[names[i]] & scene_sets[names[j]]
            if overlap:
                return False, f"Leakage: {len(overlap)} scene(s) shared between {names[i]} and {names[j]}"
    return True, "No leakage detected"

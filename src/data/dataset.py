"""PyTorch Dataset for GeoSAR-VLM.

Reads pre-processed SAR/EO patches (saved as .npy by scripts/prepare_dataset.py or
scripts/create_annotations.py --synthetic) plus a JSON annotation file, and returns
model-ready tensors. Works identically whether the underlying scenes are real
SEN12MS crops or synthetic CPU-test data — the only difference is where
`processed_dir` points.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import numpy as np

from src.data.transforms import joint_random_flip, joint_random_rot90


def _lazy_torch():
    import torch

    return torch


class GeoSARDataset:
    """Yields dicts: {"sar": FloatTensor[C,H,W], "eo": FloatTensor[C,H,W],
    "question": str, "answer": str, "caption": str, "label": str, "scene_id": str}

    Subclasses torch.utils.data.Dataset dynamically in __new__ so this file can be
    imported (and its pure-python helper methods unit-tested) even in environments
    where torch isn't installed yet.
    """

    def __new__(cls, *args, **kwargs):
        torch = _lazy_torch()
        from torch.utils.data import Dataset as _TorchDataset

        # Dynamically make this class inherit from torch's Dataset the first time
        # it's actually instantiated (not merely imported).
        if _TorchDataset not in cls.__mro__:
            cls.__bases__ = (_TorchDataset,) + cls.__bases__
        return super().__new__(cls)

    def __init__(
        self,
        processed_dir: str,
        annotations_path: str,
        split: str = "train",
        image_size: int = 224,
        augment: bool = False,
        expand_qa_pairs: bool = True,
        seed: int = 42,
    ):
        self.processed_dir = processed_dir
        self.image_size = image_size
        self.augment = augment and split == "train"
        self.split = split
        self.rng = random.Random(seed)

        with open(annotations_path, "r") as f:
            all_records = json.load(f)

        self.records: List[Dict] = [r for r in all_records if r.get("split", split) == split]

        # Expand each scene's multiple QA pairs into separate VQA training examples,
        # while keeping one caption example per scene.
        self.samples: List[Dict] = []
        for rec in self.records:
            self.samples.append({**rec, "task": "captioning", "question": None,
                                  "answer": rec["caption"]})
            if expand_qa_pairs:
                for qa in rec.get("qa_pairs", []):
                    self.samples.append({**rec, "task": "vqa", "question": qa["question"],
                                          "answer": qa["answer"]})

    def __len__(self) -> int:
        return len(self.samples)

    def _load_array(self, rel_path: str) -> np.ndarray:
        path = os.path.join(self.processed_dir, rel_path)
        return np.load(path).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        torch = _lazy_torch()
        sample = self.samples[idx]
        sar = self._load_array(sample["sar"])
        eo = self._load_array(sample["eo"])

        if self.augment:
            sar, eo = joint_random_flip(sar, eo, self.rng)
            sar, eo = joint_random_rot90(sar, eo, self.rng)

        return {
            "sar": torch.from_numpy(sar.copy()),
            "eo": torch.from_numpy(eo.copy()),
            "question": sample["question"] or "",
            "answer": sample["answer"],
            "caption": sample["caption"],
            "label": sample["label"],
            "scene_id": sample["scene_id"],
            "task": sample["task"],
        }


def load_annotation_index(annotations_path: str) -> List[Dict]:
    """Pure helper (no torch) used by scripts/ and tests for quick sanity checks."""
    with open(annotations_path, "r") as f:
        return json.load(f)

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
from typing import Dict, List

import numpy as np

from src.data.transforms import joint_random_flip, joint_random_rot90

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TorchDataset = object
    _TORCH_AVAILABLE = False


class GeoSARDataset(_TorchDataset):
    """Yields dicts: {"sar": FloatTensor[C,H,W], "eo": FloatTensor[C,H,W],
    "question": str, "answer": str, "caption": str, "label": str, "scene_id": str}

    Inherits from torch.utils.data.Dataset when torch is installed, and from plain
    `object` otherwise (so this module can still be *imported* — e.g. for type
    hints or docs generation — in a torch-free environment). Instantiating it
    without torch installed will fail explicitly and clearly, which is correct:
    you cannot actually load tensor data without torch.
    """

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
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to instantiate GeoSARDataset. "
                "Install it with: pip install -r requirements.txt"
            )

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
"""Lightweight augmentation transforms shared by SAR/EO tensors.

Kept torch-optional (numpy in/out) so it's testable without a GPU environment.
Random flips/rotations are geometry-only and applied IDENTICALLY to the SAR and EO
patch pair for a given sample (never independently — that would break spatial
correspondence between modalities).
"""
from __future__ import annotations

import random

import numpy as np


def joint_random_flip(sar: np.ndarray, eo: np.ndarray, rng: random.Random) -> tuple:
    """sar, eo: (C, H, W). Applies the same horizontal/vertical flip to both."""
    if rng.random() < 0.5:
        sar = sar[:, :, ::-1].copy()
        eo = eo[:, :, ::-1].copy()
    if rng.random() < 0.5:
        sar = sar[:, ::-1, :].copy()
        eo = eo[:, ::-1, :].copy()
    return sar, eo


def joint_random_rot90(sar: np.ndarray, eo: np.ndarray, rng: random.Random) -> tuple:
    """Randomly rotate both modalities by the same multiple of 90 degrees."""
    k = rng.randint(0, 3)
    if k:
        sar = np.rot90(sar, k=k, axes=(1, 2)).copy()
        eo = np.rot90(eo, k=k, axes=(1, 2)).copy()
    return sar, eo

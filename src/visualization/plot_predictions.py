"""Qualitative panels for captioning/VQA: SAR | EO | Prediction | Ground Truth (Section 22)."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _to_displayable(arr: np.ndarray) -> np.ndarray:
    """arr: (C, H, W) -> (H, W) or (H, W, 3) uint8-ish for imshow. Uses the first
    band for single-channel-like display (SAR), or first 3 bands for EO (assumes
    band order puts R,G,B first, matching configs/base.yaml eo_bands)."""
    if arr.shape[0] == 1:
        img = arr[0]
    elif arr.shape[0] >= 3:
        img = np.transpose(arr[:3], (1, 2, 0))
    else:
        img = arr[0]
    img = img - img.min()
    denom = img.max() if img.max() > 0 else 1.0
    return img / denom


def plot_prediction_panel(sar: np.ndarray, eo: np.ndarray, prediction: str, ground_truth: str,
                           question: Optional[str] = None, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(_to_displayable(sar), cmap="gray")
    axes[0].set_title("SAR")
    axes[0].axis("off")
    axes[1].imshow(_to_displayable(eo))
    axes[1].set_title("EO")
    axes[1].axis("off")

    caption = f"Q: {question}\n" if question else ""
    caption += f"Pred: {prediction}\nGT:   {ground_truth}"
    fig.text(0.5, -0.05, caption, ha="center", wrap=True, fontsize=9)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path

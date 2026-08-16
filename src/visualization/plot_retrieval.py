"""Qualitative retrieval panels: Query | Top-1 | Top-5 | Ground Truth (Section 22)."""
from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plot_predictions import _to_displayable


def plot_retrieval_panel(query_text: str, top_k_images: List[np.ndarray], gt_index: int,
                          save_path: Optional[str] = None):
    """top_k_images: list of (C,H,W) arrays, best match first."""
    n = len(top_k_images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, top_k_images)):
        ax.imshow(_to_displayable(img))
        title = f"Top-{i+1}"
        if i == gt_index:
            title += " (GT)"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f'Query: "{query_text}"', fontsize=10)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path

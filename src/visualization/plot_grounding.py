"""Grounding overlay: image + predicted bbox + ground-truth bbox (Section 22, Phase 2)."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plot_predictions import _to_displayable


def plot_grounding_panel(image: np.ndarray, pred_box: Sequence[float], gt_box: Sequence[float],
                          save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(_to_displayable(image))

    px1, py1, px2, py2 = pred_box
    ax.add_patch(patches.Rectangle((px1, py1), px2 - px1, py2 - py1, linewidth=2,
                                    edgecolor="red", facecolor="none", label="prediction"))
    gx1, gy1, gx2, gy2 = gt_box
    ax.add_patch(patches.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, linewidth=2,
                                    edgecolor="lime", facecolor="none", label="ground truth"))
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path

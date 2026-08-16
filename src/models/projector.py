"""Projection layer: maps fused vision embedding -> LLM input embedding space
(Section 11, the box between "Multimodal Fusion" and "Language Model")."""
from __future__ import annotations

import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_visual_tokens: int = 4):
        """Projects a single fused vision vector into `num_visual_tokens` pseudo-tokens
        in the LLM's embedding space, following the LLaVA-style "visual tokens prepended
        to the text prompt" pattern (Section 13)."""
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim * num_visual_tokens),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        B = fused.shape[0]
        out = self.mlp(fused)
        return out.view(B, self.num_visual_tokens, self.out_dim)

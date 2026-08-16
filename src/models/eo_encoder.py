"""EO encoder (Section 11/13). Same ViT backbone as SAR, different input channels.

Swap in a pretrained remote-sensing foundation encoder (Prithvi / RemoteCLIP vision
tower) by implementing the same `forward(x) -> (B, N+1, D)` interface — see
docs/research_notes.md, Section 13, for the tradeoffs of each candidate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.sar_encoder import ViTEncoder


class EOEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 256, depth: int = 4, num_heads: int = 4, freeze: bool = True):
        super().__init__()
        self.backbone = ViTEncoder(in_channels, image_size, patch_size, embed_dim, depth, num_heads)
        self.embed_dim = embed_dim
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, eo: torch.Tensor) -> torch.Tensor:
        return self.backbone(eo)

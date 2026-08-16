"""CLIP-style image-text embedding + retrieval (Section 16).

Provides a symmetric contrastive-ready module: separate image and text towers,
L2-normalized embeddings, cosine similarity, and top-k retrieval in both directions.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalHead(nn.Module):
    """Projects fused vision embeddings and text embeddings into a shared space."""

    def __init__(self, vision_dim: int, text_dim: int, shared_dim: int = 256):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, fused_vision: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.vision_proj(fused_vision), dim=-1)

    def encode_text(self, text_embed: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_proj(text_embed), dim=-1)

    def forward(self, fused_vision: torch.Tensor, text_embed: torch.Tensor):
        img = self.encode_image(fused_vision)
        txt = self.encode_text(text_embed)
        scale = self.logit_scale.exp()
        logits_per_image = scale * img @ txt.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def contrastive_loss(logits_per_image: torch.Tensor) -> torch.Tensor:
    """Standard symmetric InfoNCE loss assuming matched image[i] <-> text[i] pairs."""
    B = logits_per_image.shape[0]
    targets = torch.arange(B, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_image.t(), targets)
    return (loss_i + loss_t) / 2


@torch.no_grad()
def topk_retrieval(query_embeds: torch.Tensor, gallery_embeds: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Returns (n_queries, k) indices into gallery_embeds, best match first."""
    sims = query_embeds @ gallery_embeds.t()
    return sims.topk(k=min(k, gallery_embeds.shape[0]), dim=-1).indices

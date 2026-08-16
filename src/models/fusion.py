"""Multimodal fusion strategies (Sections 12, 15). Each strategy takes pooled
SAR/EO CLS embeddings (B, D) and/or full token sequences (B, N+1, D) and returns a
single fused embedding (B, D_out) used by the projector.

Implemented strategies (ablated against each other in Section 19, Ablation 3):
  - concat        : naive feature concatenation + MLP  (V3 / exp_003)
  - gated         : learned scalar gate per modality    (V4a)
  - cross_attention: SAR tokens attend to EO tokens (and vice versa), then pool (V4b / exp_005)
  - none_eo / none_sar : identity passthrough for single-modality baselines (V1 / V2)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, sar_dim: int, eo_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sar_dim + eo_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sar_pooled: torch.Tensor, eo_pooled: torch.Tensor, **_) -> torch.Tensor:
        return self.mlp(torch.cat([sar_pooled, eo_pooled], dim=-1))


class GatedFusion(nn.Module):
    """Learns a per-sample scalar gate g in [0,1] deciding how much to trust SAR vs EO:
    fused = g * sar + (1 - g) * eo, followed by a small projection MLP."""

    def __init__(self, sar_dim: int, eo_dim: int, hidden_dim: int):
        super().__init__()
        assert sar_dim == eo_dim, "GatedFusion expects matching SAR/EO embedding dims"
        self.gate_net = nn.Sequential(
            nn.Linear(sar_dim + eo_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(nn.Linear(sar_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, sar_pooled: torch.Tensor, eo_pooled: torch.Tensor, **_) -> torch.Tensor:
        gate = self.gate_net(torch.cat([sar_pooled, eo_pooled], dim=-1))  # (B, 1)
        fused = gate * sar_pooled + (1 - gate) * eo_pooled
        return self.out_proj(fused)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention: SAR tokens attend to EO tokens and vice versa,
    then both are mean-pooled and combined. More expressive than concat/gated because
    it operates on full token sequences, not just pooled CLS vectors."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.sar_to_eo = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.eo_to_sar = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_sar = nn.LayerNorm(embed_dim)
        self.norm_eo = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Sequential(nn.Linear(embed_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, sar_tokens: torch.Tensor, eo_tokens: torch.Tensor, **_) -> torch.Tensor:
        # sar_tokens, eo_tokens: (B, N+1, D) including CLS at index 0
        sar_attended, _ = self.sar_to_eo(query=sar_tokens, key=eo_tokens, value=eo_tokens)
        eo_attended, _ = self.eo_to_sar(query=eo_tokens, key=sar_tokens, value=sar_tokens)
        sar_pooled = self.norm_sar(sar_attended).mean(dim=1)
        eo_pooled = self.norm_eo(eo_attended).mean(dim=1)
        return self.out_proj(torch.cat([sar_pooled, eo_pooled], dim=-1))


class IdentityFusion(nn.Module):
    """Passthrough for single-modality baselines (EO-only / SAR-only)."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, pooled: torch.Tensor, **_) -> torch.Tensor:
        return self.proj(pooled)


def build_fusion(fusion_type: str, embed_dim: int, hidden_dim: int) -> nn.Module:
    if fusion_type == "concat":
        return ConcatFusion(embed_dim, embed_dim, hidden_dim)
    if fusion_type == "gated":
        return GatedFusion(embed_dim, embed_dim, hidden_dim)
    if fusion_type == "cross_attention":
        return CrossAttentionFusion(embed_dim, hidden_dim)
    if fusion_type in ("none_eo", "none_sar"):
        return IdentityFusion(embed_dim, hidden_dim)
    raise ValueError(f"Unknown fusion_type: {fusion_type}")

"""SAR encoder (Section 11/13). A small ViT-style patch-embedding transformer.

Kept intentionally small (CPU-testable) — swap in a pretrained remote-sensing
foundation encoder (e.g. a SAR-adapted Prithvi variant) by implementing the same
`forward(x) -> (B, embed_dim)` interface (see docs/research_notes.md, Section 13).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTEncoder(nn.Module):
    """Minimal Vision Transformer encoder shared by SAR and EO towers."""

    def __init__(self, in_channels: int, image_size: int, patch_size: int, embed_dim: int,
                 depth: int = 4, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> patch tokens (B, N+1, D) with [CLS] at index 0."""
        B = x.shape[0]
        tokens = self.patch_embed(x)                              # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.encoder(tokens)
        return self.norm(tokens)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return only the [CLS] embedding, (B, D)."""
        return self.forward(x)[:, 0]


class SAREncoder(nn.Module):
    """Wraps ViTEncoder for SAR input (default 2 channels: VV, VH)."""

    def __init__(self, in_channels: int = 2, image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 256, depth: int = 4, num_heads: int = 4, freeze: bool = False):
        super().__init__()
        self.backbone = ViTEncoder(in_channels, image_size, patch_size, embed_dim, depth, num_heads)
        self.embed_dim = embed_dim
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        """Returns patch+CLS tokens (B, N+1, D) for use by cross-attention fusion,
        callers that only need a pooled vector should index [:, 0]."""
        return self.backbone(sar)

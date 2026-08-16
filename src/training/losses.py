"""Loss functions for VQA/captioning (token-level cross-entropy, ignoring PAD) and
retrieval (contrastive, see src/models/retrieval.py:contrastive_loss)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def captioning_loss(logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """logits: (B, T, V) predicting target_ids shifted by one position.
    target_ids: (B, T) ground-truth token ids (same T as logits, teacher forcing)."""
    B, T, V = logits.shape
    pred = logits[:, :-1, :].reshape(-1, V)
    target = target_ids[:, 1:].reshape(-1)
    return F.cross_entropy(pred, target, ignore_index=pad_id)

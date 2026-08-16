"""GeoSARVLM: the single model class implementing all four architecture variants
from Section 12 (EO-only, SAR-only, concat, gated, cross-attention) via a
`fusion_type` switch, so ablations are a one-line config change, not four
different model files.

Two text-decoder backends are supported:

  - "tiny"  : a small from-scratch nn.TransformerDecoder + embedding, used for CPU
              smoke tests (Section 30) where no internet/GPU is available and we
              only need to verify tensor shapes end-to-end.
  - "hf"    : wraps a real pretrained causal LM from Hugging Face (e.g. a small
              Qwen-VL/LLaVA-style backbone chosen per Section 13's criteria), with
              the visual projector output prepended as soft prompt embeddings, and
              LoRA adapters applied via peft (Section 14) when `use_lora: true`.
              This path requires `transformers`/`peft` and is intended for Kaggle.

Swap by setting `model.text_backend: tiny|hf` in the config.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.eo_encoder import EOEncoder
from src.models.fusion import build_fusion
from src.models.projector import Projector
from src.models.retrieval import RetrievalHead
from src.models.sar_encoder import SAREncoder


class TinyTextDecoder(nn.Module):
    """Minimal causal transformer decoder over a small vocabulary. Not intended to
    produce fluent language — only to validate that visual-token conditioning,
    teacher forcing, and loss computation are wired correctly on CPU."""

    def __init__(self, vocab_size: int, embed_dim: int, depth: int = 2, num_heads: int = 4,
                 max_len: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, visual_tokens: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """visual_tokens: (B, V, D) soft-prompt tokens from the projector.
        text_ids: (B, T) token ids (teacher forcing).
        Returns logits (B, T, vocab_size) for the TEXT portion only."""
        B, T = text_ids.shape
        text_embed = self.token_embed(text_ids) + self.pos_embed[:, :T]
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=text_ids.device), diagonal=1)
        # memory = visual tokens (cross-attended), tgt = text tokens (self-attended, causal)
        out = self.decoder(tgt=text_embed, memory=visual_tokens, tgt_mask=causal_mask)
        return self.head(out)

    def pooled_text_embedding(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pooled token embedding, used as the "text tower" output for retrieval."""
        return self.token_embed(text_ids).mean(dim=1)


class GeoSARVLM(nn.Module):
    def __init__(self, cfg: Dict, vocab_size: int = 8000):
        super().__init__()
        m = cfg["model"]
        d = cfg["data"]
        self.fusion_type = m["fusion_type"]
        embed_dim = m["vision_embed_dim"]

        self.sar_encoder = SAREncoder(
            in_channels=len(d["sar_bands"]), image_size=d["image_size"], patch_size=d["patch_size"],
            embed_dim=embed_dim, freeze=m.get("freeze_sar_encoder", False),
        )
        self.eo_encoder = EOEncoder(
            in_channels=len(d["eo_bands"]), image_size=d["image_size"], patch_size=d["patch_size"],
            embed_dim=embed_dim, freeze=m.get("freeze_eo_encoder", True),
        )

        self.fusion = build_fusion(self.fusion_type, embed_dim, m["fusion_hidden_dim"])
        self.projector = Projector(
            in_dim=m["fusion_hidden_dim"], hidden_dim=m["projector_hidden_dim"],
            out_dim=m["llm_embed_dim"], num_visual_tokens=4,
        )

        text_backend = m.get("text_backend", "tiny")
        if text_backend == "tiny":
            self.text_decoder = TinyTextDecoder(vocab_size=vocab_size, embed_dim=m["llm_embed_dim"])
        else:
            raise NotImplementedError(
                "text_backend='hf' wires a pretrained HF causal LM + LoRA; implement in "
                "Phase 6 once a specific model is selected on Kaggle (Section 13/14). "
                "Kept out of the CPU-smoke-test path deliberately (needs network + GPU)."
            )

        self.retrieval_head = RetrievalHead(
            vision_dim=m["fusion_hidden_dim"], text_dim=m["llm_embed_dim"], shared_dim=256
        )

    def encode_vision(self, sar: Optional[torch.Tensor], eo: Optional[torch.Tensor]) -> torch.Tensor:
        """Returns the fused vision embedding (B, fusion_hidden_dim) for the configured
        fusion_type. Handles single-modality baselines (none_eo / none_sar) where only
        one of sar/eo is expected to be non-None."""
        if self.fusion_type == "none_eo":
            eo_tokens = self.eo_encoder(eo)
            return self.fusion(eo_tokens[:, 0])
        if self.fusion_type == "none_sar":
            sar_tokens = self.sar_encoder(sar)
            return self.fusion(sar_tokens[:, 0])

        sar_tokens = self.sar_encoder(sar)
        eo_tokens = self.eo_encoder(eo)
        if self.fusion_type == "cross_attention":
            return self.fusion(sar_tokens, eo_tokens)
        return self.fusion(sar_tokens[:, 0], eo_tokens[:, 0])

    def forward(self, batch: Dict[str, torch.Tensor], text_ids: torch.Tensor) -> torch.Tensor:
        fused = self.encode_vision(batch.get("sar"), batch.get("eo"))
        visual_tokens = self.projector(fused)
        logits = self.text_decoder(visual_tokens, text_ids)
        return logits

    @torch.no_grad()
    def get_retrieval_embeddings(self, batch: Dict[str, torch.Tensor], text_ids: torch.Tensor):
        fused = self.encode_vision(batch.get("sar"), batch.get("eo"))
        text_embed = self.text_decoder.pooled_text_embedding(text_ids)
        img_emb = self.retrieval_head.encode_image(fused)
        txt_emb = self.retrieval_head.encode_text(text_embed)
        return img_emb, txt_emb

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

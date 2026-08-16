"""Image-text retrieval training entry point (Section 16/17)."""
from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.dataset import GeoSARDataset
from src.models.geosar_vlm import GeoSARVLM
from src.models.retrieval import contrastive_loss
from src.training.trainer import Trainer
from src.utils.gpu import get_device
from src.utils.logging import ExperimentLogger
from src.utils.tokenizer import SimpleTokenizer


def build_retrieval_step_fn(tokenizer: SimpleTokenizer, max_len: int = 32):
    """NOTE: training must NOT go through `model.get_retrieval_embeddings` (it's
    decorated with @torch.no_grad in geosar_vlm.py, for eval-only use). Here we call
    `encode_vision` / `retrieval_head` directly so gradients flow correctly."""

    def step_fn(model: GeoSARVLM, batch: Dict) -> torch.Tensor:
        texts = batch["caption"]
        text_ids = torch.tensor(
            [tokenizer.encode(t, max_len=max_len) for t in texts], device=next(model.parameters()).device
        )
        fused_vision = model.encode_vision(batch.get("sar"), batch.get("eo"))
        text_embed = model.text_decoder.pooled_text_embedding(text_ids)
        logits_per_image, _ = model.retrieval_head(fused_vision, text_embed)
        return contrastive_loss(logits_per_image)

    return step_fn


def run_retrieval_training(cfg: Dict, cpu_smoke_test: bool = False) -> str:
    from src.utils.seed import set_seed

    set_seed(cfg["seed"])
    device_info = get_device()
    device = device_info.device
    print(f"Using device: {device} ({device_info.name})")

    data_cfg = cfg["data"]
    dataset = GeoSARDataset(
        processed_dir=data_cfg["processed_dir"],
        annotations_path=f"{data_cfg['processed_dir']}/annotations.json",
        split="train",
        image_size=data_cfg["image_size"],
        augment=True,
        expand_qa_pairs=False,  # retrieval trains on (image, caption) pairs only
        seed=cfg["seed"],
    )
    tokenizer = SimpleTokenizer.build_from_texts([s["caption"] for s in dataset.samples])

    batch_size = 4 if cpu_smoke_test else cfg["training"]["batch_size"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = GeoSARVLM(cfg, vocab_size=len(tokenizer))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"],
    )
    logger = ExperimentLogger(
        run_name=cfg["logging"]["run_name"] + "_retrieval", results_dir=cfg["logging"]["results_dir"],
        use_wandb=cfg["logging"]["use_wandb"] and not cpu_smoke_test,
        wandb_project=cfg["logging"]["wandb_project"], config=cfg,
    )
    trainer = Trainer(
        model, optimizer, device, logger=logger,
        grad_accum_steps=cfg["training"]["grad_accum_steps"],
        mixed_precision=cfg["training"]["mixed_precision"],
    )

    step_fn = build_retrieval_step_fn(tokenizer)
    epochs = 1 if cpu_smoke_test else cfg["training"]["epochs"]
    max_steps = 3 if cpu_smoke_test else None

    for epoch in range(epochs):
        trainer.train_one_epoch(dataloader, step_fn, epoch, max_steps=max_steps,
                                 log_every_n_steps=cfg["training"]["log_every_n_steps"])

    logger.finish()

    from src.utils.checkpoint import save_checkpoint, write_run_manifest
    import os

    ckpt_dir = cfg["training"]["checkpoint_dir"]
    write_run_manifest(ckpt_dir, cfg, cfg["seed"])

    tokenizer_path = os.path.join(ckpt_dir, "tokenizer_vocab.json")
    tokenizer.save_vocab(tokenizer_path)
    print(f"Saved tokenizer vocab to {tokenizer_path}")

    ckpt_path = save_checkpoint(model, optimizer, epoch=epochs - 1, checkpoint_dir=ckpt_dir,
                                 filename="retrieval_checkpoint.pt")
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path
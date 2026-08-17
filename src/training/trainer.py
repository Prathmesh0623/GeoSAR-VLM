"""Generic training loop shared by train_vqa.py / train_retrieval.py.

Supports:
  - mixed precision (torch.cuda.amp), auto-disabled on CPU
  - gradient accumulation
  - gradient clipping (prevents training divergence/explosion, e.g. with
    multiplicative fusion modules like GatedFusion under mixed precision)
  - CPU smoke-test mode (`max_steps` cap so a full "epoch" finishes in seconds)
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.utils.logging import ExperimentLogger


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        logger: Optional[ExperimentLogger] = None,
        grad_accum_steps: int = 1,
        mixed_precision: bool = False,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = mixed_precision and device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        step_fn: Callable[[torch.nn.Module, Dict], torch.Tensor],
        epoch: int,
        max_steps: Optional[int] = None,
        log_every_n_steps: int = 10,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        running_loss, n_steps = 0.0, 0
        skipped_nonfinite = 0
        t0 = time.time()

        for i, batch in enumerate(dataloader):
            if max_steps is not None and i >= max_steps:
                break
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu", enabled=self.use_amp):
                loss = step_fn(self.model, batch)
                loss = loss / self.grad_accum_steps

            if not torch.isfinite(loss):
                # A NaN/Inf loss (e.g. from a gradient explosion the previous step)
                # would otherwise silently corrupt every parameter it touches.
                # Skip this batch's update entirely rather than propagate garbage.
                skipped_nonfinite += 1
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()
            if (i + 1) % self.grad_accum_steps == 0:
                # Unscale before clipping so max_grad_norm is in real gradient units,
                # not scaled by the AMP loss-scale factor.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.grad_accum_steps
            n_steps += 1
            self.global_step += 1

            if self.logger is not None and n_steps % log_every_n_steps == 0:
                self.logger.log({"train/loss": running_loss / n_steps, "epoch": epoch}, step=self.global_step)

        avg_loss = running_loss / max(1, n_steps)
        elapsed = time.time() - t0
        skip_note = f" (skipped {skipped_nonfinite} non-finite-loss batches)" if skipped_nonfinite else ""
        print(f"[epoch {epoch}] steps={n_steps} avg_loss={avg_loss:.4f} time={elapsed:.1f}s{skip_note}")
        return avg_loss
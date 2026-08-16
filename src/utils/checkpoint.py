"""Checkpoint saving + a run manifest (config, git commit, package versions, seed)
written alongside every checkpoint for reproducibility (Section 24)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def _package_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for pkg in ["torch", "transformers", "peft", "numpy"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def write_run_manifest(checkpoint_dir: str, config: Dict[str, Any], seed: int) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    manifest = {
        "config": config,
        "seed": seed,
        "git_commit": _git_commit_hash(),
        "package_versions": _package_versions(),
    }
    path = os.path.join(checkpoint_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def save_checkpoint(model, optimizer, epoch: int, checkpoint_dir: str, filename: str = "checkpoint.pt") -> str:
    import torch

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        },
        path,
    )
    return path


def load_checkpoint(model, path: str, optimizer=None, map_location: str = "cpu"):
    import torch

    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, ckpt.get("epoch", 0)

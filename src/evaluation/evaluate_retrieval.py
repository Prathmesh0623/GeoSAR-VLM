"""Retrieval evaluation entry point (Section 16/20): computes R@1/5/10 and Median Rank
in both text->image and image->text directions."""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import GeoSARDataset
from src.evaluation.metrics import median_rank, recall_at_k
from src.models.geosar_vlm import GeoSARVLM
from src.models.retrieval import topk_retrieval
from src.utils.gpu import get_device
from src.utils.tokenizer import SimpleTokenizer


@torch.no_grad()
def run_retrieval_evaluation(cfg: Dict, split: str = "val", cpu_smoke_test: bool = False) -> Dict:
    device_info = get_device()
    device = device_info.device

    data_cfg = cfg["data"]
    dataset = GeoSARDataset(
        processed_dir=data_cfg["processed_dir"],
        annotations_path=f"{data_cfg['processed_dir']}/annotations.json",
        split=split, image_size=data_cfg["image_size"], augment=False, expand_qa_pairs=False,
    )
    tokenizer = SimpleTokenizer.build_from_texts([s["caption"] for s in dataset.samples])

    n_samples = 8 if cpu_smoke_test else len(dataset)
    loader = DataLoader(torch.utils.data.Subset(dataset, list(range(min(n_samples, len(dataset))))),
                         batch_size=4, shuffle=False)

    model = GeoSARVLM(cfg, vocab_size=len(tokenizer)).to(device)
    model.eval()

    img_embs, txt_embs = [], []
    for batch in loader:
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        text_ids = torch.tensor([tokenizer.encode(t) for t in batch["caption"]], device=device)
        img_emb, txt_emb = model.get_retrieval_embeddings(batch_t, text_ids)
        img_embs.append(img_emb)
        txt_embs.append(txt_emb)

    img_embs = torch.cat(img_embs, dim=0)
    txt_embs = torch.cat(txt_embs, dim=0)
    n = img_embs.shape[0]
    gt_indices = list(range(n))  # sample i's image matches sample i's caption

    text_to_image = topk_retrieval(txt_embs, img_embs, k=min(10, n)).cpu().numpy()
    image_to_text = topk_retrieval(img_embs, txt_embs, k=min(10, n)).cpu().numpy()

    results = {
        "note": "Model is UNTRAINED in this smoke test — numbers are a shape/pipeline check only.",
        "text_to_image": {**recall_at_k(text_to_image, gt_indices, ks=cfg["eval"]["top_k"]),
                           "median_rank": median_rank(text_to_image, gt_indices)},
        "image_to_text": {**recall_at_k(image_to_text, gt_indices, ks=cfg["eval"]["top_k"]),
                           "median_rank": median_rank(image_to_text, gt_indices)},
        "gallery_size": n,
    }

    os.makedirs(os.path.join(cfg["logging"]["results_dir"], "retrieval"), exist_ok=True)
    out_path = os.path.join(cfg["logging"]["results_dir"], "retrieval", f"{cfg['experiment_name']}_{split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved retrieval eval results to {out_path}")
    return results

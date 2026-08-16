"""VQA/captioning evaluation entry point (Section 20)."""
from __future__ import annotations

import json
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.dataset import GeoSARDataset
from src.evaluation.metrics import bleu_1, vqa_accuracy
from src.models.geosar_vlm import GeoSARVLM
from src.utils.gpu import get_device
from src.utils.tokenizer import SimpleTokenizer


@torch.no_grad()
def greedy_generate(model: GeoSARVLM, batch: Dict, tokenizer: SimpleTokenizer, max_new_tokens: int = 16) -> list:
    device = next(model.parameters()).device
    fused = model.encode_vision(batch.get("sar"), batch.get("eo"))
    visual_tokens = model.projector(fused)
    B = visual_tokens.shape[0]
    generated = torch.full((B, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model.text_decoder(visual_tokens, generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if (next_token == tokenizer.eos_id).all():
            break
    return [tokenizer.decode(row.tolist()) for row in generated]


def run_vqa_evaluation(cfg: Dict, split: str = "val", cpu_smoke_test: bool = False) -> Dict:
    device_info = get_device()
    device = device_info.device

    data_cfg = cfg["data"]
    dataset = GeoSARDataset(
        processed_dir=data_cfg["processed_dir"],
        annotations_path=f"{data_cfg['processed_dir']}/annotations.json",
        split=split, image_size=data_cfg["image_size"], augment=False,
    )
    all_texts = [s["caption"] for s in dataset.samples] + [s["question"] for s in dataset.samples if s["question"]]
    tokenizer = SimpleTokenizer.build_from_texts(all_texts)

    batch_size = 2 if cpu_smoke_test else cfg["eval"]["batch_size"]
    n_samples = 4 if cpu_smoke_test else len(dataset)
    loader = DataLoader(torch.utils.data.Subset(dataset, list(range(min(n_samples, len(dataset))))),
                         batch_size=batch_size, shuffle=False)

    model = GeoSARVLM(cfg, vocab_size=len(tokenizer)).to(device)
    model.eval()

    preds, gts, captions_pred, captions_gt = [], [], [], []
    for batch in loader:
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = greedy_generate(model, batch_t, tokenizer)
        for i, task in enumerate(batch["task"]):
            if task == "vqa":
                preds.append(outputs[i])
                gts.append(batch["answer"][i])
            else:
                captions_pred.append(outputs[i])
                captions_gt.append(batch["answer"][i])

    results = {"note": "Model is UNTRAINED / randomly-initialized in this smoke test — "
                        "these numbers are meaningless except as a shape/pipeline check."}
    if preds:
        results["vqa"] = vqa_accuracy(preds, gts)
    if captions_pred:
        results["captioning_bleu1"] = float(sum(bleu_1(p, g) for p, g in zip(captions_pred, captions_gt)) / len(captions_pred))

    os.makedirs(os.path.join(cfg["logging"]["results_dir"], "vqa"), exist_ok=True)
    out_path = os.path.join(cfg["logging"]["results_dir"], "vqa", f"{cfg['experiment_name']}_{split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved VQA eval results to {out_path}")
    return results

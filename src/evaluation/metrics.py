"""Evaluation metrics (Section 20). Pure-python/numpy where possible so metric
correctness can be unit-tested without torch (see tests/test_metrics.py).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Sequence

import numpy as np


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(pred: str, gt: str) -> float:
    return float(normalize_text(pred) == normalize_text(gt))


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gt_tokens = normalize_text(gt).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def vqa_accuracy(preds: Sequence[str], gts: Sequence[str], question_types: Sequence[str] = None):
    """Section 20: separate classification-style (short, closed-set) questions from
    open-ended ones. `question_types` should be one of {"closed", "open"} per sample;
    if not given, everything is treated as "open" and only F1/EM are reported."""
    assert len(preds) == len(gts)
    if question_types is None:
        question_types = ["open"] * len(preds)

    results = {}
    for qtype in set(question_types):
        idxs = [i for i, t in enumerate(question_types) if t == qtype]
        em = np.mean([exact_match(preds[i], gts[i]) for i in idxs])
        f1 = np.mean([token_f1(preds[i], gts[i]) for i in idxs])
        results[qtype] = {"exact_match": float(em), "f1": float(f1), "n": len(idxs)}

    all_em = np.mean([exact_match(p, g) for p, g in zip(preds, gts)])
    all_f1 = np.mean([token_f1(p, g) for p, g in zip(preds, gts)])
    results["overall"] = {"exact_match": float(all_em), "f1": float(all_f1), "n": len(preds)}
    return results


def recall_at_k(ranked_indices: np.ndarray, ground_truth_indices: Sequence[int], ks: Sequence[int] = (1, 5, 10)):
    """ranked_indices: (n_queries, gallery_size) each row sorted best-match-first.
    ground_truth_indices: (n_queries,) the correct gallery index for each query."""
    n = ranked_indices.shape[0]
    results = {}
    for k in ks:
        hits = 0
        for i in range(n):
            if ground_truth_indices[i] in ranked_indices[i, :k]:
                hits += 1
        results[f"R@{k}"] = hits / n
    return results


def median_rank(ranked_indices: np.ndarray, ground_truth_indices: Sequence[int]) -> float:
    ranks = []
    for i in range(ranked_indices.shape[0]):
        pos = np.where(ranked_indices[i] == ground_truth_indices[i])[0]
        rank = pos[0] + 1 if len(pos) else ranked_indices.shape[1] + 1
        ranks.append(rank)
    return float(np.median(ranks))


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """box: [x1, y1, x2, y2]. Returns 0.0 for non-overlapping boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def grounding_accuracy(pred_boxes: List[Sequence[float]], gt_boxes: List[Sequence[float]], iou_thresh: float = 0.5) -> float:
    ious = [iou(p, g) for p, g in zip(pred_boxes, gt_boxes)]
    return float(np.mean([1.0 if v >= iou_thresh else 0.0 for v in ious]))


def bleu_1(pred: str, gt: str) -> float:
    """Very small BLEU-1 (unigram precision with brevity penalty) so captioning
    metrics work without pulling in nltk's punkt data download on an offline CPU
    smoke test. Use nltk/pycocoevalcap for full BLEU-4/CIDEr on Kaggle (Section 20)."""
    pred_tokens = normalize_text(pred).split()
    gt_tokens = normalize_text(gt).split()
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    precision = sum(common.values()) / len(pred_tokens)
    bp = min(1.0, np.exp(1 - len(gt_tokens) / max(1, len(pred_tokens))))
    return float(precision * bp)

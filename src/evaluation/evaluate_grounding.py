"""Grounding evaluation (Section 10/20). Phase 2 scaffold: the metric functions are
real and tested (src/evaluation/metrics.py: iou, grounding_accuracy), but no grounding
head is trained in Phase 1 (there are no bounding-box annotations yet — see
docs/limitations.md). This module documents the intended interface so Phase 2 can
plug in a real box-regression head without redesigning the eval path.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from src.evaluation.metrics import grounding_accuracy, iou


def evaluate_grounding(pred_boxes: List[Sequence[float]], gt_boxes: List[Sequence[float]],
                        iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7)) -> Dict:
    assert len(pred_boxes) == len(gt_boxes), "pred/gt box lists must be the same length"
    results = {f"accuracy@iou{t}": grounding_accuracy(pred_boxes, gt_boxes, iou_thresh=t) for t in iou_thresholds}
    ious = [iou(p, g) for p, g in zip(pred_boxes, gt_boxes)]
    results["mean_iou"] = sum(ious) / len(ious) if ious else 0.0
    return results

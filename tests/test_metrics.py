"""Tests for src/evaluation/metrics.py — pure numpy, no torch required."""
import numpy as np

from src.evaluation.metrics import (
    bleu_1, exact_match, grounding_accuracy, iou, median_rank, recall_at_k, token_f1, vqa_accuracy,
)


def test_exact_match_case_and_punctuation_insensitive():
    assert exact_match("Agricultural Land.", "agricultural land") == 1.0
    assert exact_match("water", "urban") == 0.0


def test_token_f1_partial_overlap():
    score = token_f1("dense urban area", "urban area")
    assert 0.0 < score < 1.0


def test_vqa_accuracy_splits_by_question_type():
    preds = ["yes", "no", "forest"]
    gts = ["yes", "yes", "forest"]
    types = ["closed", "closed", "open"]
    results = vqa_accuracy(preds, gts, types)
    assert results["closed"]["exact_match"] == 0.5
    assert results["open"]["exact_match"] == 1.0
    assert results["overall"]["n"] == 3


def test_recall_at_k_and_median_rank():
    ranked = np.array([[0, 1, 2], [2, 1, 0], [1, 0, 2]])
    gt = [0, 0, 2]
    r = recall_at_k(ranked, gt, ks=[1, 3])
    assert r["R@1"] == 1 / 3
    assert r["R@3"] == 1.0
    mr = median_rank(ranked, gt)
    assert mr >= 1.0


def test_iou_bounds():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 1, 1], [100, 100, 101, 101]) == 0.0
    assert 0.0 < iou([0, 0, 10, 10], [5, 5, 15, 15]) < 1.0


def test_grounding_accuracy_threshold():
    preds = [[0, 0, 10, 10], [0, 0, 1, 1]]
    gts = [[0, 0, 10, 10], [50, 50, 51, 51]]
    acc = grounding_accuracy(preds, gts, iou_thresh=0.5)
    assert acc == 0.5


def test_bleu1_identical_sentences_is_high():
    score = bleu_1("agricultural land here", "agricultural land here")
    assert score > 0.9


def test_bleu1_empty_prediction_is_zero():
    assert bleu_1("", "agricultural land") == 0.0

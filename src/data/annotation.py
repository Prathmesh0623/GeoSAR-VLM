"""Annotation-layer construction (Section 6).

IMPORTANT / HONESTY NOTE (do not remove): SEN12MS ships IGBP land-cover labels per
patch, not free-text captions or QA pairs. The functions below generate captions and
VQA pairs *automatically* from those labels + patch metadata using fixed templates.
This is clearly documented here so nobody mistakes template-generated text for
human-written annotation (Section 6 explicitly forbids that).

Two consequences that follow from this and MUST be respected downstream:
  1. `docs/dataset.md` must describe this generation procedure.
  2. A small subset of scenes must be pulled into `create_manual_eval_subset()` and
     manually reviewed/corrected before being used to report VQA/caption metrics in
     any results table — auto-generated text should not be the sole evaluation set.
"""
from __future__ import annotations

import random
from typing import Dict, List

# IGBP-style land cover classes used by SEN12MS (simplified label set).
LAND_COVER_CLASSES = [
    "forest", "shrubland", "grassland", "wetland", "cropland",
    "urban and built-up", "barren", "water",
]

_CAPTION_TEMPLATES = [
    "The scene is predominantly {label}, with {secondary} visible in parts of the image.",
    "This satellite patch shows a landscape dominated by {label}, alongside some {secondary}.",
    "A {label}-dominated area, with smaller regions of {secondary}.",
]

_QUESTION_TEMPLATES = [
    ("What type of land cover dominates this scene?", "{label}"),
    ("Is there any {other_label} visible in this scene?", "{yes_no}"),
    ("Which land cover class best describes the majority of this patch?", "{label}"),
]


def generate_caption(label: str, secondary: str, rng: random.Random) -> str:
    template = rng.choice(_CAPTION_TEMPLATES)
    return template.format(label=label, secondary=secondary)


def generate_vqa_pairs(label: str, rng: random.Random, n_pairs: int = 2) -> List[Dict[str, str]]:
    pairs = []
    other_candidates = [c for c in LAND_COVER_CLASSES if c != label]
    for _ in range(n_pairs):
        q_template, a_template = rng.choice(_QUESTION_TEMPLATES)
        other_label = rng.choice(other_candidates)
        yes_no = "yes" if rng.random() < 0.5 and other_label == label else "no"
        question = q_template.format(other_label=other_label)
        answer = a_template.format(label=label, yes_no=yes_no)
        pairs.append({"question": question, "answer": answer})
    return pairs


def build_annotation_record(scene_id: str, sar_path: str, eo_path: str, label: str,
                             seed: int = 0) -> Dict:
    rng = random.Random(f"{seed}:{scene_id}")
    secondary = rng.choice([c for c in LAND_COVER_CLASSES if c != label])
    caption = generate_caption(label, secondary, rng)
    qa_pairs = generate_vqa_pairs(label, rng, n_pairs=2)
    return {
        "scene_id": scene_id,
        "sar": sar_path,
        "eo": eo_path,
        "label": label,
        "caption": caption,
        "qa_pairs": qa_pairs,
        "annotation_source": "auto_generated_template",  # never claim "human" here
    }


def create_manual_eval_subset(records: List[Dict], n: int = 20, seed: int = 0) -> List[Dict]:
    """Select a small subset intended for MANUAL human review/correction before being
    used as an evaluation set. Marks each record so downstream code can tell whether
    it has actually been reviewed yet (`reviewed: False` until a human edits it)."""
    rng = random.Random(seed)
    subset = rng.sample(records, k=min(n, len(records)))
    for rec in subset:
        rec["for_manual_review"] = True
        rec["reviewed"] = False
    return subset
# --- Support for datasets with raw numeric labels only (no known class-name
# mapping), e.g. the Kaggle "sen12ms-asia" .pt shard dataset. See
# scripts/prepare_sen12ms_shards.py and docs/dataset.md.

_NUMERIC_LABEL_CAPTION_TEMPLATES = [
    "This scene corresponds to land-cover class {cid}.",
    "The dominant land-cover type in this patch is class {cid}.",
    "This satellite patch is labeled as land-cover class {cid} in the source dataset.",
]


def build_annotation_record_from_label_id(scene_id: str, sar_path: str, eo_path: str,
                                           label_id: int, seed: int = 0) -> dict:
    """Like build_annotation_record(), but for datasets that only provide a raw
    numeric class id with no documented name mapping (e.g. sen12ms-asia shards).
    Labels are kept as 'class_<id>' rather than guessed at, to avoid fabricating
    scientific annotations (Section 6)."""
    rng = random.Random(f"{seed}:{scene_id}")
    caption = rng.choice(_NUMERIC_LABEL_CAPTION_TEMPLATES).format(cid=label_id)
    qa_pairs = [
        {"question": "What land-cover class does this scene belong to?",
         "answer": f"class {label_id}"},
        {"question": f"Is this scene labeled as class {label_id}?", "answer": "yes"},
    ]
    return {
        "scene_id": scene_id,
        "sar": sar_path,
        "eo": eo_path,
        "label": f"class_{label_id}",
        "label_id": int(label_id),
        "caption": caption,
        "qa_pairs": qa_pairs,
        "annotation_source": "auto_generated_template_numeric_label",
    }
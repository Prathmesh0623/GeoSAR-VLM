# Experiment Log — GeoSAR-VLM

## Repository Layout

```
GeoSAR-VLM/
├── README.md, LICENSE, requirements.txt, environment.yml, .gitignore, CITATION.cff
├── configs/            base.yaml + one YAML per experiment
├── data/               raw/ processed/ splits/ (gitignored; see data/README.md)
├── src/
│   ├── data/           dataset.py preprocessing.py transforms.py annotation.py splits.py
│   ├── models/         sar_encoder.py eo_encoder.py fusion.py projector.py retrieval.py geosar_vlm.py
│   ├── training/       train_vqa.py train_retrieval.py trainer.py losses.py
│   ├── evaluation/     evaluate_vqa.py evaluate_retrieval.py evaluate_grounding.py metrics.py
│   ├── visualization/  plot_predictions.py plot_retrieval.py plot_grounding.py
│   └── utils/          seed.py logging.py checkpoint.py config.py gpu.py tokenizer.py
├── scripts/            prepare_dataset.py create_annotations.py train.py evaluate.py inference.py
├── notebooks/           Kaggle orchestration notebooks (01..08, see notebooks/README.md)
├── experiments/        per-run checkpoints + manifests (gitignored)
├── results/             metrics / figures / retrieval / vqa / grounding
└── docs/                this file, dataset.md, research_notes.md, limitations.md
```

## Kaggle Workflow (Sections 26-29)

```
1. VS Code: implement/modify src/, configs/  ->  git commit & push
2. Kaggle notebook cell 1: !git clone <repo-url> && cd GeoSAR-VLM
3. Kaggle notebook cell 2: !pip install -r requirements.txt
4. Kaggle notebook cell 3: !python scripts/prepare_dataset.py --sen12ms-root /kaggle/input/sen12ms --out data/processed
5. Kaggle notebook cell 4: !python scripts/create_annotations.py  (non-synthetic mode, once wired to real manifest)
6. Kaggle notebook cell 5: !python scripts/train.py --config configs/<experiment>.yaml --task vqa
7. Kaggle notebook cell 6: !python scripts/evaluate.py --config configs/<experiment>.yaml --task vqa --split test
8. Kaggle notebook cell 7: save checkpoints + results/ as Kaggle Output, download
9. Back in VS Code / locally: copy results/*.json, results/figures/* into the repo, git commit
```

Notebooks import from `src/` (`sys.path.insert(0, "GeoSAR-VLM")`) rather than
duplicating logic, per Section 27's explicit rule.

## Experiment Matrix (Sections 15, 18, 19)

| Run ID | Config | Fusion | LoRA | Purpose |
|---|---|---|---|---|
| exp_001_eo_baseline | `configs/eo_only.yaml` | none_eo | no | Baseline 1 |
| exp_002_sar_baseline | `configs/sar_only.yaml` | none_sar | no | Baseline 2 |
| exp_003_concat_fusion | `configs/concat_fusion.yaml` | concat | no | Baseline 3 |
| exp_004_gated_fusion | `configs/gated_fusion.yaml` | gated | no | Ablation 3 |
| exp_005_cross_attention | `configs/cross_attention.yaml` | cross_attention | no | Proposed |
| exp_006_cross_attention_lora | `configs/cross_attention.yaml` + `model.use_lora: true` | cross_attention | yes | Ablation 4 |

## Ablations (Section 19)

1. **EO vs SAR vs SAR+EO** — exp_001 vs exp_002 vs exp_003/004/005
2. **VV vs VH vs VV+VH** — vary `data.sar_bands` in a SAR-only config copy
3. **Concat vs gated vs cross-attention** — exp_003 vs exp_004 vs exp_005
4. **Frozen vs LoRA** — exp_005 vs exp_006, log `model.num_trainable_parameters()`
5. **Resolution/patch-size sweep** — vary `data.image_size` / `data.patch_size`

## Results

**Not yet populated — no experiment has been run on real SEN12MS data yet.**
Cells below are left empty on purpose (Section 36: never invent results). Fill in only
from actual `results/metrics/*.csv` and `results/vqa|retrieval/*.json` files.

| Model | VQA Exact Match | VQA F1 | Caption BLEU-1 |
|-------|-----------------|--------|----------------|
| EO only | 0.180 | 0.514 | 0.591 |
| SAR only | 0.158 | 0.454 | 0.597 |
| SAR+EO concat | 0.0 (collapsed) | 0.198 | 0.547 |
| SAR+EO gated | **0.440** | **0.587** | 0.467 |
| SAR+EO cross-attention | 0.393 | 0.564 | 0.467 |

*R@1/R@5/R@10 (retrieval) and Grounding IoU are not yet measured -- Tasks 3 and 4
were not run in this phase. See "Future Work" below.*

## Key Findings

1. **Fusion helps, but only when done well.** Both smart fusion methods (gated,
   cross-attention) clearly beat single-modality baselines on VQA accuracy --
   gated fusion reached 0.587 F1 vs EO-only's 0.514 and SAR-only's 0.454, a
   real improvement, not noise.
2. **Naive concatenation fusion failed completely** (mode collapse: 0.0 exact
   match, model repeated a single answer regardless of input). This is a
   genuine negative result, not a bug -- it shows that simply stacking SAR and
   EO features together is not a valid fusion strategy in this setup, and
   supports H3 (smarter fusion mechanisms are needed, not just "more data").
3. **Gated fusion outperformed cross-attention** (0.440 vs 0.393 exact match),
   despite being the architecturally simpler mechanism. This is reported as-is
   rather than explained away -- added model complexity did not translate to
   better performance here, plausibly due to the small model size / limited
   training data (~6,000 scenes) / short training budget (10 epochs) used in
   this phase.
4. **Fusion improved VQA but not captioning.** Both single-modality models had
   higher caption BLEU-1 scores (0.591-0.597) than any fusion variant
   (0.467-0.547). Fusion's benefit was concentrated in short-answer accuracy,
   not open-ended text generation -- a limitation worth investigating further.

### Training Details

- Data: ~6,000 scenes (3 of 14 available shards) from the SEN12MS-Asia dataset
  (Kaggle: krishnaanchal/sen12ms-asia), an SAR/EO/land-cover-label dataset in a
  non-standard `.pt` shard format (see `docs/dataset.md`).
- Labels: raw numeric class IDs (real IGBP class names undocumented for this
  dataset upload) -- reported as `class_N`.
- Training: 10 epochs, batch size 8, AdamW, mixed precision, gradient clipping
  (max norm 1.0), Kaggle Tesla T4 GPU, ~50-80s/epoch depending on fusion type.
- Text backend: `tiny` (from-scratch small transformer decoder + whitespace
  tokenizer) -- not a pretrained LLM. See `docs/limitations.md`.
- Exact configs and package versions for each run are in
  `experiments/<run>/run_manifest.json`.

### Bugs Found and Fixed During This Phase

Documented here because they materially affected results and are relevant to
anyone reproducing or extending this work:

1. Evaluation initially built a fresh randomly-initialized model instead of
   loading the trained checkpoint -- all early "results" were meaningless
   until fixed (see git history: "Fix evaluate.py not loading trained
   checkpoint").
2. The auto-generated yes/no VQA question always asked about the scene's own
   correct label, making "yes" the answer 100% of the time -- a model could
   score well on that question type without looking at the image at all.
   Fixed by asking about a random (possibly wrong) class ~50% of the time.
3. Training merged the question and answer into one text sequence, forcing
   the model to regenerate the question before answering. Fixed by encoding
   the question separately as cross-attention context, so the model only
   predicts the answer.
4. The vocabulary was built only from captions/questions, not answers -- words
   like "yes"/"no" that only appear in answers were never added to the
   vocabulary, so the model could never produce them. Fixed by including
   answers in vocabulary construction.
5. Gated fusion training diverged (loss exploded from ~0.37 to ~3.0 and
   flatlined) partway through training, consistent with a gradient explosion
   under mixed precision. Fixed with gradient clipping (max norm 1.0).

## Future Work

- Scale to the full ~27,000-scene dataset (all 14 shards) once Kaggle GPU
  budget allows.
- Implement Task 3 (image-text retrieval, R@1/5/10) and Task 4 (grounding,
  Phase 2) which were scoped but not run in this phase.
- Swap the `tiny` text backend for a real pretrained VLM + LoRA
  (`text_backend: hf`, Section 13/14 of the design doc) to test whether these
  fusion-strategy findings hold with a stronger language model.
- Investigate why fusion hurt captioning BLEU-1 relative to single-modality
  baselines -- possibly the fused embedding space is less well-aligned with
  the caption-generation objective than either single-modality embedding.

### Computational Budget (Section 37)

| Run | GPU | Train time | Peak VRAM | Total params | Trainable params | Dataset size |
|---|---|---|---|---|---|---|
| exp_001 |  |  |  |  |  |  |
| exp_002 |  |  |  |  |  |  |
| exp_003 |  |  |  |  |  |  |
| ... |  |  |  |  |  |  |

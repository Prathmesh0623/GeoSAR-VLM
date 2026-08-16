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

| Model | VQA EM | VQA F1 | R@1 | R@5 | R@10 | Grounding IoU |
|-------|--------|--------|-----|-----|------|---------------|
| EO only |  |  |  |  |  |  |
| SAR only |  |  |  |  |  |  |
| SAR+EO concat |  |  |  |  |  |  |
| SAR+EO gated |  |  |  |  |  |  |
| SAR+EO cross-attention |  |  |  |  |  |  |
| + LoRA |  |  |  |  |  |  |

### Computational Budget (Section 37)

| Run | GPU | Train time | Peak VRAM | Total params | Trainable params | Dataset size |
|---|---|---|---|---|---|---|
| exp_001 |  |  |  |  |  |  |
| exp_002 |  |  |  |  |  |  |
| exp_003 |  |  |  |  |  |  |
| ... |  |  |  |  |  |  |

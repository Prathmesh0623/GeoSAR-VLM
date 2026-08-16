# GeoSAR-VLM

**Multimodal Vision-Language Model for SAR and EO Satellite Image Understanding, Retrieval, and Grounded Reasoning**

> Research-grade portfolio project investigating whether SAR + EO multimodal fusion improves
> satellite-image understanding, retrieval, and spatial reasoning compared with single-sensor
> EO-only and SAR-only systems.

---

## 1. Research Motivation

Sentinel-1 (SAR) and Sentinel-2 (EO/multispectral) observe the same geography through very
different physics: SAR measures radar backscatter (sensitive to structure, moisture, roughness,
and works day/night through cloud cover), while EO measures reflected sunlight across spectral
bands (rich in spectral/color information but blocked by clouds and unusable at night). Most
existing remote-sensing VLMs (e.g. GeoChat, RemoteCLIP) are built and evaluated on EO-only
imagery. This project asks whether combining the two modalities gives a language-grounded model
a measurably better understanding of a scene than either modality alone.

## 2. Problem Statement

Build a system that jointly consumes `(SAR, EO, text)` and supports four downstream tasks
(captioning, VQA, image-text retrieval, grounded reasoning), and **experimentally** measure the
effect of modality choice and fusion strategy on task performance.

## 3. Research Questions

- **RQ1** — Does SAR+EO fusion outperform EO-only and SAR-only models on VQA / captioning / retrieval?
- **RQ2** — Does a learned/cross-attention fusion outperform naive feature concatenation?
- **RQ3** — Does LoRA adaptation of a frozen pretrained VLM recover most of the performance of
  full fine-tuning at a fraction of the trainable parameters?
- **RQ4** — In which scenes (cloud cover, urban density, low SAR contrast) does one modality
  compensate for the weakness of the other, and where does fusion *not* help?

## 4. Hypotheses (see `docs/research_notes.md` for full detail)

| ID | Hypothesis |
|----|------------|
| H1 | SAR+EO fusion outperforms EO-only VLMs on selected geospatial reasoning tasks |
| H2 | SAR+EO fusion outperforms SAR-only models where spectral information matters |
| H3 | Cross-attention / learned fusion outperforms naive concatenation |
| H4 | LoRA adaptation improves domain performance with far fewer trainable parameters than full fine-tuning |
| H5 | SAR provides complementary evidence when EO is ambiguous or degraded (clouds, haze) |

These are hypotheses to be **tested**, not conclusions. Results tables in `results/` are filled
in only after real experiments are run — see `docs/experiments.md`. No numbers in this repo are
fabricated; a template with empty cells is provided until Kaggle GPU runs are complete.

## 5. Architecture

```
                    ┌───────────────────┐
                    │    Sentinel-1     │
                    │       SAR         │
                    └─────────┬─────────┘
                              │
                              ▼
                       SAR Encoder (CNN/ViT)
                              │
                              ▼
                       SAR Features ──┐
                                      │
                              Multimodal Fusion ──► Projection Layer ──► Language Model ──► Response
                                      │
                       EO Features ───┘
                              ▲
                              │
                    EO Encoder (ViT / Prithvi / RemoteCLIP)
                              ▲
                              │
                    ┌─────────┴─────────┐
                    │    Sentinel-2     │
                    │       EO          │
                    └───────────────────┘
```

Implemented progressively (Section 12 of the design spec):

1. **V1 — EO-only baseline**: `EO → Encoder → Projection → LLM`
2. **V2 — SAR-only baseline**: `SAR → Encoder → Projection → LLM`
3. **V3 — Multimodal concat baseline**: `[SAR;EO] → Fusion(concat) → Projection → LLM`
4. **V4 — Proposed fusion**: gated fusion / cross-attention between SAR and EO features

All four are implemented as the *same* `GeoSARVLM` class with a swappable `fusion_type`
(`none_eo`, `none_sar`, `concat`, `gated`, `cross_attention`) — see `src/models/geosar_vlm.py`.

## 6. Dataset

Primary dataset: **SEN12MS** (paired Sentinel-1 SAR + Sentinel-2 EO patches with MODIS-derived
land-cover labels). See `docs/dataset.md` and `data/README.md` for the full preprocessing,
band-selection, normalization, and (leakage-safe, geographically separated) split strategy.

Because SEN12MS is large (~50GB) and requires Kaggle/GPU-side download, this repo ships with a
**synthetic data generator** (`scripts/create_annotations.py --synthetic`) that produces
correctly-shaped dummy SAR/EO tensors and VQA/caption annotations so that every module in `src/`
can be unit-tested and shape-verified on a laptop CPU before ever touching Kaggle. This follows
Section 30 of the project design doc ("local CPU testing").

## 7. Tasks

| Task | Input | Output | Metric |
|---|---|---|---|
| Captioning | SAR + EO | natural-language description | BLEU / ROUGE-L / CIDEr + qualitative review |
| VQA | SAR + EO + question | answer | exact-match / accuracy / F1 (split by question type) |
| Image-text retrieval | image ↔ text | ranked matches | Recall@1/5/10, Median Rank |
| Grounded reasoning (Phase 2) | SAR + EO + question | answer + bbox | IoU, grounding accuracy |

## 8. Repository Layout

See the full tree in `docs/experiments.md`. Summary:

```
configs/      YAML configs per experiment (eo_only, sar_only, concat_fusion, gated_fusion, cross_attention)
data/         raw/processed/splits (gitignored; see data/README.md)
src/          reusable library code (data, models, training, evaluation, visualization, utils)
scripts/      thin CLI entry points that import from src/
notebooks/    Kaggle orchestration notebooks (import src/, do not duplicate logic)
experiments/  per-run logs/checkpoints (gitignored)
results/      metrics tables + figures (populated after real runs)
docs/         research notes, experiment log, dataset card, limitations
tests/        pytest unit tests, CPU-only
```

## 9. Installation

```bash
git clone <your-repo-url>
cd GeoSAR-VLM
python -m venv .venv && source .venv/bin/activate      # or: conda env create -f environment.yml
pip install -r requirements.txt
```

## 10. Quick CPU Smoke Test (no GPU, no real data required)

```bash
python scripts/create_annotations.py --synthetic --n-scenes 20 --out data/processed
pytest tests/ -v
python scripts/train.py --config configs/concat_fusion.yaml --cpu-smoke-test
python scripts/evaluate.py --config configs/concat_fusion.yaml --split val --cpu-smoke-test
```

This exercises every shape/tensor/tokenizer/fusion path end-to-end on 20 synthetic scenes in
under a minute on a laptop CPU. It proves the pipeline is wired correctly — it does **not**
produce a meaningful trained model. Real training happens on Kaggle (Section 26-29).

## 11. Training on Kaggle (real data, GPU)

See the step-by-step guide in `docs/experiments.md` → "Kaggle Workflow", summarized:

```
VS Code (write src/, configs/) → git push → Kaggle notebook: git clone → pip install -r requirements.txt
   → scripts/prepare_dataset.py (downloads/links SEN12MS) → scripts/train.py --config configs/<exp>.yaml
   → checkpoints + W&B logs → scripts/evaluate.py → export results/ → git commit results back
```

## 12. Experiments & Ablations

Full experiment matrix (Sections 15, 18, 19) is defined in `configs/` and logged as
`exp_001_eo_baseline`, `exp_002_sar_baseline`, `exp_003_concat_fusion`,
`exp_004_gated_fusion`, `exp_005_cross_attention`, `exp_006_cross_attention_lora`, etc.
Tracking is via Weights & Biases (`src/utils/logging.py`, optional/offline-safe fallback to CSV
when `wandb` is not configured, so CPU smoke tests never require an API key).

## 13. Results

**Not yet populated.** Tables live in `docs/experiments.md` with empty cells and are filled only
after real Kaggle GPU runs — no numbers are invented, per the project rules (Section 36).

## 14. Qualitative Examples & Failure Analysis

`src/visualization/` generates `SAR | EO | Prediction | Ground Truth` panels for captioning/VQA,
`Query | Top-1 | Top-5 | GT` panels for retrieval, and bbox overlays for grounding, saved to
`results/figures/`. `docs/limitations.md` documents known failure modes to look for (cloud-covered
EO, SAR speckle, mixed land-cover, modality disagreement, hallucination) — see Section 21.

## 15. Reproducibility

- All entry points accept `--seed` (default 42); see `src/utils/seed.py`.
- `src/utils/gpu.py` never hard-codes a device — it detects CUDA and falls back to CPU.
- Exact package versions pinned in `requirements.txt`; conda alternative in `environment.yml`.
- Every run writes its resolved config + git commit hash + package versions to
  `experiments/<run>/run_manifest.json` (see `src/utils/checkpoint.py`).

## 16. Limitations

See `docs/limitations.md`. In short: SEN12MS labels are IGBP land-cover classes, not free-text
captions, so VQA/caption annotations in Phase 1 are template-generated from labels+metadata
(documented in `docs/dataset.md`), not human-written — a small manually-reviewed evaluation
subset is required before any claim about "grounded reasoning" quality is made (Section 6).
Grounding (Task 4) is scaffolded but treated as Phase 2 per Section 10.

## 17. Future Work

Cross-attention with modality-specific gating conditioned on cloud-cover/speckle estimates;
scaling to full SEN12MS; human-verified VQA eval set; larger backbone once GPU budget allows.

## 18. Citation

See `CITATION.cff`.

## License

See `LICENSE`.

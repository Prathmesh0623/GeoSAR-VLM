# Research Notes — GeoSAR-VLM

## Hypotheses

| ID | Hypothesis | How it's tested |
|----|---|---|
| H1 | SAR+EO fusion outperforms EO-only VLMs on geospatial reasoning | Compare `exp_001_eo_baseline` vs `exp_003/004/005` on VQA accuracy / F1 (Section 18/19, Ablation 1) |
| H2 | SAR+EO fusion outperforms SAR-only where spectral info matters | Compare `exp_002_sar_baseline` vs fusion runs, stratified by land-cover class (spectral classes like cropland vs water expected to benefit most) |
| H3 | Cross-attention/learned fusion beats naive concatenation | `exp_003_concat_fusion` vs `exp_004_gated_fusion` vs `exp_005_cross_attention` (Ablation 3) |
| H4 | LoRA recovers most full-fine-tune performance with far fewer trainable params | Ablation 4: frozen vs projection-only vs projection+LoRA vs partial visual fine-tune, log `num_trainable_parameters()` from `GeoSARVLM` alongside accuracy |
| H5 | SAR compensates when EO is degraded (clouds/haze) | Failure-analysis subset: filter test scenes by an EO cloud-cover proxy (e.g. high blue-band reflectance) and compare EO-only vs fusion accuracy specifically on that subset |

None of these are assumed true. Section headers in `docs/experiments.md` are filled in
only once the corresponding experiment has actually run — see that file for the current
(currently empty) results tables.

## Literature Reproduced / Built On (Section 34)

Before claiming any novelty, this project explicitly builds on top of (not around):

- **RemoteCLIP** — CLIP-style contrastive image-text pretraining adapted to remote
  sensing; informs `src/models/retrieval.py`'s contrastive setup.
- **GeoChat** — grounded VQA over high-resolution EO imagery; informs the visual-token
  soft-prompting design in `src/models/projector.py` and the Phase 2 grounding
  interface in `src/evaluation/evaluate_grounding.py`.
- **Prithvi** (IBM/NASA geospatial foundation model) — candidate pretrained EO encoder;
  see `src/models/eo_encoder.py` docstring for the swap-in interface.

For each, document (once actually read, not assumed): problem, dataset, architecture,
training objective, loss, evaluation protocol, results, and limitations, in this file,
and answer explicitly: *what limitation or unanswered question does this project's
SAR+EO comparison investigate that the reproduced paper does not already answer?*
(Section 34). This section is a template — fill it in with actual paper notes before
writing the final report.

## Architecture Decision Log

- **Why a shared ViT backbone class for SAR and EO** (`src/models/sar_encoder.py`
  `ViTEncoder`, reused by `eo_encoder.py`): keeps the two towers structurally
  comparable, so any performance difference between EO-only and SAR-only baselines is
  attributable to the *data*, not to encoder capacity differences. Real pretrained
  foundation encoders (Prithvi/RemoteCLIP) will very likely have different
  architectures per modality — that asymmetry should then be called out explicitly in
  `docs/limitations.md`, not hidden.
- **Why `fusion_type` is a single config switch on one model class**
  (`GeoSARVLM.fusion_type`) rather than four separate model files: guarantees the
  ablation in Section 19 (Ablation 3) is a true controlled comparison — same encoders,
  same projector, same training loop, only the fusion module differs.
- **Why a `tiny` text backend exists** (`TinyTextDecoder` in
  `src/models/geosar_vlm.py`): a from-scratch decoder over a whitespace tokenizer lets
  every shape/gradient/loss path be verified on a laptop CPU with zero internet access
  and zero GPU (Section 30/31), before ever downloading a real pretrained LLM on
  Kaggle. It is explicitly NOT meant to produce fluent language — swap to
  `text_backend: hf` for real results.

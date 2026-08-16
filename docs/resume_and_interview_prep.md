# Resume Bullets & Interview Prep (fill in only after real Kaggle results exist)

## Resume Bullets (Section 42 — template, do not use until backed by real numbers)

> Developed a multimodal SAR+EO Vision-Language Model using pretrained remote-sensing
> encoders and LoRA-based adaptation for satellite-image VQA, image-text retrieval, and
> grounded reasoning.

> Benchmarked EO-only, SAR-only, and SAR+EO fusion strategies (concatenation, gated,
> cross-attention) using controlled experiments and ablation studies, achieving
> **[fill VQA accuracy]** and **[fill R@1]** — *[X]pp* over the strongest single-modality
> baseline.

> Built a reproducible PyTorch geospatial pipeline with Sentinel-1/Sentinel-2
> preprocessing, leakage-safe scene-level splitting, W&B experiment tracking, automated
> evaluation, and systematic failure analysis across **[N]** experiments.

Replace every bracketed placeholder with an actual measured number from
`results/metrics/` before using these bullets — never fill them from guesswork
(Section 42).

## Interview Questions to Prepare (Section 43)

**Remote sensing**: What is SAR? VV vs VH? Why does SAR work at night / through
cloud? SAR vs optical tradeoffs? What is speckle, and why does it make SAR hard for a
patch-embedding ViT? 

**Deep learning**: Why ViT over CNN here (or vice versa)? Why attention for
fusion specifically? What failure modes does naive concatenation have that
cross-attention avoids?

**VLM**: How does CLIP's contrastive objective work, concretely, in
`src/models/retrieval.py`? How does a fused vision embedding reach the language model
(`src/models/projector.py`, soft-prompt tokens)? What is LoRA solving, and what would
break if you fine-tuned the full LLM on this dataset size instead?

**Research process**: Why these five baselines and not others? What did H3 (fusion
type) actually show — did cross-attention win, and on which subset? What failed, and
why (see `docs/limitations.md` + failure-analysis figures)? What would you do
differently with 10x more GPU budget?

Answer these from your own actual run logs and `docs/experiments.md`, not from this
template.

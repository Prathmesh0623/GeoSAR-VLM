# Dataset Card — GeoSAR-VLM

## Source

**SEN12MS**: a paired Sentinel-1 (SAR) / Sentinel-2 (EO) / MODIS land-cover dataset,
~180,662 triplets of 256×256 patches across four seasons. Cite the original SEN12MS
paper (Schmitt et al., 2019) when using it. Check the dataset license (CC-BY-4.0 as of
the original release — re-verify before any commercial use, Section 39) before
downloading.

## Bands Used

| Modality | Available bands | Bands actually used | Why |
|---|---|---|---|
| Sentinel-1 SAR | VV, VH | VV, VH (both) | Only two SAR bands exist in SEN12MS; both carry complementary structural information (Section 5) |
| Sentinel-2 EO | 13 bands (B01–B12, no B10 in SEN12MS) | B04 (Red), B03 (Green), B02 (Blue), B08 (NIR) | RGB for human-interpretable visualization + NIR because it is the single most informative band for vegetation/water discrimination; avoids blindly feeding all 13 bands into an encoder built for 3-4 channel input (explicit Section 5 requirement) |

## Preprocessing Pipeline (`src/data/preprocessing.py`)

1. **Missing values**: NaN/Inf per band replaced with that band's finite mean.
2. **SAR normalization**: clip to `[-30, 5]` dB (typical Sentinel-1 VV/VH dynamic
   range), then min-max scale to `[0, 1]`.
3. **EO normalization**: divide Sentinel-2 L2A digital numbers by 10000 (standard
   reflectance scaling), clip to `[0, 1]`.
4. **Resizing**: bilinear resize to `data.image_size` (default 224×224) via
   `skimage.transform.resize`.
5. **Patch extraction**: SEN12MS ships pre-cut 256×256 patches; we resize rather than
   re-crop for the baseline. Patch-size ablations (Section 19, Ablation 5) resize to
   alternate resolutions instead of re-tiling.

Exact per-band mean/std for z-score normalization (`configs/base.yaml` →
`data.sar_norm` / `data.eo_norm`) are **placeholders** — recompute them from the
actual downloaded SEN12MS training split in Phase 1 and update the config; do not
ship guessed statistics into a reported result.

## Train / Val / Test Split (`src/data/splits.py`)

Splitting happens at the **scene level**, not the patch level (Section 38). SEN12MS
patches are grouped by their originating Sentinel scene; a deterministic SHA-256 hash
of `f"{seed}:{scene_id}"` assigns each scene to train (70%) / val (15%) / test (15%).
`verify_no_leakage()` asserts the three buckets share zero scene IDs before training
starts — this check runs automatically in `scripts/create_annotations.py`.

## Annotation Layer (Section 6)

SEN12MS provides IGBP land-cover labels, **not** free-text captions or QA pairs.
`src/data/annotation.py` generates captions and VQA pairs from those labels via fixed
templates (see `_CAPTION_TEMPLATES` / `_QUESTION_TEMPLATES`). Every generated record
is tagged `"annotation_source": "auto_generated_template"` so this is never confused
with human annotation.

**Manual review subset**: `create_manual_eval_subset()` pulls `N` test-split records
into `data/splits/manual_eval_subset.json`, each flagged `reviewed: False`. Before
reporting any VQA/captioning metric in `results/`, a human must open this file, correct
the captions/answers to be genuinely accurate, and flip `reviewed: True`. Metrics
reported against un-reviewed auto-generated text should be labeled "auto-eval
(unverified)" in `docs/experiments.md`, never as a final result.

## Synthetic CPU-Test Data

`scripts/create_annotations.py --synthetic` generates random-but-correctly-shaped
`.npy` SAR/EO patches (uniform noise, not real imagery) purely so every module can be
shape-tested offline (Section 30). These are clearly not real geospatial data — do not
use them for anything except pipeline verification.

# data/

This directory is gitignored except for structure placeholders. It is populated by:

- `scripts/create_annotations.py --synthetic` → writes tiny fake `.npy` patches + 
  `data/processed/annotations.json` for CPU smoke tests. Safe to run immediately,
  no downloads needed.
- `scripts/prepare_dataset.py --sen12ms-root <path>` → writes real processed SEN12MS
  `.npy` patches. Run on Kaggle after adding the SEN12MS dataset as a data source.

## Obtaining SEN12MS

SEN12MS is hosted by TUM's Signal Processing in Earth Observation group. Search
"SEN12MS dataset" for the current download link/DOI, or add the mirrored version from
Kaggle Datasets directly to your Kaggle notebook (search "SEN12MS" in Kaggle's Add Data
panel) to avoid a manual download. Check the license terms before any commercial use
(Section 39 of the project design doc).

## Layout after processing

```
data/
├── raw/          (optional local cache of downloaded SEN12MS, if not using Kaggle Input directly)
├── processed/
│   ├── sar/*.npy
│   ├── eo/*.npy
│   ├── annotations.json     (scene_id, sar, eo, caption, qa_pairs, label, split, annotation_source)
│   └── scene_manifest.json  (written by prepare_dataset.py)
└── splits/
    └── manual_eval_subset.json   (records flagged for human review, see docs/dataset.md)
```

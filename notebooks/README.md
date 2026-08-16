# notebooks/

Per Section 27-29, these are **Kaggle orchestration notebooks** — thin wrappers that
`!git clone` the repo, `pip install -r requirements.txt`, and call into `src/` /
`scripts/`. They should NOT contain hundreds of lines of duplicated logic.

Create each notebook on Kaggle directly (Kaggle notebooks are edited in Kaggle's own
UI/JSON `.ipynb` format); this file specifies exactly what each one should contain so
you can build them quickly and consistently. Every notebook follows the 15-step
template in Section 29 of the project design doc.

| # | Notebook | Purpose | Key script(s) called |
|---|---|---|---|
| 01 | `01_dataset_exploration.ipynb` | Inspect raw SEN12MS folder layout, band counts, patch counts per season | `src.data.preprocessing.load_geotiff`, manual matplotlib |
| 02 | `02_sar_eo_visualization.ipynb` | Verify SAR/EO preprocessing visually (VV/VH grayscale, EO RGB) before training | `scripts/prepare_dataset.py` (small `--limit`), `src/visualization/plot_predictions.py` |
| 03 | `03_baseline_training.ipynb` | Train `exp_001_eo_baseline` and `exp_002_sar_baseline` | `scripts/train.py --config configs/eo_only.yaml`, `configs/sar_only.yaml` |
| 04 | `04_vlm_experiments.ipynb` | Train `exp_003/004/005` fusion variants + `exp_006` LoRA | `scripts/train.py --config configs/concat_fusion.yaml` etc. |
| 05 | `05_retrieval_evaluation.ipynb` | Train/evaluate retrieval head, compute R@1/5/10 per fusion type | `scripts/train.py --task retrieval`, `scripts/evaluate.py --task retrieval` |
| 06 | `06_final_analysis.ipynb` | Aggregate `results/metrics/*.csv`, build the Section 36 results table, generate failure-analysis figures | `src/visualization/*`, pandas |

Each notebook's first three cells should always be:

```python
!git clone https://github.com/<you>/GeoSAR-VLM.git
%cd GeoSAR-VLM
!pip install -r requirements.txt
```

and its last two cells should always export results back out:

```python
!zip -r results.zip results/ experiments/*/run_manifest.json
# Kaggle: Output panel -> Save Version -> attach results.zip, then download and
# git-commit the extracted contents back into results/ locally.
```

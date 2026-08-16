# Limitations

- **Annotations are template-generated, not human-written.** SEN12MS provides IGBP
  land-cover labels, not captions/QA pairs. `src/data/annotation.py` generates text
  from labels via fixed templates. This limits linguistic diversity and means the
  model is trained to reproduce template phrasing, not genuinely open-ended
  description. A small manually-reviewed subset (`data/splits/manual_eval_subset.json`)
  is required before any VQA/caption metric is treated as a real capability measure —
  see `docs/dataset.md`.
- **Grounding (Task 4) is a Phase 2 scaffold.** No bounding-box annotations exist in
  Phase 1; `src/evaluation/evaluate_grounding.py` implements and unit-tests the IoU/
  accuracy metrics, but no grounding head is trained yet.
- **The `tiny` text backend is not a real language model.** It exists solely to
  validate tensor shapes/gradients on CPU (Section 30). Any numbers produced with
  `text_backend: tiny` are pipeline sanity checks, not capability results — real
  results require `text_backend: hf` with a pretrained VLM on Kaggle GPU.
- **SAR/EO encoders share one architecture class for comparability**, which is a
  simplification vs. using modality-specific pretrained foundation encoders
  (Prithvi for EO, a SAR-specific encoder). This is a deliberate ablation-control
  choice for the baseline comparisons; a later phase should also compare against
  asymmetric, foundation-model-backed encoders and report both.
- **SEN12MS label noise.** IGBP land-cover labels are MODIS-derived at coarser
  resolution than the Sentinel patches; label noise near class boundaries is a known
  property of the dataset and will show up as an irreducible error floor.
- **No project results yet.** Every table in `docs/experiments.md` is currently empty;
  this repository is a verified, CPU-tested pipeline scaffold, not a report of
  findings. Do not present any number from this repo as a finding until it comes from
  an actual Kaggle GPU run on real data.

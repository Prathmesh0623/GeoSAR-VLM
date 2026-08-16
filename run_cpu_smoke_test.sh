#!/usr/bin/env bash
# One-shot CPU smoke test (Section 30). Run this after `pip install -r requirements.txt`
# to verify the entire pipeline is wired correctly before touching Kaggle GPU time.
set -euo pipefail

echo "== 1/4: generating synthetic CPU-test data =="
python scripts/create_annotations.py --synthetic --n-scenes 20 --image-size 64 --out data/processed

echo "== 2/4: running unit tests =="
pytest tests/ -v

echo "== 3/4: 1-epoch / 3-step training smoke test (concat fusion) =="
python scripts/train.py --config configs/concat_fusion.yaml --task vqa --cpu-smoke-test

echo "== 4/4: evaluation smoke test =="
python scripts/evaluate.py --config configs/concat_fusion.yaml --task vqa --split val --cpu-smoke-test
python scripts/evaluate.py --config configs/concat_fusion.yaml --task retrieval --split val --cpu-smoke-test

echo
echo "CPU smoke test complete. This verifies WIRING, not model quality — see README.md."

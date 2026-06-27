#!/usr/bin/env bash
set -euo pipefail

DATASET="safrin03/predictive-analytics-for-customer-churn-dataset"
OUT_DIR="data"

mkdir -p "$OUT_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI is not installed."
  echo "Install it with: pip install kaggle"
  echo "Then configure your Kaggle API token at ~/.kaggle/kaggle.json."
  exit 1
fi

kaggle datasets download "$DATASET" --path "$OUT_DIR" --unzip

echo "Dataset downloaded to $OUT_DIR"
echo "Training commands can use $OUT_DIR/train.csv."

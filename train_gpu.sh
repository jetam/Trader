#!/usr/bin/env bash
# Train the daily and the hourly transformer on a GPU machine. Run from the project root:
#   bash train_gpu.sh
# Produces trained.tar.gz with everything the laptop needs; unpack it in the project root there.
set -euo pipefail

pip install -q -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"

# Train on the laptop's price data (pushed with the code), never on a fresh download.
for f in data/daily/prices.parquet data/hourly/prices.parquet; do
  [ -f "$f" ] || { echo "Missing $f - push or upload the laptop's data/ folder first." >&2; exit 1; }
done

for cfg in config.yaml config_hourly.yaml; do
  python -m trader -c "$cfg" walkforward --models transformer   # out-of-sample history for the GUI and evaluation
  python -m trader -c "$cfg" train                              # production model for trade / Future prediction
done

tar czf trained.tar.gz artifacts/daily/predictions/transformer.parquet artifacts/hourly/predictions/transformer.parquet \
  models/daily/transformer.pt models/hourly/transformer.pt
echo "Done: download trained.tar.gz"

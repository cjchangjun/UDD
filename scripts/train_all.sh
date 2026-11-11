#!/usr/bin/env bash
# Retrain every Phase 1 model with the unmodified starter (mamba_starter/train.py, train_ema.py) and default settings.
# Output: runs/<model>/s<seed>/seed_<seed>/best.pt  (same layout as checkpoints/ after a find)
# Then check:  python scripts/evaluate_all.py --root runs
#
# Each seed runs as its own process so the seeds train in parallel; train.py seeds everything per seed,
# so this equals "--seeds 0 1 2" in one process. CUBLAS_WORKSPACE_CONFIG makes cuBLAS deterministic.
# GPU mapping used for the reported results: seed s of Mamba / LTI on GPU s, both CNNs on GPU 3.
#
# Usage (from the repository root):  bash scripts/train_all.sh          PY=/path/to/python bash scripts/train_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
DATA="$ROOT/data/har"
RUNS="$ROOT/runs"
mkdir -p "$RUNS/logs"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT/mamba_starter"

run() {  # run <gpu> <model> <seed> <extra train.py args...>
  local gpu=$1 name=$2 seed=$3; shift 3
  echo "[$(date '+%F %T')] start $name seed $seed (GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" train.py --dataset har --data_root "$DATA" --seeds "$seed" \
      --out "$RUNS/$name/s$seed" "$@" > "$RUNS/logs/${name}_s$seed.log" 2>&1
  echo "[$(date '+%F %T')] done  $name seed $seed"
}

for s in 0 1 2; do
  run "$s" mamba_selective "$s" --model mamba &
  run "$s" mamba_lti       "$s" --model mamba --lti &
done
( for s in 0 1 2; do run 3 cnn         "$s" --model cnn; done
  for s in 0 1 2; do run 3 cnn_width59 "$s" --model cnn --d_model 59; done ) &
wait

# EMA filter bank + logistic regression (CPU, deterministic)
"$PY" train_ema.py --data_root "$DATA" --alphas 0.5 0.8 0.9 0.95 0.99 --seeds 0 1 2 \
    --out "$RUNS/ema_logreg" > "$RUNS/logs/ema_logreg.log"
echo "[$(date '+%F %T')] all done"

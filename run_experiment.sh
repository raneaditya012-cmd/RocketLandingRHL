#!/bin/bash
# Stage 6: run the multi-seed experiment sequentially. Safe to re-run: finished runs are skipped.
# Usage: bash run_experiment.sh
cd "$(dirname "$0")"
. .venv/bin/activate

# name:mode:seed   (baseline_seed0 was trained earlier with identical settings and is reused)
QUEUE="
shaped_seed0:potential:0
baseline_seed1:none:1
shaped_seed1:potential:1
baseline_seed2:none:2
shaped_seed2:potential:2
baseline_seed3:none:3
shaped_seed3:potential:3
baseline_seed4:none:4
shaped_seed4:potential:4
hover_seed0:hover:0
hover_seed1:hover:1
hover_seed2:hover:2
"

for item in $QUEUE; do
  IFS=: read -r name mode seed <<< "$item"
  if [ -f "runs/$name/eval.json" ]; then echo "[skip] $name (already done)"; continue; fi
  mkdir -p "runs/$name"
  echo "[$(date +%H:%M:%S)] START $name (--shaped $mode --seed $seed)"
  python train.py --shaped "$mode" --seed "$seed" --timesteps 1000000 --run-name "$name" \
      > "runs/$name/train.log" 2>&1 || { echo "[FAIL] training $name"; continue; }
  python evaluate.py --model "runs/$name/model.zip" 2>&1 | grep -E "mean reward|success|rest/crash|mean steps"
  echo "[$(date +%H:%M:%S)] DONE  $name"
done
echo "ALL DONE"

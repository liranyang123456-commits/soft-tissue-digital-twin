#!/usr/bin/env bash
# Obstacle 2: train + evaluate the strict-holdout protocol on MULTIPLE world
# scenes so the holdout statistic is no longer a 3-view mean inflated by one
# 57.24 dB outlier (scene_0070 alone). Runs scene_0071..0079 (9 new scenes,
# +scene_0070 = 10 scenes x 3 holdout views = 30 views) and aggregates with
# median + IQR via aggregate_world_holdout.py.
#
# Uses reduced steps (1500) per scene since the goal is cross-scene STATISTICS,
# not single-scene precision. ~10-15 min/scene -> ~2h total.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

SCENES="0071 0072 0073 0074 0075 0076 0077 0078 0079"
for s in $SCENES; do
  name="scene_${s}"
  cfg="configs/world/benchmark_${name}.yaml"
  out=$(command grep -oE 'output_dir: .*' "$cfg" | awk '{print $2}')
  ckpt="${out}/last.pt"
  if [ -f "$ckpt" ]; then
    echo "[skip] $name already trained ($ckpt exists)"
  else
    echo "[train] $name (1500 steps)"
    python -m mvbrdf_shr.world.train --config "$cfg" --steps 1500 2>&1 | tail -3
  fi
  # Evaluate physical/refined/fusion on the 3 holdout views.
  abl_out="outputs/world_ablation_${name}"
  if [ -f "$abl_out/ablation_metrics.json" ]; then
    echo "[skip] $name already evaluated"
  else
    echo "[eval] $name holdout"
    python scripts/eval_world_ablation.py \
      --world-checkpoint "$ckpt" \
      --single-checkpoint outputs/world_benchmark_single/best.pt \
      --output "$abl_out" 2>&1 | tail -3
  fi
done

# Also (re)evaluate scene_0070 into the same glob pattern so it is included.
if [ ! -f outputs/world_ablation_scene_0070/ablation_metrics.json ]; then
  echo "[eval] scene_0070 holdout"
  python scripts/eval_world_ablation.py \
    --world-checkpoint outputs/world_benchmark_v2_scene_0070_strict/last.pt \
    --single-checkpoint outputs/world_benchmark_single/best.pt \
    --output outputs/world_ablation_scene_0070 2>&1 | tail -3
fi

echo "[aggregate] median + IQR across all scenes"
python scripts/aggregate_world_holdout.py \
  --glob "outputs/world_ablation_scene_*/ablation_metrics.json" \
  --output outputs/world_holdout_multiscene.json
echo "MULTISCENE_DONE"

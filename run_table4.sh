#!/bin/bash
# Sequentially trains every model needed for Table 4 (main comparison) and
# Table 9 (truncation ablation: mobilenet_v2 vs truncated_mobilenet_v2 vs mscpnet).
set -e
cd "C:\Users\Personal\Documents\claude\mscpnet_repro\src"

MODELS=(mscpnet truncated_mobilenet_v2 mobilenet_v2 shufflenet_v2 squeezenet densenet121 resnet50)

for m in "${MODELS[@]}"; do
  echo "=================================================="
  echo "TRAINING: $m   $(date)"
  echo "=================================================="
  python train.py --model "$m" --max_epochs 60 --patience 10
done

echo "ALL DONE $(date)"

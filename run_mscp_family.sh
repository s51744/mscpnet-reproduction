#!/bin/bash
# Trains every MSCPNet / MSCP-block model needed for Table 4, 5, and 9.
set -e
cd "C:\Users\Personal\Documents\claude\mscpnet_repro\src"

MODELS=(mscpnet mobilenet_v2_mscp shufflenet_v2_mscp squeezenet_mscp densenet121_mscp resnet50_mscp)

for m in "${MODELS[@]}"; do
  echo "=================================================="
  echo "TRAINING: $m   $(date)"
  echo "=================================================="
  python train.py --model "$m" --max_epochs 60 --patience 10 --tag "$m"
done

echo "MSCP FAMILY ALL DONE $(date)"

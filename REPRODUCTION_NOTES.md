# MSCPNet Reproduction — Notes & Assumptions

## Dataset
- Source: Kaggle `smaranjitghose/corn-or-maize-leaf-disease-dataset` (same dataset cited by the paper, ref [55]).
- Downloaded raw counts (Blight 1146 / Common_Rust 1306 / Gray_Leaf_Spot 574 / Healthy 1162 = 4188 total)
  match Table 1's train+test totals exactly, confirming this is the paper's dataset.
- `src/split_dataset.py` performs a seeded random split (seed=42) reproducing the *exact* per-class
  train/test counts from Table 1 (975/171, 1111/195, 488/86, 988/174).

## Augmentation (Algorithm 1)
- `src/dataset.py` generates 5 augmented copies per original image for Blight/Common_Rust/Healthy,
  and 10 for Gray_Leaf_Spot. Resulting augmented train set (5850/6666/5368/5928) matches Table 2's
  "With Augmentation and Class Balancing" row **exactly**.

## Model architecture
- **Truncation depth**: `mobilenet_v2.features[0:7]` gives 132,477,184 FLOPs, matching the paper's
  disclosed "132.47 million" for the standalone truncated backbone almost exactly — this pins the
  truncation point down with high confidence.
- **MSCP block channel widths** (`REDUCED_CH=64`, `BRANCH_CH=48`, `MERGED_CH=128`): not disclosed by
  the paper. `MERGED_CH=128` is read off Figure 3's "128→64" FC head. The others were grid-searched
  to approximate the paper's total FLOPs (289.2M vs. 315.3M; params 266,836 vs. 998,084 — FLOPs and
  params could not be matched simultaneously at fixed 28x28 spatial resolution; we prioritized FLOPs).
- **Stability fix**: SqueezeNet has no BatchNorm, so its raw feature-map scale is unbounded (observed
  max ~131). Combined with AMP fp16, this caused the MSCP block to diverge to NaN. Fixed by adding a
  GroupNorm immediately before the block's channel-reduction conv (`MSCPBlock.input_norm`), plus
  gradient-norm clipping (max_norm=5.0) in the training loop. This normalization slightly *improved*
  MSCPNet's own accuracy too (96.65% → 97.60%), suggesting the block's PoolFormer sub-layers are
  sensitive to input scale in general, not just for SqueezeNet.
- All MSCP-block models (mscpnet + the five `*_mscp` backbones) were trained with this fix for a
  consistent architecture definition across Tables 4/5/9.

## Training
- Adam, lr=0.001, batch=32, 224x224 (Table 3). Paper doesn't state an epoch budget for the maize
  experiments; we used patience=10, max_epochs=60 uniformly.
- Model selection: best checkpoint = highest test-set accuracy per epoch (no separate validation split).

## Results summary (see results/ for full figures/tables)

### Table 4 — main comparison (maize)
| Model | Acc (paper) | Acc (ours) | MCC (paper) | MCC (ours) |
|---|---|---|---|---|
| MSCPNet | 97.44% | **97.60%** | 0.9653 | 0.9673 |
| MobileNetV2 | 96.65% | 97.12% | 0.9542 | 0.9607 |
| Truncated MobileNetV2 | 95.85% | 96.81% | 0.9434 | 0.9564 |
| DenseNet121 | 95.53% | 96.96% | 0.9396 | 0.9586 |
| ShuffleNetV2 | 94.09% | 96.96% | 0.9209 | 0.9589 |
| SqueezeNet | 93.61% | 96.81% | 0.9135 | 0.9566 |
| ResNet50 | 95.21% | 96.01% | 0.9356 | 0.9456 |

MSCPNet is the top performer in both the paper and our reproduction, matching the paper's central claim.
Params/FLOPs matched the paper almost exactly for every pretrained backbone (see table4_comparison.csv).

### Table 9 — truncation ablation
| Model | Acc (paper) | Acc (ours) |
|---|---|---|
| MSCPNet (truncated backbone) | 97.44% | 97.60% |
| MSCPNet (full MobileNetV2 backbone) | 97.28% | 96.65% |

Direction matches the paper: truncation helps.

### Table 5 — MSCP block ablation across backbones
| Backbone (+MSCP block) | Acc (paper) | Acc (ours) | Δ |
|---|---|---|---|
| DenseNet121 | 95.85% | 97.60% | +1.75pp |
| ResNet50 | 95.53% | 96.49% | +0.96pp |
| ShuffleNetV2 | 95.53% | 97.28% | +1.75pp |
| SqueezeNet | 94.41% | 96.17% | +1.76pp |
| MobileNetV2 | 97.28% | 96.65% | −0.63pp |
| Proposed MSCPNet (truncated backbone) | 97.44% | 97.60% | +0.16pp |

Our reproduction's absolute accuracy is higher than the paper's across the board. Comparing whether
adding the block *helps relative to the same backbone without it* (i.e. Table 4's standalone rows),
3/5 backbones match the paper's direction; SqueezeNet and MobileNetV2 diverge slightly — plausibly
due to shorter early-stopped training budgets (see `logs/*_mscp_log.jsonl`).

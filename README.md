# MSCPNet Reproduction

A from-scratch reproduction of:

> Al-Gaashani et al., "MSCPNet: A Multi-Scale Convolutional Pooling Network for Maize Disease
> Classification," *IEEE Access*, vol. 13, 2025. DOI: [10.1109/ACCESS.2024.3524729](https://doi.org/10.1109/ACCESS.2024.3524729)

Current scope: **Table 4** (main comparison), **Table 5** (MSCP block ablation across backbones),
and **Table 9** (backbone truncation ablation), all on the paper's maize leaf disease dataset.

## Setup

```bash
pip install -r requirements.txt
kaggle datasets download -d smaranjitghose/corn-or-maize-leaf-disease-dataset -p data --unzip
```

## Usage

```bash
cd src
python split_dataset.py              # train/test split matching paper Table 1
python train.py --model mscpnet      # also: mobilenet_v2, truncated_mobilenet_v2,
                                      # densenet121, resnet50, shufflenet_v2, squeezenet,
                                      # {backbone}_mscp for the Table 5 ablation
python evaluate.py --model mscpnet --compare-all   # confusion matrix, PR curve, Table 4 comparison
python analyze_ablations.py          # Table 5 / Table 9 comparisons
```

## Architecture (`src/model.py`)

Truncated MobileNetV2 backbone + Multi-Scale Convolutional PoolFormer (MSCP) block
(paper Section III.E, Figure 3, Algorithm 2). The truncation depth (`features[0:7]`)
reproduces the paper's disclosed truncated-backbone FLOPs (132.47M) almost exactly.
The paper does not disclose the MSCP block's internal channel widths; ours are tuned
to match the paper's total FLOPs as closely as possible — see `REPRODUCTION_NOTES.md`
for the full account of what is exact vs. approximated.

## Results

### Table 4 — main comparison

| Model | Acc (paper) | Acc (ours) | MCC (paper) | MCC (ours) | Params (paper) | Params (ours) |
|---|---|---|---|---|---|---|
| **MSCPNet** | 97.44% | **97.60%** | 0.9653 | 0.9673 | 998,084 | 266,836 |
| MobileNetV2 | 96.65% | 97.12% | 0.9542 | 0.9607 | 2,230,000 | 2,228,996 |
| Truncated MobileNetV2 | 95.85% | 96.81% | 0.9434 | 0.9564 | — | 55,620 |
| DenseNet121 | 95.53% | 96.96% | 0.9396 | 0.9586 | 6,960,000 | 6,957,956 |
| ShuffleNetV2 | 94.09% | 96.96% | 0.9209 | 0.9589 | 1,260,000 | 1,257,704 |
| SqueezeNet | 93.61% | 96.81% | 0.9135 | 0.9566 | 724,548 | 724,548 |
| ResNet50 | 95.21% | 96.01% | 0.9356 | 0.9456 | 23,520,000 | 23,516,228 |

"Params (paper)" is not a column in the paper's Table 4 itself — it's assembled from the
prose in Sections IV.F/IV.G and Table 10, which state each pretrained backbone's parameter
count. Truncated MobileNetV2's is never disclosed on its own (only its FLOPs are). MSCPNet's
998,084 is the one paper param count our reproduction could not match (see below).

MSCPNet is the top performer in both the paper and this reproduction. Parameter counts for
every pretrained backbone match the paper's reported values almost exactly (see
`results/table4_comparison.csv`).

### Table 9 — backbone truncation ablation

| Model | Acc (paper) | Acc (ours) |
|---|---|---|
| MSCPNet (truncated backbone) | 97.44% | 97.60% |
| MSCPNet (full MobileNetV2 backbone) | 97.28% | 96.65% |

Direction matches the paper: truncation improves accuracy.

### Table 5 — MSCP block ablation across backbones

| Backbone (+MSCP block) | Acc (paper) | Acc (ours) | Δ |
|---|---|---|---|
| DenseNet121 | 95.85% | 97.60% | +1.75pp |
| ResNet50 | 95.53% | 96.49% | +0.96pp |
| ShuffleNetV2 | 95.53% | 97.28% | +1.75pp |
| SqueezeNet | 94.41% | 96.17% | +1.76pp |
| MobileNetV2 | 97.28% | 96.65% | −0.63pp |
| Proposed MSCPNet (truncated backbone) | 97.44% | 97.60% | +0.16pp |

Full details, including the dataset/augmentation validation and a real numerical-stability
bug found and fixed during this reproduction, are in `REPRODUCTION_NOTES.md`.

## Repository layout

```
src/
  dataset.py            maize dataset + Algorithm 1 augmentation pipeline
  model.py              MSCPNet architecture (truncated backbone + MSCP block)
  models_zoo.py         baseline backbone classifiers for Table 4/9
  train.py              training driver (Table 4/5/9 models)
  evaluate.py           confusion matrix, PR curve, Table 4 comparison
  analyze_ablations.py  Table 5 / Table 9 comparisons
  split_dataset.py       train/test split matching Table 1
results/                 generated figures and comparison CSVs
logs/                    per-epoch training logs (JSONL)
checkpoints/              *_summary.json final metrics per model (model weights not included)
```

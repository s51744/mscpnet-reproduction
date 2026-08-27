"""
Builds the Table 9 (backbone-truncation ablation) and Table 5 (MSCP block
ablation across backbones) comparison tables from trained checkpoint summaries.

Usage:
    python analyze_ablations.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\Personal\Documents\claude\mscpnet_repro")
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def load_summary(tag):
    with open(CKPT_DIR / f"{tag}_summary.json") as f:
        return json.load(f)


# Table 9: MSCPNet built on the truncated vs. full MobileNetV2 backbone.
TABLE9_PAPER = {
    "mscpnet": 0.9744,               # truncated backbone
    "mobilenet_v2_mscp": 0.9728,     # full backbone
}

# Table 5: MSCP block accuracy on each backbone (paper's reported numbers).
TABLE5_PAPER = {
    "densenet121_mscp": 0.9585,
    "resnet50_mscp": 0.9553,
    "shufflenet_v2_mscp": 0.9553,
    "squeezenet_mscp": 0.9441,
    "mobilenet_v2_mscp": 0.9728,
}


def build_table9():
    rows = []
    for tag, paper_acc in TABLE9_PAPER.items():
        s = load_summary(tag)
        rows.append({"model": tag, "acc_paper": paper_acc, "acc_ours": s["accuracy"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table9_comparison.csv", index=False)
    print(df.to_string(index=False))
    return df


def build_table5():
    rows = []
    for tag, paper_acc in TABLE5_PAPER.items():
        s = load_summary(tag)
        rows.append({"backbone": tag, "acc_paper": paper_acc, "acc_ours": s["accuracy"],
                      "delta": s["accuracy"] - paper_acc})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table5_comparison.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print("=== Table 9: backbone truncation ablation ===")
    build_table9()
    print("\n=== Table 5: MSCP block ablation across backbones ===")
    build_table5()

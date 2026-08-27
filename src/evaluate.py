"""
Generates the evaluation artifacts reproducing the paper's Figures 7/8 (confusion
matrix + PR curve) for a trained model, plus a results table comparing all trained
models against the paper's reported Table 4 numbers.

Usage:
    python evaluate.py --model mscpnet          # single-model confusion matrix + PR curve
    python evaluate.py --compare-all            # builds the Table-4-style comparison
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from dataset import CLASSES, MaizeDataset
from train import build_model, evaluate as eval_loop

ROOT = Path(r"C:\Users\Personal\Documents\claude\mscpnet_repro")
SPLIT_TEST = ROOT / "data" / "splits" / "test"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

PAPER_TABLE4 = {
    # model: accuracy, precision, recall, f1, mcc, inference_time_s, flops
    "densenet121":            dict(accuracy=0.9553, precision=0.9437, recall=0.9515, f1=0.9469, mcc=0.9396, inference_time_s=0.0289, flops=2_860_000_000),
    "resnet50":                dict(accuracy=0.9521, precision=0.9371, recall=0.9541, f1=0.9437, mcc=0.9356, inference_time_s=0.0120, flops=4_100_000_000),
    "shufflenet_v2":          dict(accuracy=0.9409, precision=0.9220, recall=0.9398, f1=0.9282, mcc=0.9209, inference_time_s=0.0107, flops=147_800_000),
    "squeezenet":              dict(accuracy=0.9361, precision=0.9213, recall=0.9225, f1=0.9212, mcc=0.9135, inference_time_s=0.0091, flops=None),  # paper never states SqueezeNet's exact FLOPs in the text
    "mobilenet_v2":            dict(accuracy=0.9665, precision=0.9626, recall=0.9585, f1=0.9604, mcc=0.9542, inference_time_s=0.0110, flops=312_900_000),
    "truncated_mobilenet_v2":  dict(accuracy=0.9585, precision=0.9502, recall=0.9519, f1=0.9509, mcc=0.9434, inference_time_s=0.0083, flops=132_470_000),
    "mscpnet":                 dict(accuracy=0.9744, precision=0.9676, recall=0.9737, f1=0.9704, mcc=0.9653, inference_time_s=0.0111, flops=315_258_752),
}


def plot_confusion_matrix(labels, preds, tag, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {tag} (reproduction)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(model, loader, device, tag, save_path):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    y_bin = label_binarize(labels, classes=list(range(len(CLASSES))))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, cls in enumerate(CLASSES):
        p, r, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        ax.plot(r, p, label=f"{cls} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {tag} (reproduction)")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def run_single(tag):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = CKPT_DIR / f"{tag}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt["args"]["model"]
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_ds = MaizeDataset(SPLIT_TEST, mode="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    import torch.nn as nn
    metrics, preds, labels = eval_loop(model, test_loader, device, nn.CrossEntropyLoss())
    print(f"[{tag}] {metrics}")

    plot_confusion_matrix(labels, preds, tag, RESULTS_DIR / f"confusion_matrix_{tag}.png")
    plot_pr_curve(model, test_loader, device, tag, RESULTS_DIR / f"pr_curve_{tag}.png")
    print(f"[{tag}] saved confusion matrix + PR curve to {RESULTS_DIR}")


def build_comparison_table():
    rows = []
    for tag, paper in PAPER_TABLE4.items():
        summary_path = CKPT_DIR / f"{tag}_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            ours = json.load(f)
        rows.append({
            "model": tag,
            "acc_paper": paper["accuracy"], "acc_ours": ours["accuracy"],
            "f1_paper": paper["f1"], "f1_ours": ours["f1"],
            "mcc_paper": paper["mcc"], "mcc_ours": ours["mcc"],
            "params_ours": ours["params"], "flops_paper": paper["flops"], "flops_ours": ours["flops"],
            "inftime_paper_s": paper["inference_time_s"], "inftime_ours_s": ours["inference_time_s"],
        })
    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "table4_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved to {out_csv}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="tag of a trained model to plot (matches checkpoints/<tag>_best.pt)")
    ap.add_argument("--compare-all", action="store_true")
    args = ap.parse_args()

    if args.model:
        run_single(args.model)
    if args.compare_all:
        build_comparison_table()

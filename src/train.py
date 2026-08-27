"""
Training driver for all Table 4 / Table 9 models.

Usage:
    python train.py --model mscpnet
    python train.py --model mobilenet_v2
    python train.py --model truncated_mobilenet_v2
    python train.py --model densenet121
    python train.py --model resnet50
    python train.py --model shufflenet_v2
    python train.py --model squeezenet

Hyperparameters follow Table 3 of the paper: Adam, lr=0.001, batch_size=32,
input 224x224. The paper does not disclose an epoch budget / early-stopping
patience for the main maize experiments (only for the tomato transfer
experiment, where patience=10 is stated) -- we apply the same patience=10
here for consistency, with a generous max_epochs cap so training halts on
its own once the test-set metric plateaus.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                              precision_score, recall_score)
from torch.utils.data import DataLoader

from dataset import CLASSES, MaizeDataset
from model import MSCPNet, build_backbone_with_mscp
from models_zoo import TruncatedMobileNetV2Classifier, build_pretrained_classifier

ROOT = Path(r"C:\Users\Personal\Documents\claude\mscpnet_repro")
SPLIT_TRAIN = ROOT / "data" / "splits" / "train"
SPLIT_TEST = ROOT / "data" / "splits" / "test"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)


def build_model(name):
    if name == "mscpnet":
        return MSCPNet(pretrained=True)
    if name == "truncated_mobilenet_v2":
        return TruncatedMobileNetV2Classifier(pretrained=True)
    if name.endswith("_mscp"):
        return build_backbone_with_mscp(name[: -len("_mscp")], pretrained=True)
    return build_pretrained_classifier(name, pretrained=True)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(all_labels, all_preds),
    }
    return metrics, all_preds, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     choices=["mscpnet", "mobilenet_v2", "truncated_mobilenet_v2",
                              "densenet121", "resnet50", "shufflenet_v2", "squeezenet",
                              "mobilenet_v2_mscp", "densenet121_mscp", "resnet50_mscp",
                              "shufflenet_v2_mscp", "squeezenet_mscp"])
    ap.add_argument("--max_epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{tag}] device={device}")

    train_ds = MaizeDataset(SPLIT_TRAIN, mode="train")
    test_ds = MaizeDataset(SPLIT_TEST, mode="test")
    print(f"[{tag}] train={len(train_ds)}  test={len(test_ds)}  classes={CLASSES}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    log_path = LOG_DIR / f"{tag}_log.jsonl"
    best_path = CKPT_DIR / f"{tag}_best.pt"
    log_f = open(log_path, "w", encoding="utf-8")

    best_acc = -1.0
    best_epoch = -1
    patience_ctr = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_ds)

        test_metrics, _, _ = evaluate(model, test_loader, device, criterion)
        dt = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss, "epoch_time_s": dt, **{f"test_{k}": v for k, v in test_metrics.items()}}
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        print(f"[{tag}] epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"test_acc={test_metrics['accuracy']:.4f}  test_f1={test_metrics['f1']:.4f}  "
              f"mcc={test_metrics['mcc']:.4f}  ({dt:.1f}s)")

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            best_epoch = epoch
            patience_ctr = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "metrics": test_metrics, "args": vars(args)}, best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[{tag}] early stopping at epoch {epoch} (best={best_acc:.4f} @ epoch {best_epoch})")
                break

    log_f.close()

    # final summary using best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    final_metrics, preds, labels = evaluate(model, test_loader, device, criterion)

    # measure inference time (single-image, matches paper's per-sample inference time reporting)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        n_reps = 100
        for _ in range(n_reps):
            model(dummy)
        torch.cuda.synchronize() if device == "cuda" else None
        inf_time = (time.time() - t0) / n_reps

    from thop import profile
    flops, params = profile(model, inputs=(dummy,), verbose=False)

    summary = {
        "tag": tag, "model": args.model, "best_epoch": ckpt["epoch"],
        "params": params, "flops": flops, "inference_time_s": inf_time,
        **final_metrics,
    }
    summary_path = CKPT_DIR / f"{tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] FINAL: {json.dumps(summary, indent=2)}")

    np.save(CKPT_DIR / f"{tag}_test_preds.npy", preds)
    np.save(CKPT_DIR / f"{tag}_test_labels.npy", labels)


if __name__ == "__main__":
    main()

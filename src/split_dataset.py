"""
Split the raw Kaggle Corn/Maize Leaf Disease dataset into train/test sets
matching the exact per-class counts reported in the MSCPNet paper (Table 1).

Paper counts:
    Class            Train   Test
    Blight             975    171
    Common_Rust       1111    195
    Gray_Leaf_Spot     488     86
    Healthy            988    174
    Total             3562    626
"""
import os
import random
import shutil
from pathlib import Path

RAW_DIR = Path(r"C:\Users\Personal\Documents\claude\mscpnet_repro\data\data")
OUT_DIR = Path(r"C:\Users\Personal\Documents\claude\mscpnet_repro\data\splits")
SEED = 42

TARGET_COUNTS = {
    "Blight": {"train": 975, "test": 171},
    "Common_Rust": {"train": 1111, "test": 195},
    "Gray_Leaf_Spot": {"train": 488, "test": 86},
    "Healthy": {"train": 988, "test": 174},
}

def main():
    random.seed(SEED)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    summary = {}
    for cls, counts in TARGET_COUNTS.items():
        src_dir = RAW_DIR / cls
        files = sorted([f for f in src_dir.iterdir() if f.is_file()])
        random.shuffle(files)

        n_train, n_test = counts["train"], counts["test"]
        assert len(files) == n_train + n_test, (
            f"{cls}: found {len(files)} files, expected {n_train + n_test}"
        )

        train_files = files[:n_train]
        test_files = files[n_train:n_train + n_test]

        for split_name, split_files in [("train", train_files), ("test", test_files)]:
            dst_dir = OUT_DIR / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dst_dir / f.name)

        summary[cls] = {"train": len(train_files), "test": len(test_files)}
        print(f"{cls:16s}  train={len(train_files):4d}  test={len(test_files):4d}")

    total_train = sum(v["train"] for v in summary.values())
    total_test = sum(v["test"] for v in summary.values())
    print(f"{'TOTAL':16s}  train={total_train:4d}  test={total_test:4d}")

if __name__ == "__main__":
    main()

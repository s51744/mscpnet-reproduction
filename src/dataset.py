"""
Dataset + augmentation pipeline reproducing:
  - Section III.B (preprocessing/resizing): 224x224, ImageNet normalization
  - Section III.C/D + Algorithm 1 (augmentation & class balancing):
      5 augmented images per original for all classes except Gray_Leaf_Spot,
      10 augmented images per original for Gray_Leaf_Spot.
      Augmentations: random rotations, flips, brightness/contrast, motion blur,
      grid distortion (via albumentations).
"""
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

K_PER_CLASS = {
    "Blight": 5,
    "Common_Rust": 5,
    "Gray_Leaf_Spot": 10,
    "Healthy": 5,
}

# Individual augmentation ops (Algorithm 1, line 8: "Randomly apply augmentation from A")
AUG_POOL = A.OneOf([
    A.Rotate(limit=45, p=1.0),
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.RandomBrightnessContrast(p=1.0),
    A.MotionBlur(blur_limit=7, p=1.0),
    A.GridDistortion(p=1.0),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=1.0),
    A.Transpose(p=1.0),
    A.OpticalDistortion(distort_limit=0.3, p=1.0),
    A.Blur(blur_limit=5, p=1.0),
], p=1.0)

RESIZE = A.Resize(224, 224)
NORMALIZE = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def _load_rgb(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class MaizeDataset(Dataset):
    """
    mode="train": builds the augmented training set in-memory index
                  (original + K augmented copies per image, per Algorithm 1).
    mode="test":  original images only, no augmentation.
    """

    def __init__(self, split_dir, mode="train", seed=42):
        self.split_dir = Path(split_dir)
        self.mode = mode
        self.samples = []  # list of (filepath, label, aug_flag)

        rng = random.Random(seed)
        for cls in CLASSES:
            cls_dir = self.split_dir / cls
            files = sorted(cls_dir.iterdir())
            label = CLASS_TO_IDX[cls]
            for f in files:
                self.samples.append((f, label, False))
                if mode == "train":
                    k = K_PER_CLASS[cls]
                    for _ in range(k):
                        self.samples.append((f, label, True))
        if mode == "train":
            rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, do_aug = self.samples[idx]
        img = _load_rgb(path)
        img = RESIZE(image=img)["image"]
        if do_aug:
            img = AUG_POOL(image=img)["image"]
        img = NORMALIZE(image=img)["image"]
        return img, label

    def class_counts(self):
        counts = {c: 0 for c in CLASSES}
        for _, label, _ in self.samples:
            counts[CLASSES[label]] += 1
        return counts


if __name__ == "__main__":
    ds = MaizeDataset(r"C:\Users\Personal\Documents\claude\mscpnet_repro\data\splits\train", mode="train")
    print("train augmented size:", len(ds), ds.class_counts())
    ds_test = MaizeDataset(r"C:\Users\Personal\Documents\claude\mscpnet_repro\data\splits\test", mode="test")
    print("test size:", len(ds_test), ds_test.class_counts())
    x, y = ds[0]
    print("sample tensor:", x.shape, x.dtype, "label:", y)

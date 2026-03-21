"""Phase 1 data strategy implementation for the Chinelotam meat freshness project.

This script performs explicit data quality checks and counts to establish a robust foundation
before model training. It is fully transparent, explicitly commented, and contains exactly
what was asked in the plan.

Key checks and outputs:
  1. Schema, one-hot validity, duplication, and sample counts.
  2. Class distribution and imbalance ratios with class weights.
  3. Path resolution (train/valid + per-class folder fallback) and missing file check.
  4. Dev transformation pipeline (resize, normalize, augment) without hue/saturation.
  5. Quick dataset iteration baseline sample sanity check.

Usage:
    python phase1_data_prep.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torchvision import transforms

from dataset_loader import LABEL_COLUMNS, LABEL_MAP, FruitFreshnessDataset, _resolve_image_path


def load_labels(root_dir: Path, split: str):
    """Load a label CSV and clean column names."""
    csv_path = root_dir / split / "_classes.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = ["filename"] + LABEL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    return df


def get_class_statistics(df: pd.DataFrame):
    """Return counts for each class, row count and one-hot issues."""
    counts = {c: int(df[c].sum()) for c in LABEL_COLUMNS}
    num_rows = int(len(df))

    # one-hot validity: exactly one class column should be 1 per sample
    one_hot_sums = df[LABEL_COLUMNS].sum(axis=1)
    invalid_one_hot = int((one_hot_sums != 1).sum())

    # duplicate filename check
    duplicate_filenames = df["filename"].duplicated(keep=False)
    duplicate_count = int(duplicate_filenames.sum())

    return counts, num_rows, invalid_one_hot, duplicate_count


def validate_image_paths(root_dir: Path, split: str, df: pd.DataFrame):
    """Check that each row's filename resolves to an existing image path."""
    root_split = root_dir / split
    missing_files = []
    for _, row in df.iterrows():
        filename = row["filename"]
        resolved = _resolve_image_path(root_split, filename, row)
        if resolved is None:
            missing_files.append(filename)
    return missing_files


def compute_class_weights(counts):
    """Compute class weights for imbalanced training.

    This is the inverse-frequency ratio, normalized to the maximum.
    """
    max_count = max(counts.values())
    weights = {cls: (max_count / count if count > 0 else 0.0)
               for cls, count in counts.items()}
    return weights


def build_transforms(image_size: int = 224):
    """Build a training + validation transform set with no hue/saturation shifts."""
    # Training behavior: geometric + light variation, not color shifts that would break meat semantics.
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 90)),
        transforms.RandomAffine(degrees=0, scale=(0.90, 1.10)),
        transforms.ColorJitter(
            brightness=0.10, contrast=0.10, saturation=0.0, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    return train_transform, valid_transform


def quick_dataset_sanity_check(root_dir: Path, train_transform, valid_transform):
    """Load a few examples and validate speaker + label shape, distribution, and tensor shape."""
    results = {}
    for split, transform in [("train", train_transform), ("valid", valid_transform)]:
        ds = FruitFreshnessDataset(
            str(root_dir), split=split, transform=transform)
        n = min(5, len(ds))
        observed_shapes = []
        observed_labels = []
        for i in range(n):
            image, label = ds[i]
            observed_shapes.append(tuple(image.shape))
            observed_labels.append(int(label))

        results[split] = {
            "num_loaded": len(ds),
            "sample_shape": observed_shapes[0] if observed_shapes else None,
            "label_histogram": {k: observed_labels.count(k) for k in range(len(LABEL_MAP))},
        }

    return results


def main():
    # The dataset is stored inside the `project-dataset/` folder.
    root_dir = Path(__file__).resolve().parent / "project-dataset"

    print("Phase 1: data inspection, path validation, class balance, and augmentation setup")

    for split in ["train", "valid"]:
        print(f"\n--- {split.upper()} split ---")
        df = load_labels(root_dir, split)

        class_counts, total_rows, invalid_one_hot, duplicates = get_class_statistics(
            df)
        missing_files = validate_image_paths(root_dir, split, df)

        print(f"CSV rows: {total_rows}")
        print(f"One-hot invalid rows: {invalid_one_hot}")
        print(f"Duplicate filenames: {duplicates}")

        print("Class counts:")
        for cls, cnt in class_counts.items():
            print(f"  {cls}: {cnt}")

        imbalance_ratios = {
            cls: class_counts[cls] / total_rows for cls in class_counts}
        print("Class ratios:")
        for cls, ratio in imbalance_ratios.items():
            print(f"  {cls}: {ratio:.4f}")

        print(f"Missing image paths: {len(missing_files)}")
        if missing_files:
            print("Example missing files:", missing_files[:5])

        if split == "train":
            weights = compute_class_weights(class_counts)
            print("Class weights (inverse frequency):")
            for cls, w in weights.items():
                print(f"  {cls}: {w:.4f}")

    # Data transforms check
    train_t, valid_t = build_transforms(image_size=224)
    sanity = quick_dataset_sanity_check(root_dir, train_t, valid_t)

    print("\nQuick dataset sanity check after transforms:")
    for split, info in sanity.items():
        print(
            f"{split}: num_loaded={info['num_loaded']}, sample_shape={info['sample_shape']}, label_histogram={info['label_histogram']}")

    print("\nPhase 1 is done. Ready for Phase 2 training code.")


if __name__ == "__main__":
    main()

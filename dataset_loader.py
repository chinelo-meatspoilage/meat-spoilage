import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


LABEL_COLUMNS = ["Fresh", "Spoiled"]
LABEL_MAP = {"Fresh": 0, "Spoiled": 1}


class FruitFreshnessDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root = Path(root_dir)
        assert split in ["train", "valid"], "split must be 'train' or 'valid'"
        self.split = split
        self.transform = transform

        self.csv_path = self.root / split / "_classes.csv"
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing label file: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = self.df.columns.str.strip()

        required = ["filename"] + LABEL_COLUMNS
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: {missing}")

        self.samples: List[Tuple[Path, int]] = []
        for _, row in self.df.iterrows():
            fn = row["filename"]
            image_path = self._find_image_path(fn, row)
            self.samples.append((image_path, self._row_to_label(row)))

    def _find_image_path(self, fn: str, row) -> Path:
        root_split = self.root / self.split

        candidates = [root_split / fn]

        # Prefer class derived from label one-hot.
        for cls, value in row.items():
            if cls in LABEL_MAP and int(value) == 1:
                class_dir = cls.lower().replace("-", "-")
                candidates.append(root_split / class_dir / fn)

        # Fallback from filename prefix (e.g., FRESH-xxx)
        prefix = fn.split("-", 1)[0].strip().lower()
        if prefix in ["fresh", "spoiled", "half-fresh"]:
            candidates.append(root_split / prefix / fn)

        # check all class folders if not found yet
        for fallback_cls in ["fresh", "spoiled"]:
            candidates.append(root_split / fallback_cls / fn)

        for p in candidates:
            if p.exists():
                return p

        # If still not found, keep estimated path and warn, but keep dataset length
        raise FileNotFoundError(
            f"Image not found for '{fn}'. Tried: {[str(p) for p in candidates]}")

    def _row_to_label(self, row) -> int:
        for col, idx in LABEL_MAP.items():
            if row.get(col, 0) == 1:
                return idx
        raise ValueError(f"No one-hot label found row: {row}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _resolve_image_path(root_split: Path, fn: str, row) -> Path:
    candidates = [root_split / fn]

    for cls, value in row.items():
        if cls in LABEL_MAP and int(value) == 1:
            candidates.append(root_split / cls.lower() / fn)

    prefix = fn.split("-", 1)[0].strip().lower()
    if prefix in ["fresh", "spoiled", "half-fresh"]:
        candidates.append(root_split / prefix / fn)

    for fallback_cls in ["fresh", "spoiled"]:
        candidates.append(root_split / fallback_cls / fn)

    return next((p for p in candidates if p.exists()), None)


def summarize_dataset(root_dir: str, split: str = "train") -> Dict:
    root = Path(root_dir)
    csv_path = root / split / "_classes.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    class_counts = {c: int(df[c].sum()) for c in LABEL_COLUMNS}
    num_samples = len(df)
    root_split = root / split
    filepath_exists = df.apply(
        lambda r: _resolve_image_path(
            root_split, r["filename"], r) is not None,
        axis=1)
    missing = int((~filepath_exists).sum())

    return {
        "split": split,
        "num_samples": num_samples,
        "class_counts": class_counts,
        "missing_files": missing,
    }


if __name__ == "__main__":
    # The dataset lives under the "project-dataset" directory within this repo.
    data_root = Path(__file__).resolve().parent / "project-dataset"
    for subset in ["train", "valid"]:
        summary = summarize_dataset(data_root, split=subset)
        print(f"{subset.upper()} summary:")
        print(f"  num_samples: {summary['num_samples']}")
        print("  class_counts:")
        for c, v in summary["class_counts"].items():
            print(f"    {c}: {v}")
        print(f"  missing_files: {summary['missing_files']}")
        print()

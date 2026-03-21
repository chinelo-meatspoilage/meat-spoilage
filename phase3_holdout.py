"""Run a quick holdout/test evaluation on a folder of images.

This script is intended for a small set of "unseen" images that were not used
for training or validation. It computes a simple accuracy plus confusion matrix
based on folder labels (folder names should match the class labels: Fresh, Half-Fresh, Spoiled).

Example:
  python phase3_holdout.py --dir holdout --best

Folder layout (recommended):
  holdout/
    Fresh/...
    Half-Fresh/...
    Spoiled/...

If your directory contains images directly (no subfolders), the script will run inference
but will not compute accuracy/confusion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

from phase2_training import LABEL_COLUMNS
from phase3_inference import _find_best_model_name, load_model, predict


def _gather_images(holdout_dir: Path) -> Tuple[List[Path], Optional[List[str]]]:
    """Gather images and, if possible, their ground truth labels.

    Returns:
      - list of image paths
      - list of ground truth labels (or None if labels cannot be inferred)
    """
    image_paths: List[Path] = []
    labels: List[str] = []

    # If directory has subfolders matching known classes, use those labels.
    subdirs = [p for p in sorted(holdout_dir.iterdir()) if p.is_dir()]
    valid_subdirs = [p for p in subdirs if p.name in LABEL_COLUMNS]

    if valid_subdirs:
        for subdir in valid_subdirs:
            for img in sorted(subdir.iterdir()):
                if img.is_file():
                    image_paths.append(img)
                    labels.append(subdir.name)
        return image_paths, labels

    # Otherwise, just collect all files and do not assign labels.
    for img in sorted(holdout_dir.iterdir()):
        if img.is_file():
            image_paths.append(img)
    return image_paths, None


def write_report(out_path: Path, report: Dict) -> None:
    md = ["# Holdout evaluation report\n"]
    md.append(f"**Model**: {report['model']}\n")
    md.append(f"**Images evaluated**: {report['num_images']}\n")

    if report.get("accuracy") is not None:
        md.append(f"**Accuracy**: {report['accuracy']:.4f}\n")
        md.append("\n## Confusion matrix (rows=true, cols=predicted)\n")
        cm = report["confusion_matrix"]
        df = pd.DataFrame(cm, index=LABEL_COLUMNS, columns=LABEL_COLUMNS)
        md.append(df.to_markdown())
        md.append("\n")

    if report.get("raw_predictions") is not None:
        md.append("\n## Predictions\n")
        for r in report["raw_predictions"]:
            md.append(
                f"- {r['image']} -> {r['prediction']} ({r['confidence']:.1%})")

    out_path.write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a holdout/test evaluation.")
    parser.add_argument("--dir", required=True,
                        help="Holdout directory to evaluate")
    parser.add_argument("--model", help="Model name to use (e.g., ResNet50)")
    parser.add_argument("--best", action="store_true",
                        help="Use the best model determined by evaluation")
    parser.add_argument("--output", help="Path to write a markdown report",
                        default="phase3_reports/holdout_report.md")
    args = parser.parse_args()

    if args.model and args.best:
        raise SystemExit("Please provide only one of --model or --best")

    if not args.model and not args.best:
        raise SystemExit("Please provide --model or --best")

    model_name = args.model or _find_best_model_name()
    if model_name is None:
        raise SystemExit(
            "Could not determine best model; run phase3_evaluation.py --best first.")

    model = load_model(model_name)

    holdout_dir = Path(args.dir)
    if not holdout_dir.exists():
        raise SystemExit(f"Holdout directory does not exist: {holdout_dir}")

    image_paths, labels = _gather_images(holdout_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {holdout_dir}")

    results = predict(model, image_paths)

    report: Dict = {
        "model": model_name,
        "num_images": len(image_paths),
        "raw_predictions": results,
    }

    if labels is not None and len(labels) == len(results):
        y_true = np.array([LABEL_COLUMNS.index(l) for l in labels])
        y_pred = np.array([LABEL_COLUMNS.index(r["prediction"])
                          for r in results])

        report["accuracy"] = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(len(LABEL_COLUMNS))))
        report["confusion_matrix"] = cm.tolist()

        print(f"Holdout accuracy: {report['accuracy']:.4f}")
        print("Confusion matrix (rows=true, cols=predicted):")
        print(np.array(cm))

    else:
        print("Holdout directory did not contain labeled subfolders; only predictions are shown.")

    write_report(Path(args.output), report)
    print(f"Wrote holdout report to {args.output}")


if __name__ == "__main__":
    main()

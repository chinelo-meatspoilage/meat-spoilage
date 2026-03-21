"""Phase 3: Evaluation & model selection.

This script loads trained models from `phase2_models/`, runs evaluation on the validation
set, and generates plots + a short report for each model.

Usage:
    python phase3_evaluation.py --all
    python phase3_evaluation.py --model ResNet50
    python phase3_evaluation.py --best

Outputs:
  - `phase3_reports/<model>/confusion_matrix.png`
  - `phase3_reports/<model>/roc_curve.png`
  - `phase3_reports/<model>/report.md`
  - `phase3_reports/selected_model.txt` (when using --best)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

# Reuse helpers from phase2_training to keep preprocessing consistent.
from phase2_training import (LABEL_COLUMNS, MODEL_OUTPUT_DIR, IMAGE_SIZE, BATCH_SIZE,
                             build_path_label_pairs, make_dataset)


REPORT_ROOT = Path(__file__).resolve().parent / "phase3_reports"
REPORT_ROOT.mkdir(exist_ok=True, parents=True)


def _load_eval_report(model_dir: Path) -> Dict:
    report_path = model_dir / "eval_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing eval report: {report_path}")
    return json.loads(report_path.read_text())


def _best_model_by_macro_f1() -> Optional[str]:
    """Pick the best model based on macro-average F1 score from eval_report.json."""
    best = None
    best_f1 = -1.0
    for model_dir in MODEL_OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        try:
            report = _load_eval_report(model_dir)
            macro_f1 = report["classification_report"]["macro avg"]["f1-score"]
        except Exception:
            continue
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best = model_dir.name
    return best


def _find_model_file(model_dir: Path) -> Path:
    """Find the saved best model file (prefers .keras, falls back to .h5)."""
    for name in ["best_model.keras", "best_model.h5"]:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No saved model file found in {model_dir} (expected best_model.keras or best_model.h5)"
    )


def _safe_load_model(model_dir: Path) -> tf.keras.Model:
    model_path = _find_model_file(model_dir)
    return tf.keras.models.load_model(str(model_path))


def _plot_confusion_matrix(conf: np.ndarray, labels: List[str], out_path: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_roc_curves(y_true_onehot: np.ndarray, y_pred_proba: np.ndarray, labels: List[str], out_path: Path):
    plt.figure(figsize=(7, 6))

    # Plot per-class ROC curve
    for idx, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, idx], y_pred_proba[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

    # Diagonal reference
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _make_output_dir(model_name: str, root: Optional[Path] = None) -> Path:
    root = root or REPORT_ROOT
    out_dir = root / model_name
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


def evaluate_and_report(model_name: str, output_root: Optional[Path] = None) -> Dict:
    out_dir = _make_output_dir(model_name, root=output_root)

    model_dir = MODEL_OUTPUT_DIR / model_name
    model = _safe_load_model(model_dir)

    # Reload data to ensure identical evaluation to training
    valid_pairs = build_path_label_pairs("valid")
    valid_ds = make_dataset(valid_pairs, batch=BATCH_SIZE, training=False)

    # Compute predictions on validation split
    y_true = []
    y_pred = []
    y_pred_proba = []
    for x_batch, y_batch in valid_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred_proba.extend(preds.tolist())
        y_pred.extend(np.argmax(preds, axis=-1).tolist())
        y_true.extend(y_batch.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    conf = confusion_matrix(y_true, y_pred)
    y_true_onehot = tf.keras.utils.to_categorical(
        y_true, num_classes=len(LABEL_COLUMNS))

    # Generate plots
    _plot_confusion_matrix(conf, LABEL_COLUMNS,
                           out_dir / "confusion_matrix.png")
    _plot_roc_curves(y_true_onehot, y_pred_proba,
                     LABEL_COLUMNS, out_dir / "roc_curve.png")

    # Save a small report, including a fresh evaluation (in case models and eval reports are out of sync)
    class_report = classification_report(
        y_true, y_pred, target_names=LABEL_COLUMNS, output_dict=True
    )
    roc_auc = roc_auc_score(y_true_onehot, y_pred_proba, multi_class="ovo")

    report = {
        "model": model_name,
        "num_validation_samples": int(len(y_true)),
        "confusion_matrix": conf.tolist(),
        "roc_auc": float(roc_auc),
        "classification_report": class_report,
    }

    # Note: we include the existing JSON report (if it exists) for quick reference.
    try:
        report_json = _load_eval_report(model_dir)
        report["eval_report_json"] = report_json
    except Exception:
        pass

    # Write report markdown
    md = [f"# Evaluation report: {model_name}\n"]
    md.append("## Key outputs")
    md.append(f"- Confusion matrix: `confusion_matrix.png`")
    md.append(f"- ROC curves: `roc_curve.png`")
    md.append("\n## Metrics from validation run")
    cr = report["classification_report"]
    md.append("| Class | Precision | Recall | F1 | Support |")
    md.append("|---|---|---|---|---|")
    for cls in LABEL_COLUMNS:
        r = cr.get(cls, {})
        md.append(
            f"| {cls} | {r.get('precision', '-'):.3f} | {r.get('recall', '-'):.3f} | {r.get('f1-score', '-'):.3f} | {int(r.get('support', 0))} |"
        )
    macro = cr.get("macro avg", {})
    md.append(
        f"\n**Macro avg F1**: {macro.get('f1-score', '-'): .3f}  \n**Accuracy**: {cr.get('accuracy', '-'):.3f}"
    )

    (out_dir / "report.md").write_text("\n".join(md))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 evaluation: plots + reports for trained models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Evaluate all available models")
    group.add_argument("--model", type=str,
                       help="Evaluate a single model by name")
    group.add_argument("--best", action="store_true",
                       help="Evaluate and mark the best model by macro F1")

    args = parser.parse_args()

    if args.best:
        best = _best_model_by_macro_f1()
        if not best:
            raise SystemExit(
                "Could not determine best model (no evaluation reports found).")
        selected_path = REPORT_ROOT / "selected_model.txt"
        selected_path.write_text(best)
        print(
            f"Best model (macro F1) is: {best} (recorded in {selected_path})")
        print("Generating evaluation artifacts for the best model...")
        evaluate_and_report(best)
        return

    if args.model:
        evaluate_and_report(args.model)
        return

    # --all
    for model_dir in MODEL_OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        print(f"Evaluating {model_dir.name}...")
        evaluate_and_report(model_dir.name)


if __name__ == "__main__":
    main()

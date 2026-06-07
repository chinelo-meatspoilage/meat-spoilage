"""Generate training/validation accuracy and loss curves for the ResNet50 model.

Reads the saved training history from phase2_models/ResNet50/training_history.json
and saves two publication-ready PNG files to phase3_reports/ResNet50/:

  - resnet50_accuracy_curve.png
  - resnet50_loss_curve.png

These correspond to Section 4.3.3 (Analysis of Training Curves) in the report.

Usage:
    python generate_training_curves.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HISTORY_PATH = Path(__file__).resolve().parent / "phase2_models" / "ResNet50" / "training_history.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "phase3_reports" / "ResNet50"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

STYLE = {
    "train_color": "#1f77b4",   # blue
    "val_color":   "#d62728",   # red
    "linewidth":   2.0,
    "markersize":  5,
    "figsize":     (8, 5),
    "dpi":         150,
    "font_size":   12,
}

plt.rcParams.update({
    "font.size":        STYLE["font_size"],
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})


def load_history(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_accuracy(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.plot(epochs, history["accuracy"],
            color=STYLE["train_color"], linewidth=STYLE["linewidth"],
            marker="o", markersize=STYLE["markersize"], label="Training Accuracy")
    ax.plot(epochs, history["val_accuracy"],
            color=STYLE["val_color"], linewidth=STYLE["linewidth"],
            marker="s", markersize=STYLE["markersize"], linestyle="--",
            label="Validation Accuracy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("ResNet50 — Training and Validation Accuracy")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(0.70, 1.01)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_loss(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["loss"]) + 1)

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.plot(epochs, history["loss"],
            color=STYLE["train_color"], linewidth=STYLE["linewidth"],
            marker="o", markersize=STYLE["markersize"], label="Training Loss")
    ax.plot(epochs, history["val_loss"],
            color=STYLE["val_color"], linewidth=STYLE["linewidth"],
            marker="s", markersize=STYLE["markersize"], linestyle="--",
            label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("ResNet50 — Training and Validation Loss")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history not found at {HISTORY_PATH}.\n"
            "Run extract_resnet50_history.py first."
        )

    history = load_history(HISTORY_PATH)
    n_epochs = len(history["accuracy"])
    print(f"Loaded training history: {n_epochs} epochs")

    plot_accuracy(history, OUTPUT_DIR / "resnet50_accuracy_curve.png")
    plot_loss(history,    OUTPUT_DIR / "resnet50_loss_curve.png")

    print("\nDone. Graphs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

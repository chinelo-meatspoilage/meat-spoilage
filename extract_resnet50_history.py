"""Extract per-epoch training history for ResNet50 only.

Saves history to phase2_models/ResNet50/training_history.json
WITHOUT overwriting the existing best_model.keras.
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import LABEL_COLUMNS, _resolve_image_path
from phase2_training import (
    build_model,
    build_path_label_pairs,
    compute_class_weights,
    make_dataset,
)

DATA_ROOT = Path(__file__).resolve().parent / "project-dataset"
MODEL_OUTPUT_DIR = Path(__file__).resolve().parent / "phase2_models"
IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42
EPOCHS = 30

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def run():
    print("Loading data...")
    train_pairs = build_path_label_pairs("train")
    valid_pairs = build_path_label_pairs("valid")

    class_weights = compute_class_weights(train_pairs)

    train_ds = make_dataset(train_pairs, batch=BATCH_SIZE, training=True)
    valid_ds = make_dataset(valid_pairs, batch=BATCH_SIZE, training=False)

    print("Building ResNet50 model...")
    model = build_model("ResNet50")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # Save to a TEMP location — never overwrites existing best_model.keras
    model_dir = MODEL_OUTPUT_DIR / "ResNet50"
    model_dir.mkdir(exist_ok=True, parents=True)
    temp_checkpoint = str(model_dir / "temp_history_run.keras")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        temp_checkpoint,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print("Training ResNet50 (history extraction run)...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    # Save per-epoch history to JSON
    history_data = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    history_path = model_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)

    print(f"\nSaved training history to {history_path}")
    print(f"Epochs completed: {len(history_data['loss'])}")

    # Print a quick summary table
    print(f"\n{'Epoch':<7} {'Train Acc':<12} {'Val Acc':<12} {'Train Loss':<13} {'Val Loss'}")
    print("-" * 58)
    for i, (ta, va, tl, vl) in enumerate(zip(
        history_data["accuracy"],
        history_data["val_accuracy"],
        history_data["loss"],
        history_data["val_loss"],
    ), start=1):
        print(f"{i:<7} {ta:<12.6f} {va:<12.6f} {tl:<13.6f} {vl:.6f}")

    # Remove the temp checkpoint so we don't clutter the directory
    if Path(temp_checkpoint).exists():
        Path(temp_checkpoint).unlink()

    return history_data


if __name__ == "__main__":
    run()

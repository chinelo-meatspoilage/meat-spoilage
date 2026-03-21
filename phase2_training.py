"""Phase 2: Train and evaluate multiple TensorFlow/Keras models for meat freshness.

This script trains three transfer-learning models (MobileNetV2, ResNet50, EfficientNetB0)
using the dataset in `project-dataset/` and compares them using validation metrics.

Key behaviors:
  - Uses the same clean file mapping logic as the existing PyTorch dataset loader.
  - Applies safe augmentations (no hue/saturation shifts) and resizes to 224x224.
  - Computes class weights from training class distribution.
  - Trains each model with EarlyStopping and ModelCheckpoint.
  - Reports accuracy, precision, recall, F1, confusion matrix, and ROC-AUC.

Usage:
    python phase2_training.py

Outputs:
  - Saved model files under `phase2_models/`.
  - A summary CSV report in `phase2_models/summary.csv`.
"""

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_auc_score)

from dataset_loader import LABEL_COLUMNS, LABEL_MAP, _resolve_image_path


# === Config ===
DATA_ROOT = Path(__file__).resolve().parent / "project-dataset"
IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42
EPOCHS = 30
MODEL_OUTPUT_DIR = Path(__file__).resolve().parent / "phase2_models"
MODEL_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# --- Helper utilities ---

def read_labels(split: str) -> pd.DataFrame:
    """Read label CSV and clean columns."""
    path = DATA_ROOT / split / "_classes.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = ["filename"] + LABEL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def build_path_label_pairs(split: str) -> List[Tuple[str, int]]:
    """Return (image_path, label) pairs for a split."""
    df = read_labels(split)

    # Map one-hot to integer label
    label_map = {"Fresh": 0, "Spoiled": 1}

    pairs: List[Tuple[str, int]] = []
    for _, row in df.iterrows():
        filename = row["filename"]
        path = _resolve_image_path(DATA_ROOT / split, filename, row)
        if path is None:
            raise FileNotFoundError(
                f"Could not resolve image path for {filename}")

        # Determine label (one-hot => index)
        for k, v in label_map.items():
            if int(row.get(k, 0)) == 1:
                pairs.append((str(path), v))
                break
        else:
            raise ValueError(f"No one-hot label found for {filename}")

    return pairs


def compute_class_weights(pairs: List[Tuple[str, int]]) -> Dict[int, float]:
    """Compute class weights to address imbalance during training."""
    labels = [label for (_, label) in pairs]
    counts = np.bincount(labels, minlength=len(LABEL_COLUMNS))
    max_count = counts.max()
    return {i: float(max_count / c) if c > 0 else 0.0 for i, c in enumerate(counts)}


def build_augmenter(image_size: int) -> Callable[[tf.Tensor], tf.Tensor]:
    """Build a function to apply safe augmentations to a tensor image."""

    def augmenter(image: tf.Tensor) -> tf.Tensor:
        # Random flips
        image = tf.image.random_flip_left_right(image, seed=SEED)
        image = tf.image.random_flip_up_down(image, seed=SEED)

        # Random 90° rotations (0/90/180/270)
        k = tf.random.uniform([], minval=0, maxval=4,
                              dtype=tf.int32, seed=SEED)
        image = tf.image.rot90(image, k)

        # Random zoom: crop and resize
        scale = tf.random.uniform([], 0.90, 1.0, seed=SEED)
        orig_size = tf.cast(tf.shape(image)[:2], tf.float32)
        crop_size = tf.cast(orig_size * scale, tf.int32)
        image = tf.image.random_crop(
            image, size=tf.concat([crop_size, [3]], axis=0))
        image = tf.image.resize(image, [image_size, image_size])

        # Lightness: brightness + contrast (small range)
        image = tf.image.random_brightness(image, max_delta=0.10, seed=SEED)
        image = tf.image.random_contrast(
            image, lower=0.90, upper=1.10, seed=SEED)

        return image

    return augmenter


def preprocess_image(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """Decode/resize/normalize an image tensor.

    This function is intentionally robust:
    - Accepts JPEG/PNG and grayscale inputs.
    - Ensures the output is always 3-channel RGB.
    - Avoids "Number of channels requested does not match input" errors.
    """
    # tf.io.decode_image can handle JPEG/PNG and automatically converts grayscale/alpha.
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    # Ensure the result has a static 3-channel shape for downstream layers.
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    return image


def make_dataset(pairs: List[Tuple[str, int]], batch: int, training: bool) -> tf.data.Dataset:
    """Build and return a tf.data.Dataset from (path, label) pairs."""
    paths, labels = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _load(path, label):
        img = tf.io.read_file(path)
        img = preprocess_image(img, IMAGE_SIZE)
        if training:
            img = augmenter(img)
        # Normalize to ImageNet range [-1, 1] (works for all the chosen models)
        img = tf.keras.applications.imagenet_utils.preprocess_input(
            img * 255.0)
        return img, label

    augmenter = build_augmenter(IMAGE_SIZE)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(name: str, input_shape=(224, 224, 3), num_classes=2) -> tf.keras.Model:
    """Build a transfer-learning model based on the given name."""
    base: tf.keras.Model
    if name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
    elif name == "ResNet50":
        base = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
    elif name == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
    else:
        raise ValueError(f"Unsupported model name: {name}")

    base.trainable = False  # freeze base weights for transfer learning

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def evaluate_model(model: tf.keras.Model, ds: tf.data.Dataset) -> Dict[str, float]:
    """Compute evaluation metrics on the validation set."""
    y_true = []
    y_pred = []
    y_pred_proba = []

    for x_batch, y_batch in ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred_proba.extend(preds.tolist())
        y_pred.extend(np.argmax(preds, axis=-1).tolist())
        y_true.extend(y_batch.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    report = classification_report(
        y_true, y_pred, target_names=LABEL_COLUMNS, output_dict=True)
    conf = confusion_matrix(y_true, y_pred)

    # Multi-class ROC AUC
    y_true_onehot = tf.keras.utils.to_categorical(
        y_true, num_classes=len(LABEL_COLUMNS))
    roc_auc = roc_auc_score(y_true_onehot, y_pred_proba, multi_class="ovo")

    return {
        "classification_report": report,
        "confusion_matrix": conf.tolist(),
        "roc_auc": float(roc_auc),
    }


def convert_h5_to_keras(model_root: Path):
    """Convert any best_model.h5 files to best_model.keras format.

    - Skips conversion if best_model.keras already exists.
    - Keeps the original .h5 file untouched.
    """
    converted = []
    for model_dir in model_root.iterdir():
        if not model_dir.is_dir():
            continue
        h5_path = model_dir / "best_model.h5"
        keras_path = model_dir / "best_model.keras"
        if h5_path.exists() and not keras_path.exists():
            print(f"Converting {h5_path} -> {keras_path}")
            model = tf.keras.models.load_model(str(h5_path))
            model.save(str(keras_path))
            converted.append((h5_path, keras_path))
    if not converted:
        print("No .h5 models found to convert, or all models already have .keras versions.")
    return converted


def train_and_evaluate():
    # Prepare data
    train_pairs = build_path_label_pairs("train")
    valid_pairs = build_path_label_pairs("valid")

    class_weights = compute_class_weights(train_pairs)

    train_ds = make_dataset(train_pairs, batch=BATCH_SIZE, training=True)
    valid_ds = make_dataset(valid_pairs, batch=BATCH_SIZE, training=False)

    # model_names = ["MobileNetV2", "ResNet50", "EfficientNetB0"]
    model_names = [ "ResNet50"]
    summary_rows = []

    for model_name in model_names:
        print(f"\n--- Training {model_name} ---")
        model = build_model(model_name)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        )

        model_dir = MODEL_OUTPUT_DIR / model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            str(model_dir / "best_model.keras"),  # Changed from .h5 to .keras
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=[checkpoint, early_stop],
            verbose=1,
        )

        eval_metrics = evaluate_model(model, valid_ds)

        summary_rows.append({
            "model": model_name,
            "best_val_loss": min(history.history["val_loss"]),
            "best_val_accuracy": max(history.history["val_accuracy"]),
            "roc_auc": eval_metrics["roc_auc"],
            "confusion_matrix": eval_metrics["confusion_matrix"],
            "classification_report": eval_metrics["classification_report"],
        })

        # Save evaluation report for this model
        report_path = model_dir / "eval_report.json"
        pd.Series(eval_metrics).to_json(report_path, indent=2)

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(MODEL_OUTPUT_DIR / "summary.csv", index=False)
    print(f"\nSaved summary to {MODEL_OUTPUT_DIR / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train/convert models for the meat freshness project.")
    parser.add_argument("--convert-only", action="store_true",
                        help="Convert existing .h5 models to .keras without retraining")
    parser.add_argument("--train", action="store_true",
                        help="Run training/evaluation pipeline")
    args = parser.parse_args()

    if args.convert_only:
        convert_h5_to_keras(MODEL_OUTPUT_DIR)
        return

    if args.train:
        train_and_evaluate()
        return

    # Default behavior: train and then ensure .keras exists for any existing .h5
    train_and_evaluate()
    convert_h5_to_keras(MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    main()

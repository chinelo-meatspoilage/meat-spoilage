"""Phase 3: Inference helper for the meat freshness classifier.

This script loads a trained model from `phase2_models/` and runs inference on one or
more images, printing class probabilities and the top predicted label.

Usage:
    python phase3_inference.py --image path/to/img.jpg --model ResNet50
    python phase3_inference.py --dir some_images/ --best

By default, it will reuse the same preprocessing used during training (resize to 224x224,
normalize to ImageNet range [-1, 1]).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from PIL import Image

from phase2_training import IMAGE_SIZE, LABEL_COLUMNS, preprocess_image


PHASE2_MODEL_ROOT = Path(__file__).resolve().parent / "phase2_models"
PHASE3_REPORTS = Path(__file__).resolve().parent / "phase3_reports"


def _find_saved_model(model_dir: Path) -> Path:
    for candidate in ["best_model.keras", "best_model.h5"]:
        p = model_dir / candidate
        if p.exists():
            return p
    raise FileNotFoundError(f"No saved model found in {model_dir}")


def _find_best_model_name() -> Optional[str]:
    selected = PHASE3_REPORTS / "selected_model.txt"
    if selected.exists():
        return selected.read_text().strip()

    # fallback: use highest macro F1 from available eval reports
    best_name = None
    best_f1 = -1.0
    for model_dir in PHASE2_MODEL_ROOT.iterdir():
        if not model_dir.is_dir():
            continue
        report_path = model_dir / "eval_report.json"
        if not report_path.exists():
            continue
        try:
            data = report_path.read_text()
            import json

            report = json.loads(data)
            macro = report["classification_report"]["macro avg"]["f1-score"]
            if macro > best_f1:
                best_f1 = macro
                best_name = model_dir.name
        except Exception:
            continue
    return best_name


def load_model(model_name: str) -> tf.keras.Model:
    model_dir = PHASE2_MODEL_ROOT / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    model_path = _find_saved_model(model_dir)
    return tf.keras.models.load_model(str(model_path))


def preprocess_path(path: Path) -> tf.Tensor:
    """Load an image and return a preprocessed tensor ready for the model.

    This helper is tolerant to a wide range of image formats (JPEG/PNG/WEBP/etc.)
    and ensures the output is always a 3-channel RGB tensor.
    """
    try:
        img_bytes = tf.io.read_file(str(path))
        img = preprocess_image(img_bytes, IMAGE_SIZE)
    except Exception:
        # Fall back to Pillow for unusual image formats or encoding issues.
        pil_img = Image.open(str(path)).convert("RGB")
        pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(pil_img).astype(np.float32) / 255.0
        img = tf.convert_to_tensor(arr)

    # `preprocess_image` returns float32 in [0, 1]; train pipeline scaled to [-1, 1]
    img = tf.keras.applications.imagenet_utils.preprocess_input(img * 255.0)
    return img


def predict(model: tf.keras.Model, image_paths: List[Path]) -> List[dict]:
    preprocessed = []
    for p in image_paths:
        try:
            preprocessed.append(preprocess_path(p))
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image {p}: {e}") from e

    inputs = tf.stack(preprocessed)
    probs = model.predict(inputs, verbose=0)

    results = []
    for path, proba in zip(image_paths, probs):
        top_idx = int(np.argmax(proba))
        results.append(
            {
                "image": str(path),
                "prediction": LABEL_COLUMNS[top_idx],
                "confidence": float(proba[top_idx]),
                "probabilities": {k: float(v) for k, v in zip(LABEL_COLUMNS, proba)},
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained meat freshness model.")
    parser.add_argument("--image", type=str,
                        help="Path to a single image to classify.")
    parser.add_argument("--dir", type=str,
                        help="Directory of images to classify.")
    parser.add_argument(
        "--model", type=str, help="Model name (e.g., ResNet50). If not provided, --best is required.")
    parser.add_argument("--best", action="store_true",
                        help="Use the best model determined by evaluation.")
    args = parser.parse_args()

    if not args.image and not args.dir:
        raise SystemExit("Please provide --image or --dir")

    if args.model and args.best:
        raise SystemExit("Please provide only one of --model or --best")

    model_name = None
    if args.best:
        model_name = _find_best_model_name()
        if not model_name:
            raise SystemExit(
                "Could not determine best model; run phase3_evaluation.py --best first.")
    elif args.model:
        model_name = args.model

    if model_name is None:
        raise SystemExit("No model specified. Use --model or --best.")

    model = load_model(model_name)

    image_paths: List[Path] = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.dir:
        p = Path(args.dir)
        image_paths.extend([x for x in sorted(p.iterdir()) if x.is_file()])

    results = predict(model, image_paths)

    for r in results:
        print(f"{r['image']}")
        for label, score in r["probabilities"].items():
            print(f"  {label:10s}: {score*100:5.1f}%")
        print(
            f"  -> Predicted: {r['prediction']} ({r['confidence']*100:5.1f}%)")
        print()


if __name__ == "__main__":
    main()

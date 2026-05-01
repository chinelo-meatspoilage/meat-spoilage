# Meat Freshness Classifier — Complete Project Documentation

> **Prepared by:** damilareadekeye  
> **Date:** May 2026  
> **Project name:** Chinelotam — Meat Freshness Classification System  
> **Supervisor/Owner:** Mr. Benedict Ogazi  
> **Framework:** TensorFlow 2.21 / Keras — ResNet50 transfer learning  
> **Deployment:** Streamlit Cloud (web) + Tkinter (local desktop)

---

## Table of Contents

1. [Project Overview and Background](#1-project-overview-and-background)
2. [Problem Statement](#2-problem-statement)
3. [Classification Decision: Binary Labels](#3-classification-decision-binary-labels)
4. [Dataset Description](#4-dataset-description)
5. [Data Strategy — The Color Trap](#5-data-strategy--the-color-trap)
6. [Phase 1 — Data Validation (`phase1_data_prep.py`)](#6-phase-1--data-validation)
7. [Phase 2 — Model Training (`phase2_training.py`)](#7-phase-2--model-training)
8. [Phase 3 — Evaluation (`phase3_evaluation.py`)](#8-phase-3--evaluation)
9. [Phase 3 — Inference CLI (`phase3_inference.py`)](#9-phase-3--inference-cli)
10. [Phase 3 — Holdout Evaluation (`phase3_holdout.py`)](#10-phase-3--holdout-evaluation)
11. [Phase 4 — Streamlit Web App (`app.py`)](#11-phase-4--streamlit-web-app)
12. [Phase 4 — Tkinter Desktop App (`app_desktop.py`)](#12-phase-4--tkinter-desktop-app)
13. [Repository and File Structure](#13-repository-and-file-structure)
14. [Dependencies and Requirements](#14-dependencies-and-requirements)
15. [Git and GitHub LFS Setup](#15-git-and-github-lfs-setup)
16. [How to Run Everything (Step-by-Step)](#16-how-to-run-everything-step-by-step)
17. [Streamlit Cloud Deployment](#17-streamlit-cloud-deployment)
18. [Final Results Summary](#18-final-results-summary)
19. [Image Conversion Utilities](#19-image-conversion-utilities)
20. [Dataset Integrity Notes](#20-dataset-integrity-notes)

---

## 1. Project Overview and Background

This project is an end-to-end deep learning system designed to classify raw meat images as either **Fresh** or **Spoiled**. It was built as an academic deliverable, beginning from raw data collection and going all the way through model training, rigorous evaluation, and final deployment as a publicly accessible web application.

The central use case is food safety. A consumer or retailer points a camera at a piece of meat and receives an instant AI-powered verdict — whether the meat is safe to eat or has spoiled. The system is designed to be highly sensitive to spoiled meat, meaning it is tuned to minimize false negatives (classifying spoiled meat as fresh), because the cost of that error is a health risk.

The entire pipeline — from data validation to training to deployment — has been implemented in Python using TensorFlow/Keras for the machine learning components, Streamlit for the web deployment, and Tkinter for the local desktop GUI.

---

## 2. Problem Statement

Raw meat spoilage is detectable through visual cues: color changes (fresh red meat turning brown, gray, or green), surface texture changes (slimy or dry), the presence of mold growth, and discoloration from oxidation. While a trained human eye can often identify these indicators, the goal of this project is to automate that detection using a convolutional neural network.

The key engineering challenge in this domain — unlike generic image classification — is that **color is the primary diagnostic signal**. A model that is casually trained on image augmentation techniques that randomly shift hues will learn incorrect biological markers. For example, artificially turning a bright red fresh steak to a brown-gray during training teaches the model the wrong patterns entirely. This constraint shaped every design decision in the augmentation pipeline.

A secondary challenge is **class imbalance**. The raw training dataset has approximately 762 Fresh images and 1,154 Spoiled images. If left unhandled, a naive model will learn to predict "Spoiled" most of the time and still achieve high raw accuracy due to the imbalance. The solution used here is **class-weight-based compensation** during training (described in detail in the training section).

---

## 3. Classification Decision: Binary Labels

### The original three-class dataset

The raw dataset sourced for this project originally contained three label categories:
- **Fresh** — bright red, firm texture, typical of recently processed meat.
- **Half-Fresh** — meat in an intermediate state, showing some color degradation but not outright spoilage.
- **Spoiled** — meat showing visible signs of spoilage: discoloration, possible mold, surface changes.

### Why Half-Fresh was merged into Spoiled

From a food safety perspective, the distinction between "Half-Fresh" and "Spoiled" is not meaningful — both represent meat that is **not safe to consume**. Presenting a three-class output (Fresh / Half-Fresh / Spoiled) would introduce ambiguity for the end user: what should someone do with a "Half-Fresh" result? The answer is the same as for "Spoiled" — do not eat it.

The decision was therefore made to collapse the classification into a **binary problem**: `Fresh` (safe to eat) vs `Spoiled` (not safe to eat). All Half-Fresh labeled images were merged into the Spoiled class at the data preparation stage by updating the `_classes.csv` label CSV files. This makes the model's output actionable and unambiguous.

This binary design is reflected throughout the entire codebase:
- `dataset_loader.py`: `LABEL_COLUMNS = ["Fresh", "Spoiled"]`
- `phase2_training.py`: 2-class softmax output head, `SparseCategoricalCrossentropy` loss
- `phase3_evaluation.py`: all metrics computed against 2 classes
- Both apps: output is either ✅ Fresh or ⚠️ Spoiled

---

## 4. Dataset Description

### Folder structure on disk

```
project-dataset/
├── train/
│   ├── _classes.csv          ← label manifest (1,916 rows + header = 1,917 lines)
│   ├── fresh/                ← 762 fresh meat images
│   └── spoiled/              ← 1,154 spoiled meat images (incl. merged Half-Fresh)
└── valid/
    ├── _classes.csv          ← 451 validation images
    ├── fresh/
    └── spoiled/
```

### _classes.csv format

Each CSV has three columns: `filename`, `Fresh`, `Spoiled`. Labels are one-hot encoded — exactly one of `Fresh` or `Spoiled` is `1` per row, the other is `0`.

```csv
filename,Fresh,Spoiled
FRESH-202-_JPG.rf.4f13e1b8c1559b63d140d294269709e1.jpg,1,0
SPOILED-207-_JPG.rf.4e9025c18a5ae21defb30242a670ab05.jpg,0,1
```

The filename column does not include the subfolder name — filenames are resolved to the correct `fresh/` or `spoiled/` subfolder based on the label value (implemented in `dataset_loader.py:_resolve_image_path`).

### Split sizes

| Split | Fresh | Spoiled | Total |
|-------|-------|---------|-------|
| Train | 762 | 1,154 | 1,916 |
| Validation | 178 | 273 | 451 |
| Holdout (manual test) | 32 | 14 | 46 |

### Class imbalance

The training set has approximately 1.51× more Spoiled images than Fresh. This is an inherent imbalance arising from the original dataset and the merging of Half-Fresh into Spoiled. This imbalance is **not corrected by removing images** — instead it is handled by applying **inverse-frequency class weights** during training. This ensures the model is penalized proportionally for errors on the minority (Fresh) class.

Computed class weights for training (approximate):
- **Fresh weight ≈ 1.515** (Fresh is underrepresented, so it gets a higher weight)
- **Spoiled weight ≈ 1.0** (Spoiled is the majority class, gets baseline weight)

### Holdout images

A small set of 46 manually curated images were placed in `holdout/` (organized into `holdout/Fresh/` and `holdout/Spoiled/` subfolders). These images were collected independently of the Roboflow-sourced training set and represent a "real-world" evaluation. The model was never trained or validated on them. The final holdout accuracy was **100%** (46/46 correct), which confirms the model's generalization ability.

### Newly added training images (during project iteration)

Four additional images were added to the training set during project development:
1. `project-dataset/train/fresh/fresh-beef-tenderloin-on-tray-ready-for-cooking-photo.jpg` — labeled Fresh
2. `project-dataset/train/spoiled/new sm.jpg` — labeled Spoiled
3. `project-dataset/train/spoiled/new sm 2.webp` — labeled Spoiled
4. `project-dataset/train/spoiled/vsm.jpg` — labeled Spoiled

These were added to `_classes.csv` manually (or via script), and the models were retrained to include them.

### Image formats in the dataset

Most images are JPEG. Some are WEBP. An AVIF file (`new sm.avif`) was encountered during development and was converted to JPEG using `fix_avif.py` before being added to the dataset. The training pipeline handles JPEG and PNG natively through `tf.io.decode_image` with `channels=3`. The inference pipeline adds a PIL-based fallback for unusual formats.

---

## 5. Data Strategy — The Color Trap

This is the most important design decision in the entire project. It is documented here at length because any future contributor who naively applies "standard" data augmentation will break the model's ability to detect spoilage.

### Why color matters more than usual

In most image classification tasks (e.g., cats vs. dogs, objects vs. background), the color of the subject is not the primary discriminating feature. The model can learn from shape, texture, and spatial patterns. For those tasks, randomly shifting the hue, saturation, and brightness of images during training is a common and highly effective regularization technique — it teaches the model to be color-invariant.

**Meat freshness classification is the opposite.** The primary biological markers of spoilage are:

- **Color shift from red to brown/gray/green:** Myoglobin in fresh meat (oxymyoglobin) gives it the bright red color. Spoilage bacteria produce by-products that oxidize the myoglobin, turning the meat brown (metmyoglobin), then gray, then green with severe spoilage.
- **Texture changes:** Slimy surface, collapse of muscle fibers (visible on close inspection).
- **Mold:** Green, white, or black fungal growth.

If you apply a random hue shift during training that changes a bright red fresh steak to brown-gray, you have just labeled a brown image as "Fresh" in the training set. The model then learns that brown meat is Fresh — the exact opposite of the truth. Similarly, shifting a spoiled gray image to a red hue teaches the model that red meat is Spoiled.

### What augmentations are used and why

The augmentation pipeline is deliberately conservative. Only **geometry and minor lightness** augmentations are applied:

| Augmentation | Value | Reason |
|---|---|---|
| Random horizontal flip | 50% probability | Meat orientation is arbitrary |
| Random vertical flip | 50% probability | Rotation invariance |
| Random 90° rotation | 0/90/180/270° | Meat can be at any angle |
| Random zoom (crop+resize) | 90%–100% scale | Handle varying frame distances |
| Random brightness | ±10% | Handle different lighting conditions |
| Random contrast | 90%–110% range | Handle different lighting conditions |

**Explicitly excluded:**
- Hue shifts (`tf.image.random_hue`) — would corrupt the color-based spoilage signal
- Saturation shifts (`tf.image.random_saturation`) — would change the vibrancy of the meat color
- Heavy color jitter — same reason

This policy is implemented in `phase2_training.py:build_augmenter()` and in the PyTorch-equivalent in `phase1_data_prep.py:build_transforms()` (where `ColorJitter` is called with `saturation=0.0, hue=0.0` explicitly).

---

## 6. Phase 1 — Data Validation

**Script:** `phase1_data_prep.py`  
**Run with:** `python phase1_data_prep.py`

### Purpose

Phase 1 is a data quality gate. It must be run before any training to confirm that:
1. The CSV files are correctly structured.
2. Every row in the CSV has exactly one label set to 1 (valid one-hot encoding).
3. No filenames are duplicated.
4. Every filename in the CSV resolves to an actual image file on disk.
5. Class counts and imbalance ratios are printed and understood.
6. Class weights are computed and displayed.
7. The augmentation pipeline produces tensors of the correct shape.

### What it checks and prints

**For each split (train and valid):**
- CSV row count
- Count of rows with invalid one-hot encoding (should be 0)
- Count of duplicate filenames (should be 0)
- Per-class counts and their ratios
- Count of missing image files (should be 0)
- Class weights (for training split only)

**Augmentation sanity check:**
- Loads 5 samples from each split using `FruitFreshnessDataset` with the full transform pipeline.
- Prints the tensor shape (should be `torch.Size([3, 224, 224])` for each image).
- Prints a label histogram of those 5 samples.

### Important note on Phase 1's framework

Phase 1 uses PyTorch (`torchvision.transforms`) for its transform pipeline demonstration. This is intentional — Phase 1 was written as a design document and exploration step before the final training was moved to TensorFlow. The `FruitFreshnessDataset` class in `dataset_loader.py` is a PyTorch `Dataset` with a graceful fallback (`Dataset = object`) when PyTorch is not installed. This class is **not used by the TensorFlow training pipeline** in Phase 2 — Phase 2 uses `tf.data` directly. However, the path resolution logic (`_resolve_image_path`) from `dataset_loader.py` IS imported and used by Phase 2.

If you do not have PyTorch installed, Phase 1 will fail because it imports `torch`. This is acceptable — Phase 1 is a diagnostic/exploration script, not a production requirement. Phases 2, 3, and 4 do not require PyTorch at all.

### The `_resolve_image_path` function

This function (`dataset_loader.py:89–103`) is the path resolver used by both Phase 1 and Phase 2. Given a `split` root directory, a `filename`, and the CSV row, it tries multiple candidate paths:

1. `<split_root>/<filename>` — file in the split root directly
2. `<split_root>/<class_folder>/<filename>` — derived from the one-hot label (e.g., `train/fresh/FRESH-001.jpg`)
3. `<split_root>/<prefix>/<filename>` — derived from the filename prefix (e.g., `FRESH-` maps to `fresh/`)
4. `<split_root>/fresh/<filename>` and `<split_root>/spoiled/<filename>` — exhaustive fallback

This makes the data loader resilient to datasets where files are organized in different ways. It returns the first path that exists, or `None` if no candidate resolves.

---

## 7. Phase 2 — Model Training

**Script:** `phase2_training.py`  
**Run with:** `python phase2_training.py`

### Overview

Phase 2 trains three transfer-learning models in sequence and saves the best checkpoint for each. At the end, it writes a summary CSV comparing all three models on validation metrics.

### The three architectures

All three models are loaded from TensorFlow's `tf.keras.applications` module with `weights="imagenet"`, meaning they start with weights pre-trained on ImageNet (1.2 million images, 1,000 classes). This gives the models a powerful feature extraction capability out of the box, particularly for texture and color gradients, which are exactly the features needed for meat freshness detection.

**1. MobileNetV2 (The Baseline)**
- Architecture: Inverted residuals with linear bottlenecks, depthwise separable convolutions.
- Parameters: ~2.2M (very lightweight).
- Purpose: Establishes a fast, low-resource baseline. Optimized for mobile devices.
- Performance: Validation accuracy 86.25%, macro F1 0.855, ROC-AUC 0.945.
- Verdict: Acceptable but clearly weaker than the alternatives. Higher misclassification rates across both classes.

**2. ResNet50 (The Industry Standard) — SELECTED**
- Architecture: Deep residual network with 50 layers and skip connections that prevent vanishing gradients.
- Parameters: ~23.5M.
- Purpose: The industry-standard feature extractor for complex visual tasks.
- Performance: Validation accuracy 99.33%, macro F1 0.993, ROC-AUC 0.9998.
- Verdict: Overwhelmingly the best performer. Only 3 misclassifications (all 3 were Spoiled predicted as Fresh — no Fresh items were incorrectly labeled Spoiled) out of 451 validation images.

**3. EfficientNetB0 (The Modern Challenger)**
- Architecture: Compound scaling (depth + width + resolution scaled together), MBConv blocks.
- Parameters: ~4.0M.
- Purpose: The modern efficient architecture, expected to balance MobileNetV2 and ResNet50.
- Performance: Validation accuracy 96.23%, macro F1 0.960, ROC-AUC 0.993.
- Verdict: Very good, but consistently below ResNet50 on every metric.

### The model architecture (all three follow the same pattern)

```
Input: 224×224×3 image tensor (ImageNet-normalized, float32)
  ↓
Pre-trained base (MobileNetV2 / ResNet50 / EfficientNetB0)
  base.trainable = False  ← all ImageNet weights frozen
  pooling = "avg"  ← Global Average Pooling at the end of the base
  ↓
Dense(256, activation="relu")  ← custom classification head
  ↓
Dropout(0.4)  ← 40% dropout for regularization
  ↓
Dense(2, activation="softmax")  ← 2-class output: [P(Fresh), P(Spoiled)]
```

The base model weights are **frozen** (not updated during training). Only the custom head (Dense + Dropout + Dense) is trained. This is called "feature extraction" style transfer learning, as opposed to "fine-tuning" (where the top layers of the base are also unfrozen). Freezing the base is appropriate here because:
- The dataset is relatively small (~1,916 training images).
- ImageNet features (edges, textures, colors) are still highly relevant.
- Fine-tuning on a small dataset risks overfitting.

### Training configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam, learning rate = 1e-4 |
| Loss function | SparseCategoricalCrossentropy |
| Batch size | 32 |
| Max epochs | 30 |
| Early stopping patience | 5 (monitors `val_loss`) |
| EarlyStopping behavior | Restores best weights when stopping |
| ModelCheckpoint | Saves only the best epoch (`val_loss` minimized) |
| Class weights | Inverse-frequency (Fresh ≈ 1.515, Spoiled ≈ 1.0) |
| Input size | 224×224 pixels |
| ImageNet normalization | `imagenet_utils.preprocess_input(img * 255.0)` → maps to [-1, 1] range |

**Why Adam at 1e-4?** A low learning rate is intentional. With pre-trained ImageNet weights frozen, only the custom head (2 dense layers) is being updated. Using a high learning rate here can destabilize the training and fail to converge to a good optimum. 1e-4 is a standard safe choice for transfer learning fine-tuning of a classification head.

**Why SparseCategoricalCrossentropy?** Labels are stored as integers (0 = Fresh, 1 = Spoiled) rather than one-hot vectors inside the `tf.data` pipeline. `SparseCategoricalCrossentropy` accepts integer labels directly. The CSV uses one-hot for storage, but the training pipeline converts this to integers via `build_path_label_pairs()`.

### Augmentation in the training pipeline

The `build_augmenter()` function in `phase2_training.py` returns a callable that applies augmentations directly to `tf.Tensor` objects (not PIL images). This is important — it keeps the augmentation inside the `tf.data` pipeline for GPU-accelerated preprocessing.

The augmentations applied during training only (validation data is **not** augmented):
1. `random_flip_left_right` (seed=42)
2. `random_flip_up_down` (seed=42)
3. `rot90` with random k ∈ {0, 1, 2, 3} (seed=42)
4. Random crop (scale 0.90–1.00) then resize back to 224×224
5. `random_brightness` max_delta=0.10 (seed=42)
6. `random_contrast` lower=0.90, upper=1.10 (seed=42)

Seeds are fixed at `SEED = 42` for reproducibility.

### ImageNet normalization

The preprocessing function `preprocess_image()` decodes the image bytes using `tf.io.decode_image` (handles JPEG/PNG/grayscale/alpha) and resizes to 224×224 as float32 in [0, 1]. Before feeding to the model, the pipeline then calls:

```python
tf.keras.applications.imagenet_utils.preprocess_input(img * 255.0)
```

This maps the [0, 1] float to [0, 255] and then applies the standard ImageNet normalization used by all three architectures: subtract the ImageNet mean channel-wise and divide by standard deviation (mode `"caffe"` for ResNet50 and MobileNetV2, mode `"torch"` for EfficientNetB0 — `imagenet_utils` selects the correct mode based on the model's architecture automatically). The output range is approximately [-1, 1] to [-128, 128] depending on the model.

### Output files from Phase 2

For each trained model:
- `phase2_models/<ModelName>/best_model.keras` — the saved Keras model weights (tracked by Git LFS)
- `phase2_models/<ModelName>/eval_report.json` — a JSON file with the full classification report, confusion matrix, and ROC-AUC score computed on the validation set immediately after training

Additionally:
- `phase2_models/summary.csv` — one row per model with `model`, `best_val_loss`, `best_val_accuracy`, `roc_auc`, `confusion_matrix`, `classification_report`

### `.keras` vs `.h5` format

Model checkpoints are saved in TensorFlow's modern native `.keras` format (introduced in TF 2.12). This format is preferred over the legacy HDF5 `.h5` format because:
- It is fully self-describing.
- It supports custom objects more reliably.
- It is the format expected by Streamlit Cloud (TF 2.21+ only supports `.keras` for `model.save()`).

A conversion utility is available: `python phase2_training.py --convert-only` will look for any `.h5` files in `phase2_models/` and convert them to `.keras` without retraining.

---

## 8. Phase 3 — Evaluation

**Script:** `phase3_evaluation.py`  
**Run with:** `python phase3_evaluation.py --all` or `--best` or `--model ResNet50`

### Purpose

Phase 3 loads the saved model checkpoints from `phase2_models/`, runs inference on the validation set, computes all evaluation metrics from scratch (independent of what was stored in `eval_report.json`), and generates publication-quality plots and a markdown report.

### Usage modes

```bash
# Evaluate all models and generate reports for each
python phase3_evaluation.py --all

# Find the best model (by macro F1) and generate its report + write selected_model.txt
python phase3_evaluation.py --best

# Evaluate one specific model by name
python phase3_evaluation.py --model ResNet50
```

The `--best` flag does two things:
1. Reads all existing `eval_report.json` files and picks the model with the highest `classification_report → macro avg → f1-score`.
2. Writes `phase3_reports/selected_model.txt` containing just the model name (e.g., `ResNet50`). This file is read by both inference apps to know which model to load.

### Metrics computed

**Classification Report (per class):**
- **Precision** — Of all images predicted as Fresh/Spoiled, how many actually are?
- **Recall** — Of all actually Fresh/Spoiled images, how many did the model correctly identify?
- **F1-Score** — Harmonic mean of precision and recall. The primary ranking metric.
- **Support** — Number of true instances of that class in the validation set.

**Why Spoiled Recall is the safety-critical metric:**
A false negative on Spoiled (predicting "Fresh" when the meat is actually Spoiled) means someone consumes unsafe meat. This is the most dangerous possible error. The confusion matrix and Spoiled Recall are therefore the primary safety checks, and any model with poor Spoiled Recall would be rejected regardless of overall accuracy.

**Macro-average F1:**
Computed as the simple average of Fresh F1 and Spoiled F1 (not weighted by class support). This is the **primary model selection criterion** because it treats both classes equally and is not inflated by the Spoiled class majority.

**ROC-AUC (multi-class OvO):**
Area under the Receiver Operating Characteristic curve. A score of 1.0 means perfect separation between classes at all thresholds. Computed using sklearn's `roc_auc_score` with `multi_class="ovo"` (One-vs-One).

### Output files

For each evaluated model, reports are written to `phase3_reports/<ModelName>/`:

1. **`confusion_matrix.png`** — A seaborn heatmap of the 2×2 confusion matrix (rows = true class, columns = predicted class). Annotated with raw counts.
2. **`roc_curve.png`** — Per-class ROC curves plotted on the same axes, with AUC scores in the legend and a diagonal reference line (random classifier).
3. **`report.md`** — A markdown-formatted table of precision/recall/F1/support per class, plus macro F1 and accuracy.

Additionally: `phase3_reports/selected_model.txt` is written when `--best` is used.

### Results for all three models

#### MobileNetV2

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Fresh | 0.837 | 0.809 | 0.823 | 178 |
| Spoiled | 0.878 | 0.897 | 0.888 | 273 |

- Macro F1: 0.855
- Accuracy: 86.25%
- ROC-AUC: 0.945
- Confusion matrix: 144 Fresh correct, 34 Fresh predicted Spoiled (false negative), 28 Spoiled predicted Fresh (false positive), 245 Spoiled correct.
- **Assessment:** 28 dangerous false negatives (Spoiled called Fresh). Unacceptable for food safety use.

#### EfficientNetB0

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Fresh | 0.971 | 0.933 | 0.951 | 178 |
| Spoiled | 0.957 | 0.982 | 0.969 | 273 |

- Macro F1: 0.960
- Accuracy: 96.23%
- ROC-AUC: 0.993
- Confusion matrix: 166 Fresh correct, 12 Fresh predicted Spoiled, 5 Spoiled predicted Fresh, 268 Spoiled correct.
- **Assessment:** Good, but still 5 dangerous Spoiled-as-Fresh misclassifications. Beaten by ResNet50 on every metric.

#### ResNet50 — **SELECTED**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Fresh | 0.983 | 1.000 | 0.992 | 178 |
| Spoiled | 1.000 | 0.989 | 0.994 | 273 |

- Macro F1: 0.993
- Accuracy: 99.33%
- ROC-AUC: 0.9998
- Confusion matrix: 178 Fresh correct (zero false negatives on Fresh), 0 Fresh predicted Spoiled, 3 Spoiled predicted Fresh (false negatives), 270 Spoiled correct.
- **Assessment:** The 3 remaining Spoiled-as-Fresh errors represent edge cases. All other metrics are near-perfect. Selected as the production model.

---

## 9. Phase 3 — Inference CLI

**Script:** `phase3_inference.py`  
**Run with:** `python phase3_inference.py --image <path> --best`

### Purpose

A command-line tool for running predictions on arbitrary images or batches of images without launching a full GUI. Useful for batch testing and for integration into other scripts.

### Usage

```bash
# Classify a single image using the best model
python phase3_inference.py --image path/to/meat.jpg --best

# Classify a single image using a specific model
python phase3_inference.py --image path/to/meat.jpg --model ResNet50

# Classify an entire folder of images (no labels needed)
python phase3_inference.py --dir path/to/images --best
```

### Model resolution logic

The function `_find_best_model_name()` determines which model to load:
1. First checks if `phase3_reports/selected_model.txt` exists and reads it.
2. If not, falls back to reading all `eval_report.json` files in `phase2_models/` and picking the one with the highest macro F1.

This means after running `python phase3_evaluation.py --best`, the inference CLI will automatically use ResNet50 without needing to specify `--model ResNet50`.

### Preprocessing pipeline for inference

The `preprocess_path()` function in `phase3_inference.py`:
1. Reads the image file bytes using `tf.io.read_file`.
2. Calls `preprocess_image()` from `phase2_training.py` (the same function used during training, ensuring identical preprocessing).
3. Applies `imagenet_utils.preprocess_input(img * 255.0)` — the identical ImageNet normalization used during training.
4. Falls back to PIL/Pillow if TensorFlow's decoder fails (handles unusual formats like AVIF, BMP, TIFF).

This strict reuse of the training preprocessing is critical. Any difference between training-time and inference-time preprocessing would degrade model performance.

### Output format

```
path/to/meat.jpg
  Fresh     :  0.5%
  Spoiled   : 99.5%
  -> Predicted: Spoiled (99.5%)
```

---

## 10. Phase 3 — Holdout Evaluation

**Script:** `phase3_holdout.py`  
**Run with:** `python phase3_holdout.py --dir holdout --best`

### Purpose

The holdout set is a small collection of images that were **never used for training or validation**. They were manually curated from independent sources. Running the model on these images provides the most realistic estimate of how well it will perform on real-world, unseen images.

### Folder structure requirement

For accuracy and confusion matrix computation, holdout images must be organized into class-named subfolders:

```
holdout/
├── Fresh/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Spoiled/
    ├── image1.jpg
    └── ...
```

If no subfolders are present (just a flat folder of images), the script will still run inference but cannot compute accuracy since there are no ground truth labels.

### Holdout results

The holdout set contained 46 images: 32 Fresh, 14 Spoiled.

| | Predicted Fresh | Predicted Spoiled |
|---|---|---|
| **True Fresh** | 32 | 0 |
| **True Spoiled** | 0 | 14 |

**Accuracy: 100.0% (46/46 correct)**

This result, combined with the 99.33% validation accuracy, provides very strong confidence in the model's generalization ability. The holdout images included a range of meat types (beef, pork, poultry), lighting conditions, and image qualities. All were classified correctly with high confidence — most Fresh images were classified at 99.5–100%, and Spoiled images ranged from 58% to 100% confidence (lower confidence on borderline cases, which is expected and correct behavior).

The holdout report is saved at `phase3_reports/holdout_report.md`.

---

## 11. Phase 4 — Streamlit Web App

**Script:** `app.py`  
**Run locally:** `streamlit run app.py`  
**Deploy:** Push to GitHub → connect at [share.streamlit.io](https://share.streamlit.io)

### Overview

`app.py` is the primary deployment artifact. It is a Streamlit web application that provides a user-facing interface to the trained model. It runs in any modern web browser, works on both desktop and mobile devices, and is hosted for free on Streamlit Cloud.

### UI structure

The app has a single page with two tabs:

**Tab 1 — 📂 Upload Image**
- A file uploader widget accepting JPEG, PNG, WEBP, BMP, and TIFF files.
- When an image is uploaded, it is displayed in the left column.
- The right column shows the classification result (verdict + probability bars).

**Tab 2 — 📷 Take a Photo**
- Uses Streamlit's `st.camera_input` widget, which accesses the device's camera directly through the browser.
- On a laptop: uses the webcam.
- On a phone: can use the front or rear camera.
- No OpenCV or any native library is required for this — it is entirely browser-based.

### Image preprocessing in the app

`app.py` does not call the model directly with the PIL image. Instead, it performs the following preprocessing:

1. **Alpha compositing:** If the image has transparency (RGBA, P, PA, or LA mode), it is composited onto a white background. This handles PNG files with transparent backgrounds and WEBP images with alpha channels.
2. **RGB conversion:** The image is converted to RGB mode (drops the alpha channel if present, converts grayscale to 3 channels).
3. **Temporary file:** The processed PIL image is saved as a temporary JPEG file on disk.
4. **Inference:** `phase3_inference.predict()` is called with the temporary file path, which applies the full training-identical preprocessing pipeline.
5. **Cleanup:** The temporary file is deleted after inference, even if an exception occurs (the `finally` block ensures this).

This pattern ensures the inference pipeline is not duplicated — `app.py` delegates all preprocessing to `phase3_inference.py`, which in turn uses the exact same functions as training.

### Result display

- If Fresh probability ≥ Spoiled probability: **green success banner** with `✅ FRESH — X.X% confidence`
- If Spoiled probability > Fresh probability: **red error banner** with `⚠️ SPOILED — X.X% confidence`
- Two progress bars show both probabilities side by side (Fresh and Spoiled).

### Model caching

```python
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    ...
```

The `@st.cache_resource` decorator ensures the model is loaded only once per Streamlit session (or server process restart). Loading a ResNet50 model (~90 MB) takes approximately 5–15 seconds on first load. After the first inference, subsequent classifications are near-instant because the model stays in memory.

### Required files for Streamlit Cloud deployment

The following files must be present in the GitHub repository for Streamlit Cloud to deploy successfully:

```
app.py                                  ← entry point
requirements.txt                        ← dependencies
runtime.txt                             ← Python 3.12 pin
dataset_loader.py                       ← LABEL_COLUMNS, LABEL_MAP, _resolve_image_path
phase2_training.py                      ← IMAGE_SIZE, LABEL_COLUMNS, preprocess_image
phase3_inference.py                     ← load_model, predict, _find_best_model_name
phase2_models/
    ResNet50/
        best_model.keras                ← THE MODEL — must be pushed via Git LFS
    summary.csv
phase3_reports/
    selected_model.txt                  ← "ResNet50" — tells the app which model to use
    ResNet50/
        eval_report.json
```

The training images in `project-dataset/` are **not** needed for the deployed app. The app only needs the trained model weights and the code files.

---

## 12. Phase 4 — Tkinter Desktop App

**Script:** `app_desktop.py`  
**Run with:** `python app_desktop.py`

### Overview

`app_desktop.py` is a local desktop GUI application built with Python's built-in `tkinter` library. It provides the same meat freshness classification functionality as the Streamlit app but runs entirely offline on a local machine, with no internet connection required after the first setup.

### UI design

The UI uses a dark theme (`#1e1e2e` background, similar to VS Code's dark theme). It has three sections:

1. **Image preview panel (400×400 px):** Displays the loaded image, thumbnailed to fit.
2. **Button row:**
   - `📂 Upload Image` — opens a file dialog to select any image from disk (JPEG, PNG, WEBP, AVIF, BMP, TIFF).
   - `📷 Capture from Camera` — opens a live OpenCV webcam window.
3. **Result panel:**
   - Large label showing ✅ Fresh (green) or ❌ Spoiled (red) with the confidence percentage.
   - Horizontal probability bar for each class.
4. **Status bar:** Shows the current state (e.g., "Loaded: my_image.jpg — classifying…", "Classified as Fresh (99.5%)").

### Camera capture workflow

Camera functionality requires `opencv-python` to be installed (`pip install opencv-python`). If OpenCV is not installed, clicking the camera button shows an informational message box with the install command instead of throwing an error.

When OpenCV is available:
1. `cv2.VideoCapture(0)` opens the default webcam.
2. A live preview window is displayed: `"Camera — SPACE: capture  |  ESC: cancel"`.
3. The user presses **SPACE** to capture the current frame, or **ESC** to cancel.
4. The captured OpenCV frame (BGR) is converted to RGB and then to a PIL Image.
5. Classification runs automatically.

### Model loading

The app uses lazy model loading — `_get_model()` is called on the first classification, not on startup. This keeps the app window responsive while the model loads on demand. The model is cached in a module-level global variable (`_model`) so subsequent classifications reuse the loaded model.

### Image preprocessing

Identical to the Streamlit app:
1. Transparent images are composited onto white.
2. Converted to RGB.
3. Saved to a temporary JPEG file.
4. `phase3_inference.predict()` is called.
5. Temporary file deleted.

---

## 13. Repository and File Structure

```
Chinelotam/                              ← project root
│
├── app.py                               ← Streamlit web app (main deployment file)
├── app_desktop.py                       ← Tkinter desktop app
│
├── phase1_data_prep.py                  ← Phase 1: data validation and sanity checks
├── phase2_training.py                   ← Phase 2: model training (3 architectures)
├── phase3_evaluation.py                 ← Phase 3: evaluation reports and model selection
├── phase3_inference.py                  ← Phase 3: inference CLI and shared predict() function
├── phase3_holdout.py                    ← Phase 3: holdout/test set evaluation
│
├── dataset_loader.py                    ← Shared: label map, path resolver, PyTorch Dataset
├── generate_model_selection_justification.py  ← Generates model_selection_justification.md
│
├── requirements.txt                     ← Python dependencies (pinned versions)
├── runtime.txt                          ← Python version pin for Streamlit Cloud (python-3.12)
├── .python-version                      ← Python version pin for local tooling (3.12)
├── .gitattributes                       ← Git LFS tracking: *.keras files tracked via LFS
├── .gitignore                           ← Ignored: *.zip, __pycache__, *.pyc, *.bak, .claude/, logs
│
├── convert_avif_to_jpg.py               ← Utility: batch convert any image format to JPEG
├── fix_avif.py                          ← Utility: convert a single AVIF file to JPEG
├── rebalance_dataset.py                 ← Utility: (historical) moved HALF-FRESH images in/out
│
├── COMMANDS.md                          ← Quick command reference for all scripts
├── how-to.md                            ← Phase-by-phase explanation of the project
├── model_selection_justification.md     ← Why ResNet50 was selected
├── PROJECT_DOCUMENTATION.md            ← This file — comprehensive project knowledge base
├── tasks_completed_uncompleted.md       ← Phase completion status
│
├── phase2_models/                       ← Trained model checkpoints and evaluation JSONs
│   ├── summary.csv                      ← One row per model with all validation metrics
│   ├── EfficientNetB0/
│   │   ├── best_model.keras             ← EfficientNetB0 weights (Git LFS)
│   │   └── eval_report.json             ← EfficientNetB0 validation metrics
│   ├── MobileNetV2/
│   │   ├── best_model.keras             ← MobileNetV2 weights (Git LFS)
│   │   └── eval_report.json             ← MobileNetV2 validation metrics
│   └── ResNet50/
│       ├── best_model.keras             ← ResNet50 weights — the production model (Git LFS)
│       └── eval_report.json             ← ResNet50 validation metrics
│
├── phase3_reports/                      ← Evaluation plots and reports
│   ├── selected_model.txt               ← Contains "ResNet50" — the deployed model name
│   ├── holdout_report.md                ← 100% accuracy on 46 holdout images
│   ├── EfficientNetB0/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── report.md
│   ├── MobileNetV2/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── report.md
│   └── ResNet50/
│       ├── confusion_matrix.png         ← 178/0 Fresh, 3/270 Spoiled
│       ├── roc_curve.png                ← AUC 0.9998
│       └── report.md
│
├── project-dataset/                     ← Training and validation images
│   ├── train/
│   │   ├── _classes.csv                 ← 1,916 labeled training images
│   │   ├── _classes.csv.bak             ← Backup of CSV (ignored by git)
│   │   ├── fresh/                       ← 762 fresh meat images
│   │   └── spoiled/                     ← 1,154 spoiled meat images
│   └── valid/
│       ├── _classes.csv                 ← 451 labeled validation images
│       ├── fresh/
│       └── spoiled/
│
└── holdout/                             ← Manually curated unseen test images (46 total)
    ├── Fresh/                           ← 32 fresh images
    └── Spoiled/                         ← 14 spoiled images
```

---

## 14. Dependencies and Requirements

All dependencies are listed in `requirements.txt` with version constraints. The Python version is pinned to **3.12** (via `runtime.txt` and `.python-version`) because TensorFlow 2.21 does not have Python 3.13+ wheels at the time of this project.

### Core ML dependencies

| Package | Version | Role |
|---|---|---|
| `tensorflow` | `==2.21.0` | Model architecture, training, inference |
| `numpy` | `==2.4.3` | Numerical arrays (numpy 2.x required — do NOT downgrade) |
| `pandas` | `>=2.2.0` | CSV loading and DataFrame operations |
| `scikit-learn` | `>=1.4.0` | `classification_report`, `roc_auc_score`, `confusion_matrix` |
| `Pillow` | `==11.3.0` | Image loading and preprocessing in inference |
| `pillow-avif-plugin` | latest | Only needed when converting .avif images |
| `matplotlib` | `>=3.9.0` | Plotting confusion matrix and ROC curves (3.9+ for numpy 2.x) |
| `seaborn` | `>=0.13.2` | Heatmap style for confusion matrix |
| `streamlit` | `>=1.30.0` | Web app |

### Why numpy 2.x is required and must not be downgraded

The `requirements.txt` includes a comment warning: **numpy 2.x required — do NOT downgrade**. The reason is that numpy 2.0 introduced a breaking ABI change. Several packages that are installed from binary wheels (OpenCV, matplotlib, etc.) were compiled against numpy 2.x. Downgrading numpy to 1.x causes import errors for those packages due to the ABI mismatch. The entire dependency set — tensorflow, matplotlib, scikit-learn, pandas — is pinned to versions that are compatible with numpy 2.4.x.

### OpenCV (optional)

`opencv-python` is **not** listed in `requirements.txt` because it is not required for the Streamlit app or training scripts. It is only needed for the **camera capture** feature of the desktop Tkinter app (`app_desktop.py`). If you want to use the camera feature of the desktop app, install it manually:

```bash
pip install opencv-python
```

The desktop app handles the absence of OpenCV gracefully — clicking the camera button shows an install prompt instead of crashing.

### PyTorch (only for Phase 1)

`torch` and `torchvision` are not listed in `requirements.txt` because they are only used by `phase1_data_prep.py` (the data exploration script). The production pipeline (training, evaluation, inference, apps) is entirely TensorFlow-based. The `dataset_loader.py` guards the torch import:

```python
try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object  # fallback when torch is not installed
```

This means `dataset_loader.py` can be imported safely in a TensorFlow-only environment.

---

## 15. Git and GitHub LFS Setup

### Why Git LFS is required

The trained model checkpoint files (`best_model.keras`) are large binary files. Specifically:
- `MobileNetV2/best_model.keras` — approximately 9 MB
- `EfficientNetB0/best_model.keras` — approximately 16 MB
- `ResNet50/best_model.keras` — approximately 90–100 MB

GitHub has a **100 MB hard limit per file**. The ResNet50 checkpoint is close to or exceeds this limit. Even below the limit, GitHub's performance degrades with large binary files (they are stored inefficiently in git's object store). Git LFS (Large File Storage) is the standard solution — it stores the binary content in GitHub's LFS storage and keeps a small pointer file in the git history.

### How LFS is configured in this project

The `.gitattributes` file at the root of the repository contains:

```
*.keras filter=lfs diff=lfs merge=lfs -text
```

This tells git to route any file matching `*.keras` through the LFS filter. This was set up by running:

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
```

**You do not need to run these setup commands again** — the `.gitattributes` file is already committed and the LFS tracking is already configured.

### Verifying LFS status

```bash
git lfs version        # Check LFS is installed (should print version)
git lfs status         # Shows which files are tracked by LFS vs git
git lfs ls-files       # Lists all LFS-tracked files in the repository
```

In `git lfs status` output, files tracked by LFS show `(LFS: <hash> -> File: <hash>)` instead of `(Git: <hash> -> File: <hash>)`.

### Pushing with LFS (complete workflow)

After making changes, the full push workflow is:

```bash
# Stage all changes
git add .gitignore
git add phase2_training.py
git add phase2_models/
git add phase3_reports/ResNet50/
git add project-dataset/train/_classes.csv
git add how-to.md tasks_completed_uncompleted.md
git add fix_avif.py rebalance_dataset.py
git add "project-dataset/train/spoiled/new sm.jpg"
git add "project-dataset/train/spoiled/new sm 2.webp"
git add project-dataset/train/spoiled/vsm.jpg
git add project-dataset/train/fresh/fresh-beef-tenderloin-on-tray-ready-for-cooking-photo.jpg
git add PROJECT_DOCUMENTATION.md

# Commit
git commit -m "your commit message"

# Push (git automatically uses LFS for *.keras files)
git push origin main
```

LFS upload happens transparently during `git push`. Git will first push the LFS objects (model files) to the LFS storage endpoint, then push the regular git objects. You will see output like:

```
Uploading LFS objects: 100% (3/3), 120 MB | 2.5 MB/s, done.
```

### What is in .gitignore

```
*.zip              ← The Chinelotam.zip archive is excluded (too large, not needed)
__pycache__/       ← Python bytecode cache directories
*.pyc              ← Compiled Python bytecode files
*.bak              ← Backup files (e.g., _classes.csv.bak)
.claude/           ← Claude Code IDE internal state (not project code)
push.txt           ← Personal notes about push commands
Q-A.md             ← Personal Q&A notes
logs-*.txt         ← Streamlit log files
```

Note: `project-dataset/` images are tracked by git (not in .gitignore). The training and validation image files were committed as part of the repository. Only non-essential or privacy-sensitive files are excluded.

---

## 16. How to Run Everything (Step-by-Step)

### Prerequisites

1. Python 3.12 installed (check with `python --version`).
2. Git LFS installed (check with `git lfs version`). Download from [git-lfs.github.com](https://git-lfs.github.com) if needed.
3. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Chinelotam
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 1: Validate the dataset

```bash
python phase1_data_prep.py
```

Expected output: CSV row counts, class counts, class weights, zero missing files, zero invalid one-hot rows, sample tensor shapes of `(3, 224, 224)`.

### Step 2: Train all models

```bash
python phase2_training.py
```

This will train MobileNetV2, ResNet50, and EfficientNetB0 in sequence. Each model trains for up to 30 epochs with early stopping. On a modern GPU, each model takes approximately 5–20 minutes. On CPU, this can take several hours. The script will print validation loss and accuracy at the end of each epoch.

Outputs:
- `phase2_models/MobileNetV2/best_model.keras`
- `phase2_models/ResNet50/best_model.keras`
- `phase2_models/EfficientNetB0/best_model.keras`
- `phase2_models/<ModelName>/eval_report.json` (for each model)
- `phase2_models/summary.csv`

### Step 3: Select the best model and generate reports

```bash
python phase3_evaluation.py --best
```

This will:
1. Read all `eval_report.json` files and find the model with the highest macro F1 (will be ResNet50).
2. Write `phase3_reports/selected_model.txt` containing `ResNet50`.
3. Generate `phase3_reports/ResNet50/confusion_matrix.png`, `roc_curve.png`, and `report.md`.

### Step 4 (optional): Evaluate all models

```bash
python phase3_evaluation.py --all
```

Generates evaluation artifacts for all three models in their respective subfolders.

### Step 5 (optional): Run the holdout evaluation

```bash
python phase3_holdout.py --dir holdout --best
```

Runs inference on all 46 holdout images and writes a report to `phase3_reports/holdout_report.md`.

### Step 6: Test inference on a single image

```bash
python phase3_inference.py --image path/to/your/image.jpg --best
```

### Step 7: Launch the web app locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your default browser. Upload an image or use the camera tab.

### Step 8: Launch the desktop app

```bash
python app_desktop.py
```

Opens a Tkinter window. Click "Upload Image" to select a file, or "Capture from Camera" if OpenCV is installed.

---

## 17. Streamlit Cloud Deployment

### One-time deployment setup

1. Ensure all required files (listed in Section 11) are committed and pushed to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Sign in with your GitHub account.
4. Click "New app".
5. Select the repository, the branch (`main`), and set the main file path to `app.py`.
6. Click "Deploy".

Streamlit Cloud will:
- Read `runtime.txt` to select Python 3.12.
- Run `pip install -r requirements.txt`.
- Start the Streamlit server with `streamlit run app.py`.

The app will be available at a URL like `https://<yourapp>.streamlit.app`.

### Redeployment after pushing changes

Streamlit Cloud automatically redeploys whenever you push to the connected branch. There is no manual step needed — push your changes to `main`, and within a few minutes the live app will reflect the update.

If the redeployment does not trigger automatically, you can force it by going to the app's management page on Streamlit Cloud and clicking "Reboot app".

### Important: Streamlit Cloud and Git LFS

Streamlit Cloud **supports Git LFS**. When it clones your repository, it will also fetch the LFS objects (the `.keras` model files). This is why the `.gitattributes` file must be committed to the repository — it tells git (and Streamlit's cloning process) which files to fetch from LFS.

If the `.keras` files are not in LFS and are simply too large for GitHub, the push will be rejected. If they are in LFS but the `.gitattributes` is not committed, the LFS pointer file (a small text file) will be cloned instead of the actual model weights, and the app will fail with a model loading error.

### How the app knows which model to use

`phase3_reports/selected_model.txt` contains the single word `ResNet50`. When `app.py` loads:
1. `get_model()` is called.
2. It calls `_find_best_model_name()` from `phase3_inference.py`.
3. That function reads `phase3_reports/selected_model.txt` and returns `"ResNet50"`.
4. `load_model("ResNet50")` loads `phase2_models/ResNet50/best_model.keras`.

This indirection means you could swap to a different model by simply updating `selected_model.txt` without changing any app code.

---

## 18. Final Results Summary

### Model comparison (validation set, 451 images)

| Model | Val Accuracy | Macro F1 | ROC-AUC | Spoiled Recall | Fresh Recall |
|---|---|---|---|---|---|
| MobileNetV2 | 86.25% | 0.855 | 0.945 | 89.7% | 80.9% |
| EfficientNetB0 | 96.23% | 0.960 | 0.993 | 98.2% | 93.3% |
| **ResNet50** | **99.33%** | **0.993** | **0.9998** | **98.9%** | **100%** |

**ResNet50 confusion matrix (validation set):**

|  | Predicted Fresh | Predicted Spoiled |
|---|---|---|
| **True Fresh** | 178 | 0 |
| **True Spoiled** | 3 | 270 |

Zero Fresh images were incorrectly labeled Spoiled. Only 3 Spoiled images slipped through as Fresh — a 98.9% Spoiled recall.

### Holdout set (46 completely unseen images)

| | Predicted Fresh | Predicted Spoiled |
|---|---|---|
| **True Fresh** | 32 | 0 |
| **True Spoiled** | 0 | 14 |

**Holdout Accuracy: 100% (46/46)**

### Key design choices validated by results

1. **No hue/saturation augmentation** — Correct. The model learned genuine biological color markers of spoilage, not artifacts of augmented colors.
2. **Class weighting instead of resampling** — Correct. The model performs well on both classes despite the 1:1.51 imbalance.
3. **ResNet50 over EfficientNetB0** — Correct. The 3.1 percentage point accuracy gap on validation, combined with 0 vs 5 dangerous false negatives, makes ResNet50 the clear choice.
4. **Transfer learning (frozen base)** — Correct. ImageNet pre-training gave the model strong texture and color feature detectors that transferred directly to the meat freshness domain.

---

## 19. Image Conversion Utilities

### `convert_avif_to_jpg.py`

A batch conversion utility. Given a folder path, it converts every image in that folder (regardless of format — AVIF, PNG, WEBP, BMP, TIFF) to JPEG with a white background. White background compositing handles transparent images (PNG with alpha, WEBP with alpha) by filling the transparent areas with white instead of black.

```bash
python convert_avif_to_jpg.py path/to/folder
```

The JPEG files are written to the same directory with the `.jpg` extension. The original files are preserved.

### `fix_avif.py`

A single-file version of the above. Used during development when one specific AVIF file (`new sm.avif`) needed to be converted to JPEG to be included in training. Outputs `new sm.jpg` next to the input file.

### Why these utilities exist

During dataset curation, some images were obtained in AVIF format. AVIF is a modern image format (based on AV1 codec) that is not natively supported by `tf.io.decode_image`. Attempting to add AVIF files to the dataset directly would cause the training pipeline to fail. Converting to JPEG first was the simplest solution.

The `pillow-avif-plugin` package in `requirements.txt` provides AVIF decoding capability to Pillow, which these utilities use.

---

## 20. Dataset Integrity Notes

### The rebalance operation (historical — reverted)

During development, an experiment was attempted where 392 HALF-FRESH labeled images were moved out of `project-dataset/train/spoiled/` into a temporary folder `project-dataset/train/_excluded_half_spoiled/`. The intent was to create a 1:1 class balance (762 Fresh / 762 Spoiled). However, this approach was abandoned for the following reasons:

1. Class weighting already handles imbalance without discarding data.
2. Removing 392 data points reduces the training set by ~20%.
3. The 392 excluded images contain valid visual spoilage patterns that the model should learn.
4. The experiment was reversed: all 392 images were moved back to `train/spoiled/`, and `_classes.csv` was restored to its full 1,916-image state.

The script `rebalance_dataset.py` was used for this operation and is preserved in the repository as a historical artifact. It should not be run again without a deliberate decision to revisit the rebalance experiment.

### `_classes.csv.bak`

This is a backup of the `_classes.csv` file taken before any modifications. It is excluded from git via `.gitignore` (`*.bak`). Its purpose is to allow restoration of the original CSV if manual edits to the label file go wrong. The `.bak` file is always kept on disk but never committed to the repository.

### File naming in the dataset

The bulk of the dataset files are named with a structured convention from the Roboflow export format:
- Fresh images: `FRESH-<number>-_JPG.rf.<hash>.jpg`
- Spoiled images: `SPOILED-<number>-_JPG.rf.<hash>.jpg`

The prefix (`FRESH-` / `SPOILED-`) is used as a fallback resolution hint in `_resolve_image_path()` when the label cannot be inferred from the CSV row alone. The 4 manually added images do not follow this naming convention but are still resolved correctly because their CSV entries have explicit one-hot labels.

### "Deleted" files in git status (explained)

During the rebalance experiment, moving files from `train/spoiled/` to `train/_excluded_half_spoiled/` caused git to record those 392 files as "deleted" from their tracked path. The files were still on disk but in a different location. After the operation was reversed and all files were moved back, git saw them as restored (no longer deleted). This explains the observation of large numbers of deletions in git status at an earlier point in the project.

---

*End of documentation — Sections 1–20.*

> This document was generated on **May 1, 2026** and reflects the complete state of the project at that date. All phases (1 through 4) are implemented and working. The selected production model is ResNet50 with 99.33% validation accuracy and 100% holdout accuracy.

---

## 21. Transfer Learning — Theory and Justification

### What is transfer learning?

Transfer learning is a machine learning technique where a model trained on one large task (the *source* task) is repurposed as the starting point for a different but related task (the *target* task). Instead of training a deep convolutional neural network from scratch on your own dataset — which typically requires millions of labeled images and days of GPU compute — transfer learning allows you to leverage the feature representations already learned from a large dataset.

In this project, all three models (MobileNetV2, ResNet50, EfficientNetB0) were pre-trained on **ImageNet**: a benchmark dataset of 1.2 million images across 1,000 object categories. These models learned to detect edges, textures, color gradients, shapes, and semantic structures across a huge variety of visual inputs. Meat freshness, while not one of ImageNet's 1,000 categories, is a visual discrimination task that depends precisely on those low- and mid-level features (color gradients, surface texture, mold patterns).

### Why the convolutional features transfer

In a CNN, earlier layers learn generic, low-level features:
- **Layer 1–3:** Edge detectors (vertical, horizontal, diagonal), color blobs
- **Layer 4–8:** Simple textures, curves, corners
- **Layer 9–20+:** Complex textures, pattern combinations, part detectors

These low-level features are universal across image domains. The ResNet50 base's first 40+ layers contain color-gradient detectors, texture filters, and surface-response filters — all of which are exactly what distinguishes fresh red meat (smooth, uniform texture, saturated red) from spoiled meat (dull brown/gray, slimy film, mold patches). The final few layers of the ImageNet-trained base encode high-level object semantics (recognizing "this is a cat"), which are less relevant — but they too contribute context.

### Feature extraction vs. fine-tuning

Two transfer learning strategies exist:

**Feature extraction (used in this project):**
- The pre-trained base weights are **frozen** (`base.trainable = False`).
- Only the custom classification head (Dense → Dropout → Dense) is trained.
- The base acts as a fixed feature extractor: raw image → rich feature vector.
- Preferred when: small dataset, strong domain overlap with ImageNet, risk of overfitting.

**Fine-tuning:**
- The top N layers of the base are unfrozen and trained with a very small learning rate.
- Both the head and the unfrozen base layers adapt to the target domain.
- Preferred when: larger dataset, strong need for domain adaptation, more compute budget.

Feature extraction was chosen here because the training set has only ~1,916 images. With 23.5M parameters in ResNet50, fine-tuning risks severe overfitting on such a small dataset. The validation accuracy of 99.33% achieved with pure feature extraction confirms this choice was correct — fine-tuning would likely have lowered generalization performance.

### Why ImageNet features work for meat

Despite ImageNet containing no "meat" category, the convolutional features transfer for the following reasons:

1. **Texture detectors** in the base respond strongly to the fibrous, glossy, or slimy surface textures of meat.
2. **Color-gradient filters** directly encode the red-to-brown color shift caused by myoglobin oxidation (the primary biological spoilage marker).
3. **Mold and stain detectors** (originally trained to recognize fur, bark, and fabric patterns) happen to respond well to fungal growth on meat surfaces.
4. **Lighting and shadow detectors** help the model ignore photographic variations and focus on the meat's intrinsic properties.

This is why ResNet50 achieved 99.33% accuracy with a training set of under 2,000 images — a feat that would be impossible with a randomly initialized network of the same depth.

---

## 22. Neural Network Architecture — Deep Dive

### ResNet50 architecture

ResNet50 (He et al., 2015, "Deep Residual Learning for Image Recognition") was a landmark architecture that solved the *degradation problem* in very deep networks: when adding more layers to a CNN, accuracy eventually degrades because gradients vanish during backpropagation before reaching the earlier layers.

The solution is the **skip connection** (also called residual connection or shortcut connection):

```
x ──────────────────────────────┐
  │                              │
  ↓                              │
Conv2D → BatchNorm → ReLU       │  (residual block)
  ↓                              │
Conv2D → BatchNorm → ReLU       │
  ↓                              │
Conv2D → BatchNorm ─────────────+  (element-wise addition)
                                 │
                                 ↓
                               ReLU → next block
```

Instead of learning `H(x)` (the desired mapping), the block learns the **residual** `F(x) = H(x) - x`. If adding more layers is not helpful, the block can trivially learn `F(x) = 0`, making the skip connection an identity mapping. This means deeper networks can never be worse than shallower ones — the extra layers simply learn to be no-ops if they cannot improve performance. This property enabled training networks of 50, 101, and even 152 layers reliably for the first time.

### Full architecture of this project's model

```
Input layer:          224 × 224 × 3 (H × W × Channels)
                                ↓
ResNet50 base:        Frozen ImageNet weights
  - Stage 1:          7×7 Conv, 64 filters, stride 2 → 112×112×64
                      MaxPool 3×3, stride 2 → 56×56×64
  - Stage 2 (×3):     Bottleneck blocks, 256 output channels → 56×56×256
  - Stage 3 (×4):     Bottleneck blocks, 512 output channels → 28×28×512
  - Stage 4 (×6):     Bottleneck blocks, 1024 output channels → 14×14×1024
  - Stage 5 (×3):     Bottleneck blocks, 2048 output channels → 7×7×2048
                                ↓
GlobalAveragePooling2D:   7×7×2048 → 2048 (spatial averaging)
                                ↓
Dense(256, relu):     2048 → 256 (custom head, learned)
                                ↓
Dropout(0.4):         Randomly zeros 40% of neurons during training only
                                ↓
Dense(2, softmax):    256 → 2 (output: [P(Fresh), P(Spoiled)])
```

### Parameter count breakdown

| Component | Parameters | Trainable? |
|---|---|---|
| ResNet50 base | ~23,587,712 | No (frozen) |
| Dense(256) head | 2048×256 + 256 = 524,544 | Yes |
| Dropout | 0 | N/A |
| Dense(2) output | 256×2 + 2 = 514 | Yes |
| **Total** | **~24,112,770** | **525,058 trainable** |

Only 525,058 parameters (2.2% of total) are updated during training. This is why training converges quickly and reliably with only ~1,916 images.

### GlobalAveragePooling vs. Flatten

The ResNet50 base produces a 7×7×2048 feature tensor at its final layer. Two common approaches exist to collapse this into a vector:

**Flatten:** concatenates all values → 7 × 7 × 2048 = 100,352-dimensional vector. This feeds into the Dense head but creates a very large weight matrix and risks overfitting.

**GlobalAveragePooling2D (used here):** computes the average across the 7×7 spatial dimensions for each of the 2048 channels → 2048-dimensional vector. This is spatially invariant, dramatically reduces parameter count, acts as a regularizer, and is the standard approach for transfer learning heads since it was introduced in the Network-in-Network paper.

### Dropout as regularization

`Dropout(rate=0.4)` randomly sets 40% of the 256-dimensional head activations to zero during each training step. Each neuron must therefore not rely on any single partner neuron — it must learn redundant, robust representations. At inference time, dropout is turned off and all neurons fire, but their outputs are scaled by (1 − rate) = 0.6 to preserve expected activation magnitude. This technique, introduced by Srivastava et al. (2014), is one of the most effective regularization methods for neural networks.

---

## 23. Evaluation Metrics — Mathematical Treatment

### Confusion matrix definitions

For a binary classifier with classes Positive (Spoiled) and Negative (Fresh):

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

For ResNet50 on the validation set:
- TP = 270 (Spoiled correctly identified)
- FN = 3 (Spoiled incorrectly called Fresh — the dangerous error)
- FP = 0 (Fresh incorrectly called Spoiled — false alarm)
- TN = 178 (Fresh correctly identified)

### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

Spoiled precision = 270 / (270 + 0) = **1.0000**  
Fresh precision = 178 / (178 + 3) = 178/181 = **0.9834**

Precision answers: "Of all the images I predicted as class X, what fraction actually belong to class X?" High Spoiled precision means that when the model says "spoiled," it is always right.

### Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Spoiled recall = 270 / (270 + 3) = 270/273 = **0.9890**  
Fresh recall = 178 / (178 + 0) = **1.0000**

Recall answers: "Of all the images that actually belong to class X, what fraction did I correctly identify?" Spoiled Recall is the safety-critical metric: 0.9890 means 98.90% of all actually spoiled meat was correctly flagged.

### F1-Score

$$\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

Spoiled F1 = 2 × (1.0000 × 0.9890) / (1.0000 + 0.9890) = **0.9945**  
Fresh F1 = 2 × (0.9834 × 1.0000) / (0.9834 + 1.0000) = **0.9916**

The F1-score is the harmonic mean of precision and recall. It penalizes extreme imbalances between the two. A model that achieves high precision by only predicting the most obvious examples (low recall) will have a low F1. A model that achieves high recall by predicting positive for everything (low precision) will also have a low F1.

### Macro-average F1

$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^{K} F1_k$$

With K=2 classes:  
Macro F1 = (0.9916 + 0.9945) / 2 = **0.9931**

"Macro" averaging computes each class's F1 independently and then takes the unweighted mean. This treats each class equally regardless of its sample count. This is the preferred metric when class balance is imperfect — it does not let the majority class dominate the score.

**Contrast with weighted-average F1** (where each class's F1 is weighted by its support):  
Weighted F1 = (0.9916 × 178 + 0.9945 × 273) / 451 = 0.9933 — slightly higher because Spoiled (the easier majority class) gets more weight. Macro F1 is more conservative and more informative.

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{270 + 178}{451} = \frac{448}{451} = 0.9933$$

Accuracy is **not the primary metric** here because of class imbalance. A model that always predicts "Spoiled" would achieve 273/451 = 60.5% accuracy — acceptable by naive count but catastrophically wrong for Fresh classification. Macro F1 is insensitive to this failure mode.

### ROC-AUC

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Recall) against the False Positive Rate (FP / (FP + TN)) at every possible classification threshold (from 0 to 1). The **Area Under the Curve (AUC)** summarizes the model's discriminative ability in a single number:

- AUC = 1.0: Perfect classifier at all thresholds
- AUC = 0.5: Random classifier (diagonal line)
- AUC < 0.5: Worse than random (predictions are inverted)

ResNet50 AUC = **0.9998** — the model maintains near-perfect True Positive Rate with near-zero False Positive Rate at virtually every threshold. This means even if you changed the decision threshold (e.g., requiring 80% confidence before calling something "Spoiled"), the model would still classify almost all Spoiled images correctly.

---

## 24. The `tf.data` Pipeline — Explained

TensorFlow's `tf.data` API builds a lazy, highly optimized data pipeline that feeds training examples to the GPU/CPU. Unlike loading all images into RAM upfront (which would require ~several GB for this dataset at float32), `tf.data` streams images from disk, processes them on-the-fly, and overlaps loading/preprocessing with model training.

### Pipeline construction (Phase 2)

```python
# 1. Build list of (filepath, label) pairs from CSV
paths, labels = build_path_label_pairs(csv_path, split_root)

# 2. Create a dataset of file paths and integer labels
dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

# 3. Shuffle the dataset (training only)
dataset = dataset.shuffle(buffer_size=len(paths), seed=SEED)

# 4. Map: decode + resize + normalize each image
dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# 5. Augment (training only)
dataset = dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

# 6. Batch
dataset = dataset.batch(BATCH_SIZE)

# 7. Prefetch: load next batch while GPU trains on current batch
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### Key operations explained

**`from_tensor_slices`:** Creates a dataset where each element is one (path, label) pair. Does not read any files yet.

**`shuffle(buffer_size)`:** Maintains a buffer of `buffer_size` elements in RAM and samples randomly from it. A buffer equal to the full dataset size gives perfectly uniform shuffling. This is critical to prevent the model from memorizing the order of examples.

**`map(fn, num_parallel_calls=AUTOTUNE)`:** Applies a function to each element. `AUTOTUNE` tells TensorFlow to choose the number of parallel worker threads dynamically based on available CPU cores. This is where file I/O and image decoding happen.

**`batch(32)`:** Groups 32 examples into a single tensor of shape `(32, 224, 224, 3)` for the model.

**`prefetch(AUTOTUNE)`:** While the GPU/CPU trains on the current batch, the pipeline asynchronously prepares the next batch. This overlaps compute and I/O, dramatically improving throughput.

### Why not use `ImageDataGenerator`?

TensorFlow's older `ImageDataGenerator` API is commonly seen in tutorials but is now deprecated in favor of `tf.data`. The `tf.data` pipeline is:
- Fully parallelizable (multi-threaded map operations)
- Compatible with `@tf.function` (compiled execution graphs)
- Composable (operations chain cleanly)
- More memory-efficient (no need to hold all augmented data in RAM)

`ImageDataGenerator` reads files sequentially from disk with a Python GIL-bound loop. For a dataset of 1,916 images, the difference is small but `tf.data` is the correct modern approach.

---

## 25. Class Imbalance — Theory and Solution

### Why class imbalance is a problem

When classes are not equally represented, an uncorrected model learns to minimize loss by favoring the majority class. With 762 Fresh and 1,154 Spoiled images (ratio ≈ 1:1.51), a model that always predicts "Spoiled" would achieve:

- Accuracy: 1,154 / 1,916 = 60.2%
- Fresh Recall: 0% (never identifies Fresh)
- Spoiled Recall: 100% (always predicts Spoiled)
- Macro F1: ≈ 0.38 (very poor)

The loss function (`SparseCategoricalCrossentropy`) averages loss over all examples. If Spoiled has 51.6% more examples, any given gradient step is more influenced by Spoiled examples, pushing the decision boundary away from a balanced position.

### The solution: class weights

Class weights assign a **multiplier** to the loss for each class. A higher multiplier means the model is penalized more for errors on that class. By setting the weight inversely proportional to class frequency, every class contributes equally to the total gradient regardless of sample count.

**Formula:**

$$w_k = \frac{N}{K \cdot n_k}$$

Where:
- $N$ = total training samples (1,916)
- $K$ = number of classes (2)
- $n_k$ = samples in class $k$

**Computed weights:**
- Fresh weight: 1,916 / (2 × 762) = 1,916 / 1,524 ≈ **1.257**
- Spoiled weight: 1,916 / (2 × 1,154) = 1,916 / 2,308 ≈ **0.830**

*(The exact values printed by `phase1_data_prep.py` may differ slightly depending on the sklearn version's implementation of `compute_class_weight`, which uses a slightly different normalization. The principle is the same.)*

These weights are passed to `model.fit(..., class_weight={0: w_fresh, 1: w_spoiled})`. TensorFlow multiplies the loss for each example by the weight of its true class before summing. The net effect is that the model treats a Fresh error as ~1.5× more costly than a Spoiled error, compensating for the imbalance.

### Why not oversample or undersample?

Two alternative approaches exist:
1. **Oversampling (SMOTE, random repeat):** Artificially duplicate minority class images until classes are balanced. Risk: the model memorizes the oversampled images, causing overfitting.
2. **Undersampling:** Remove majority class images to balance counts. Risk: wastes 392 real Spoiled training images, losing genuine diversity.

Class weighting achieves the same mathematical correction during training without touching the actual data distribution — all images remain available to the model. It is the preferred approach for moderate imbalance ratios (< 10:1).

---

## 26. ImageNet Normalization — Why and How

### The purpose of input normalization

Neural networks are sensitive to the scale of their inputs. During training, the optimizer adjusts weights using gradient descent. If input values span very different magnitudes (e.g., 0 to 255 for raw pixel values), the gradients will have inconsistent scales across input dimensions, slowing convergence and destabilizing training.

More importantly for transfer learning: the pre-trained ResNet50 weights were trained with a specific normalization applied to inputs. Using a different normalization at inference time would shift the activations in every layer, effectively invalidating all the learned features. **The normalization used during transfer learning must exactly match the normalization used when the base model was originally trained on ImageNet.**

### ImageNet channel statistics

The ImageNet training set has the following pixel statistics (computed across all 1.2M images):

| Channel | Mean | Std |
|---|---|---|
| Red | 123.675 | 58.395 |
| Green | 116.280 | 57.120 |
| Blue | 103.530 | 57.375 |

*(Values in the 0–255 range.)*

### The normalization applied in this project

```python
tf.keras.applications.imagenet_utils.preprocess_input(img * 255.0)
```

Step 1: Multiply float32 image in [0, 1] → [0, 255].  
Step 2: `preprocess_input` subtracts the per-channel ImageNet means:
- R channel: R - 123.68
- G channel: G - 116.779
- B channel: B - 103.939

This is the **`"caffe"` mode** normalization (used by ResNet50 and MobileNetV2). The result is a zero-centered input tensor in approximately the range [-128, 128].

EfficientNetB0 uses **`"torch"` mode**: divides by 255 then applies mean subtraction and division by standard deviation. `imagenet_utils.preprocess_input` selects the correct mode automatically based on which model it was called from, but since this project calls it directly (not through the model's preprocessing layer), it defaults to `"caffe"` mode consistently for all three architectures.

### Why this matters for inference

The `app.py` and `phase3_inference.py` scripts both import and call `preprocess_image()` from `phase2_training.py` directly. This single source-of-truth approach guarantees that inference inputs are processed identically to training inputs. If someone replaced `preprocess_input(img * 255.0)` with plain `img` (no normalization), the model's accuracy would degrade significantly — the raw [0,1] inputs would fall far outside the input distribution the model was trained on.

---

## 27. Limitations

This section documents the known limitations of the current system. Acknowledging limitations is a standard requirement for academic work and demonstrates critical analysis.

### 1. Binary classification only

The model outputs only "Fresh" or "Spoiled." It cannot indicate the *degree* of freshness or estimate days until spoilage. A continuous freshness score or a regression output would be more informative for practical supply chain use, but training such a model requires time-stamped freshness data that was not available.

### 2. Visual features only — no olfactory or chemical input

Spoilage is a multi-modal phenomenon. While visual cues (color, texture, mold) are reliable indicators, they are not the only or always the first indicator. Some bacterial contamination (e.g., Salmonella, early Listeria) produces no visible changes. The model cannot detect:
- Off-odors (the most reliable human spoilage indicator)
- pH changes
- Bacterial colony counts
- Surface temperature

In a production food safety system, the visual classifier would be one component of a multi-modal sensor array, not the sole decision-maker.

### 3. Dataset source and geographic specificity

The training dataset was sourced from Roboflow and supplemented with a small number of manually added images. The images are predominantly of one to two meat types (primarily beef/pork) and were photographed under studio or indoor lighting conditions. The model may perform less reliably on:
- Poultry, fish, or game meats (different color baselines)
- Images taken in highly unusual lighting (e.g., fluorescent yellow light)
- Very unusual spoilage patterns not represented in the training set

### 4. Small holdout set

The holdout evaluation achieved 100% accuracy, but the holdout set contains only 46 images (32 Fresh, 14 Spoiled). While the 99.33% validation accuracy is from a larger 451-image set, 46 images is a small sample for drawing strong statistical conclusions. A proper external test set would have at least 500 images per class.

### 5. No fine-tuning

The ResNet50 base is entirely frozen (feature extraction only). While this was the correct choice for this dataset size, a larger dataset could benefit from fine-tuning the top stages of ResNet50 — potentially pushing accuracy above 99.5% and further reducing the 3 remaining false negatives.

### 6. CPU-only inference on Streamlit Cloud

Streamlit Cloud does not provide GPU instances. All inference runs on CPU. For ResNet50 on a 224×224 image, CPU inference takes approximately 100–300ms per image (depending on server load). For batch use cases (e.g., scanning an entire refrigerator of products), this latency would need to be addressed through GPU deployment or model quantization/distillation.

### 7. Static model — no continual learning

The deployed model is static. It does not learn from new images uploaded by users. If new spoilage patterns emerge (e.g., a novel mold species) or the user population submits images of meat types not in the training set, performance will silently degrade without retraining. A production system would implement monitoring of prediction confidence distributions to detect drift and trigger retraining pipelines.

---

## 28. Future Work

### 1. Multi-class freshness grading

Replace the binary Fresh/Spoiled output with a four-level grading scale: **Very Fresh → Acceptable → Borderline → Spoiled**. This requires constructing or sourcing a dataset with time-stamped images of meat at known stages of aging under controlled conditions. It would transform the system from a binary detector into a quantitative freshness estimator.

### 2. Fine-tuning the ResNet50 base

Unfreeze the last one or two residual stages of ResNet50 and train them with a very low learning rate (1e-5 to 1e-6) on the full dataset. This would allow the model to adapt the high-level feature detectors to meat-specific patterns rather than relying entirely on generic ImageNet features. Expected to reduce the 3 remaining false negatives toward zero.

### 3. Real-time webcam classification

Extend `app_desktop.py` to capture frames from a USB or built-in camera in real time. Each frame would be passed through the model and the prediction displayed as an overlay on the live feed. This would create a genuinely useful "point and detect" tool for butchers and retailers.

### 4. Mobile app deployment

Export the ResNet50 model to TensorFlow Lite (`TFLiteConverter`) and deploy on Android/iOS. On a modern mobile SoC with a neural processing unit (NPU), inference would take under 10ms. This would make the tool usable in settings without internet connectivity (rural markets, fieldwork).

### 5. Multi-modal sensor fusion

Integrate the visual classifier with a low-cost electronic nose (e-nose) sensor — a gas sensor array that detects volatile organic compounds (VOCs) emitted by spoiling meat (hydrogen sulfide, trimethylamine, ammonia). Fusing visual and chemical signals would dramatically reduce false negatives and make the system viable for regulatory use.

### 6. Explainability — Grad-CAM visualization

Implement **Gradient-weighted Class Activation Maps (Grad-CAM)** to visualize which regions of the image the model is using to make its prediction. This would:
- Provide a heatmap overlay showing "this part of the meat appears spoiled."
- Allow human experts to verify that the model is using biologically meaningful features (discolored regions, mold patches) rather than image artifacts.
- Increase trust and regulatory acceptance of the system.
- Provide a more informative output to the end user.

---

## 29. Deployment Architecture

### Overview

The system is deployed as a Streamlit web application on Streamlit Cloud's free tier. The following describes the complete server-side inference flow from the moment a user uploads an image.

### Inference flow (server-side)

```
User uploads image (JPEG/PNG/WEBP)
          │
          ▼
Streamlit receives bytes via st.file_uploader()
          │
          ▼
PIL.Image.open() → validates image, converts to RGB (drops alpha channel if PNG)
          │
          ▼
Pillow saves to in-memory BytesIO buffer as JPEG
          │
          ▼
TensorFlow tf.io.read_file() equivalent: reads bytes
tf.io.decode_jpeg() → uint8 tensor [H, W, 3]
          │
          ▼
tf.image.resize([224, 224]) → float32 tensor [224, 224, 3] in [0.0, 1.0]
          │
          ▼
imagenet_utils.preprocess_input(img * 255.0)
→ float32 tensor [224, 224, 3] in approx [-128, 128]
          │
          ▼
tf.expand_dims(img, axis=0) → shape [1, 224, 224, 3]
          │
          ▼
model.predict(batch) → [[P(Fresh), P(Spoiled)]]
          │
          ▼
argmax → predicted_class (0=Fresh, 1=Spoiled)
confidence = max(P(Fresh), P(Spoiled))
          │
          ▼
Display result: ✅ Fresh (99.3%) or ⚠️ Spoiled (97.1%)
```

### Model loading strategy

The model is loaded once per Streamlit session using `@st.cache_resource`:

```python
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("phase2_models/ResNet50/best_model.keras")
```

`@st.cache_resource` stores the loaded model object in Streamlit's server-side cache. On the first request, the 91MB `.keras` file is deserialized from disk (takes ~2–5 seconds). On all subsequent requests within the same session, the cached model object is returned instantly without re-loading from disk. This is critical for user experience — without caching, every image upload would trigger a 5-second model reload.

### Git LFS and model serving

The `.keras` file is stored in GitHub via **Git Large File Storage (LFS)**. When Streamlit Cloud clones the repository at deploy time, the LFS pointer files are resolved and the actual binary content (91MB) is downloaded from the LFS storage server. This is transparent to the app — the `.keras` file appears at the expected path on the Streamlit Cloud server's filesystem.

Without Git LFS, the `.keras` file would exceed GitHub's 100MB file size limit and the push would fail. Alternatively, the file could be served from an external object store (AWS S3, Google Cloud Storage) and downloaded at app startup, but Git LFS is simpler for an academic project.

### Environment on Streamlit Cloud

Streamlit Cloud provisions a Linux container with:
- Python version: specified in `runtime.txt` (`python-3.12`)
- Packages: installed from `requirements.txt` via pip at deploy time
- RAM: ~1GB available (free tier)
- CPU: shared, no GPU
- Persistent storage: none (stateless — uploaded images are not saved)
- Network: outbound internet access allowed

The app is stateless by design. No user-uploaded images are stored on the server. Each Streamlit session is independent.

---

## 30. Frequently Asked Questions (FAQ)

**Q1: Why did you use ResNet50 instead of a more modern architecture like EfficientNetV2 or Vision Transformer?**

ResNet50 was chosen because it is the best-performing architecture among the three evaluated in this study. EfficientNetV2 and Vision Transformers were not included in the comparison because they require larger datasets and more compute to realize their advantages. For a dataset of ~1,916 training images, ResNet50 is the appropriate choice — it is well-understood, well-documented, and achieves near-perfect performance on this task. Adding more candidate architectures would not have changed the conclusion.

---

**Q2: Why is the validation accuracy 99.33% but the model still gets 3 examples wrong?**

99.33% accuracy on 451 images means 448/451 correct. The 3 incorrect predictions are all Spoiled images classified as Fresh (false negatives). These 3 images are edge cases — likely borderline examples where spoilage is in early stages or the visual cues are subtle (slight discoloration only, no mold). No model achieves 100% accuracy on a real-world dataset without overfitting. The 100% holdout accuracy represents a smaller 46-image sample and does not contradict the 99.33% validation figure.

---

**Q3: What does the Dropout layer do during inference vs. training?**

During **training**: Dropout randomly zeroes 40% of the 256-unit layer's activations on each forward pass. This forces the network to learn redundant representations and prevents co-adaptation of neurons (a form of overfitting).

During **inference** (prediction): Dropout is automatically disabled by Keras when `model.predict()` or `model(x, training=False)` is called. All 256 neurons contribute to the output. Their activations are scaled by (1 − 0.4) = 0.6 to preserve the expected magnitude of the output distribution, ensuring consistency between training and inference.

---

**Q4: Why does the training pipeline use `img * 255.0` before `preprocess_input`?**

`tf.io.decode_image` returns a `uint8` tensor, but after `tf.image.resize`, the tensor is cast to `float32` in the range [0.0, 1.0]. TensorFlow's `imagenet_utils.preprocess_input` was written assuming input values in [0, 255] (matching the range of raw uint8 pixels). Multiplying by 255.0 restores the expected input range before the mean subtraction is applied. Without this step, the mean subtraction would apply approximately −123/255 ≈ −0.48 to the already-[0,1]-normalized inputs, resulting in inputs centered around −0.48 rather than 0 — completely wrong.

---

**Q5: What is the difference between `_classes.csv` filenames and the actual subfolder structure?**

The `_classes.csv` file stores filenames without subfolder prefixes (e.g., `FRESH-001.jpg` not `fresh/FRESH-001.jpg`). The `_resolve_image_path()` function in `dataset_loader.py` tries multiple path candidates to resolve the actual file location, including using the one-hot label value to determine whether to look in `fresh/` or `spoiled/`. This design makes the data loader robust to different dataset organizations.

---

**Q6: Why does Phase 1 use PyTorch but Phase 2–4 use TensorFlow?**

Phase 1 was written as an exploratory data analysis and pipeline design step, using PyTorch because it offers more flexible dataset handling for prototyping. After confirming the data loading and augmentation logic was correct, the training pipeline was re-implemented in TensorFlow for consistency with the deployment stack (Streamlit + TF model serving). The `FruitFreshnessDataset` class in `dataset_loader.py` is only used by Phase 1 — Phases 2–4 bypass it and use `tf.data` directly. If PyTorch is not installed, Phase 1 will fail gracefully with an ImportError, but all other phases work correctly without PyTorch.

---

**Q7: Why 224×224 pixels? Why not a higher resolution?**

224×224 is the standard input size for ImageNet-trained models including ResNet50, MobileNetV2, and EfficientNetB0. It was chosen when these architectures were designed as the minimal resolution that preserves enough spatial detail for 1,000-class object recognition. Using a higher resolution (e.g., 512×512) would require retraining the base from scratch with the new input size, multiplying compute requirements approximately 5×, and providing diminishing returns for this task — the diagnostic visual features of meat spoilage (color zones, surface texture) are visible at 224×224.

---

**Q8: How does Git LFS work for the model files?**

Git LFS (Large File Storage) intercepts the storage of files matching specified patterns (in this case, `*.keras` files as configured in `.gitattributes`). Instead of storing the binary file contents in the Git object database, it stores a text pointer file containing the file's hash and size. The actual binary content is uploaded to a separate LFS server (GitHub's LFS backend). When anyone clones the repository, the pointer files are automatically resolved and the binary content is downloaded from the LFS server. This keeps the Git history lightweight while still allowing large model files to be versioned.

---

**Q9: What happens when a user uploads a non-meat image to the app?**

The model will still produce a prediction — it has no "reject" class. For a completely unrelated image (e.g., a landscape photo), the model will output softmax probabilities that sum to 1.0 and classify it as either Fresh or Spoiled with some confidence. The confidence might be lower for truly out-of-distribution inputs (the softmax outputs might be closer to [0.6, 0.4] rather than [0.99, 0.01]), but there is no hard rejection mechanism. This is a known limitation of closed-set classifiers. A production system would add an out-of-distribution (OOD) detection module to identify and reject non-meat inputs.

---

**Q10: What does `@st.cache_resource` do and why is it needed?**

`@st.cache_resource` is a Streamlit decorator that stores the return value of a function in a global, session-persistent cache on the server. Without it, every time a user interacts with the Streamlit app (uploading a file, clicking a button), the entire Python script reruns from top to bottom — including `tf.keras.models.load_model()`, which takes 2–5 seconds. With `@st.cache_resource`, the model is loaded only once per server startup and then reused for all subsequent interactions. The cached value is shared across all user sessions on the same server instance.

---

**Q11: Why SparseCategoricalCrossentropy instead of BinaryCrossentropy?**

Even though this is a binary classification problem (two classes), the model outputs a softmax vector of length 2 (not a single sigmoid probability). `SparseCategoricalCrossentropy` is the correct loss function for multi-class softmax outputs with integer labels. `BinaryCrossentropy` is used when the model output is a single sigmoid neuron (scalar probability). The choice of 2-class softmax over 1-class sigmoid is intentional — it makes the architecture identical across all three models (which were originally tested on a 3-class version) and makes the class probability outputs symmetric and interpretable.

---

**Q12: Why does the app use a confidence threshold and what is it set to?**

The app does not apply a threshold — it simply reports the class with the higher softmax probability (argmax). The displayed confidence percentage is the softmax probability of the predicted class. This is appropriate for a consumer-facing tool. A threshold (e.g., "only report Spoiled if P(Spoiled) > 0.8") would introduce a second hyperparameter requiring careful calibration and could cause the model to output "uncertain" for edge cases, which is confusing for users. The current behavior — always committing to a prediction while showing confidence — is standard for production classifiers.

---

**Q13: What is the Streamlit Cloud URL and how is it deployed?**

The app is deployed at: `https://chinelo-meat-spoilage.streamlit.app/`

Deployment steps:
1. Push the repository (including `app.py`, `requirements.txt`, `runtime.txt`, and the `.keras` model file) to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), connect the `chinelo-meatspoilage/meat-spoilage` repository, specify `main` as the branch and `app.py` as the entry point.
3. Streamlit Cloud automatically builds the environment (installs packages from `requirements.txt`) and launches the app.
4. Any subsequent `git push` to `main` triggers an automatic redeploy.

---

**Q14: What is the half-fresh merging decision and why can't it be reversed?**

The merge is implemented in the `_classes.csv` label files — the `Spoiled` column was set to `1` for all rows originally labeled `half-fresh`. The original three-class CSV files were overwritten. The images themselves remain in the `half-fresh/` subfolders (they were not moved or deleted), but their labels in the CSV now point them to the `Spoiled` class. To revert to three-class classification, a new CSV would need to be reconstructed from the original Roboflow export (which can be re-downloaded if the dataset is still accessible), and the model would need to be retrained from scratch with 3-class softmax output.

---

## 31. Glossary

**Accuracy** — The fraction of all predictions that are correct: (TP + TN) / (TP + TN + FP + FN). Not the primary metric here due to class imbalance.

**AUC (Area Under the Curve)** — The area under the ROC curve. Ranges from 0 to 1. A value of 1.0 indicates perfect classification at all thresholds.

**Augmentation** — Artificially expanding the training dataset by applying random transformations (flips, rotations, brightness changes) to existing images, improving model generalization.

**Batch size** — The number of training examples used in a single forward/backward pass during training. Set to 32 in this project.

**Binary classification** — A classification task with exactly two possible output classes. Here: Fresh vs. Spoiled.

**Class imbalance** — A condition where training examples are not equally distributed across classes. Here: 762 Fresh vs. 1,154 Spoiled.

**Class weights** — Multipliers applied to the loss function to compensate for class imbalance. Higher weight = more penalty for errors on that class.

**Confusion matrix** — A 2×2 table showing the counts of TP, FN, FP, TN for a binary classifier.

**Dropout** — A regularization technique that randomly zeroes a fraction of neuron activations during training to prevent overfitting.

**Early stopping** — A training callback that halts training when a monitored metric (e.g., val_loss) stops improving for a specified number of epochs (patience=5 here).

**EfficientNetB0** — A compact, modern CNN architecture using compound scaling. Parameters: ~4.0M. Validation accuracy here: 96.23%.

**Epoch** — One complete pass through the entire training dataset.

**F1-score** — The harmonic mean of precision and recall. Balances false positives and false negatives.

**False negative (FN)** — A Spoiled image classified as Fresh. The dangerous error in this application.

**False positive (FP)** — A Fresh image classified as Spoiled. The nuisance error (unnecessary waste).

**Feature extraction** — A transfer learning strategy where the pre-trained base's weights are frozen and only the custom classification head is trained.

**Fine-tuning** — A transfer learning strategy where the top layers of the pre-trained base are unfrozen and trained alongside the custom head.

**`@st.cache_resource`** — A Streamlit decorator that persists the return value of a function in server memory across reruns, used here to avoid reloading the model on every user action.

**Git LFS (Large File Storage)** — A Git extension that stores large binary files (e.g., model weights) on a separate server while maintaining lightweight pointer files in the main repository.

**GlobalAveragePooling2D** — A layer that computes the spatial mean across each feature map, converting a 3D tensor (H × W × C) to a 1D vector of length C.

**ImageNet** — A benchmark dataset of 1.2 million images across 1,000 categories used to pre-train deep CNNs. The source of pre-trained weights used in this project.

**ImageNet normalization** — Subtracting the ImageNet dataset's per-channel mean pixel values from the input image, required for models pre-trained on ImageNet.

**Keras** — TensorFlow's high-level deep learning API, used to define, compile, train, and save all models in this project.

**Macro-average** — Averaging a metric (e.g., F1) equally across all classes, regardless of class support. Treats each class as equally important.

**MobileNetV2** — A lightweight CNN architecture optimized for mobile devices. Parameters: ~2.2M. Validation accuracy here: 86.25%.

**Model checkpoint** — A saved snapshot of model weights at a particular epoch, used to restore the best-performing state.

**Myoglobin** — The oxygen-binding protein in muscle tissue responsible for the red color of fresh meat. Its oxidation to metmyoglobin causes the brown/gray color change associated with spoilage.

**One-hot encoding** — Representing a categorical label as a vector of zeros with a single 1 at the index corresponding to the class. Used in `_classes.csv`.

**Overfitting** — When a model learns to memorize the training data rather than generalize to unseen data. Regularized by dropout, weight freezing, and data augmentation.

**Precision** — Of all images predicted as class X, the fraction that truly belong to class X. TP / (TP + FP).

**Prefetching** — Asynchronously preparing the next data batch while the model trains on the current batch, improving throughput.

**Recall (Sensitivity)** — Of all images that truly belong to class X, the fraction the model correctly identifies. TP / (TP + FN).

**ResNet50** — A 50-layer deep residual network with skip connections, pre-trained on ImageNet. Parameters: ~23.5M. Selected as the production model. Validation accuracy: 99.33%.

**ROC curve** — A plot of True Positive Rate vs. False Positive Rate at every classification threshold. Used to evaluate classifier performance independent of threshold choice.

**Skip connection (residual connection)** — A direct pathway in ResNet that adds the input of a block to its output, allowing gradients to flow directly to earlier layers and preventing vanishing gradients.

**Softmax** — An activation function that converts a vector of raw scores (logits) to a probability distribution summing to 1.0. Used in the output layer of all models here.

**SparseCategoricalCrossentropy** — A loss function for multi-class classification where labels are integer indices (not one-hot vectors).

**Streamlit** — An open-source Python library for building interactive web applications. Used for the project's web deployment.

**`tf.data`** — TensorFlow's data pipeline API. Provides a lazy, parallelizable, memory-efficient way to feed data to model training.

**Transfer learning** — Using a model pre-trained on a large dataset (ImageNet) as the starting point for training on a new, related task.

**Validation set** — A held-out subset of labeled data not used during training, used to evaluate model performance and tune hyperparameters. Here: 451 images.

**Weighted-average F1** — Averaging F1 across classes weighted by class support (sample count). Favors the majority class. Macro-average F1 is preferred here.

---

*End of documentation — Sections 1–31 complete.*

> This document was generated on **May 1, 2026** and reflects the complete state of the project at that date. All phases (1 through 4) are implemented and working. The selected production model is ResNet50 with 99.33% validation accuracy and 100% holdout accuracy.

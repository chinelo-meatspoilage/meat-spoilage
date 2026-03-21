# Command Cheat Sheet (Quick Reference)

Concise reference for every script in this project. Two classes: **Fresh** and **Spoiled**.

---

## Setup
Install all dependencies before running anything:
```bash
pip install -r requirements.txt
```

---

## Phase 1 — Data sanity checks
Validates CSV structure, one-hot encoding, file existence, class balance, and augmentation pipeline.
```bash
python phase1_data_prep.py
```

---

## Phase 2 — Training + checkpoints
Trains transfer-learning models, saves the best checkpoint per model, outputs validation metrics.
```bash
python phase2_training.py
```
Convert any legacy `.h5` checkpoints to `.keras` format without retraining:
```bash
python phase2_training.py --convert-only
```

---

## Phase 3 — Evaluation (validation set)
Generate evaluation artifacts (confusion matrix, ROC curve, report) for all trained models:
```bash
python phase3_evaluation.py --all
```
Select the best model (highest macro F1) and generate its report:
```bash
python phase3_evaluation.py --best
```
Evaluate a specific model:
```bash
python phase3_evaluation.py --model ResNet50
```

---

## Phase 3 — Inference (command line)
Classify a single image:
```bash
python phase3_inference.py --image path/to/img.jpg --best
```
Batch-classify a folder (no label required):
```bash
python phase3_inference.py --dir path/to/images --best
```

---

## Phase 3 — Holdout evaluation
Evaluates images in class-named subfolders (`Fresh/`, `Spoiled/`) and produces accuracy + confusion matrix:
```bash
python phase3_holdout.py --dir holdout --best
```

---

## Web app — Streamlit (image upload + camera)
Launches the Streamlit web app locally:
```bash
streamlit run app.py
```
The camera tab uses the browser's built-in camera — no OpenCV required.

To deploy publicly (free):
1. Push the project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set main file to `app.py` and click Deploy.

---

## Desktop app — Tkinter (image upload + camera)
Launches the local desktop GUI:
```bash
python app_desktop.py
```
Camera capture requires OpenCV (`pip install opencv-python`). The app works without it — the camera button will show an install prompt.

---

## Misc — Image conversion
Convert all non-JPEG images in a folder to JPEG with a white background:
```bash
python convert_avif_to_jpg.py path/to/folder
```

## Misc — Model selection justification
Generates/refreshes the document explaining why the selected model was chosen:
```bash
python generate_model_selection_justification.py
```

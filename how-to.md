# How This Project Works

> **Current implementation:** Binary classification — **Fresh** vs **Spoiled**.
> Half-Fresh was merged into the Spoiled class (both are unsafe to eat).
> Selected model: **ResNet50** (val accuracy 98.9%, macro F1 0.988).

---

### **PHASE 1: Data Strategy & The "Color Trap" (Crucial)**
The training set contains **1,916 images** (762 Fresh / 1,154 Spoiled) and the validation set 451 images across 2 classes (`Fresh`, `Spoiled`).
Here is how we handle it:

1.  **Check for Class Imbalance:** The training set has ~762 Fresh and ~1,154 Spoiled images. We handle the imbalance using **Class Weights** (inverse-frequency) so the model pays proportional attention to both classes.
2.  **Image Sizing:** The images are 416x416. While great for detail, standard CNNs (ResNet, MobileNet) are optimized for **224x224**. We will resize the images to 224x224. This speeds up training massively without losing the semantic information needed to detect spoilage.
3.  **Data Augmentation (The Trap to Avoid):** 
    *   *What we WILL do:* Random rotations (0-90 degrees), horizontal/vertical flips, and slight zooming. (Code will be shared with Chinelo).
    *   *What we MUST AVOID:* **Extreme Hue/Color Shifts.** In standard AI projects, engineers randomly change image colors to make the model robust. **Do not do this for meat.** If you artificially change the hue of fresh red meat to brown/green during augmentation, you are literally teaching the AI the wrong biological markers for spoilage. We will strictly limit color augmentation to very slight brightness and contrast adjustments.

### **PHASE 2: The "Multiple Choice" Model Selection Strategy**
Three architectures are trained using Transfer Learning (pre-trained on ImageNet) and the best is selected.

> **Note:** Model checkpoints are saved in TensorFlow's modern `.keras` format (not `.h5`).

1.  **MobileNetV2 (The Baseline):** Extremely fast, lightweight.
2.  **ResNet50 (The Industry Standard):** Deeper network, great at complex texture features. **Current winner.**
3.  **EfficientNetB0 (The Modern Option):** Balances speed and accuracy.

**The Training Approach:**
*   **Loss Function:** `SparseCategoricalCrossentropy` (2 classes, integer labels).
*   **Optimizer:** `Adam` with a low learning rate (`1e-4`) to preserve pre-trained ImageNet weights.
*   **Callbacks:** `EarlyStopping` (patience=5, monitors val_loss) + `ModelCheckpoint` (saves the best epoch only).

### **PHASE 3: Evaluation & Proving "The Best" Model**
Once all three models are trained, we prove to the supervisor *why* we picked the winner:

1.  **Accuracy, Precision, Recall, and F1-Score:** **Spoiled Recall is the most safety-critical metric** — a false negative on spoiled meat is dangerous. Macro F1 is used as the primary ranking metric.
2.  **The Confusion Matrix (The Most Important Graph):** Shows exactly where the model is confused. A model that marks `Spoiled` as `Fresh` is unacceptable regardless of overall accuracy.
3.  **Binary ROC-AUC Curve:** Shows how well the model separates Fresh from Spoiled across all confidence thresholds.

**Phase 3 also includes two inference apps — see Phase 4.**

*Whichever model has the highest F1-Score and the safest Confusion Matrix becomes our "Chosen Model" for the web app.*

#### ✅ Running the Phase 3 evaluation scripts (in this repo)

Run all evaluation artifacts (plots + report markdown) for every trained model:
```
python phase3_evaluation.py --all
```

Pick the best model (macro F1) and generate its report:
```
python phase3_evaluation.py --best
```

Run inference against a single image (uses the best model by default):
```
python phase3_inference.py --image path/to/test.jpg --best
```

Run inference on an entire folder of images:
```
python phase3_inference.py --dir path/to/images --best
```

The figures and markdown reports are written under `phase3_reports/`.

### **PHASE 4: Deployment (Streamlit Web App + Tkinter Desktop App)** ✅
Two fully working inference apps are included:

#### `app.py` — Streamlit Web App (for deployment)
*   **Tab 1 — Upload Image:** Accepts JPEG, PNG, WEBP, BMP, TIFF.
*   **Tab 2 — Take a Photo:** Browser-based camera via `st.camera_input` (desktop and mobile, no OpenCV needed).
*   **Preprocessing:** White-background compositing → resize to 224×224 → ImageNet normalisation.
*   **Output:** Predicted class with confidence % and probability bars, colour-coded (green = Fresh, red = Spoiled).

**Run locally:**
```bash
streamlit run app.py
```

**Deploy for free (Streamlit Cloud):**
1. Push the project to a public GitHub repository (see required files below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set the main file to `app.py` and click **Deploy**.
4. Streamlit Cloud reads `requirements.txt` automatically — no extra setup needed.

**Files that MUST be in the GitHub repo for deployment to work:**

```
app.py                                  ← the Streamlit app (main entry point)
requirements.txt                        ← tells Streamlit Cloud what to install
runtime.txt                             ← pins Python to 3.12 (required — TF has no Python 3.14 wheels)
phase3_inference.py                     ← inference logic used by app.py
phase2_training.py                      ← IMAGE_SIZE, LABEL_COLUMNS, preprocess_image
dataset_loader.py                       ← label map used by training scripts
phase2_models/
    ResNet50/
        best_model.keras                ← the trained model weights  ⚠️ required
    summary.csv
phase3_reports/
    selected_model.txt                  ← tells app.py which model is "best"
    ResNet50/
        eval_report.json
```

> ⚠️ **Important — model file size:** `best_model.keras` is typically 90–120 MB.
> GitHub has a 100 MB per-file limit. If the file exceeds that, use
> [Git Large File Storage (LFS)](https://git-lfs.github.com/):
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes
> ```
> Then commit and push as normal. Streamlit Cloud supports Git LFS.

**Files you do NOT need to push** (heavy / not needed at runtime):
- `project-dataset/` — training images, not needed for inference
- `holdout/` — evaluation images
- `phase3_reports/*/confusion_matrix.png`, `roc_curve.png` — plots only
- `app_desktop.py` — Tkinter app, not used by Streamlit Cloud
- `phase3_evaluation.py`, `phase3_holdout.py`, `phase1_data_prep.py`, `phase2_training.py` — training/evaluation scripts (optional to include for completeness, but not required to run the app)

#### `app_desktop.py` — Tkinter Desktop App (local use)
*   **Upload Image button:** File dialog, supports all image formats.
*   **Capture from Camera button:** Live webcam preview via OpenCV — press SPACE to capture, ESC to cancel. Requires `pip install opencv-python`; shows a friendly install prompt if missing.
*   Same dark-theme UI with colour-coded result and probability bars.

**Run:**
```bash
python app_desktop.py
```

### **PHASE 5: Deliverables for Chinelo (The Knowledge Transfer)**
Here is exactly what I will will hand over:

1.  **The "Flowchart":** A diagram showing: `Raw Image` → `Resize/Normalize` → `Data Augmentation` → `Three CNN Models (MobileNet, ResNet, EfficientNet)` → `Evaluation Selection` → `Best Model (ResNet50)` → `Streamlit App Prediction`. (Use draw.io or Canva to build this visually.)
2.  **The Python Scripts:** Commented code showing the data loading (using `_classes.csv`), augmentation, training of 3 models, and evaluation graphs.
3.  **The Streamlit App Code (`app.py`):** The clean deployment code.
4.  **The Saved Best Model (`best_model.keras`):** The final weights at `phase2_models/ResNet50/best_model.keras`.

---

> **Framework used:** TensorFlow/Keras (ResNet50). All phases are complete.

---

## 🔧 Quick Command Reference (all scripts)

### Phase 1 — Data sanity checks
Run the full dataset validation script to ensure labels are correct, images exist, and augmentation/preprocessing is sane.
```bash
python phase1_data_prep.py
```

### Phase 2 — Training + checkpoints
Train all models (MobileNetV2, ResNet50, EfficientNetB0) and save the best checkpoints.
```bash
python phase2_training.py
```

Convert any existing `.h5` checkpoints to modern `.keras` format without retraining.
```bash
python phase2_training.py --convert-only
```

### Phase 3 — Evaluation (validation set)
Generate evaluation reports (confusion matrix, ROC curves, and markdown report) for all trained models.
```bash
python phase3_evaluation.py --all
```

Select the best model (by macro F1) and generate its report.
```bash
python phase3_evaluation.py --best
```

Evaluate a specific model by name (useful for comparing models manually).
```bash
python phase3_evaluation.py --model ResNet50
```

### Phase 3 — Inference (individual images / folders)
Run inference on a single image using the best model (or a specific model if desired).
```bash
python phase3_inference.py --image path/to/img.jpg --best
```

Run inference on a folder of images. This is useful for batch predictions; it does not compute accuracy.
```bash
python phase3_inference.py --dir path/to/images --best
```

### Phase 3 — Holdout evaluation (accuracy + confusion matrix)
Evaluate a holdout directory that is organized into class-named subfolders (e.g., `holdout/Fresh/`, `holdout/Spoiled/`). This computes accuracy and confusion matrix.
```bash
python phase3_holdout.py --dir holdout --best
```

### Misc (reports & justification)
Regenerate the model selection justification document, which explains why the best model was chosen.
```bash
python generate_model_selection_justification.py
```

> ✅ Note: `phase3_inference.py` is the general inference tool (any set of images, no labels needed). `phase3_holdout.py` is the “evaluation” tool: it can compute accuracy/confusion, but only when images are organized into class-named folders so it can infer the correct labels.

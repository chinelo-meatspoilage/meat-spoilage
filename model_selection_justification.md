# Model Selection Justification
**Selected model:** `ResNet50`
**Task:** Binary classification — Fresh vs Spoiled
**Date:** May 2026

This document explains why ResNet50 was chosen as the production model for the
meat freshness classification system. All metrics are computed on the validation
set (451 images: 178 Fresh, 273 Spoiled) and are sourced directly from
`phase2_models/ResNet50/eval_report.json`.

---

## Selection Criterion

The **macro-average F1-score** was used as the primary selection criterion because:
- It averages F1 equally across both classes, regardless of class size.
- It is not inflated by the majority (Spoiled) class.
- It balances precision and recall, which are both critical for food safety.

Secondary criteria: Spoiled Recall (the safety-critical metric — a false negative
means labeling spoiled meat as fresh, which is dangerous), and ROC-AUC.

---

## Full Validation Metrics — All Three Models

### MobileNetV2

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Fresh | 0.8372 | 0.8090 | 0.8229 | 178 |
| Spoiled | 0.8781 | 0.8974 | 0.8877 | 273 |
| **Macro avg** | **0.8577** | **0.8532** | **0.8553** | 451 |

- **Accuracy:** 86.25%
- **ROC-AUC:** 0.9455
- **Confusion matrix:** [[144, 34], [28, 245]]
  - 34 Fresh images falsely predicted Spoiled (false positives)
  - 28 Spoiled images falsely predicted Fresh (false negatives — dangerous)
- **Verdict:** Unacceptable for food safety. 28 dangerous misclassifications.

---

### EfficientNetB0

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Fresh | 0.9708 | 0.9326 | 0.9513 | 178 |
| Spoiled | 0.9571 | 0.9817 | 0.9693 | 273 |
| **Macro avg** | **0.9640** | **0.9571** | **0.9603** | 451 |

- **Accuracy:** 96.23%
- **ROC-AUC:** 0.9926
- **Confusion matrix:** [[166, 12], [5, 268]]
  - 12 Fresh images falsely predicted Spoiled
  - 5 Spoiled images falsely predicted Fresh (dangerous)
- **Verdict:** Good, but 5 dangerous false negatives. Beaten by ResNet50 on all metrics.

---

### ResNet50 ⭐ — SELECTED

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Fresh | 0.9834 | **1.0000** | 0.9916 | 178 |
| Spoiled | **1.0000** | 0.9890 | 0.9945 | 273 |
| **Macro avg** | **0.9917** | **0.9945** | **0.9931** | 451 |

- **Accuracy:** 99.33%
- **ROC-AUC:** 0.9998
- **Confusion matrix:** [[178, 0], [3, 270]]
  - **0** Fresh images falsely predicted Spoiled (zero false alarms)
  - **3** Spoiled images falsely predicted Fresh (only 3 dangerous misclassifications in 273)
- **Verdict:** Near-perfect. Selected as production model.

---

## Key Reasoning

| Criterion | MobileNetV2 | EfficientNetB0 | **ResNet50** |
|---|---|---|---|
| Macro F1 | 0.8553 | 0.9603 | **0.9931** |
| Spoiled Recall | 89.74% | 98.17% | **98.90%** |
| Fresh Recall | 80.90% | 93.26% | **100.00%** |
| Dangerous false negatives | 28 | 5 | **3** |
| ROC-AUC | 0.9455 | 0.9926 | **0.9998** |

ResNet50 wins on every single metric. Its 50-layer deep residual architecture with
skip connections provides richer feature representations than the lighter MobileNetV2
(~2.2M parameters) and EfficientNetB0 (~4.0M parameters), despite ResNet50's larger
size (~23.5M parameters). For a food safety application, the extra computational cost
is entirely justified by the reduction in dangerous false negatives.

---

## Key outputs
- Confusion matrix: `phase3_reports/ResNet50/confusion_matrix.png`
- ROC curve: `phase3_reports/ResNet50/roc_curve.png`
- Full validation report: `phase3_reports/ResNet50/report.md`
- Holdout evaluation (46 unseen images): `phase3_reports/holdout_report.md` — **100% accuracy**
# Model Selection Justification
**Selected model:** `ResNet50`
**Task:** Binary classification — Fresh vs Spoiled

This document summarizes why ResNet50 was chosen as the best model for the
meat freshness classification task. Selection is based on evaluation metrics
computed on the held-out validation set (451 images: 178 Fresh, 273 Spoiled).

## Key reasoning
- **Highest macro-average F1 score** across all three trained architectures.
- **Spoiled Recall = 98.9%** — the safety-critical metric (missing spoiled meat is dangerous).
- **Fresh Recall = 98.9%** — minimal false alarms.
- Lowest validation loss (0.039) compared to MobileNetV2 (0.348) and EfficientNetB0 (0.125).

## Final validation metrics (ResNet50 — binary)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Fresh | 0.983 | 0.989 | 0.986 | 178 |
| Spoiled | 0.993 | 0.989 | 0.991 | 273 |

**Macro avg F1**: 0.988  
**Accuracy**: 98.9%  
**ROC-AUC**: 0.999

## Key outputs
- Confusion matrix: `phase3_reports/ResNet50/confusion_matrix.png`
- ROC curve: `phase3_reports/ResNet50/roc_curve.png`
- Full report: `phase3_reports/ResNet50/report.md`
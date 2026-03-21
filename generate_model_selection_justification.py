"""Generate a short justification document explaining why a particular model was chosen.

This script reads `phase3_reports/selected_model.txt` and the corresponding
`phase3_reports/<model>/report.md` (or the JSON report) and writes a small
`model_selection_justification.md` file.

It is meant to be re-run whenever you retrain/evaluate models.
"""

from pathlib import Path


REPORT_ROOT = Path(__file__).resolve().parent / "phase3_reports"
OUTPUT = Path(__file__).resolve().parent / "model_selection_justification.md"


def _read_selected_model() -> str:
    sel = REPORT_ROOT / "selected_model.txt"
    if not sel.exists():
        raise FileNotFoundError(f"Expected selected_model.txt at {sel}")
    return sel.read_text().strip()


def _read_report_md(model_name: str) -> str:
    report_md = REPORT_ROOT / model_name / "report.md"
    if not report_md.exists():
        raise FileNotFoundError(
            f"Expected report.md for {model_name} at {report_md}")
    return report_md.read_text()


def main() -> None:
    model = _read_selected_model()
    report_md = _read_report_md(model)

    justification_lines = [
        "# Model Selection Justification\n",
        f"**Selected model:** `{model}`\n",
        "\n",
        "This document summarizes why the above model was chosen as the best model for the\n",
        "meat freshness classification task. The selection is based on the evaluation metrics\n",
        "computed on the validation set (see `phase3_reports/<model>/report.md`).\n",
        "\n",
        "## Key reasoning\n",
        "- The model selected is the one with the **highest macro-average F1 score**.\n",
        "- Macro F1 is preferred because it treats both classes equally, which is important\n",
        "  for this binary Fresh vs Spoiled classification task.\n",
        "- A strong confusion matrix (low cross-class errors) is also used as a tie-breaker.\n",
        "\n",
        "## Metrics for the selected model\n",
        "(Copied from the evaluation report below.)\n",
        "\n",
        report_md,
    ]

    OUTPUT.write_text("".join(justification_lines))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

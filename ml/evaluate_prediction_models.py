"""Print the saved held-out metrics for the research and PDF-deployable HGB models.

This is read-only: it retrains nothing and does not modify any model or output.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def show(label: str, row: pd.Series) -> None:
    print(f"\n{label}")
    print(f"  evaluation: {row.get('validation', 'fixed 80/20 stratified held-out test')}")
    print(f"  features: {row.get('feature_set', row.get('features', 'not recorded'))}")
    for metric in METRICS:
        print(f"  {metric}: {float(row[metric]):.4f}")


def main() -> None:
    research = pd.read_csv(ROOT / "outputs" / "classification_benchmark_metrics.csv")
    deployed = pd.read_csv(ROOT / "outputs" / "model_optimization" / "model_leaderboard.csv")
    print("Research benchmark: fixed stratified 80/20 held-out test")
    for _, row in research.sort_values("model").iterrows():
        show(f"  {row['model']}", row)
    print("\nPDF/document candidates: fixed stratified 80/20 held-out test; selection uses 3-fold training-only CV")
    for _, row in deployed.sort_values("model").iterrows():
        show(f"  {row['model']}", row)
    print("\nNote: the research HGB is not a safe PDF replacement because backward_citations is not reliably PDF-extractable and has unresolved prediction-time availability.")


if __name__ == "__main__":
    main()

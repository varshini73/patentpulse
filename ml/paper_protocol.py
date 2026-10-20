"""Approximate Han et al. (2022) three-tier classification protocol.

This is deliberately separate from PatentPulse's binary >=10-citation predictor.
The workbook reproduces the paper's cohort and tier counts, but not all 13 final
indicators (notably cited-patent IPC and Web of Science/JCR measures), nor its
published random seed or exact parameter optimization.  Results are therefore
an approximate, reproducible comparison rather than an exact reproduction.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dataset.xlsx"
OUT = ROOT / "outputs" / "v3"
SEED = 2026
TIERS = ["C0", "C1", "C2"]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(RAW)
    pub_year = numeric(raw["PY"].astype(str).str[:4])
    citations = numeric(raw["CITED TIME IN 3 YEARS"])
    eligible = raw[pub_year.between(2000, 2016) & (citations >= 10)].copy()
    tier = pd.cut(citations.loc[eligible.index], bins=[9, 40, 62, np.inf], labels=TIERS).astype(str)
    # Direct/near-direct workbook equivalents. VTciting is retained only as a
    # proxy because the paper's final 13 selected TCciting, which cannot be
    # reconstructed without IPC data for every cited patent.
    features = pd.DataFrame({
        "VC_ipc_count_approx": eligible["IPC"].fillna("").astype(str).map(lambda v: 0 if not v or v == "-" else len(v.split("|"))),
        "FM_family_size": numeric(eligible["FAMILY PATENTS NUMBER"]),
        "PG_pages": numeric(eligible["PAGES"]),
        "CL_claims": numeric(eligible["CLAIMS NUMBER"]),
        "NPL_count": numeric(eligible["NUMBER OF NPL"]),
        "IV_inventors": numeric(eligible["NUMBER OF INVENTORS"]),
        "VTciting_proxy": numeric(eligible["NUMBER OF CITATION"]),
    })
    x_train, x_test, y_train, y_test, pn_train, pn_test = train_test_split(
        features, tier, eligible["PN"], test_size=.20, stratify=tier, random_state=SEED
    )
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())])
    models = {
        "majority_baseline": Pipeline([("prep", prep), ("model", DummyClassifier(strategy="most_frequent"))]),
        "rbf_svm": Pipeline([("prep", prep), ("model", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED))]),
        "random_forest": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=SEED, n_jobs=1))]),
    }
    rows, predictions = [], []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, pred, labels=TIERS, zero_division=0)
        row = {"model": name, "protocol": "approximate paper three-tier 80/20 stratified split", "seed": SEED,
               "train_records": len(x_train), "test_records": len(x_test), "accuracy": accuracy_score(y_test, pred),
               "balanced_accuracy": balanced_accuracy_score(y_test, pred), "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
               "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
               "c0_precision": precision[0], "c0_recall": recall[0], "c0_f1": f1[0], "c0_support": support[0],
               "c1_precision": precision[1], "c1_recall": recall[1], "c1_f1": f1[1], "c1_support": support[1],
               "c2_precision": precision[2], "c2_recall": recall[2], "c2_f1": f1[2], "c2_support": support[2],
               "feature_set": "; ".join(features.columns),
               "xgboost_status": "not evaluated: xgboost is not installed", 
               "reproduction_status": "approximate: workbook cohort/tier thresholds match; exact 13 paper indicators and seed unavailable"}
        rows.append(row)
        cm = confusion_matrix(y_test, pred, labels=TIERS)
        fig, ax = plt.subplots(figsize=(4.5, 3.8)); image = ax.imshow(cm, cmap="Blues")
        for (r, c), value in np.ndenumerate(cm): ax.text(c, r, str(value), ha="center", va="center")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=TIERS, yticklabels=TIERS, xlabel="Predicted tier", ylabel="Actual tier", title=f"Three-tier protocol: {name}")
        fig.colorbar(image, ax=ax, label="Patents"); fig.tight_layout(); fig.savefig(OUT / f"{name}_confusion_matrix.png", dpi=160); plt.close(fig)
        predictions.append(pd.DataFrame({"pn": pn_test.values, "actual_tier": y_test.values, "predicted_tier": pred, "model": name}))
        print(f"[paper-protocol] {name}: accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, balanced-accuracy={row['balanced_accuracy']:.4f}")
    pd.DataFrame(rows).sort_values("model").to_csv(OUT / "three_tier_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "three_tier_predictions.csv", index=False)
    mapping = {
        "cohort": "published 2000-2016, cited >=10 in three years", "counts": tier.value_counts().sort_index().to_dict(),
        "paper_protocol": "paper reports 80/20 random train/test; seed and full tuning unavailable", "available_features": list(features.columns),
        "unavailable_paper_features": ["bottom-level IPC novelty/combinations", "figures", "cited-patent IPC diversity/novelty", "Web of Science/JCR science-citation features", "verified inventor-history VPaverage"],
        "important_limitation": "C0 is 997/1149 (86.77%); accuracy close to this majority baseline does not demonstrate strong C1/C2 discrimination.",
        "timing_limitation": "Family size and NUMBER OF CITATION are retained only as retrospective workbook fields; their values at the intended prediction time are not verified. TCciting is a non-equivalent proxy for the paper's cited-patent IPC indicator.",
        "paper_reported_reference": {"SVM": .861, "Random Forest": .852, "XGBoost": .839},
    }
    (OUT / "three_tier_protocol_metadata.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"[paper-protocol] cohort={len(eligible)}, tiers={mapping['counts']}, outputs={OUT}")


if __name__ == "__main__":
    main()

"""Run PatentPulse Review 2 data-mining experiments against the SQLite warehouse.

The classification task reproduces the dataset's defined three-year citation
impact label; it is not a claim to predict future citations in deployment.
"""
from __future__ import annotations

import re
import sqlite3
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score,
                             root_mean_squared_error, roc_auc_score, silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "warehouse" / "patentpulse.db"
RAW = ROOT / "data" / "raw" / "dataset.xlsx"
OUT = ROOT / "outputs"
SEED = 2026
NUMERIC = ["pages", "claims", "inventors", "family_size", "npl_count", "backward_citations", "ipc_count"]
CATEGORICAL = ["country_code", "kind_code", "ipc_subclass"]
SAFE_NUMERIC = [name for name in NUMERIC if name != "family_size"]
ENRICHED_NUMERIC = ["ipc_subclass_diversity", "ipc_section_diversity", "ipc_pair_count",
                    "new_ipc_subclass_count", "new_ipc_pair_count", "inventor_prior_patent_mean",
                    "inventor_prior_patent_max", "title_word_count", "abstract_word_count"]


def warehouse_data() -> pd.DataFrame:
    """Return only fields needed by Review 2, joined to readable dimensions."""
    query = """
    SELECT f.pn, t.year AS pub_year, f.pages, f.claims, f.inventors,
           f.family_size, f.npl_count, f.backward_citations, f.ipc_count,
           f.is_highly_cited, c.country_code, k.kind_code, g.ipc_subclass,
           g.ipc_section
    FROM fact_patent f
    JOIN dim_time t ON t.time_key = f.pub_time_key
    JOIN dim_country c ON c.country_key = f.country_key
    JOIN dim_kind k ON k.kind_key = f.kind_key
    JOIN dim_tech g ON g.tech_key = f.tech_key
    """
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(query, con)


def enriched_classification_data(df: pd.DataFrame) -> pd.DataFrame:
    """Derive publication-time IPC, inventor-history, and text-length features from raw data.

    Historical values use only strictly earlier application years: patents from the
    same year never contribute to one another's history.
    """
    raw = pd.read_excel(RAW, usecols=["PN", "APP YEAR", "IPC", "INVENTORS", "TITLE", "ABSTRACT"])
    raw = raw.drop_duplicates("PN").copy()
    raw["app_year"] = pd.to_numeric(raw["APP YEAR"], errors="coerce")
    raw["ipc_codes"] = raw["IPC"].astype(str).str.upper().apply(lambda value: set(re.findall(r"[A-H]\d{2}[A-Z]", value)))
    raw["inventor_names"] = raw["INVENTORS"].astype(str).apply(lambda value: {name.strip() for name in value.split("|") if name.strip() and name.strip() != "-"})
    raw["ipc_subclass_diversity"] = raw["ipc_codes"].apply(len)
    raw["ipc_section_diversity"] = raw["ipc_codes"].apply(lambda codes: len({code[0] for code in codes}))
    raw["ipc_pair_count"] = raw["ipc_codes"].apply(lambda codes: len(list(combinations(sorted(codes), 2))))
    raw["title_word_count"] = raw["TITLE"].astype(str).str.findall(r"\b\w+\b").str.len()
    raw["abstract_word_count"] = raw["ABSTRACT"].astype(str).str.findall(r"\b\w+\b").str.len()
    historical = {name: [] for name in ["new_ipc_subclass_count", "new_ipc_pair_count", "inventor_prior_patent_mean", "inventor_prior_patent_max"]}
    seen_codes, seen_pairs, inventor_counts = set(), set(), Counter()
    for _, group in raw.sort_values("app_year").groupby("app_year", sort=True):
        for row in group.itertuples():
            pairs = set(combinations(sorted(row.ipc_codes), 2))
            prior = [inventor_counts[name] for name in row.inventor_names]
            historical["new_ipc_subclass_count"].append((row.Index, len(row.ipc_codes - seen_codes)))
            historical["new_ipc_pair_count"].append((row.Index, len(pairs - seen_pairs)))
            historical["inventor_prior_patent_mean"].append((row.Index, float(np.mean(prior)) if prior else 0.0))
            historical["inventor_prior_patent_max"].append((row.Index, float(max(prior)) if prior else 0.0))
        for row in group.itertuples():
            seen_codes.update(row.ipc_codes)
            seen_pairs.update(combinations(sorted(row.ipc_codes), 2))
            inventor_counts.update(row.inventor_names)
    for name, values in historical.items():
        raw[name] = 0.0
        raw.loc[[index for index, _ in values], name] = [value for _, value in values]
    features = raw[["PN"] + ENRICHED_NUMERIC].rename(columns={"PN": "pn"})
    return df.merge(features, on="pn", how="left", validate="one_to_one")


def classification(df: pd.DataFrame) -> None:
    """Compare current and safely enriched features under the fixed split."""
    enriched = enriched_classification_data(df).dropna(subset=["is_highly_cited"]).copy()
    y = enriched["is_highly_cited"].astype(int)
    train_index, test_index = train_test_split(enriched.index, test_size=0.20, random_state=SEED, stratify=y)
    print(f"[classification] baseline={NUMERIC + CATEGORICAL}; enriched={SAFE_NUMERIC + ENRICHED_NUMERIC + CATEGORICAL}")
    print(f"[classification] enriched missing={enriched[ENRICHED_NUMERIC].isna().sum().to_dict()}; labels={y.value_counts().sort_index().to_dict()}")
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

    def svm_prep(numeric):
        return ColumnTransformer([
            ("num", Pipeline([( "impute", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([( "impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), CATEGORICAL),
        ])

    def hgb_prep(numeric):
        return ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([( "impute", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan))]), CATEGORICAL),
        ])

    feature_sets = {"baseline": NUMERIC, "enriched_timing_safe": SAFE_NUMERIC + ENRICHED_NUMERIC}
    experiments = {}
    baseline_numeric = feature_sets["baseline"]
    enriched_numeric = feature_sets["enriched_timing_safe"]
    experiments["baseline_svm"] = (baseline_numeric, Pipeline([( "prepare", svm_prep(baseline_numeric)), ("model", SVC(kernel="rbf", random_state=SEED))]), {"model__C": [0.5, 1.0], "model__gamma": ["scale"]}, "log1p/scaled numeric plus grouped one-hot categorical")
    for label, numeric in [("baseline", baseline_numeric), ("enriched_timing_safe", enriched_numeric)]:
        experiments[f"{label}_hgb"] = (numeric, Pipeline([( "prepare", hgb_prep(numeric)), ("model", HistGradientBoostingClassifier(categorical_features=[False] * len(numeric) + [True] * len(CATEGORICAL), max_iter=100, max_leaf_nodes=7, random_state=SEED))]), {"model__l2_regularization": [0.0]}, "median numeric plus native categorical boosting")
    searches = {}
    for name, (numeric, pipeline, grid, _) in experiments.items():
        x_train, y_train = enriched.loc[train_index, numeric + CATEGORICAL], y.loc[train_index]
        search = GridSearchCV(pipeline, grid, scoring={"accuracy": "accuracy", "f1": "f1"}, refit="accuracy", cv=cv, n_jobs=1)
        search.fit(x_train, y_train)
        searches[name] = search
        print(f"[classification] {name}: CV accuracy={search.best_score_:.3f}; params={search.best_params_}", flush=True)

    models = {name: search.best_estimator_ for name, search in searches.items()}
    models["majority_baseline"] = Pipeline([( "prepare", svm_prep(NUMERIC)), ("model", DummyClassifier(strategy="most_frequent"))])
    models["baseline_random_forest"] = Pipeline([( "prepare", svm_prep(NUMERIC)), ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=1))])
    feature_map = {name: numeric for name, (numeric, _, _, _) in experiments.items()}
    feature_map["majority_baseline"] = NUMERIC
    feature_map["baseline_random_forest"] = NUMERIC
    feature_notes = {name: note for name, (_, _, _, note) in experiments.items()}
    feature_notes["majority_baseline"] = "current baseline feature set"
    feature_notes["baseline_random_forest"] = "current baseline feature set; class-weighted random forest"
    cv_accuracy = {name: search.best_score_ for name, search in searches.items()}
    cv_f1 = {name: search.cv_results_["mean_test_f1"][search.best_index_] for name, search in searches.items()}
    cv_accuracy["majority_baseline"] = np.nan; cv_f1["majority_baseline"] = np.nan
    cv_accuracy["baseline_random_forest"] = np.nan; cv_f1["baseline_random_forest"] = np.nan

    metric_rows, fitted = [], {}
    for name, pipeline in models.items():
        numeric = feature_map[name]
        x_train, x_test = enriched.loc[train_index, numeric + CATEGORICAL], enriched.loc[test_index, numeric + CATEGORICAL]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        score = pipeline.predict_proba(x_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.decision_function(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        metric_rows.append({
            "model": name, "train_records": len(x_train), "test_records": len(x_test),
            "positive_train": int(y_train.sum()), "positive_test": int(y_test.sum()),
            "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, score), "pr_auc": average_precision_score(y_test, score),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "cv_accuracy_mean": cv_accuracy[name], "cv_f1_mean": cv_f1[name],
            "feature_preprocessing": feature_notes[name],
            "selected_hyperparameters": str(searches[name].best_params_) if name in searches else ("n_estimators=300, min_samples_leaf=2, class_weight=balanced" if name == "baseline_random_forest" else "most_frequent"),
            "paper_reported_svm_accuracy": 0.861,
        })
        fitted[name] = (pipeline, pred, score)
    # Accuracy is the requested primary target; CV F1 breaks any exact accuracy tie.
    selected_name = max((row for row in metric_rows if row["model"] in searches), key=lambda row: (row["cv_accuracy_mean"], row["cv_f1_mean"]))["model"]
    for row in metric_rows: row["selected_by_training_cv"] = row["model"] == selected_name
    selected_model, selected_pred, selected_score = fitted[selected_name]
    pd.DataFrame(metric_rows).sort_values("model").to_csv(OUT / "classification_metrics.csv", index=False)
    pd.DataFrame({"pn": enriched.loc[test_index, "pn"].values, "actual_label": y.loc[test_index].values,
                  "predicted_label": selected_pred, "predicted_probability": selected_score,
                  "selected_model": selected_name}).to_csv(
        OUT / "classification_predictions.csv", index=False)
    cm = confusion_matrix(y_test, selected_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (row, col), val in np.ndenumerate(cm):
        ax.text(col, row, str(val), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not highly cited", "Highly cited"],
           yticklabels=["Not highly cited", "Highly cited"], xlabel="Predicted label", ylabel="Actual label",
           title=f"{selected_name.replace('_', ' ').title()}: Highly Cited Patent Classification")
    fig.colorbar(im, ax=ax, label="Patent count")
    fig.tight_layout(); fig.savefig(OUT / "classification_confusion_matrix.png", dpi=160); plt.close(fig)
    comparison = pd.DataFrame(metric_rows).sort_values("cv_f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(comparison["model"], comparison["accuracy"], label="Held-out accuracy", color="#4472c4")
    ax.scatter(comparison["cv_accuracy_mean"], comparison["model"], color="#c55a11", zorder=3, label="Training CV accuracy")
    ax.set(title="Classification Model Comparison", xlabel="Accuracy", ylabel="Model")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "classification_model_comparison.png", dpi=160); plt.close(fig)
    chosen = next(row for row in metric_rows if row["model"] == selected_name)
    print(f"[classification] train={len(train_index):,}; test={len(test_index):,}; selected={selected_name}; CV accuracy={chosen['cv_accuracy_mean']:.3f}; test accuracy={chosen['accuracy']:.3f}; F1={chosen['f1']:.3f}")


def chronological_classification(df: pd.DataFrame) -> None:
    """Evaluate a frozen 2013 information-cutoff model on later cohorts."""
    labelled = df.dropna(subset=["is_highly_cited"]).copy()
    train = labelled[labelled["pub_year"].between(2000, 2010)].copy()
    test = labelled[labelled["pub_year"].between(2011, 2013)].copy()
    if train.empty or test.empty:
        raise ValueError("The warehouse does not contain the required 2000-2013 chronological cohorts.")
    numeric = NUMERIC  # family_size is retained only for comparison and flagged below.
    x_train, y_train = train[numeric + CATEGORICAL], train["is_highly_cited"].astype(int)
    x_test, y_test = test[numeric + CATEGORICAL], test["is_highly_cited"].astype(int)
    print(f"[chronological] information cutoff=end-2013; train publications=2000-2010 ({len(train):,}, classes={y_train.value_counts().sort_index().to_dict()}); test publications=2011-2013 ({len(test):,}, classes={y_test.value_counts().sort_index().to_dict()})")
    svm_prep = ColumnTransformer([
        ("num", Pipeline([( "impute", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([( "impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), CATEGORICAL),
    ])
    hgb_prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([( "impute", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan))]), CATEGORICAL),
    ])
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    experiments = {
        "chronological_log_numeric_svm": (Pipeline([( "prepare", svm_prep), ("model", SVC(kernel="rbf", random_state=SEED))]), {"model__C": [0.5, 1.0], "model__gamma": ["scale", 0.01]}, "log1p/scaled numeric + grouped one-hot categorical"),
        "chronological_hgb": (Pipeline([( "prepare", hgb_prep), ("model", HistGradientBoostingClassifier(categorical_features=[False] * len(numeric) + [True] * len(CATEGORICAL), max_iter=100, random_state=SEED))]), {"model__max_leaf_nodes": [3, 7], "model__l2_regularization": [0.0, 1.0]}, "median numeric + native categorical boosting"),
    }
    rows, fitted = [], {}
    for name, (pipeline, grid, preprocessing) in experiments.items():
        search = GridSearchCV(pipeline, grid, scoring={"accuracy": "accuracy", "f1": "f1"}, refit="accuracy", cv=cv, n_jobs=1)
        search.fit(x_train, y_train)
        model = search.best_estimator_; pred = model.predict(x_test)
        score = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({"model": name, "train_publication_years": "2000-2010", "test_publication_years": "2011-2013", "training_label_observable_by": "end-2013", "test_label_observable_by": "end-2016", "train_records": len(train), "test_records": len(test), "positive_train": int(y_train.sum()), "positive_test": int(y_test.sum()), "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0), "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0), "roc_auc": roc_auc_score(y_test, score), "pr_auc": average_precision_score(y_test, score), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "cv_accuracy_mean": search.best_score_, "cv_f1_mean": search.cv_results_["mean_test_f1"][search.best_index_], "feature_preprocessing": preprocessing, "selected_hyperparameters": str(search.best_params_), "timing_note": "family_size retained for comparability but timing-uncertain; citation-window fields excluded"})
        fitted[name] = (pred, score)
        print(f"[chronological] {name}: CV accuracy={search.best_score_:.3f}; test accuracy={rows[-1]['accuracy']:.3f}")
    selected_name = max(rows, key=lambda row: (row["cv_accuracy_mean"], row["cv_f1_mean"]))["model"]
    for row in rows: row["selected_by_training_cv"] = row["model"] == selected_name
    selected_pred, selected_score = fitted[selected_name]
    pd.DataFrame(rows).sort_values("model").to_csv(OUT / "chronological_classification_metrics.csv", index=False)
    pd.DataFrame({"pn": test["pn"].values, "pub_year": test["pub_year"].values, "actual_label": y_test.values, "predicted_label": selected_pred, "predicted_score": selected_score, "selected_model": selected_name}).to_csv(OUT / "chronological_classification_predictions.csv", index=False)
    cm = confusion_matrix(y_test, selected_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(cm, cmap="Blues")
    for (row, col), value in np.ndenumerate(cm): ax.text(col, row, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not highly cited", "Highly cited"], yticklabels=["Not highly cited", "Highly cited"], xlabel="Predicted label", ylabel="Actual label", title=f"Chronological: {selected_name.replace('_', ' ').title()}")
    fig.colorbar(image, ax=ax, label="Patent count"); fig.tight_layout(); fig.savefig(OUT / "chronological_classification_confusion_matrix.png", dpi=160); plt.close(fig)
    chosen = next(row for row in rows if row["model"] == selected_name)
    print(f"[chronological] selected={selected_name}; CV accuracy={chosen['cv_accuracy_mean']:.3f}; test accuracy={chosen['accuracy']:.3f}; F1={chosen['f1']:.3f}")


def svm_optimization_experiment(df: pd.DataFrame) -> None:
    """Run a separate, training-CV-selected SVM experiment on the fixed random split."""
    labelled = df.dropna(subset=["is_highly_cited"]).copy()
    x, y = labelled[NUMERIC + CATEGORICAL], labelled["is_highly_cited"].astype(int)
    x_train, x_test, y_train, y_test, _, id_test = train_test_split(
        x, y, labelled["pn"], test_size=0.20, random_state=SEED, stratify=y
    )
    print("[svm] paper mapping: VC=ipc_count; FM=family_size (timing-uncertain); PG=pages; CL=claims; VTciting=backward_citations; NPL=npl_count; IV=inventors. Missing: IPC novelty/combination at paper granularity, figures, cited-patent IPC diversity/novelty, science-citation indicators, and inventor history.")
    print(f"[svm] fixed random split: train={len(x_train):,} ({y_train.value_counts().sort_index().to_dict()}), test={len(x_test):,} ({y_test.value_counts().sort_index().to_dict()})")

    def make_prep(log_numeric: bool):
        numeric_steps = [("impute", SimpleImputer(strategy="median"))]
        if log_numeric:
            numeric_steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        numeric_steps.append(("scale", StandardScaler()))
        return ColumnTransformer([
            ("num", Pipeline(numeric_steps), NUMERIC),
            ("cat", Pipeline([( "impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), CATEGORICAL),
        ])

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    # Tune the primary log-numeric RBF SVM; keep the two direct ablations fixed
    # so the focused experiment remains practical for the review environment.
    variants = {
        "svm_rbf_standard_unweighted": (Pipeline([( "prepare", make_prep(False)), ("model", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED))]), {}, "standard numeric preprocessing; class_weight=unweighted"),
        "svm_rbf_log_numeric_unweighted": (Pipeline([( "prepare", make_prep(True)), ("model", SVC(kernel="rbf", random_state=SEED))]), {"model__C": [0.5, 1.0], "model__gamma": ["scale", 0.01]}, "log-numeric preprocessing; class_weight=unweighted"),
        "svm_rbf_log_numeric_balanced": (Pipeline([( "prepare", make_prep(True)), ("model", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=SEED))]), {}, "log-numeric preprocessing; class_weight=balanced"),
    }
    rows, models = [], {}
    for name, (pipeline, grid, note) in variants.items():
        search = GridSearchCV(pipeline, grid, scoring={"accuracy": "accuracy", "f1": "f1"}, refit="accuracy", cv=cv, n_jobs=1)
        search.fit(x_train, y_train)
        model = search.best_estimator_; pred = model.predict(x_test); score = model.decision_function(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({"model": name, "candidate_type": "SVM tuned by training CV", "train_records": len(x_train), "test_records": len(x_test), "positive_train": int(y_train.sum()), "positive_test": int(y_test.sum()), "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0), "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0), "roc_auc": roc_auc_score(y_test, score), "pr_auc": average_precision_score(y_test, score), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "cv_accuracy_mean": search.best_score_, "cv_f1_mean": search.cv_results_["mean_test_f1"][search.best_index_], "preprocessing": note, "selected_hyperparameters": str(search.best_params_) if grid else "fixed C=1.0, gamma=scale", "timing_note": "family_size timing-uncertain; all citation-window outcome fields excluded"})
        models[name] = (model, pred, score)
        print(f"[svm] {name}: CV accuracy={search.best_score_:.3f}; test accuracy={rows[-1]['accuracy']:.3f}")

    # Existing non-SVM baselines evaluated on exactly the same held-out records.
    hgb_prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMERIC), ("cat", Pipeline([( "impute", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan))]), CATEGORICAL)])
    comparators = {
        "majority_baseline": (Pipeline([( "prepare", make_prep(True)), ("model", DummyClassifier(strategy="most_frequent"))]), "Majority-class baseline"),
        "baseline_hgb": (Pipeline([( "prepare", hgb_prep), ("model", HistGradientBoostingClassifier(categorical_features=[False] * len(NUMERIC) + [True] * len(CATEGORICAL), max_iter=100, max_leaf_nodes=7, l2_regularization=0.0, random_state=SEED))]), "Existing HGB configuration"),
        "baseline_random_forest": (Pipeline([( "prepare", make_prep(True)), ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=1))]), "Existing class-weighted RF configuration"),
    }
    for name, (model, note) in comparators.items():
        model.fit(x_train, y_train); pred = model.predict(x_test)
        score = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({"model": name, "candidate_type": "Fixed existing comparator", "train_records": len(x_train), "test_records": len(x_test), "positive_train": int(y_train.sum()), "positive_test": int(y_test.sum()), "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0), "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0), "roc_auc": roc_auc_score(y_test, score), "pr_auc": average_precision_score(y_test, score), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "cv_accuracy_mean": np.nan, "cv_f1_mean": np.nan, "preprocessing": note, "selected_hyperparameters": "fixed existing configuration", "timing_note": "family_size timing-uncertain; all citation-window outcome fields excluded"})

    selected_name = max((row for row in rows if row["candidate_type"] == "SVM tuned by training CV"), key=lambda row: (row["cv_accuracy_mean"], row["cv_f1_mean"]))["model"]
    for row in rows: row["selected_by_training_cv"] = row["model"] == selected_name
    selected_model, selected_pred, selected_score = models[selected_name]
    pd.DataFrame(rows).sort_values("model").to_csv(OUT / "svm_optimization_metrics.csv", index=False)
    pd.DataFrame({"pn": id_test.values, "actual_label": y_test.values, "predicted_label": selected_pred, "decision_score": selected_score, "selected_model": selected_name}).to_csv(OUT / "svm_optimization_predictions.csv", index=False)
    cm = confusion_matrix(y_test, selected_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(cm, cmap="Blues")
    for (row, col), value in np.ndenumerate(cm): ax.text(col, row, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not highly cited", "Highly cited"], yticklabels=["Not highly cited", "Highly cited"], xlabel="Predicted label", ylabel="Actual label", title=f"Tuned SVM: {selected_name.replace('_', ' ').title()}")
    fig.colorbar(image, ax=ax, label="Patent count"); fig.tight_layout(); fig.savefig(OUT / "svm_optimization_confusion_matrix.png", dpi=160); plt.close(fig)
    comparison = pd.DataFrame(rows).sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(9, 5)); ax.barh(comparison["model"], comparison["accuracy"], color="#4472c4")
    ax.axvline(.861, color="#c55a11", linestyle="--", label="Paper SVM reference (non-equivalent)")
    ax.set(title="Focused SVM Experiment: Held-out Accuracy", xlabel="Accuracy", ylabel="Model"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "svm_optimization_comparison.png", dpi=160); plt.close(fig)
    chosen = next(row for row in rows if row["model"] == selected_name)
    print(f"[svm] selected={selected_name}; CV accuracy={chosen['cv_accuracy_mean']:.3f}; test accuracy={chosen['accuracy']:.3f}; F1={chosen['f1']:.3f}")


def clustering(df: pd.DataFrame) -> None:
    # Excludes citation outcomes and all impact labels to keep clusters structural/technological.
    x = df[NUMERIC + CATEGORICAL]
    prep = ColumnTransformer([
        ("num", Pipeline([( "impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([( "impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    matrix = prep.fit_transform(x)
    sample = np.random.default_rng(SEED).choice(len(df), size=min(3000, len(df)), replace=False)
    scores = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(matrix)
        scores[k] = silhouette_score(matrix[sample], labels[sample])
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit_predict(matrix)
    assignments = df[["pn"]].copy(); assignments["cluster_id"] = labels
    assignments.to_csv(OUT / "cluster_assignments.csv", index=False)
    summary_source = df[NUMERIC + ["ipc_section"]].copy(); summary_source["cluster_id"] = labels
    summary = summary_source.groupby("cluster_id").agg(
        patents=("ipc_section", "size"), avg_pages=("pages", "mean"), avg_claims=("claims", "mean"),
        avg_inventors=("inventors", "mean"), avg_family_size=("family_size", "mean"),
        avg_backward_citations=("backward_citations", "mean"), dominant_ipc_section=("ipc_section", lambda s: s.mode().iat[0])
    ).reset_index()
    summary["selected_k"] = best_k; summary["silhouette_score"] = scores[best_k]
    summary.round(3).to_csv(OUT / "cluster_summary.csv", index=False)
    coords = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=5, alpha=.55)
    ax.set(title=f"Patent Clusters (K-Means, k={best_k}; silhouette={scores[best_k]:.3f})", xlabel="SVD component 1", ylabel="SVD component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster ID"); fig.tight_layout()
    fig.savefig(OUT / "patent_clusters.png", dpi=160); plt.close(fig)
    print(f"[clustering] patents={len(df):,}; selected k={best_k}; silhouette={scores[best_k]:.3f}")


def ipc_transactions() -> list[list[str]]:
    raw = pd.read_excel(RAW, usecols=["IPC"])
    transactions = []
    for value in raw["IPC"].dropna().astype(str):
        codes = set(re.findall(r"[A-H]\d{2}[A-Z]", value.upper()))
        if codes:
            transactions.append(sorted(codes))
    return transactions


def rules() -> None:
    transactions = ipc_transactions()
    # Pairs answer the IPC co-occurrence question while preventing needless
    # combinatorial expansion from the multi-valued original IPC transactions.
    encoder = TransactionEncoder()
    encoded = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)
    numeric_itemsets = apriori(encoded, min_support=0.02, use_colnames=True, max_len=2)
    itemsets = numeric_itemsets.copy()
    itemsets["itemsets"] = itemsets["itemsets"].apply(lambda v: " | ".join(sorted(v)))
    itemsets.sort_values("support", ascending=False).to_csv(OUT / "frequent_itemsets.csv", index=False)
    discovered = association_rules(numeric_itemsets, metric="confidence", min_threshold=0.35)
    if discovered.empty:
        result = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    else:
        discovered = discovered[(discovered["lift"] > 1) & (discovered["antecedents"].apply(len) == 1) & (discovered["consequents"].apply(len) == 1)]
        result = discovered[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
        for col in ["antecedents", "consequents"]: result[col] = result[col].apply(lambda v: " | ".join(sorted(v)))
        result = result.sort_values(["lift", "confidence"], ascending=False)
    result.to_csv(OUT / "association_rules.csv", index=False)
    if not result.empty:
        strongest = result.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, len(strongest) * .42)))
        ax.barh(strongest["antecedents"] + " → " + strongest["consequents"], strongest["lift"], color="#4472c4")
        ax.set(title="Strongest IPC Subclass Association Rules", xlabel="Lift", ylabel="Rule")
        fig.tight_layout(); fig.savefig(OUT / "association_rules.png", dpi=160); plt.close(fig)
    print(f"[rules] transactions={len(transactions):,}; itemsets={len(itemsets):,}; rules={len(result):,}")


def prediction(df: pd.DataFrame) -> None:
    valid = df[df["pub_year"].between(1900, 2025)].copy()
    years = np.arange(int(valid.pub_year.min()), int(valid.pub_year.max()) + 1)
    domains = ["All technology sections"] + sorted(valid["ipc_section"].dropna().unique())
    series = []
    for domain in domains:
        subset = valid if domain == "All technology sections" else valid[valid["ipc_section"] == domain]
        counts = subset.groupby("pub_year").size().reindex(years, fill_value=0)
        series.append(pd.DataFrame({"pub_year": years, "actual_count": counts.values, "technology_domain": domain}))
    annual_by_domain = pd.concat(series, ignore_index=True)
    annual = annual_by_domain[annual_by_domain["technology_domain"] == "All technology sections"].copy()
    split = max(3, int(np.ceil(len(annual) * .2)))
    train, test = annual.iloc[:-split], annual.iloc[-split:]
    future_years = np.arange(int(annual.pub_year.max()) + 1, int(annual.pub_year.max()) + 4)
    output = []
    total_test = None
    for domain, domain_annual in annual_by_domain.groupby("technology_domain", sort=False):
        domain_annual = domain_annual.sort_values("pub_year").reset_index(drop=True)
        domain_train, domain_test = domain_annual.iloc[:-split], domain_annual.iloc[-split:].copy()
        model = LinearRegression().fit(domain_train[["pub_year"]], domain_train["actual_count"])
        domain_test["linear_regression_prediction"] = np.maximum(0, model.predict(domain_test[["pub_year"]]))
        domain_test["previous_year_baseline"] = domain_annual["actual_count"].shift(1).iloc[-split:].values
        future = pd.DataFrame({"pub_year": future_years, "actual_count": np.nan, "technology_domain": domain,
                               "linear_regression_prediction": np.maximum(0, model.predict(pd.DataFrame({"pub_year": future_years}))),
                               "previous_year_baseline": np.nan, "row_type": "future_forecast"})
        history = domain_annual.assign(linear_regression_prediction=np.nan, previous_year_baseline=np.nan, row_type="historical")
        output.extend([history, domain_test.assign(row_type="held_out_test"), future])
        if domain == "All technology sections": total_test = domain_test
    results = pd.concat(output, ignore_index=True)
    results.to_csv(OUT / "forecast_results.csv", index=False)
    lr_mae = mean_absolute_error(total_test["actual_count"], total_test["linear_regression_prediction"])
    lr_rmse = root_mean_squared_error(total_test["actual_count"], total_test["linear_regression_prediction"])
    baseline_mae = mean_absolute_error(total_test["actual_count"], total_test["previous_year_baseline"])
    print(f"[prediction] years={len(annual):,}; train={len(train):,}; test={len(total_test):,}; linear MAE={lr_mae:.2f}; RMSE={lr_rmse:.2f}; baseline MAE={baseline_mae:.2f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(annual.pub_year, annual.actual_count, marker="o", label="Observed historical count")
    total_future = results[(results["technology_domain"] == "All technology sections") & (results["row_type"] == "future_forecast")]
    ax.plot(total_test.pub_year, total_test.linear_regression_prediction, marker="s", linestyle="--", label="Held-out linear forecast")
    ax.plot(total_future.pub_year, total_future.linear_regression_prediction, marker="s", linestyle="--", label="Future linear forecast")
    ax.axvline(train.pub_year.max() + .5, color="gray", linestyle=":", label="Chronological split")
    ax.set(title="EV Patent Publication Volume: Historical and Forecast", xlabel="Publication year", ylabel="Patent count")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "patent_trend_forecast.png", dpi=160); plt.close(fig)


def main(classification_only: bool = False, chronological_only: bool = False, svm_optimization_only: bool = False) -> None:
    if not DB.exists() or not RAW.exists():
        raise FileNotFoundError("Both data/warehouse/patentpulse.db and data/raw/dataset.xlsx are required.")
    OUT.mkdir(exist_ok=True)
    df = warehouse_data()
    print(f"[validation] warehouse patents={len(df):,}; labelled={df.is_highly_cited.notna().sum():,}; highly cited={df.is_highly_cited.sum():.0f}")
    if svm_optimization_only:
        svm_optimization_experiment(df)
    elif chronological_only:
        chronological_classification(df)
    else:
        classification(df)
    if not classification_only and not chronological_only and not svm_optimization_only:
        clustering(df); rules(); prediction(df)
    mode = "Focused SVM optimization" if svm_optimization_only else ("Chronological classification" if chronological_only else ("Classification" if classification_only else "Review 2"))
    print(f"[complete] {mode} outputs saved to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PatentPulse Review 2 experiments.")
    parser.add_argument("--classification-only", action="store_true", help="Update classification outputs without rerunning other mining tasks.")
    parser.add_argument("--chronological-only", action="store_true", help="Run the separate chronological classification experiment only.")
    parser.add_argument("--svm-optimization-only", action="store_true", help="Run the focused SVM optimization experiment only.")
    main(**vars(parser.parse_args()))

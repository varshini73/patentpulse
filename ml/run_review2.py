"""Run PatentPulse Review 2 data-mining experiments against the SQLite warehouse.

The classification task reproduces the dataset's defined three-year citation
impact label; it is not a claim to predict future citations in deployment.
"""
from __future__ import annotations

import re
import sqlite3
import argparse
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "warehouse" / "patentpulse.db"
RAW = ROOT / "data" / "raw" / "dataset.xlsx"
OUT = ROOT / "outputs"
SEED = 2026
NUMERIC = ["pages", "claims", "inventors", "family_size", "npl_count", "backward_citations", "ipc_count"]
CATEGORICAL = ["country_code", "kind_code", "ipc_subclass"]


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


def classification(df: pd.DataFrame) -> None:
    """Evaluate leakage-free individual and ensemble classifiers on one fixed split."""
    labelled = df.dropna(subset=["is_highly_cited"]).copy()
    x = labelled[NUMERIC + CATEGORICAL]
    y = labelled["is_highly_cited"].astype(int)
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        x, y, labelled["pn"], test_size=0.20, random_state=SEED, stratify=y
    )
    prep = ColumnTransformer([
        ("num", Pipeline([( "impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([( "impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    # Two folds keep the nested ensemble experiment lightweight while preserving
    # stratified, training-only model selection.
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

    # Tune unweighted and class-weighted RBF SVMs separately; no test data is used here.
    svm_searches = {}
    for suffix, class_weight in [("unweighted", None), ("balanced", "balanced")]:
        search = GridSearchCV(
            Pipeline([( "prepare", prep), ("model", SVC(kernel="rbf", class_weight=class_weight, random_state=SEED))]),
            {"model__C": [0.5, 1.0], "model__gamma": ["scale"]},
            scoring="f1", cv=cv, n_jobs=1, refit=True,
        )
        search.fit(x_train, y_train)
        svm_searches[suffix] = search
        print(f"[classification] tuned SVM {suffix}: CV F1={search.best_score_:.3f}", flush=True)
    best_svm_key = max(svm_searches, key=lambda key: svm_searches[key].best_score_)
    best_svm = svm_searches[best_svm_key]
    best_svm_params = best_svm.best_params_
    svm_weight = "balanced" if best_svm_key == "balanced" else None

    def calibrated_svm() -> CalibratedClassifierCV:
        return CalibratedClassifierCV(
            SVC(kernel="rbf", C=best_svm_params["model__C"], gamma=best_svm_params["model__gamma"],
                class_weight=svm_weight, random_state=SEED), cv=2, method="sigmoid"
        )

    # These are the existing RF and a simple LR; all probability-enabled base models
    # are encapsulated in pipelines, including within stacking, to avoid preprocessing leakage.
    def random_forest() -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                      class_weight="balanced", random_state=SEED, n_jobs=1)

    def logistic() -> LogisticRegression:
        return LogisticRegression(C=0.25, max_iter=1500, class_weight="balanced", random_state=SEED)

    def feature_pipeline(model):
        return Pipeline([( "prepare", prep), ("model", model)])

    base_estimators = [feature_pipeline(calibrated_svm()), feature_pipeline(logistic()), feature_pipeline(random_forest())]
    # Select soft-voting weights from out-of-fold training probabilities once,
    # avoiding repeated fitting of identical calibrated base estimators.
    oof_probabilities = np.column_stack([
        cross_val_predict(estimator, x_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        for estimator in base_estimators
    ])
    soft_candidates = [(1, 1, 1), (2, 1, 1), (1, 1, 2)]
    soft_scores = {
        weights: f1_score(y_train, (np.average(oof_probabilities, axis=1, weights=weights) >= 0.5).astype(int), zero_division=0)
        for weights in soft_candidates
    }
    selected_weights = max(soft_scores, key=soft_scores.get)
    soft = VotingClassifier(
        estimators=[("svm", feature_pipeline(calibrated_svm())), ("lr", feature_pipeline(logistic())), ("rf", feature_pipeline(random_forest()))],
        voting="soft", weights=selected_weights,
    )
    print(f"[classification] selected soft-voting weights={selected_weights}: OOF F1={soft_scores[selected_weights]:.3f}", flush=True)
    stacking = StackingClassifier(
        estimators=[("svm", feature_pipeline(calibrated_svm())), ("lr", feature_pipeline(logistic())), ("rf", feature_pipeline(random_forest()))],
        final_estimator=LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED),
        cv=cv, stack_method="predict_proba", n_jobs=1,
    )

    models = {
        "majority_baseline": feature_pipeline(DummyClassifier(strategy="most_frequent")),
        "random_forest": feature_pipeline(random_forest()),
        "svm_rbf_unweighted": svm_searches["unweighted"].best_estimator_,
        "svm_rbf_balanced": svm_searches["balanced"].best_estimator_,
        "soft_voting": soft,
        "stacking": stacking,
    }
    cv_scores = {
        "majority_baseline": 0.0,
        "random_forest": cross_val_score(models["random_forest"], x_train, y_train, scoring="f1", cv=cv, n_jobs=1).mean(),
        "svm_rbf_unweighted": svm_searches["unweighted"].best_score_,
        "svm_rbf_balanced": svm_searches["balanced"].best_score_,
        "soft_voting": soft_scores[selected_weights],
        "stacking": cross_val_score(stacking, x_train, y_train, scoring="f1", cv=cv, n_jobs=1).mean(),
    }
    print(f"[classification] stacking CV F1={cv_scores['stacking']:.3f}", flush=True)
    hyperparameters = {
        "majority_baseline": "most_frequent", "random_forest": "n_estimators=300, min_samples_leaf=2, max_features=sqrt, class_weight=balanced",
        "svm_rbf_unweighted": str(svm_searches["unweighted"].best_params_),
        "svm_rbf_balanced": str(svm_searches["balanced"].best_params_),
        "soft_voting": f"weights={selected_weights}, selected from OOF training probabilities", "stacking": "OOF base probabilities; logistic-regression meta-classifier",
    }
    metric_rows, fitted = [], {}
    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        probability = pipeline.predict_proba(x_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.decision_function(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        metric_rows.append({
            "model": name, "train_records": len(x_train), "test_records": len(x_test),
            "positive_train": int(y_train.sum()), "positive_test": int(y_test.sum()),
            "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability), "pr_auc": average_precision_score(y_test, probability),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "cv_f1_mean": cv_scores[name],
            "selected_hyperparameters": hyperparameters[name], "paper_reported_svm_accuracy": 0.861,
        })
        fitted[name] = (pipeline, pred, probability)
    # The positive-class F1 objective favors detection of highly cited patents;
    # every choice above, including voting weights, is made on training CV only.
    selected_name = max(metric_rows[1:], key=lambda row: row["cv_f1_mean"])["model"]
    for row in metric_rows: row["selected_by_training_cv"] = row["model"] == selected_name
    selected_model, selected_pred, selected_prob = fitted[selected_name]
    pd.DataFrame(metric_rows).sort_values("model").to_csv(OUT / "classification_metrics.csv", index=False)
    pd.DataFrame({"pn": id_test.values, "actual_label": y_test.values,
                  "predicted_label": selected_pred, "predicted_probability": selected_prob,
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
    ax.barh(comparison["model"], comparison["f1"], label="Held-out F1", color="#4472c4")
    ax.scatter(comparison["cv_f1_mean"], comparison["model"], color="#c55a11", zorder=3, label="Training CV F1")
    ax.set(title="Classification Model Comparison", xlabel="Positive-class F1", ylabel="Model")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "classification_model_comparison.png", dpi=160); plt.close(fig)
    chosen = next(row for row in metric_rows if row["model"] == selected_name)
    print(f"[classification] labelled={len(labelled):,}; class-1={y.sum():,} ({y.mean():.1%}); train={len(x_train):,}; test={len(x_test):,}; selected={selected_name}; F1={chosen['f1']:.3f}; accuracy={chosen['accuracy']:.3f}")


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


def main(classification_only: bool = False) -> None:
    if not DB.exists() or not RAW.exists():
        raise FileNotFoundError("Both data/warehouse/patentpulse.db and data/raw/dataset.xlsx are required.")
    OUT.mkdir(exist_ok=True)
    df = warehouse_data()
    print(f"[validation] warehouse patents={len(df):,}; labelled={df.is_highly_cited.notna().sum():,}; highly cited={df.is_highly_cited.sum():.0f}")
    classification(df)
    if not classification_only:
        clustering(df); rules(); prediction(df)
    print(f"[complete] {'Classification' if classification_only else 'Review 2'} outputs saved to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PatentPulse Review 2 experiments.")
    parser.add_argument("--classification-only", action="store_true", help="Update classification outputs without rerunning other mining tasks.")
    main(**vars(parser.parse_args()))

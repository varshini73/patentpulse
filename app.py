"""PatentPulse interactive dashboard. Run with: streamlit run app.py"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "warehouse" / "patentpulse.db"
RAW = ROOT / "data" / "raw" / "dataset.xlsx"
OUT = ROOT / "outputs"
MODEL = ROOT / "models" / "classification_benchmark_model.joblib"

st.set_page_config(page_title="PatentPulse", page_icon="◉", layout="wide")
st.markdown("""<style>
    .stApp { background: #fff; color: #111; }
    [data-testid="stSidebar"] { background: #111; }
    [data-testid="stSidebar"] * { color: #fff; }
    h1, h2, h3 { color: #111; letter-spacing: -.02em; }
    div[data-testid="stMetric"] { border: 1px solid #d5d5d5; padding: 12px; }
    .stButton>button { background:#111; color:#fff; border:0; border-radius:0; }
    .muted { color: #575757; }
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def warehouse() -> pd.DataFrame:
    query = """
    SELECT f.pn, t.year AS pub_year, f.pages, f.claims, f.inventors, f.family_size,
           f.npl_count, f.backward_citations, f.ipc_count, f.cited_3yr,
           f.is_highly_cited, c.country_code, k.kind_code, g.ipc_subclass, g.ipc_section
    FROM fact_patent f JOIN dim_time t ON f.pub_time_key=t.time_key
    JOIN dim_country c ON f.country_key=c.country_key JOIN dim_kind k ON f.kind_key=k.kind_key
    JOIN dim_tech g ON f.tech_key=g.tech_key
    """
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(query, con)


@st.cache_data(show_spinner=False)
def raw_data() -> pd.DataFrame:
    return pd.read_excel(RAW)


@st.cache_data(show_spinner=False)
def output_csv(name: str) -> pd.DataFrame:
    path = OUT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def saved_artifact():
    return joblib.load(MODEL) if MODEL.exists() else None


def show_image(name: str, caption: str) -> None:
    path = OUT / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Saved output not found: {name}")


def metric_grid(df: pd.DataFrame) -> None:
    labelled = df[df.is_highly_cited.notna()]
    positive = int(labelled.is_highly_cited.sum())
    cols = st.columns(4)
    cols[0].metric("Patent records", f"{len(df):,}")
    cols[1].metric("Labelled records", f"{len(labelled):,}")
    cols[2].metric("Highly cited", f"{positive:,}")
    cols[3].metric("Highly cited rate", f"{positive / len(labelled):.1%}")


def home(df: pd.DataFrame) -> None:
    st.title("PatentPulse")
    st.subheader("A data-warehouse and data-mining framework for EV patent impact")
    st.markdown("PatentPulse estimates whether an eligible electric-vehicle patent will be highly cited: at least 10 forward citations in its following three years. It combines a SQLite star-schema warehouse with classification, clustering, association rules, and publication-volume forecasting.")
    metric_grid(df)
    st.markdown("#### Interpretation")
    st.markdown("A prediction is a model estimate, not a guarantee of patent value. The primary benchmark uses a stratified random split; the separate chronological evaluation is more realistic for later cohorts and should be considered when judging deployment readiness. Family size is excluded from the deployed model because its observation time is unclear; backward citations are retained but flagged as timing-uncertain.")


def prediction(df: pd.DataFrame) -> None:
    st.title("Patent prediction")
    artifact = saved_artifact()
    if artifact is None:
        st.error("The saved model is unavailable. Run: python ml/run_review2.py --benchmark-only")
        return
    st.caption(f"Loaded model: {artifact['selected_model'].replace('_', ' ').title()}. Inputs exactly match its saved feature schema.")
    countries = sorted(df.country_code.dropna().unique())
    kinds = sorted(df.kind_code.dropna().unique())
    subclasses = sorted(df.ipc_subclass.dropna().unique())
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            pub_year = st.number_input("Publication year", min_value=1976, max_value=2035, value=2016)
            pages = st.number_input("Pages", min_value=0, value=10)
            claims = st.number_input("Claims", min_value=0, value=10)
            inventors = st.number_input("Number of inventors", min_value=0, value=2)
            npl_count = st.number_input("NPL count", min_value=0, value=0)
            backward = st.number_input("Backward patent citations", min_value=0, value=0)
        with right:
            ipc_count = st.number_input("IPC code count", min_value=0, value=1)
            country = st.selectbox("Primary country", countries, index=countries.index("US") if "US" in countries else 0)
            kind = st.selectbox("Kind code", kinds)
            subclass = st.selectbox("Main IPC subclass", subclasses)
            title = st.text_input("Title", placeholder="Optional patent title")
            abstract = st.text_area("Abstract", placeholder="Optional patent abstract")
        submitted = st.form_submit_button("Predict citation class")
    if submitted:
        if pub_year > 2016:
            st.warning("The labelled benchmark covers patents published through 2016. This future-facing prediction is extrapolative.")
        record = pd.DataFrame([{
            "pages": pages, "claims": claims, "inventors": inventors, "npl_count": npl_count,
            "backward_citations": backward, "ipc_count": ipc_count,
            "title_word_count": len(title.split()), "abstract_word_count": len(abstract.split()),
            "pub_year": pub_year, "country_code": country, "kind_code": kind, "ipc_subclass": subclass,
        }])[artifact["features"]]
        model = artifact["model"]
        label = int(model.predict(record)[0])
        probability = float(model.predict_proba(record)[0, 1])
        st.success("Highly cited" if label else "Not highly cited")
        st.metric("Model probability of highly cited", f"{probability:.1%}")
        st.caption("This is a model probability, not a guarantee. Backward-citation timing may differ for newly filed patents.")
        st.dataframe(record.T.rename(columns={0: "value"}).astype(str), use_container_width=True)


def dashboard(df: pd.DataFrame) -> None:
    st.title("Analytics dashboard")
    years = sorted(df.pub_year.dropna().astype(int).unique())
    c1, c2, c3 = st.columns(3)
    year_range = c1.slider("Publication year", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
    country = c2.selectbox("Country", ["All"] + sorted(df.country_code.unique()))
    section = c3.selectbox("IPC section", ["All"] + sorted(df.ipc_section.unique()))
    filtered = df[df.pub_year.between(*year_range)].copy()
    if country != "All": filtered = filtered[filtered.country_code == country]
    if section != "All": filtered = filtered[filtered.ipc_section == section]
    metric_grid(filtered)
    left, right = st.columns(2)
    with left:
        trend = filtered.groupby("pub_year").size().reset_index(name="patents")
        st.plotly_chart(px.line(trend, x="pub_year", y="patents", markers=True, title="Publication trend", template="simple_white"), use_container_width=True)
        by_country = filtered.groupby("country_code").size().nlargest(12).reset_index(name="patents")
        st.plotly_chart(px.bar(by_country, x="country_code", y="patents", title="Top countries", template="simple_white"), use_container_width=True)
    with right:
        by_section = filtered.groupby("ipc_section").size().reset_index(name="patents")
        st.plotly_chart(px.bar(by_section, x="ipc_section", y="patents", title="IPC sections", template="simple_white"), use_container_width=True)
        benchmark = output_csv("classification_benchmark_metrics.csv")
        if not benchmark.empty:
            plot = benchmark.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"], var_name="metric", value_name="score")
            st.plotly_chart(px.bar(plot, x="model", y="score", color="metric", barmode="group", title="Classification benchmark", template="simple_white"), use_container_width=True)
    show_image("classification_benchmark_confusion_matrix.png", "Selected benchmark confusion matrix")


def mining() -> None:
    st.title("Data-mining modules")
    tabs = st.tabs(["Classification", "K-Means", "Apriori", "Forecasting"])
    with tabs[0]:
        st.markdown("**Purpose:** identify patents likely to meet the three-year highly-cited threshold. The benchmark uses a fixed stratified 80/20 split and training-only cross-validation.")
        metrics = output_csv("classification_benchmark_metrics.csv")
        if not metrics.empty: st.dataframe(metrics[["model", "accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "cv_accuracy_mean"]], use_container_width=True)
        show_image("classification_benchmark_comparison.png", "Held-out benchmark comparison")
    with tabs[1]:
        st.markdown("**Purpose:** group patents by structural and technology attributes using K-Means.")
        summary = output_csv("cluster_summary.csv")
        if not summary.empty: st.dataframe(summary, use_container_width=True)
        show_image("patent_clusters.png", "Patent clusters")
    with tabs[2]:
        st.markdown("**Purpose:** discover frequent IPC-subclass co-occurrences using Apriori association rules.")
        rules = output_csv("association_rules.csv")
        if not rules.empty: st.dataframe(rules.head(20), use_container_width=True)
        show_image("association_rules.png", "Top association rules")
    with tabs[3]:
        st.markdown("**Purpose:** forecast annual patent publication volume with linear regression.")
        forecast = output_csv("forecast_results.csv")
        if not forecast.empty: st.dataframe(forecast.tail(15), use_container_width=True)
        show_image("patent_trend_forecast.png", "Historical volume and forecast")


def explorer(df: pd.DataFrame) -> None:
    st.title("Dataset explorer")
    raw = raw_data()
    search = st.text_input("Search patent number or title")
    years = sorted(pd.to_numeric(raw["PY"].astype(str).str[:4], errors="coerce").dropna().astype(int).unique())
    yr = st.slider("Publication year", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
    display = raw.copy()
    pub = pd.to_numeric(display["PY"].astype(str).str[:4], errors="coerce")
    display = display[pub.between(*yr)]
    if search:
        mask = display["PN"].astype(str).str.contains(search, case=False, na=False) | display["TITLE"].astype(str).str.contains(search, case=False, na=False)
        display = display[mask]
    st.caption(f"{len(display):,} matching records")
    st.dataframe(display.head(200), use_container_width=True, height=420)
    st.download_button("Download filtered CSV", display.to_csv(index=False).encode("utf-8"), "patentpulse_filtered.csv", "text/csv")
    missing = raw.isna().sum().sort_values(ascending=False).rename("missing values").to_frame()
    st.subheader("Missing values")
    st.dataframe(missing, use_container_width=True)


def main() -> None:
    if not DB.exists():
        st.error("Warehouse database not found. Run: python etl/build_warehouse.py")
        st.stop()
    df = warehouse()
    page = st.sidebar.radio("PatentPulse", ["Home", "Patent Prediction", "Analytics Dashboard", "Data Mining", "Dataset Explorer"])
    st.sidebar.caption("Black-and-white academic dashboard")
    if page == "Home": home(df)
    elif page == "Patent Prediction": prediction(df)
    elif page == "Analytics Dashboard": dashboard(df)
    elif page == "Data Mining": mining()
    else: explorer(df)


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz
import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "warehouse" / "patentpulse.db"
RAW = ROOT / "data" / "raw" / "dataset.xlsx"
OUT = ROOT / "outputs"
MODEL_FILE = ROOT / "models" / "classification_benchmark_model.joblib"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="PatentPulse API", version="2.0.0")
origins = [origin.strip() for origin in os.getenv("PATENTPULSE_FRONTEND_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


class PatentFeatures(BaseModel):
    publication_year: int | None = Field(default=None, ge=1900, le=2100)
    pages: float | None = Field(default=None, ge=0)
    claims: float | None = Field(default=None, ge=0)
    inventors: float | None = Field(default=None, ge=0)
    npl_count: float | None = Field(default=None, ge=0)
    backward_citations: float | None = Field(default=None, ge=0)
    ipc_count: float | None = Field(default=None, ge=0)
    country_code: str | None = None
    kind_code: str | None = None
    ipc_subclass: str | None = None
    title: str = ""
    abstract: str = ""


@lru_cache(maxsize=1)
def model_artifact() -> dict[str, Any]:
    if not MODEL_FILE.exists():
        raise HTTPException(503, "Saved model artifact is unavailable. Run the classification benchmark first.")
    return joblib.load(MODEL_FILE)


@app.on_event("startup")
def preload_model() -> None:
    """Fail fast if the persisted inference pipeline is missing or unreadable."""
    model_artifact()


def sql(query: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(query, con, params=params)


def clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def extract_pdf(text: str, pages: int) -> dict[str, Any]:
    compact = re.sub(r"\s+", " ", text).strip()
    def first(patterns: list[str]) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, compact, flags=re.I | re.S)
            if match:
                return clean(match.group(1))
        return None
    title = first([r"(?:^|\b)TITLE\s*[:\-]?\s*(.{3,180}?)(?=\s+(?:ABSTRACT|INVENTOR|FIELD OF)|$)"])
    abstract = first([r"ABSTRACT\s*[:\-]?\s*(.{20,2500}?)(?=\s+(?:CLAIMS?|WHAT IS CLAIMED|DESCRIPTION|BACKGROUND)\b|$)"])
    publication_number = first([r"(?:publication\s*(?:no\.?|number)|pub\.\s*no\.?)\s*[:#]?\s*([A-Z]{2}\s?\d{6,}[A-Z]\d?)"])
    date_value = first([r"(?:publication\s*date|published)\s*[:\-]?\s*([^;]{4,40})"])
    year_match = re.search(r"(?:19|20)\d{2}", date_value or compact[:500])
    inventors_line = first([r"INVENTORS?\s*[:\-]?\s*(.{3,500}?)(?=\s+(?:APPLICANT|ASSIGNEE|ABSTRACT|CLAIMS?)\b|$)"])
    inventor_count = None
    if inventors_line:
        inventor_count = len([item for item in re.split(r";|,|\band\b", inventors_line, flags=re.I) if item.strip()])
    ipc_codes = sorted(set(re.findall(r"\b([A-HY]\d{2}[A-Z])\d*(?:/\d+)?\b", compact.upper())))
    claims_section = first([r"(?:CLAIMS?|WHAT IS CLAIMED)\s*[:\-]?\s*(.{0,12000})"])
    claim_count = len(re.findall(r"(?m)(?:^|\s)(?:\d{1,3})\s*[\.)]", claims_section or "")) or None
    country_match = re.search(r"\b(US|WO|EP|JP|CN|KR|DE|GB|FR|CA|IN)\s?\d{5,}", compact.upper())
    kind_match = re.search(r"\b(?:US\s?)?\d{6,}\s?([A-Z]\d)\b", compact.upper())
    warnings = []
    if not title: warnings.append("Title was not reliably identified; please enter or verify it.")
    if not abstract: warnings.append("Abstract was not reliably identified; please enter or verify it.")
    if not ipc_codes: warnings.append("No IPC/CPC subclass was identified; select one before prediction.")
    warnings.append("Backward citations and NPL count are not reliably extractable from every PDF; verify them before prediction.")
    return {"title": title or "", "abstract": abstract or "", "publication_number": publication_number or "", "publication_year": int(year_match.group()) if year_match else None, "inventor_names": inventors_line or "", "inventors": inventor_count, "pages": pages, "claims": claim_count, "ipc_codes": ipc_codes, "ipc_count": len(ipc_codes) or None, "ipc_subclass": ipc_codes[0] if ipc_codes else None, "country_code": country_match.group(1) if country_match else None, "kind_code": kind_match.group(1) if kind_match else None, "warnings": warnings}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "available" if MODEL_FILE.exists() else "missing"}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    df = sql("SELECT is_highly_cited FROM fact_patent")
    labelled = df[df.is_highly_cited.notna()]
    return {"total_patents": len(df), "labelled_patents": len(labelled), "highly_cited": int(labelled.is_highly_cited.sum()), "non_highly_cited": int((labelled.is_highly_cited == 0).sum())}


@app.get("/api/models")
def models() -> dict[str, Any]:
    artifact = model_artifact()
    metrics = pd.read_csv(OUT / "classification_benchmark_metrics.csv")
    return {"selected_model": artifact["selected_model"], "features": artifact["features"], "timing_note": artifact["timing_note"], "metrics": metrics.to_dict(orient="records")}


@app.post("/api/patents/extract")
async def extract(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in {"application/pdf", "application/x-pdf"} and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are supported.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "PDF exceeds the 10 MB upload limit.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "The uploaded file is not a valid PDF.")
    try:
        document = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        text = "".join(page.get_text("text") for page in document)
        page_count = document.page_count
        document.close()
    except Exception as exc:
        raise HTTPException(400, "The PDF could not be parsed.") from exc
    if len(text.strip()) < 40:
        raise HTTPException(422, "No extractable text was found. The PDF may be scanned or image-only; please use a text-based patent PDF.")
    return extract_pdf(text, page_count)


@app.post("/api/predict")
def predict(payload: PatentFeatures) -> dict[str, Any]:
    artifact = model_artifact()
    values = payload.model_dump()
    warnings = []
    for field in ["publication_year", "pages", "claims", "inventors", "npl_count", "backward_citations", "ipc_count", "country_code", "kind_code", "ipc_subclass"]:
        if values[field] is None:
            warnings.append(f"{field} is missing; the training-time imputer will handle it, so prediction confidence may be lower.")
    values["title_word_count"] = len(values.pop("title").split())
    values["abstract_word_count"] = len(values.pop("abstract").split())
    values["pub_year"] = values.pop("publication_year")
    frame = pd.DataFrame([values])[artifact["features"]]
    model = artifact["model"]
    probability = float(model.predict_proba(frame)[0, 1])
    label = int(model.predict(frame)[0])
    return {"predicted_class": label, "meaning": "Predicted highly cited (>=10 forward citations within three years)" if label else "Predicted not highly cited", "probability_highly_cited": probability, "model": artifact["selected_model"], "model_version": "benchmark-2026", "features_used": frame.iloc[0].where(frame.iloc[0].notna(), None).to_dict(), "warnings": warnings + ["Probability is an estimate, not a guarantee. The historical target was measured over three years; new patents may differ from the study cohort.", artifact["timing_note"]]}


@app.get("/api/analytics/overview")
def analytics_overview(year_start: int | None = None, year_end: int | None = None, country: str | None = None, ipc_section: str | None = None) -> dict[str, Any]:
    df = sql("""SELECT t.year AS pub_year,c.country_code,g.ipc_section,f.is_highly_cited FROM fact_patent f JOIN dim_time t ON f.pub_time_key=t.time_key JOIN dim_country c ON f.country_key=c.country_key JOIN dim_tech g ON f.tech_key=g.tech_key""")
    if year_start is not None: df = df[df.pub_year >= year_start]
    if year_end is not None: df = df[df.pub_year <= year_end]
    if country: df = df[df.country_code == country]
    if ipc_section: df = df[df.ipc_section == ipc_section]
    labelled = df[df.is_highly_cited.notna()]
    return {"summary": {"total_patents": len(df), "labelled_patents": len(labelled), "highly_cited": int(labelled.is_highly_cited.sum())}, "years": df.groupby("pub_year").size().reset_index(name="count").to_dict(orient="records"), "countries": df.groupby("country_code").size().nlargest(12).reset_index(name="count").to_dict(orient="records"), "sections": df.groupby("ipc_section").size().reset_index(name="count").to_dict(orient="records")}


@app.get("/api/analytics/{name}")
def analytics_file(name: str) -> list[dict[str, Any]]:
    files = {"classification": "classification_benchmark_metrics.csv", "clusters": "cluster_summary.csv", "association-rules": "association_rules.csv", "forecast": "forecast_results.csv"}
    if name not in files: raise HTTPException(404, "Unknown analytics resource.")
    path = OUT / files[name]
    if not path.exists(): raise HTTPException(404, "The requested analytics output is unavailable.")
    return pd.read_csv(path).where(pd.notna, None).to_dict(orient="records")


@app.get("/api/patents")
def patents(search: str = "", limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)) -> dict[str, Any]:
    raw = pd.read_excel(RAW, usecols=["PN", "TITLE", "PY", "COUNTRY", "IPC", "NUMBER OF INVENTORS", "CLAIMS NUMBER"])
    if search:
        mask = raw["PN"].astype(str).str.contains(search, case=False, na=False) | raw["TITLE"].astype(str).str.contains(search, case=False, na=False)
        raw = raw[mask]
    return {"total": len(raw), "records": raw.iloc[offset:offset + limit].where(pd.notna, None).to_dict(orient="records")}

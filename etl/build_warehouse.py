"""PatentPulse ETL: S1 File (xlsx) -> SQLite star schema."""
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/dataset.xlsx"
DB = ROOT / "data/warehouse/patentpulse.db"
SCHEMA = ROOT / "sql/schema.sql"

REGION = {
    **dict.fromkeys(["US", "CA", "MX"], "North America"),
    **dict.fromkeys(["JP", "KR", "CN", "TW", "IN", "TH", "SG", "HK", "IL", "MY"], "Asia"),
    **dict.fromkeys(["DE", "FR", "GB", "IT", "SE", "NL", "CH", "AT", "ES", "BE", "DK",
                     "FI", "NO", "RU", "PL", "CZ", "LU", "IE"], "Europe"),
}
KIND_TYPE = {"A1": "Application publication", "B1": "Granted patent",
             "B2": "Granted patent", "P0": "Provisional application"}


def to_int(s):
    """'-' placeholders -> NaN -> nullable Int64."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


#EXTRACT
def extract():
    df = pd.read_excel(RAW)
    print(f"[extract]   {len(df):,} rows, {df.shape[1]} columns")
    return df


#TRANSFORM
def transform(df):
    df = df.drop_duplicates(subset="PN").copy()
    df["pub_year"] = to_int(df["PY"].astype(str).str[:4])        # '-' -> NA
    df["app_year"] = to_int(df["APP YEAR"])
    for src, dst in [("PAGES", "pages"), ("CLAIMS NUMBER", "claims"),
                     ("NUMBER OF INVENTORS", "inventors")]:
        df[dst] = to_int(df[src])
    df["family_size"] = to_int(df["FAMILY PATENTS NUMBER"])
    df["npl_count"] = to_int(df["NUMBER OF NPL"])
    df["backward_citations"] = to_int(df["NUMBER OF CITATION"])
    df["cited_3yr"] = to_int(df["CITED TIME IN 3 YEARS"])

    ipc = df["IPC"].fillna("-").astype(str)
    df["ipc_count"] = ipc.apply(lambda x: 0 if x.strip() in ("-", "") else len(x.split("|")))
    main = ipc.str.split("|").str[0].str.strip()
    df["ipc_subclass"] = main.str.extract(r"^([A-HY]\d{2}[A-Z])")[0].fillna("UNKNOWN")
    df["ipc_section"] = df["ipc_subclass"].str[0].where(df["ipc_subclass"] != "UNKNOWN", "?")

    # some rows list several countries ("DE | JP"): keep the first as the primary country
    df["country_code"] = (df["COUNTRY"].fillna("-").astype(str).str.split("|").str[0]
                          .str.strip().replace({"-": "UNKNOWN", "": "UNKNOWN"}))
    df["region"] = df["country_code"].map(REGION).fillna("Other")
    df.loc[df["country_code"] == "UNKNOWN", "region"] = "Unknown"

    df["kind_code"] = df["PN"].str.extract(r"(A1|B1|B2|P0)$")[0].fillna("OTHER")
    df["patent_type"] = df["kind_code"].map(KIND_TYPE).fillna("Other")

    # Label from the base paper: published 2000-2016, highly cited = cited_3yr >= 10
    labelled = df["pub_year"].between(2000, 2016).fillna(False).astype(bool)
    df["is_highly_cited"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[labelled, "is_highly_cited"] = (df.loc[labelled, "cited_3yr"] >= 10).astype("Int64")

    def impact(lab, hc, c):
        if not lab:
            return 0                      # Not labelled
        if hc == 0:
            return 1                      # Not highly cited
        return 2 if c <= 40 else (3 if c <= 62 else 4)   # C0 / C1 / C2 (paper Table 2)

    df["impact_key"] = [impact(l, h, c) for l, h, c in
                        zip(labelled, df["is_highly_cited"], df["cited_3yr"].fillna(0))]
    print(f"[transform] labelled={labelled.sum():,}  highly cited={(df.is_highly_cited == 1).sum():,}")
    return df


#LOAD
def load(df):
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA.read_text())

    years = sorted(set(df["pub_year"].dropna().astype(int)) | set(df["app_year"].dropna().astype(int)))
    rows = [(-1, None, "Unknown", "Unknown")]
    for y in years:
        b = y // 5 * 5
        rows.append((y, y, f"{b}-{b+4}", f"{y//10*10}s"))
    con.executemany("INSERT INTO dim_time VALUES (?,?,?,?)", rows)

    con.executemany("INSERT INTO dim_country(country_code,region) VALUES (?,?)",
                    df[["country_code", "region"]].drop_duplicates("country_code").values.tolist())
    con.executemany("INSERT INTO dim_kind(kind_code,patent_type) VALUES (?,?)",
                    df[["kind_code", "patent_type"]].drop_duplicates("kind_code").values.tolist())
    con.executemany("INSERT INTO dim_tech(ipc_subclass,ipc_section) VALUES (?,?)",
                    df[["ipc_subclass", "ipc_section"]].drop_duplicates("ipc_subclass").values.tolist())
    con.executemany("INSERT INTO dim_impact VALUES (?,?,?)", [
        (0, "Not labelled", "n/a"), (1, "Not highly cited", "n/a"),
        (2, "Highly cited", "C0"), (3, "Highly cited", "C1"), (4, "Highly cited", "C2")])

    ck = dict(con.execute("SELECT country_code,country_key FROM dim_country"))
    kk = dict(con.execute("SELECT kind_code,kind_key FROM dim_kind"))
    tk = dict(con.execute("SELECT ipc_subclass,tech_key FROM dim_tech"))

    def nn(v):
        return None if pd.isna(v) else int(v)

    fact = []
    for r in df.itertuples(index=False):
        fact.append((r.PN,
                     -1 if pd.isna(r.pub_year) else int(r.pub_year),
                     -1 if pd.isna(r.app_year) else int(r.app_year),
                     ck[r.country_code], kk[r.kind_code], tk[r.ipc_subclass], int(r.impact_key),
                     nn(r.pages), nn(r.claims), nn(r.inventors), nn(r.family_size),
                     nn(r.npl_count), nn(r.backward_citations), int(r.ipc_count),
                     nn(r.cited_3yr), nn(r.is_highly_cited)))
    con.executemany("INSERT INTO fact_patent VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", fact)
    con.commit()
    for t in ["dim_time", "dim_country", "dim_kind", "dim_tech", "dim_impact", "fact_patent"]:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"[load]      {t:13s} {n:>7,} rows")
    con.close()


if __name__ == "__main__":
    load(transform(extract()))

"""PatentPulse - Review 1: OLAP operations on the star schema."""
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/warehouse/patentpulse.db"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

con = sqlite3.connect(DB)

# Common join: every query starts from the fact table and joins the dimensions it needs
BASE = """
FROM fact_patent f
JOIN dim_time    t ON f.pub_time_key = t.time_key
JOIN dim_country c ON f.country_key  = c.country_key
JOIN dim_tech    g ON f.tech_key     = g.tech_key
JOIN dim_kind    k ON f.kind_key     = k.kind_key
JOIN dim_impact  i ON f.impact_key   = i.impact_key
"""


def run(title, sql, save):
    df = pd.read_sql(sql, con)
    print(f"\n=== {title} ===\n{df.to_string(index=False)}")
    df.to_csv(OUT / f"{save}.csv", index=False)
    return df


# 0. Baseline - reproduces the base paper's numbers
run("Baseline: label distribution (paper: 5,665 labelled, 1,149 highly cited)", f"""
SELECT i.impact_label, i.impact_class, COUNT(*) AS patents {BASE}
GROUP BY i.impact_key ORDER BY i.impact_key""", "00_baseline")

# 1. SLICE - fix ONE dimension value (publication year = 2010)
run("SLICE: publication year = 2010", f"""
SELECT c.country_code, COUNT(*) AS patents,
       COALESCE(SUM(f.is_highly_cited),0) AS highly_cited, ROUND(AVG(f.pages),1) AS avg_pages {BASE}
WHERE t.year = 2010
GROUP BY c.country_code ORDER BY patents DESC LIMIT 8""", "01_slice")

# 2. DICE - restrict SEVERAL dimensions at once
run("DICE: country in (US, JP, KR) AND year 2010-2016 AND section B (transport)", f"""
SELECT c.country_code, t.year, COUNT(*) AS patents,
       COALESCE(SUM(f.is_highly_cited),0) AS highly_cited, ROUND(AVG(f.claims),1) AS avg_claims {BASE}
WHERE c.country_code IN ('US','JP','KR') AND t.year BETWEEN 2010 AND 2016
  AND g.ipc_section = 'B'
GROUP BY c.country_code, t.year ORDER BY c.country_code, t.year""", "02_dice")

# 3. ROLL-UP - go UP a hierarchy: year -> five-year block -> decade ; country -> region
run("ROLL-UP (time): decade level", f"""
SELECT t.decade, COUNT(*) AS patents, COUNT(f.is_highly_cited) AS labelled, COALESCE(SUM(f.is_highly_cited),0) AS highly_cited {BASE}
WHERE t.year > 0 GROUP BY t.decade ORDER BY t.decade""", "03a_rollup_decade")
run("ROLL-UP (geography): region level", f"""
SELECT c.region, COUNT(*) AS patents, COALESCE(SUM(f.is_highly_cited),0) AS highly_cited,
       ROUND(100.0*SUM(f.is_highly_cited)/NULLIF(COUNT(f.is_highly_cited),0),1) AS pct_highly_cited {BASE}
GROUP BY c.region ORDER BY patents DESC""", "03b_rollup_region")

# 4. DRILL-DOWN - go DOWN a hierarchy: five-year block -> year ; region -> country
run("DRILL-DOWN: five-year block 2010-2014 -> individual years", f"""
SELECT t.five_year_block, t.year, COUNT(*) AS patents, COALESCE(SUM(f.is_highly_cited),0) AS highly_cited {BASE}
WHERE t.five_year_block = '2010-2014' GROUP BY t.five_year_block, t.year ORDER BY t.year""", "04a_drilldown_year")
run("DRILL-DOWN: region Asia -> country", f"""
SELECT c.region, c.country_code, COUNT(*) AS patents, COALESCE(SUM(f.is_highly_cited),0) AS highly_cited {BASE}
WHERE c.region = 'Asia' GROUP BY c.region, c.country_code ORDER BY patents DESC LIMIT 6""", "04b_drilldown_country")

# 5. PIVOT - rotate the cube: countries as rows, five-year blocks as columns
raw = pd.read_sql(f"""
SELECT c.country_code, t.five_year_block, COALESCE(SUM(f.is_highly_cited),0) AS highly_cited {BASE}
WHERE t.year BETWEEN 2000 AND 2016 AND c.country_code IN ('US','JP','KR','CN','DE','FR')
GROUP BY c.country_code, t.five_year_block""", con)
pv = raw.pivot(index="country_code", columns="five_year_block", values="highly_cited").fillna(0).astype(int)
print("\n=== PIVOT: highly cited patents, country (rows) x period (columns) ===\n", pv)
pv.to_csv(OUT / "05_pivot.csv")

con.close()
print(f"\nResults saved as CSV in {OUT}")

-- PatentPulse star schema (SQLite)
DROP TABLE IF EXISTS fact_patent;
DROP TABLE IF EXISTS dim_time;
DROP TABLE IF EXISTS dim_country;
DROP TABLE IF EXISTS dim_kind;
DROP TABLE IF EXISTS dim_tech;
DROP TABLE IF EXISTS dim_impact;

CREATE TABLE dim_time (            -- hierarchy: year -> five_year_block -> decade
    time_key        INTEGER PRIMARY KEY,   -- the year itself (-1 = unknown)
    year            INTEGER,
    five_year_block TEXT,
    decade          TEXT
);
CREATE TABLE dim_country (         -- hierarchy: country -> region
    country_key     INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code    TEXT UNIQUE,
    region          TEXT
);
CREATE TABLE dim_kind (            -- hierarchy: kind_code -> patent_type
    kind_key        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind_code       TEXT UNIQUE,
    patent_type     TEXT
);
CREATE TABLE dim_tech (            -- hierarchy: ipc_section -> ipc_subclass (main IPC)
    tech_key        INTEGER PRIMARY KEY AUTOINCREMENT,
    ipc_subclass    TEXT UNIQUE,
    ipc_section     TEXT
);
CREATE TABLE dim_impact (          -- label from the base paper
    impact_key      INTEGER PRIMARY KEY,
    impact_label    TEXT,
    impact_class    TEXT
);
CREATE TABLE fact_patent (
    pn                   TEXT PRIMARY KEY,
    pub_time_key         INTEGER REFERENCES dim_time(time_key),
    app_time_key         INTEGER REFERENCES dim_time(time_key),
    country_key          INTEGER REFERENCES dim_country(country_key),
    kind_key             INTEGER REFERENCES dim_kind(kind_key),
    tech_key             INTEGER REFERENCES dim_tech(tech_key),
    impact_key           INTEGER REFERENCES dim_impact(impact_key),
    -- measures
    pages                INTEGER,
    claims               INTEGER,
    inventors            INTEGER,
    family_size          INTEGER,
    npl_count            INTEGER,
    backward_citations   INTEGER,
    ipc_count            INTEGER,
    cited_3yr            INTEGER,
    is_highly_cited      INTEGER      -- 1 / 0, NULL if not labelled
);
CREATE INDEX ix_fact_time    ON fact_patent(pub_time_key);
CREATE INDEX ix_fact_country ON fact_patent(country_key);
CREATE INDEX ix_fact_tech    ON fact_patent(tech_key);

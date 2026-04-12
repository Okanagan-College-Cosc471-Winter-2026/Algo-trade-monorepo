-- stg_raw.market_data
CREATE TABLE IF NOT EXISTS stg_raw.market_data (
    ingest_id   BIGSERIAL PRIMARY KEY,
    symbol      TEXT,
    ts          TIMESTAMPTZ,
    open        NUMERIC(18,6),
    high        NUMERIC(18,6),
    low         NUMERIC(18,6),
    close       NUMERIC(18,6),
    volume      NUMERIC(20,4),
    asset_type  TEXT,
    source      TEXT,
    ingest_time TIMESTAMPTZ DEFAULT now(),
    raw_payload JSONB,
    CONSTRAINT unique_symbol_ts_source UNIQUE (symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_stg_raw_symbol_ts    ON stg_raw.market_data (symbol, ts);
CREATE INDEX IF NOT EXISTS idx_stg_raw_ingest_time  ON stg_raw.market_data (ingest_time);

-- stg_raw.ingest_errors
CREATE TABLE IF NOT EXISTS stg_raw.ingest_errors (
    error_id    BIGSERIAL PRIMARY KEY,
    ingest_id   BIGINT,
    symbol      TEXT,
    ts          TIMESTAMPTZ,
    asset_type  TEXT,
    source      TEXT,
    error_type  TEXT,
    error_detail TEXT,
    raw_payload JSONB,
    log_time    TIMESTAMPTZ DEFAULT now()
);

-- core_dbms.market_data_5m
CREATE TABLE IF NOT EXISTS core_dbms.market_data_5m (
    market_data_id BIGSERIAL PRIMARY KEY,
    symbol         TEXT        NOT NULL,
    ts             TIMESTAMPTZ NOT NULL,
    open           NUMERIC(18,6),
    high           NUMERIC(18,6),
    low            NUMERIC(18,6),
    close          NUMERIC(18,6),
    volume         BIGINT,
    asset_type     TEXT        NOT NULL,
    source         TEXT,
    created_at     TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT unique_symbol_ts UNIQUE (symbol, ts)
);

-- stg_transform.market_data
CREATE TABLE IF NOT EXISTS stg_transform.market_data (
    symbol  TEXT        NOT NULL,
    ts      TIMESTAMPTZ NOT NULL,
    open    NUMERIC(18,6),
    high    NUMERIC(18,6),
    low     NUMERIC(18,6),
    close   NUMERIC(18,6),
    volume  BIGINT,
    vwap    NUMERIC(20,6),
    PRIMARY KEY (symbol, ts)
);

-- stg_transform.transform_errors
CREATE TABLE IF NOT EXISTS stg_transform.transform_errors (
    error_id     BIGSERIAL PRIMARY KEY,
    symbol       TEXT,
    ts           TIMESTAMPTZ,
    error_type   TEXT,
    error_detail TEXT,
    log_time     TIMESTAMPTZ DEFAULT now()
);

-- operation_logs.*
CREATE TABLE IF NOT EXISTS operation_logs.authority_conflicts (
    conflict_id      BIGSERIAL PRIMARY KEY,
    symbol           TEXT,
    ts               TIMESTAMPTZ,
    source_a         TEXT,
    source_b         TEXT,
    preferred_source TEXT,
    log_time         TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operation_logs.backup_logs (
    backup_id    BIGSERIAL PRIMARY KEY,
    backup_type  TEXT,
    file_path    TEXT,
    status       TEXT,
    started_at   TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    log_time     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operation_logs.cast_errors (
    error_id     BIGSERIAL PRIMARY KEY,
    symbol       TEXT,
    column_name  TEXT,
    raw_value    TEXT,
    target_type  TEXT,
    error_detail TEXT,
    log_time     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operation_logs.concurrency_issues (
    issue_id   BIGSERIAL PRIMARY KEY,
    table_name TEXT,
    lock_type  TEXT,
    pid        INTEGER,
    details    TEXT,
    log_time   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operation_logs.data_quality_errors (
    error_id     BIGSERIAL PRIMARY KEY,
    symbol       TEXT,
    ts           TIMESTAMPTZ,
    error_type   TEXT,
    error_detail TEXT,
    log_time     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operation_logs.dedup_conflicts (
    conflict_id  BIGSERIAL PRIMARY KEY,
    symbol       TEXT,
    ts           TIMESTAMPTZ,
    existing_row JSONB,
    incoming_row JSONB,
    resolution   TEXT,
    log_time     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operation_logs.ingestion_log (
    log_id      BIGSERIAL PRIMARY KEY,
    symbol      TEXT,
    start_date  DATE,
    end_date    DATE,
    rows_loaded INTEGER,
    status      TEXT,
    error_msg   TEXT,
    logged_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operation_logs.pipeline_logs (
    log_id         BIGSERIAL PRIMARY KEY,
    pipeline_stage TEXT        NOT NULL,
    status         TEXT        NOT NULL,
    message        TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_pipeline_logs_status CHECK (status IN ('running', 'success', 'failed', 'warning'))
);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_stage_time ON operation_logs.pipeline_logs (pipeline_stage, created_at DESC);

CREATE TABLE IF NOT EXISTS operation_logs.pipeline_watermarks (
    pipeline_name    TEXT PRIMARY KEY,
    last_processed_ts TIMESTAMPTZ,
    status           TEXT,
    updated_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operation_logs.upsert_failures (
    failure_id   BIGSERIAL PRIMARY KEY,
    table_name   TEXT,
    symbol       TEXT,
    ts           TIMESTAMPTZ,
    error_detail TEXT,
    log_time     TIMESTAMPTZ DEFAULT now()
);

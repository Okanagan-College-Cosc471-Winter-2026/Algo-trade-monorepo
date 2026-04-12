-- staging.market_data_5m (DW API real-time ingest target)
CREATE TABLE IF NOT EXISTS staging.market_data_5m (
    market_data_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol         TEXT        NOT NULL,
    ts             TIMESTAMPTZ NOT NULL,
    open           NUMERIC(18,6) NOT NULL,
    high           NUMERIC(18,6) NOT NULL,
    low            NUMERIC(18,6) NOT NULL,
    close          NUMERIC(18,6) NOT NULL,
    volume         BIGINT      NOT NULL,
    asset_type     TEXT        NOT NULL,
    source         TEXT        NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (symbol, ts)
);

-- staging.ingestion_progress (tracks last ingested ts per symbol)
CREATE TABLE IF NOT EXISTS staging.ingestion_progress (
    symbol           TEXT PRIMARY KEY,
    last_ingested_ts TIMESTAMPTZ NOT NULL DEFAULT '1899-12-31 16:00:00+00',
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes            TEXT
);

-- dw.market_data_15m (aggregated 15-min bars with all ML features)
CREATE TABLE IF NOT EXISTS dw.market_data_15m (
    agg_id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol               TEXT        NOT NULL,
    window_ts            TIMESTAMPTZ NOT NULL,
    trade_date           DATE,
    open                 NUMERIC(18,6),
    high                 NUMERIC(18,6),
    low                  NUMERIC(18,6),
    close                NUMERIC(18,6),
    volume               BIGINT,
    slot_count           INTEGER     NOT NULL DEFAULT 0,
    status               TEXT        NOT NULL DEFAULT 'provisional',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    lag_close_1          NUMERIC(18,6),
    lag_close_5          NUMERIC(18,6),
    lag_close_10         NUMERIC(18,6),
    close_diff_1         NUMERIC(18,6),
    close_diff_5         NUMERIC(18,6),
    pct_change_1         NUMERIC(18,6),
    pct_change_5         NUMERIC(18,6),
    log_return_1         NUMERIC(18,6),
    sma_close_5          NUMERIC(18,6),
    sma_close_10         NUMERIC(18,6),
    sma_close_20         NUMERIC(18,6),
    sma_volume_5         NUMERIC(18,6),
    day_of_week          SMALLINT,
    hour_of_day          SMALLINT,
    month_of_year        SMALLINT,
    day_monday           SMALLINT DEFAULT 0,
    day_tuesday          SMALLINT DEFAULT 0,
    day_wednesday        SMALLINT DEFAULT 0,
    day_thursday         SMALLINT DEFAULT 0,
    day_friday           SMALLINT DEFAULT 0,
    quarter_1            SMALLINT DEFAULT 0,
    quarter_2            SMALLINT DEFAULT 0,
    quarter_3            SMALLINT DEFAULT 0,
    quarter_4            SMALLINT DEFAULT 0,
    hour_early_morning   SMALLINT DEFAULT 0,
    hour_mid_morning     SMALLINT DEFAULT 0,
    hour_afternoon       SMALLINT DEFAULT 0,
    hour_late_afternoon  SMALLINT DEFAULT 0,
    previous_close       NUMERIC(18,6),
    overnight_gap_pct    NUMERIC(18,6),
    overnight_log_return NUMERIC(18,6),
    is_gap_up            SMALLINT,
    is_gap_down          SMALLINT,
    UNIQUE (symbol, window_ts)
);
CREATE INDEX IF NOT EXISTS idx_dw_15m_symbol_ts ON dw.market_data_15m (symbol, window_ts DESC);

"""
TEST 00 — DB Schema Bootstrap
──────────────────────────────
Verifies that all schemas, tables, views, and stored procedures expected
by the pipeline exist in the database after `docker compose up -d db`.

These tests should pass with a freshly initialised DB and no data.
They are the foundation: if any of these fail, all downstream tests will too.
"""
import pytest
from sqlalchemy import text


# ── Expected objects ─────────────────────────────────────────────────────────

EXPECTED_SCHEMAS = [
    "stg_raw", "core_dbms", "dw", "staging", "ml",
    "historical", "market", "operation_logs", "stg_transform",
]

EXPECTED_TABLES = [
    ("stg_raw",        "market_data"),
    ("stg_raw",        "ingest_errors"),
    ("core_dbms",      "market_data_5m"),
    ("dw",             "market_data_15m"),
    ("staging",        "market_data_5m"),
    ("staging",        "ingestion_progress"),
    ("market",         "stocks"),
    ("market",         "daily_prices"),
    ("operation_logs", "pipeline_logs"),
    ("operation_logs", "pipeline_watermarks"),
]

EXPECTED_VIEWS = [
    ("historical", "market_data_5m"),   # → core_dbms.market_data_5m
    ("ml",         "market_data_15m"),  # → dw.market_data_15m
]

EXPECTED_PROCEDURES = [
    ("dw", "process_15min_window"),
    ("dw", "build_warehouse_data"),
]

EXPECTED_COLUMNS = {
    # staging.market_data_5m must have ingested_at
    ("staging", "market_data_5m"): ["symbol", "ts", "open", "high", "low", "close",
                                     "volume", "created_at", "ingested_at"],
    # dw.market_data_15m must have overnight feature columns
    ("dw", "market_data_15m"):     ["symbol", "window_ts", "trade_date",
                                     "previous_close", "overnight_gap_pct",
                                     "overnight_log_return", "is_gap_up", "is_gap_down"],
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSchemas:
    def test_all_schemas_exist(self, engine):
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT schema_name FROM information_schema.schemata"
            )).fetchall()
        existing = {r[0] for r in rows}
        for schema in EXPECTED_SCHEMAS:
            assert schema in existing, f"Schema '{schema}' is missing"


class TestTables:
    @pytest.mark.parametrize("schema,table", EXPECTED_TABLES)
    def test_table_exists(self, engine, schema, table):
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = :s AND table_name = :t"
            ), {"s": schema, "t": table}).fetchone()
        assert row is not None, f"Table '{schema}.{table}' is missing"


class TestViews:
    @pytest.mark.parametrize("schema,view", EXPECTED_VIEWS)
    def test_view_exists(self, engine, schema, view):
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT 1 FROM information_schema.views "
                "WHERE table_schema = :s AND table_name = :v"
            ), {"s": schema, "v": view}).fetchone()
        assert row is not None, f"View '{schema}.{view}' is missing"


class TestProcedures:
    @pytest.mark.parametrize("schema,proc", EXPECTED_PROCEDURES)
    def test_procedure_exists(self, engine, schema, proc):
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT 1 FROM information_schema.routines "
                "WHERE routine_schema = :s AND routine_name = :p"
            ), {"s": schema, "p": proc}).fetchone()
        assert row is not None, f"Procedure '{schema}.{proc}' is missing"


class TestColumns:
    @pytest.mark.parametrize("schema_table,cols", [
        (k, v) for k, v in EXPECTED_COLUMNS.items()
    ])
    def test_columns_exist(self, engine, schema_table, cols):
        schema, table = schema_table
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = :s AND table_name = :t"
            ), {"s": schema, "t": table}).fetchall()
        existing = {r[0] for r in rows}
        for col in cols:
            assert col in existing, f"Column '{col}' missing from {schema}.{table}"

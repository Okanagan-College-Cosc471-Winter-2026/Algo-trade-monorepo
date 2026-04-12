"""
Shared fixtures for the algo-trade integration test suite.

All tests connect to the running `db` container via the host port (5433).
Run from repo root:
    docker compose up -d db
    pytest tests/ -v
"""
from __future__ import annotations

import datetime as dt
import os
from datetime import timezone
from decimal import Decimal

import psycopg2
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ── Connection settings ──────────────────────────────────────────────────────
# Use host-side port (5433 → container's 5432) when running tests on the host.
# Use port 5432 when running tests inside the docker network.
DB_HOST     = os.getenv("TEST_DB_HOST", "localhost")
DB_PORT     = int(os.getenv("TEST_DB_PORT", "5433"))
DB_NAME     = os.getenv("POSTGRES_DB", "algotrade")
DB_USER     = os.getenv("POSTGRES_USER", "appuser")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "changeme")

SQLALCHEMY_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── Dummy data constants ─────────────────────────────────────────────────────
TEST_SYMBOL = "AAPL"

# A 15-min window well within market hours (Mon Apr 7 2026, 10:00 ET = 14:00 UTC)
WINDOW_UTC = dt.datetime(2026, 4, 7, 14, 0, 0, tzinfo=timezone.utc)

# 3 consecutive 5-min bars that make up the 10:00–10:15 window
BARS_5MIN = [
    # (ts_utc, open, high, low, close, volume)
    (dt.datetime(2026, 4, 7, 14,  0, 0, tzinfo=timezone.utc), 175.00, 175.50, 174.80, 175.20, 100_000),
    (dt.datetime(2026, 4, 7, 14,  5, 0, tzinfo=timezone.utc), 175.20, 175.80, 175.00, 175.60, 120_000),
    (dt.datetime(2026, 4, 7, 14, 10, 0, tzinfo=timezone.utc), 175.60, 176.00, 175.40, 175.90, 110_000),
]


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(SQLALCHEMY_URL, echo=False)
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def session_factory(engine):
    return sessionmaker(bind=engine)


@pytest.fixture
def db_session(session_factory):
    """Per-test session that rolls back after the test."""
    with session_factory() as session:
        yield session
        session.rollback()


@pytest.fixture(scope="session")
def raw_conn():
    """Plain psycopg2 connection for direct SQL assertions."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
    )
    conn.autocommit = True
    yield conn
    conn.close()


def count_rows(conn, schema: str, table: str, where: str = "") -> int:
    q = f"SELECT COUNT(*) FROM {schema}.{table}"
    if where:
        q += f" WHERE {where}"
    with conn.cursor() as cur:
        cur.execute(q)
        return cur.fetchone()[0]

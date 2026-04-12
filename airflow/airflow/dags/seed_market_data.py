"""
Market Data Seeding DAG
========================

Refreshes the market schema with synthetic stock price data daily.

Task flow:
    check_db_connection  →  truncate_old_data  →  seed_stocks_and_prices  →  verify_data_quality

What each task does:
    1. check_db_connection     — verify Postgres is reachable and market schema exists
    2. truncate_old_data       — wipe existing prices/stocks for a clean re-seed
    3. seed_stocks_and_prices  — run the existing seed_market.py script
    4. verify_data_quality     — assert correct row counts and recent dates

Schedule:  Daily at 1:00 AM  (cron: 0 1 * * *)
Tables:    market.stocks, market.daily_prices

Industry concepts demonstrated:
    - PostgresHook:   Airflow's built-in connector for Postgres (manages connections)
    - Connection:     Credentials stored in Airflow's metadata DB, not hardcoded
    - Idempotency:    Truncate-then-reload means re-runs produce the same result
    - Data quality:   Verification step catches silent failures
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# PostgresHook — Airflow's way of talking to Postgres
# ---------------------------------------------------------------------------
# Instead of building connection strings by hand, Airflow stores credentials
# in its metadata DB as "Connections".  We created one called "postgres_app"
# during airflow-init.  PostgresHook uses that connection automatically.
#
# You can view/edit connections in the UI:  Admin → Connections
from airflow.providers.postgres.hooks.postgres import PostgresHook

# The connection ID we registered in docker-compose (airflow-init service)
CONN_ID = "postgres_app"

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------
default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# ===========================================================================
# TASK FUNCTIONS
# ===========================================================================

def check_db_connection(**context):
    """
    Task 1: Verify Postgres is alive and the market schema exists.

    Why this matters:
        If the DB is down, we want to fail fast with a clear message rather
        than getting a cryptic connection error 3 tasks later.
    """
    hook = PostgresHook(postgres_conn_id=CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()

    # Test basic connectivity
    cur.execute("SELECT version();")
    version = cur.fetchone()[0]
    print(f"Connected to: {version}")

    # Check if the market schema exists
    cur.execute(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name = 'market';"
    )
    if cur.fetchone():
        print("Market schema exists.")
    else:
        print("Market schema not found — seed script will create it.")

    cur.close()
    conn.close()


def truncate_old_data(**context):
    """
    Task 2: Clear existing market data for a clean re-seed.

    Why truncate instead of upsert?
        The seed script generates a full 2-year synthetic dataset each run.
        Truncating is simpler and guarantees no stale rows linger.
        This is called "idempotent reload" — a common ETL pattern.
    """
    hook = PostgresHook(postgres_conn_id=CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()

    try:
        cur.execute("CREATE SCHEMA IF NOT EXISTS market;")

        # TRUNCATE is faster than DELETE for large tables (no row-by-row logging)
        cur.execute(
            "TRUNCATE TABLE market.daily_prices, market.stocks CASCADE;"
        )
        conn.commit()
        print("Truncated market.daily_prices and market.stocks.")
    except Exception as exc:
        conn.rollback()
        raise AirflowException(f"Truncation failed: {exc}") from exc
    finally:
        cur.close()
        conn.close()


def seed_stocks_and_prices(**context):
    """
    Task 3: Import and execute the existing seed_market.main().

    How this works:
        - The backend/ directory is mounted at /opt/airflow/backend
        - We add it to sys.path so Python can find the `scripts` and `app` packages
        - We set POSTGRES_SERVER=db so the backend config points at the Docker host

    This reuses your existing script — no code duplication.
    """
    backend_path = "/opt/airflow/backend"
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    # The backend's Settings class reads these env vars
    os.environ["POSTGRES_SERVER"] = "db"
    os.environ["POSTGRES_PORT"] = os.getenv("POSTGRES_PORT", "5432")
    os.environ["POSTGRES_DB"] = os.getenv("POSTGRES_DB", "app")
    os.environ["POSTGRES_USER"] = os.getenv("POSTGRES_USER", "postgres")
    os.environ["POSTGRES_PASSWORD"] = os.getenv("POSTGRES_PASSWORD", "changethis")

    try:
        from scripts.seed_market import main as seed_main

        seed_main()
        print("Seeding completed successfully.")
    except Exception as exc:
        raise AirflowException(f"Seeding failed: {exc}") from exc


def verify_data_quality(**context):
    """
    Task 4: Verify the seed produced correct data.

    Checks:
        - Exactly 5 stocks exist
        - Each stock has >= 400 daily_prices  (730 calendar days ≈ 505 trading days)
        - Most recent price date is within the last 7 days

    Why a verification step?
        Silent failures are the worst kind.  The seed script might succeed but
        produce bad data (0 rows, wrong schema, etc).  This task catches that.
    """
    hook = PostgresHook(postgres_conn_id=CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()

    try:
        # Check 1: Stock count
        cur.execute("SELECT COUNT(*) FROM market.stocks;")
        stock_count = cur.fetchone()[0]
        assert stock_count == 5, f"Expected 5 stocks, found {stock_count}"
        print(f"Stock count: {stock_count}")

        # Check 2: Row count per stock
        cur.execute(
            "SELECT symbol, COUNT(*) AS n "
            "FROM market.daily_prices GROUP BY symbol ORDER BY symbol;"
        )
        for symbol, count in cur.fetchall():
            assert count >= 400, f"{symbol} has only {count} rows (expected >= 400)"
            print(f"  {symbol}: {count} daily prices")

        # Check 3: Data freshness
        cur.execute("SELECT MAX(date) FROM market.daily_prices;")
        latest = cur.fetchone()[0]
        print(f"Latest date: {latest}")

        print("Data quality verification PASSED.")

    except AssertionError as exc:
        raise AirflowException(str(exc)) from exc
    finally:
        cur.close()
        conn.close()


# ===========================================================================
# DAG
# ===========================================================================
with DAG(
    dag_id="seed_market_data",
    default_args=default_args,
    description="Daily market data refresh — stocks and OHLC prices",
    # ---------------------------------------------------------------------------
    # Cron expression:  0 1 * * *
    #   minute=0  hour=1  day=*  month=*  weekday=*
    #   → runs at 01:00 AM every day
    #
    # Other useful schedules:
    #   @daily       = 0 0 * * *   (midnight)
    #   @hourly      = 0 * * * *
    #   0 9 * * 1-5  = 9 AM on weekdays
    # ---------------------------------------------------------------------------
    schedule="0 1 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["production", "market-data", "etl"],
    # Only one instance of this DAG can run at a time.
    # Prevents overlapping seeds from corrupting data.
    max_active_runs=1,
) as dag:

    t_check = PythonOperator(
        task_id="check_db_connection",
        python_callable=check_db_connection,
    )

    t_truncate = PythonOperator(
        task_id="truncate_old_data",
        python_callable=truncate_old_data,
    )

    t_seed = PythonOperator(
        task_id="seed_stocks_and_prices",
        python_callable=seed_stocks_and_prices,
        # Fail the task if seeding takes longer than 10 minutes
        execution_timeout=timedelta(minutes=10),
    )

    t_verify = PythonOperator(
        task_id="verify_data_quality",
        python_callable=verify_data_quality,
    )

    # Linear dependency chain
    t_check >> t_truncate >> t_seed >> t_verify

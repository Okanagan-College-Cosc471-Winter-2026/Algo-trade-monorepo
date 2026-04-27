"""
Airflow-owned intraday data pipeline (collect -> export -> aggregate).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.operators.python import PythonOperator

from market_calendar_utils import expected_last_closed_window_start_utc


REPO_ROOT = Path("/data/projects/Algo-trade-monorepo")
PIPELINE_SCRIPT = REPO_ROOT / "services" / "collector" / "src" / "run_15min_pipeline.py"
PIPELINE_PYTHON = REPO_ROOT / "pipeline-venv" / "bin" / "python"
FRESHNESS_FILE = Path(
    os.getenv("INTRADAY_FRESHNESS_FILE", str(REPO_ROOT / "logs" / "intraday_data_freshness.json"))
)


def _read_latest_window() -> dt.datetime | None:
    if not FRESHNESS_FILE.exists():
        return None
    try:
        payload = json.loads(FRESHNESS_FILE.read_text())
        raw = payload.get("latest_window_ts_utc")
        if not raw:
            return None
        return dt.datetime.fromisoformat(raw)
    except Exception:
        return None


def task_run_intraday_data_pipeline(**_ctx) -> None:
    expected_window = expected_last_closed_window_start_utc(now=dt.datetime.now(dt.timezone.utc))
    if expected_window is None:
        raise AirflowSkipException("Market gate closed (holiday/outside session/no closed RTH window).")

    latest_window = _read_latest_window()
    if latest_window is not None and latest_window >= expected_window:
        raise AirflowSkipException(
            f"Idempotency guard: expected={expected_window.isoformat()} already covered by "
            f"{latest_window.isoformat()}."
        )

    env = os.environ.copy()
    env["INTRADAY_FRESHNESS_FILE"] = str(FRESHNESS_FILE)
    # Map POSTGRES_* env vars (Airflow convention) to DB_* (pipeline script convention)
    env.setdefault("DB_HOST",     env.get("POSTGRES_SERVER", "localhost"))
    env.setdefault("DB_PORT",     env.get("POSTGRES_PORT",   "5432"))
    env.setdefault("DB_NAME",     env.get("POSTGRES_DB",     ""))
    env.setdefault("DB_USER",     env.get("POSTGRES_USER",   ""))
    env.setdefault("DB_PASSWORD", env.get("POSTGRES_PASSWORD", ""))
    proc = subprocess.run(
        [str(PIPELINE_PYTHON), str(PIPELINE_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise AirflowException(
            "run_15min_pipeline.py failed\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )

    latest_window = _read_latest_window()
    if latest_window is None or latest_window < expected_window:
        raise AirflowException(
            "Pipeline finished but freshness marker missing/stale. "
            f"expected>={expected_window.isoformat()} got={latest_window}."
        )


with DAG(
    dag_id="intraday_data_pipeline",
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "email_on_failure": False,
        "retries": 1,
        "retry_delay": dt.timedelta(minutes=2),
    },
    description="Airflow-owned intraday data pipeline (collect, export, aggregate) every 15 minutes.",
    schedule="*/15 * * * 1-5",
    start_date=dt.datetime(2026, 4, 22),
    catchup=False,
    max_active_runs=1,
    tags=["market-data", "intraday", "orchestration"],
) as dag:
    run_data_pipeline = PythonOperator(
        task_id="run_data_pipeline",
        python_callable=task_run_intraday_data_pipeline,
        execution_timeout=dt.timedelta(minutes=15),
    )


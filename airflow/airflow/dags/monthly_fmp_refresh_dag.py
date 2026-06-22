"""Monthly FMP refresh DAG for ml.market_data_15m.

Operational note:
    This DAG is intended for the monthly-refresh workflow. Existing intraday
    freshness DAGs should be paused in environments that switch to monthly-only
    ingestion.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator


REPO_ROOT = Path(os.getenv("REPO_ROOT", "/data/projects/Algo-trade-monorepo"))
SCRIPT_DIR = REPO_ROOT / "ml" / "ml" / "scripts"
REPORT_DIR = REPO_ROOT / "ml" / "ml" / "data" / "monthly_refresh_reports"

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _ensure_script_import_path() -> None:
    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def compute_target_month(**context) -> str:
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}
    override = conf.get("month_override")
    _ensure_script_import_path()
    from monthly_fmp_refresh import previous_calendar_month_label

    month = override or previous_calendar_month_label()
    print(f"Target month: {month}")
    return month


def run_monthly_refresh(**context) -> None:
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}
    month = context["ti"].xcom_pull(task_ids="compute_target_month")
    _ensure_script_import_path()
    from monthly_fmp_refresh import main

    argv: list[str] = []
    if conf.get("bootstrap"):
        argv.extend(["--mode", "bootstrap"])
        if conf.get("history_years"):
            argv.extend(["--history-years", str(conf["history_years"])])
    else:
        argv.extend(["--mode", "monthly-refresh", "--month", month])

    if conf.get("start_date") and conf.get("end_date"):
        argv.extend(["--start-date", conf["start_date"], "--end-date", conf["end_date"]])
    if conf.get("symbols_override"):
        argv.extend(["--symbols", conf["symbols_override"]])
    if conf.get("limit_symbols") is not None:
        argv.extend(["--limit-symbols", str(conf["limit_symbols"])])
    if conf.get("chunk_days") is not None:
        argv.extend(["--chunk-days", str(conf["chunk_days"])])
    if conf.get("max_retries") is not None:
        argv.extend(["--max-retries", str(conf["max_retries"])])
    if conf.get("resume_from_checkpoint"):
        argv.append("--resume-from-checkpoint")
    if conf.get("dry_run"):
        argv.append("--dry-run")
    if conf.get("skip_daily_prices"):
        argv.append("--skip-daily-prices")
    if conf.get("verbose"):
        argv.append("--verbose")

    print(f"monthly_fmp_refresh argv={argv}")
    main(argv)


def validate_reports(**context) -> None:
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}
    dry_run = bool(conf.get("dry_run"))
    summary_path = REPORT_DIR / ("dry_run_summary.json" if dry_run else "last_success_summary.json")
    unresolved_path = REPORT_DIR / "unresolved_intraday_gaps.json"
    nulls_path = REPORT_DIR / "null_ohlcv_rows.json"
    daily_failures_path = REPORT_DIR / "daily_price_failures.json"

    if unresolved_path.exists():
        payload = json.loads(unresolved_path.read_text())
        if payload:
            raise AirflowException(f"Unresolved intraday gaps reported in {unresolved_path}")
    if nulls_path.exists():
        payload = json.loads(nulls_path.read_text())
        if payload:
            raise AirflowException(f"Null OHLCV rows reported in {nulls_path}")
    if daily_failures_path.exists():
        payload = json.loads(daily_failures_path.read_text())
        if payload:
            raise AirflowException(f"Daily price failures reported in {daily_failures_path}")
    if not summary_path.exists():
        raise AirflowException(f"Expected summary report not found: {summary_path}")


def publish_summary(**context) -> None:
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}
    dry_run = bool(conf.get("dry_run"))
    summary_path = REPORT_DIR / ("dry_run_summary.json" if dry_run else "last_success_summary.json")
    payload = json.loads(summary_path.read_text())
    print(json.dumps(payload, indent=2))


with DAG(
    dag_id="monthly_fmp_refresh",
    default_args=default_args,
    description="Monthly FMP refresh for ml.market_data_15m and market.daily_prices",
    schedule="0 10 7 * *",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["production", "market-data", "monthly-refresh"],
) as dag:
    t_compute = PythonOperator(
        task_id="compute_target_month",
        python_callable=compute_target_month,
    )

    t_refresh = PythonOperator(
        task_id="run_monthly_refresh",
        python_callable=run_monthly_refresh,
        execution_timeout=timedelta(hours=6),
    )

    t_validate = PythonOperator(
        task_id="validate_reports",
        python_callable=validate_reports,
    )

    t_publish = PythonOperator(
        task_id="publish_summary",
        python_callable=publish_summary,
    )

    t_compute >> t_refresh >> t_validate >> t_publish

"""
nibi_base_fallback_dag.py
=========================

Weekday fallback guard for the NIBI base model.

At 12:00 AM ET, if either local serving pointer is missing or incomplete:
  - model_artifacts/current_base
  - model_artifacts/current_base_eod

then this DAG triggers the existing `nibi_daily_warm_refresh` DAG so it can
train a fresh base bundle from the latest data currently available.

If the base bundle is already healthy, or the training DAG is already queued or
running, this DAG skips cleanly and does nothing.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.models import DagRun
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.session import provide_session
from airflow.utils.state import DagRunState

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/data/projects/Algo-trade-monorepo/model_artifacts"))
TARGET_DAG_ID = "nibi_base_only_training"
MARKET_TZ = ZoneInfo("America/New_York")


def _bundle_ready(bundle_path: Path) -> tuple[bool, str]:
    if not bundle_path.exists():
        return False, f"{bundle_path.name} path missing"

    manifest = bundle_path / "models" / "model_manifest.json"
    if not manifest.exists():
        return False, f"{bundle_path.name} missing models/model_manifest.json"

    return True, f"{bundle_path.name} ready"


def task_check_base_bundle(**ctx) -> None:
    current_base = ARTIFACTS_DIR / "current_base"
    current_base_eod = ARTIFACTS_DIR / "current_base_eod"

    checks = [
        _bundle_ready(current_base),
        _bundle_ready(current_base_eod),
    ]
    failures = [message for ok, message in checks if not ok]
    if not failures:
        raise AirflowSkipException("Base bundles already present; fallback training not needed.")

    reason = "; ".join(failures)
    print(f"Fallback base training required: {reason}")
    ctx["ti"].xcom_push(key="fallback_reason", value=reason)


@provide_session
def task_skip_if_training_active(*, session=None, **ctx) -> None:
    active = (
        session.query(DagRun)
        .filter(
            DagRun.dag_id == TARGET_DAG_ID,
            DagRun.state.in_([DagRunState.RUNNING, DagRunState.QUEUED]),
        )
        .first()
    )
    if active is not None:
        raise AirflowSkipException(
            f"{TARGET_DAG_ID} already active via run_id={active.run_id}; not triggering another run."
        )

    reason = ctx["ti"].xcom_pull(task_ids="check_base_bundle", key="fallback_reason")
    print(f"No active {TARGET_DAG_ID} run found. Proceeding with fallback trigger. reason={reason}")


with DAG(
    dag_id="nibi_base_model_fallback",
    default_args={
        "owner": "ml-team",
        "depends_on_past": False,
        "retries": 0,
        "email_on_failure": False,
    },
    description="At 12:00 AM ET, trigger base training if current_base/current_base_eod are missing.",
    schedule="0 0 * * 1-5",
    start_date=dt.datetime(2026, 5, 4, tzinfo=MARKET_TZ),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "nibi", "fallback"],
) as dag:
    t1_check = PythonOperator(
        task_id="check_base_bundle",
        python_callable=task_check_base_bundle,
        execution_timeout=dt.timedelta(minutes=1),
    )

    t2_guard = PythonOperator(
        task_id="skip_if_training_active",
        python_callable=task_skip_if_training_active,
        execution_timeout=dt.timedelta(minutes=1),
    )

    t3_trigger = TriggerDagRunOperator(
        task_id="trigger_daily_training",
        trigger_dag_id=TARGET_DAG_ID,
        wait_for_completion=False,
        reset_dag_run=False,
        conf={
            "trigger_reason": "missing_current_base_midnight_fallback",
            "requested_by_dag": "nibi_base_model_fallback",
        },
    )

    t1_check >> t2_guard >> t3_trigger

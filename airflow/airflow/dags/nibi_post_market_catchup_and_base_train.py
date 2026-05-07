"""
nibi_post_market_catchup_and_base_train.py
==========================================

Orchestrates a full data catch-up for the current trading day followed by a 
NIBI base model training job.

Workflow:
  1. catchup_today_data — Force collection of any missing data for 'today'
                          up to market close, export to core, and re-aggregate
                          all 26 intraday windows for feature engineering.
  2. export_parquet      — Snapshot the fully-loaded ml.market_data_15m.
  3. sync_to_nibi        — Send code and parquet to NIBI.
  4. submit_base_job     — Run sim_base_train.sbatch --base-only.
  5. poll_and_promote    — Wait for completion, pull artifacts, and update symlink.

Usage:
  Trigger manually after market close (16:00 ET) to ensure a fresh base model 
  is ready for the next day's intraday warm-refreshes.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator

# ── Config ─────────────────────────────────────────────────────────────────
NIBI_USER = os.getenv("NIBI_USER", "harshsaw")
NIBI_HOST = os.getenv("NIBI_HOST", "nibi.sharcnet.ca")
NIBI_SIM_DIR = os.getenv("NIBI_SIM_DIR", "/home/harshsaw/projects/def-youry/test_simulation")

DB_HOST = os.getenv("POSTGRES_SERVER", os.getenv("OLD_DB_HOST", "localhost"))
DB_PORT = int(os.getenv("POSTGRES_PORT", os.getenv("OLD_DB_PORT", "5432")))
DB_NAME = os.getenv("POSTGRES_DB", os.getenv("OLD_DB_NAME", "market_data"))
DB_USER = os.getenv("POSTGRES_USER", os.getenv("OLD_DB_USER", "mluser"))
DB_PASS = os.getenv("POSTGRES_PASSWORD", os.getenv("OLD_DB_PASSWORD", ""))

REPO_ROOT = Path("/data/projects/Algo-trade-monorepo")
DATASETS_DIR = REPO_ROOT / "datasets"
ARTIFACTS_DIR = REPO_ROOT / "model_artifacts"
ML_SRC = REPO_ROOT / "ml" / "ml"
COLLECTOR_SRC = REPO_ROOT / "services" / "collector" / "src"

NIBI_DATA_DIR = f"{NIBI_SIM_DIR}/data"
NIBI_RUN_ROOT = f"{NIBI_SIM_DIR}/run_root"
NIBI_SBATCH = f"{NIBI_SIM_DIR}/ml/ml/nibi/sim_base_train.sbatch"

NIBI_KEY = Path(os.getenv("NIBI_SSH_KEY", str(Path.home() / ".ssh" / "nibi_key")))
NIBI_SOCKET_DIR = Path(os.getenv("NIBI_SOCKET_DIR", str(Path.home() / ".ssh" / "cm")))
NIBI_SOCKET_PATH = str(NIBI_SOCKET_DIR / f"nibi-{NIBI_USER}@{NIBI_HOST}:22")
NIBI_VENV = os.getenv("NIBI_VENV", f"/home/{NIBI_USER}/ENV")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

PIPELINE_PYTHON_PATH = REPO_ROOT / "pipeline-venv" / "bin" / "python"
if PIPELINE_PYTHON_PATH.exists():
    PIPELINE_PYTHON = PIPELINE_PYTHON_PATH
else:
    PIPELINE_PYTHON = Path(sys.executable)

MARKET_TZ = ZoneInfo("America/New_York")
TRAINING_RTH_BAR_COUNT = 26
MARKET_OPEN_ET = dt.time(9, 30)
MARKET_CLOSE_ET = dt.time(16, 0)


def _ssh(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    r = subprocess.run(
        [
            "ssh",
            "-i", str(NIBI_KEY),
            "-o", f"ControlPath={NIBI_SOCKET_PATH}",
            "-o", "ControlMaster=no",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            f"{NIBI_USER}@{NIBI_HOST}",
            cmd,
        ],
        capture_output=True, text=True, timeout=timeout + 5,
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def task_ssh_health_check(**ctx) -> None:
    rc, out, err = _ssh("echo pong && sbatch --version | head -1", timeout=20)
    if rc != 0:
        raise AirflowException(f"NIBI SSH failed (rc={rc}): {err}")
    print(f"NIBI OK: {out.replace(chr(10), ' | ')}")


def task_catchup_today_data(**ctx):
    """
    1. Runs intraday_data_collection.py to fill stg_raw.
    2. Runs export_staging_to_core.
    3. Re-aggregates all 15-min slots for today.
    """
    import exchange_calendars as xcals
    from sqlalchemy import create_engine, text

    now_et = dt.datetime.now(MARKET_TZ)
    today = now_et.date()

    xnys = xcals.get_calendar("XNYS")
    if not xnys.is_session(str(today)):
        print(f"{today} is not a trading day. Skipping catch-up.")
        return

    # ── Stage 1: Collection ──────────────────────────────────────────
    print(f"Running catch-up collection for {today}...")
    fetch_env = os.environ.copy()
    fetch_env["PYTHONPATH"] = str(COLLECTOR_SRC)
    fetch_env.update({
        "DB_HOST": DB_HOST,
        "DB_PORT": str(DB_PORT),
        "DB_NAME": DB_NAME,
        "DB_USER": DB_USER,
        "DB_PASSWORD": DB_PASS,
        "MARKET_TZ": "America/New_York",
        "MARKET_OPEN": "04:00",
        "MARKET_CLOSE": "21:00", # Catch everything up to late evening if needed
    })

    fetch_proc = subprocess.run(
        [str(PIPELINE_PYTHON), str(COLLECTOR_SRC / "intraday_data_collection.py")],
        cwd=str(COLLECTOR_SRC),
        env=fetch_env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if fetch_proc.returncode != 0:
        raise AirflowException(f"intraday_data_collection.py failed:\n{fetch_proc.stderr}")
    print("Collection OK.")

    # ── Stage 2: Export ──────────────────────────────────────────────
    print("Exporting stg_raw -> core_dbms.market_data_5m...")
    sys.path.insert(0, str(COLLECTOR_SRC))
    from model.orm_db import get_engine as _get_engine, get_session_factory
    from utils.scheduled_pipeline import export_staging_to_core

    engine = _get_engine(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        summary = export_staging_to_core(session)
        session.commit()
    engine.dispose()
    print(f"Export OK: exported {summary.exported_rows} rows.")

    # ── Stage 3: Aggregation ─────────────────────────────────────────
    print(f"Re-aggregating 15-min windows for {today}...")
    # Slots from 09:30 to 15:45 inclusive (26 slots)
    slots = []
    current = dt.datetime.combine(today, MARKET_OPEN_ET, tzinfo=MARKET_TZ).astimezone(dt.timezone.utc)
    close_utc = dt.datetime.combine(today, MARKET_CLOSE_ET, tzinfo=MARKET_TZ).astimezone(dt.timezone.utc)
    
    while current < close_utc:
        slots.append(current)
        current += dt.timedelta(minutes=15)

    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    ok_count = 0
    with engine.begin() as conn:
        for ts in slots:
            try:
                conn.execute(text("CALL dw.process_15min_window(:ts)"), {"ts": ts})
                ok_count += 1
            except Exception as exc:
                print(f"  WARNING: slot {ts.isoformat()} failed: {exc}")
    engine.dispose()
    print(f"Aggregation OK: {ok_count}/{len(slots)} windows processed.")


def task_export_parquet(**ctx) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    sim_date = ctx["ds"]
    out_path = DATASETS_DIR / f"snapshot_{sim_date}.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Always rebuild for post-market catch-up to ensure today's data is included
    if out_path.exists():
        out_path.unlink()
    if tmp_path.exists():
        tmp_path.unlink()

    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    chunks = []
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM ml.market_data_15m")).scalar()
        print(f"Total rows: {total:,}")
        for chunk in pd.read_sql(text("SELECT * FROM ml.market_data_15m ORDER BY symbol, window_ts"), conn, chunksize=200_000):
            chunks.append(chunk)
            print(f"  Loaded {sum(len(c) for c in chunks):,} / {total:,} rows ...")
    engine.dispose()

    df = pd.concat(chunks, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(df), tmp_path)
    tmp_path.rename(out_path)
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {df['symbol'].nunique()} symbols)")
    ctx["ti"].xcom_push(key="parquet_path", value=str(out_path))


def task_sync_to_nibi(**ctx) -> None:
    parquet_path = Path(ctx["ti"].xcom_pull(task_ids="export_parquet", key="parquet_path"))
    _ssh(f"mkdir -p {NIBI_SIM_DIR}/ml/ml {NIBI_DATA_DIR}")

    # Sync code
    ssh_e = f"ssh -i {NIBI_KEY} -o ControlPath={NIBI_SOCKET_PATH} -o ControlMaster=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    subprocess.run(
        ["rsync", "-az", "--delete", "-e", ssh_e, str(ML_SRC) + "/", f"{NIBI_USER}@{NIBI_HOST}:{NIBI_SIM_DIR}/ml/ml/"],
        check=True, timeout=120,
    )

    # SCP parquet
    subprocess.run(
        ["scp", "-i", str(NIBI_KEY), "-o", f"ControlPath={NIBI_SOCKET_PATH}", "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", str(parquet_path), f"{NIBI_USER}@{NIBI_HOST}:{NIBI_DATA_DIR}/{parquet_path.name}"],
        check=True, timeout=600,
    )
    print("Code and Parquet synced to NIBI.")
    ctx["ti"].xcom_push(key="remote_parquet", value=f"{NIBI_DATA_DIR}/{parquet_path.name}")


def task_submit_base_job(**ctx) -> None:
    sim_date = ctx["ds"]
    remote_parq = ctx["ti"].xcom_pull(task_ids="sync_to_nibi", key="remote_parquet")
    
    # We use --fast to speed up base training if it's just a catch-up
    cmd = f"sbatch {NIBI_SBATCH} --parquet {remote_parq} --sim-date {sim_date} --base-only"
    print(f"Submitting: {cmd}")
    rc, out, err = _ssh(cmd, timeout=30)
    if rc != 0:
        raise AirflowException(f"sbatch failed: {err}")

    job_id = next((tok for tok in out.split() if tok.isdigit()), None)
    if not job_id:
        raise AirflowException(f"Could not parse job ID from sbatch output: {out}")

    print(f"Base job submitted: {job_id}")
    ctx["ti"].xcom_push(key="job_id", value=job_id)


class NibiBaseJobSensor(BaseSensorOperator):
    def __init__(self, job_id_task: str, **kwargs):
        super().__init__(poke_interval=120, timeout=14_400, mode="reschedule", **kwargs)
        self.job_id_task = job_id_task

    def poke(self, context) -> bool:
        job_id = context["ti"].xcom_pull(task_ids=self.job_id_task, key="job_id")
        rc, out, _ = _ssh(f"sacct -j {job_id} --format=State --noheader 2>/dev/null | head -1", timeout=20)
        state = out.strip().split()[0] if out.strip() else ""
        print(f"  Job {job_id} state: {state}")
        if state.startswith("COMPLETED"): return True
        if any(state.startswith(s) for s in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"]):
            raise AirflowException(f"Job {job_id} failed with state {state}")
        return False


def _atomic_symlink(symlink: Path, target: Path) -> None:
    tmp = symlink.parent / (symlink.name + ".new")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target.name)
    tmp.rename(symlink)


def task_promote_and_reload(**ctx) -> None:
    sim_date = ctx["ds"]
    warm_dest = ARTIFACTS_DIR / f"warm_{sim_date}"
    if warm_dest.exists(): shutil.rmtree(warm_dest)
    warm_dest.mkdir(parents=True, exist_ok=True)

    ssh_e = f"ssh -i {NIBI_KEY} -o ControlPath={NIBI_SOCKET_PATH} -o ControlMaster=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    subprocess.run(
        ["rsync", "-az", "--compress", "-e", ssh_e, f"{NIBI_USER}@{NIBI_HOST}:{NIBI_RUN_ROOT}/current/", str(warm_dest) + "/"],
        check=True, timeout=1800,
    )

    _atomic_symlink(ARTIFACTS_DIR / "current_base", warm_dest)
    _atomic_symlink(ARTIFACTS_DIR / "current_base_eod", warm_dest)
    print(f"Promoted {warm_dest.name} to current_base")
    ctx["ti"].xcom_push(key="warm_artifact_dir", value=str(warm_dest))


def task_generate_base_predictions(**ctx) -> None:
    """
    Generate all 26 step prediction CSVs locally after base model is promoted.
    Uses the freshly promoted current_base model + recent DB data.
    Non-fatal — a failure logs a warning but does not block the backend reload.
    """
    import tempfile

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    sim_date = ctx["ds"]
    base_bundle = ARTIFACTS_DIR / "current_base"
    sim_dest = ARTIFACTS_DIR / f"simulation_{sim_date}"

    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"base_pred_{sim_date}_"))
        slice_path = tmp_dir / "slices" / "slice_1945.parquet"
        slice_path.parent.mkdir(parents=True)

        db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url, pool_pre_ping=True)
        print(f"Pulling 15 trading days for base predictions (sim_date={sim_date}) ...")
        with engine.connect() as conn:
            df = pd.read_sql(
                text("""
                    SELECT * FROM ml.market_data_15m
                    WHERE trade_date >= (
                        SELECT MIN(trade_date) FROM (
                            SELECT DISTINCT trade_date FROM ml.market_data_15m
                            ORDER BY trade_date DESC LIMIT 15
                        ) r
                    )
                    ORDER BY symbol, window_ts
                """),
                conn,
            )
        engine.dispose()
        pq.write_table(pa.Table.from_pandas(df), slice_path)
        print(f"  {len(df):,} rows, {df['symbol'].nunique()} symbols")

        run_root = tmp_dir / "run_root"
        for i in range(26):
            step_dir = run_root / f"step_{i:02d}"
            step_dir.mkdir(parents=True)
            (step_dir / "models").symlink_to((base_bundle / "models").resolve())
            (step_dir / "feature_names.json").symlink_to((base_bundle / "feature_names.json").resolve())
            (step_dir / "predictions").mkdir()

        gen_script = REPO_ROOT / "ml" / "ml" / "nibi" / "gen_step_predictions.py"
        venv_python = Path("/data/env/bin/python3")
        python_bin = venv_python if venv_python.exists() else Path(sys.executable)

        result = subprocess.run(
            [str(python_bin), str(gen_script), "--run-root", str(run_root), "--sim-date", sim_date],
            capture_output=True, text=True, timeout=3600, cwd=str(REPO_ROOT),
        )
        print(result.stdout[-3000:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            raise RuntimeError(f"gen_step_predictions failed (rc={result.returncode}):\n{result.stderr[-1000:]}")

        sim_dest.mkdir(parents=True, exist_ok=True)
        import datetime as _dt, json as _json
        copied = 0
        for i in range(26):
            src = run_root / f"step_{i:02d}" / "predictions" / "predictions.csv"
            dst_dir = sim_dest / f"step_{i:02d}" / "predictions"
            dst_dir.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst_dir / "predictions.csv")
                copied += 1

        if copied < 26:
            print(f"WARNING: only {copied}/26 steps — skipping promote")
            return

        d = _dt.date.fromisoformat(sim_date)
        utc_open = _dt.datetime(d.year, d.month, d.day, 13, 30, tzinfo=_dt.timezone.utc)
        summary = {
            "sim_date": sim_date,
            "status": "success",
            "source": "base_model_local",
            "steps": [
                {
                    "step": i,
                    "as_of_ts": (utc_open + _dt.timedelta(minutes=15 * i)).isoformat(),
                    "et_label": (utc_open + _dt.timedelta(minutes=15 * i - 4 * 60)).strftime("%H:%M"),
                    "status": "ok",
                }
                for i in range(26)
            ],
        }
        (sim_dest / "simulation_summary.json").write_text(_json.dumps(summary, indent=2))
        _atomic_symlink(ARTIFACTS_DIR / "current_simulation", sim_dest)
        print(f"Promoted: current_simulation → {sim_dest.name}")

    except Exception as exc:
        print(f"WARNING: generate_base_predictions failed (non-fatal): {exc}")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def task_reload_backend(**ctx) -> None:
    for endpoint in [
        "/api/v1/inference/admin/reload-model",
        "/api/v1/inference/admin/reload-base-model",
        "/api/v1/simulation/admin/reload-simulation",
    ]:
        try:
            requests.post(f"{BACKEND_URL}{endpoint}", timeout=30)
        except Exception as exc:
            print(f"WARNING: reload {endpoint} failed: {exc}")


with DAG(
    dag_id="nibi_post_market_catchup_and_base_train",
    default_args={
        "owner": "ml-team",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": dt.timedelta(minutes=5),
    },
    description="Force catch-up for today's data then train a fresh base model on NIBI.",
    schedule=None, # Manual trigger only
    start_date=dt.datetime(2026, 5, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "catchup", "nibi"],
) as dag:

    t1_health = PythonOperator(task_id="ssh_health_check", python_callable=task_ssh_health_check)
    t2_catchup = PythonOperator(task_id="catchup_today_data", python_callable=task_catchup_today_data, execution_timeout=dt.timedelta(minutes=60))
    t3_export = PythonOperator(task_id="export_parquet", python_callable=task_export_parquet, execution_timeout=dt.timedelta(minutes=20))
    t4_sync = PythonOperator(task_id="sync_to_nibi", python_callable=task_sync_to_nibi, execution_timeout=dt.timedelta(minutes=20))
    t5_submit = PythonOperator(task_id="submit_base_job", python_callable=task_submit_base_job)
    t6_poll = NibiBaseJobSensor(task_id="poll_job", job_id_task="submit_base_job")
    t7_promote = PythonOperator(task_id="promote_and_reload", python_callable=task_promote_and_reload, execution_timeout=dt.timedelta(minutes=30))
    t8_gen_preds = PythonOperator(task_id="generate_base_predictions", python_callable=task_generate_base_predictions, execution_timeout=dt.timedelta(hours=1), retries=0)
    t9_reload = PythonOperator(task_id="reload_backend", python_callable=task_reload_backend, execution_timeout=dt.timedelta(minutes=2), retries=0)

    t1_health >> t2_catchup >> t3_export >> t4_sync >> t5_submit >> t6_poll >> t7_promote >> t8_gen_preds >> t9_reload

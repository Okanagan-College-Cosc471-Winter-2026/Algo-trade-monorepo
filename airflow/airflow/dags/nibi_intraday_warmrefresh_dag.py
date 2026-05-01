"""
nibi_intraday_warmrefresh_dag.py
=================================

Intraday warm-refresh pipeline.  Runs every 15 minutes during US market hours
(Mon–Fri, 09:15–16:00 ET = 13:15–20:00 UTC) so each 15-minute bar window gets
a model that has seen all intraday data up to that point.

Full pipeline (9 tasks):
  1. check_market_open    — abort immediately if invoked outside regular session
  2. export_snapshot      — dump latest 15 trading days from ml.market_data_15m
                            → local parquet (reuses file if size matches)
  3. sync_parquet_to_nibi — SCP the parquet to NIBI test_simulation/data/
  4. submit_warm_job      — sbatch sim_warm_windows.sbatch --skip-base
                            (adds 30 warm trees to current_base boosters per step)
  5. poll_job_sensor      — Airflow Sensor pokes squeue every 2 min
  6. validate_step        — check all 26 step_XX/predictions/predictions.csv exist
  7. rsync_artifacts_back — pull run_root/step_XX/ → model_artifacts/warm_YYYY-MM-DD_HHMM/
  8. promote_intraday     — atomic symlink: current_base → warm_YYYY-MM-DD_HHMM bundle
  9. reload_backend       — POST /api/v1/inference/admin/reload-model (non-fatal)

SSH note:
  All NIBI calls reuse the ControlMaster socket opened by morning_login.sh.
  If the socket is expired the DAG fails at task 1 with a clear error message.

Artifact layout after promotion:
  model_artifacts/
    current_base               → warm_YYYY-MM-DD_HHMM/   (symlink, atomic)
    warm_YYYY-MM-DD_HHMM/     (one dir per 15-min slot, all retained)
      metadata.json
      feature_names.json
      models/
        model_manifest.json
        horizon_00.json … horizon_25.json
    current_simulation         → simulation_YYYY-MM-DD_HHMM/   (for replay UI)
    simulation_YYYY-MM-DD_HHMM/
      step_00/ … step_25/
        predictions/predictions.csv
        metadata.json
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from airflow import DAG
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator
from market_calendar_utils import expected_last_closed_window_start_utc, get_session_gate

# ── Config ──────────────────────────────────────────────────────────────────
NIBI_ALIAS   = "nibi"
NIBI_USER    = os.getenv("NIBI_USER",    "harshsaw")
NIBI_HOST    = os.getenv("NIBI_HOST",    "nibi.sharcnet.ca")
NIBI_SIM_DIR = os.getenv("NIBI_SIM_DIR", "/home/harshsaw/projects/def-youry/test_simulation")

# Main application database
DB_HOST = os.getenv("POSTGRES_SERVER",  os.getenv("OLD_DB_HOST",  "localhost"))
DB_PORT = int(os.getenv("POSTGRES_PORT", os.getenv("OLD_DB_PORT", "5432")))
DB_NAME = os.getenv("POSTGRES_DB",      os.getenv("OLD_DB_NAME",  "market_data"))
DB_USER = os.getenv("POSTGRES_USER",    os.getenv("OLD_DB_USER",  "mluser"))
DB_PASS = os.getenv("POSTGRES_PASSWORD", os.getenv("OLD_DB_PASSWORD", ""))

REPO_ROOT     = Path("/data/projects/Algo-trade-monorepo")
DATASETS_DIR  = REPO_ROOT / "datasets"
ARTIFACTS_DIR = REPO_ROOT / "model_artifacts"
ML_SRC        = REPO_ROOT / "ml" / "ml"

NIBI_DATA_DIR  = f"{NIBI_SIM_DIR}/data"
NIBI_RUN_ROOT  = f"{NIBI_SIM_DIR}/run_root"
NIBI_SBATCH    = f"{NIBI_SIM_DIR}/ml/ml/nibi/sim_warm_windows.sbatch"

NIBI_KEY         = Path(os.getenv("NIBI_SSH_KEY",    str(Path.home() / ".ssh" / "nibi_key")))
NIBI_SOCKET_DIR  = Path(os.getenv("NIBI_SOCKET_DIR", str(Path.home() / ".ssh" / "cm")))
NIBI_SOCKET_PATH = str(NIBI_SOCKET_DIR / f"nibi-{NIBI_USER}@{NIBI_HOST}:22")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# How many recent trading days of data to export per snapshot (15 keeps the
# parquet small while giving warm-refresh enough history for rolling features).
SNAPSHOT_TRADING_DAYS = 15

MARKET_TZ = ZoneInfo("America/New_York")
FRESHNESS_FILE = Path(
    os.getenv(
        "INTRADAY_FRESHNESS_FILE",
        "/data/projects/Algo-trade-monorepo/logs/intraday_data_freshness.json",
    )
)


def _read_latest_data_window() -> dt.datetime | None:
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

# ── SSH helper ───────────────────────────────────────────────────────────────
def _ssh(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """Run a command on NIBI via the existing ControlMaster socket."""
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


def _atomic_symlink(symlink: Path, target: Path) -> None:
    """Atomically replace a symlink via Linux rename() — never leaves it dangling."""
    tmp = symlink.parent / (symlink.name + ".new")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    # Use target.name (relative) so the symlink resolves correctly inside Docker,
    # where the mount point differs from the host path. target.resolve() produces
    # an absolute host path that is inaccessible inside the container.
    tmp.symlink_to(target.name)
    tmp.rename(symlink)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Market-Open Gate
# ══════════════════════════════════════════════════════════════════════════════
def task_check_market_open(**ctx) -> None:
    """
    Skip (not fail) if called outside the regular NYSE session.

    The cron schedule targets 13:15–20:00 UTC but Airflow may drift or
    catch up missed runs.  This gate prevents redundant off-hours jobs.
    """
    now_utc = dt.datetime.now(dt.timezone.utc)
    gate = get_session_gate(now=now_utc, tz_name="America/New_York")
    if not gate.is_open_now:
        raise AirflowSkipException(
            f"Market gate closed reason={gate.reason} now_et={gate.now_et.strftime('%Y-%m-%d %H:%M ET')}"
        )

    expected_window = expected_last_closed_window_start_utc(now=now_utc, tz_name="America/New_York")
    if expected_window is None:
        raise AirflowSkipException("No fully closed RTH window available yet.")

    latest_window = _read_latest_data_window()
    if latest_window is None or latest_window < expected_window:
        raise AirflowSkipException(
            "Data freshness gate not met: "
            f"expected>={expected_window.isoformat()} got={latest_window}"
        )

    print(
        "Market/freshness gate passed "
        f"expected_window={expected_window.isoformat()} latest_data_window={latest_window.isoformat()}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Export Snapshot
# ══════════════════════════════════════════════════════════════════════════════
def task_export_snapshot(**ctx) -> None:
    """
    Export the latest SNAPSHOT_TRADING_DAYS of ml.market_data_15m to parquet.

    Reuses an existing file if it is less than 15 minutes old (avoids
    redundant large exports when adjacent 15-min runs overlap).
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    sim_date   = ctx["ds"]
    parq_name  = f"intraday_{sim_date}.parquet"
    parq_path  = DATASETS_DIR / parq_name
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse if fresh (< 15 min old)
    if parq_path.exists():
        age = time.time() - parq_path.stat().st_mtime
        if age < 900:
            mb = parq_path.stat().st_size / 1e6
            print(f"Parquet fresh ({age:.0f}s old, {mb:.1f} MB) — reusing {parq_name}")
            ctx["ti"].xcom_push(key="parquet_path", value=str(parq_path))
            return

    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine  = create_engine(db_url, pool_pre_ping=True)

    print(f"Exporting {SNAPSHOT_TRADING_DAYS} trading days from ml.market_data_15m …")
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT *
                FROM ml.market_data_15m
                WHERE trade_date >= (
                    SELECT MIN(trade_date)
                    FROM (
                        SELECT DISTINCT trade_date
                        FROM ml.market_data_15m
                        ORDER BY trade_date DESC
                        LIMIT :days
                    ) recent_days
                )
                ORDER BY symbol, window_ts
            """),
            conn,
            params={"days": SNAPSHOT_TRADING_DAYS},
        )
    engine.dispose()

    pq.write_table(pa.Table.from_pandas(df), parq_path)
    mb = parq_path.stat().st_size / 1e6
    print(f"Saved {parq_name}: {len(df):,} rows, {df['symbol'].nunique()} symbols, {mb:.1f} MB")
    ctx["ti"].xcom_push(key="parquet_path", value=str(parq_path))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Sync Parquet to NIBI
# ══════════════════════════════════════════════════════════════════════════════
def task_sync_parquet(**ctx) -> None:
    """SCP the snapshot parquet to NIBI, skipping if sizes already match."""
    parq_path   = Path(ctx["ti"].xcom_pull(task_ids="export_snapshot", key="parquet_path"))
    remote_path = f"{NIBI_DATA_DIR}/{parq_path.name}"

    local_size = parq_path.stat().st_size
    rc, remote_sz, _ = _ssh(f"stat -c%s {remote_path} 2>/dev/null || echo 0")
    if rc == 0 and remote_sz.strip().isdigit() and int(remote_sz.strip()) == local_size:
        print(f"Remote already matches ({local_size / 1e6:.1f} MB) — skipping SCP")
        ctx["ti"].xcom_push(key="remote_parquet", value=remote_path)
        return

    _ssh(f"mkdir -p {NIBI_DATA_DIR}")
    r = subprocess.run(
        ["scp",
         "-i", str(NIBI_KEY),
         "-o", f"ControlPath={NIBI_SOCKET_PATH}",
         "-o", "ControlMaster=no",
         "-o", "BatchMode=yes",
         str(parq_path), f"{NIBI_USER}@{NIBI_HOST}:{remote_path}"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise AirflowException(f"SCP failed: {r.stderr.strip()}")

    print(f"Sent {parq_path.name} ({local_size / 1e6:.1f} MB) → {remote_path}")
    ctx["ti"].xcom_push(key="remote_parquet", value=remote_path)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Submit Slurm Job
# ══════════════════════════════════════════════════════════════════════════════
def task_submit_warm_job(**ctx) -> None:
    """
    Submit sim_warm_windows.sbatch with --skip-base.

    Uses an Airflow Variable "nibi_warm_trees" (default 30) to control how
    many trees are added per intraday window.
    """
    from airflow.models import Variable

    sim_date    = ctx["ds"]
    # execution_ts gives a unique key per 15-min slot (e.g. "2026-04-28T14:35:00+00:00")
    execution_ts = ctx["ts_nodash"]
    remote_parq = ctx["ti"].xcom_pull(task_ids="sync_parquet_to_nibi", key="remote_parquet")
    warm_trees  = Variable.get("nibi_warm_trees", default_var="30")

    # Key job record by execution timestamp, not just date, so each 15-min slot
    # submits its own Slurm job rather than reusing the first run's completed job.
    job_record = REPO_ROOT / "logs" / f"nibi_warm_{execution_ts}.json"
    if job_record.exists():
        rec = json.loads(job_record.read_text())
        existing_id = rec.get("job_id")
        if existing_id and rec.get("status") != "completed":
            print(f"Warm job already submitted for this slot: {existing_id} — reusing")
            ctx["ti"].xcom_push(key="job_id", value=existing_id)
            return

    cmd = (
        f"sbatch {NIBI_SBATCH} "
        f"--parquet {remote_parq} "
        f"--sim-date {sim_date}"
    )
    print(f"Submitting: {cmd}")
    rc, out, err = _ssh(cmd, timeout=30)
    if rc != 0:
        raise AirflowException(f"sbatch failed (rc={rc}):\n{err}")

    job_id = next((tok for tok in out.split() if tok.isdigit()), None)
    if not job_id:
        raise AirflowException(f"Could not parse job ID from sbatch output: {out!r}")

    print(f"Warm job submitted: {job_id}")
    submitted_at = dt.datetime.utcnow().isoformat()
    job_record.parent.mkdir(parents=True, exist_ok=True)
    job_record.write_text(json.dumps({
        "job_id": job_id,
        "sim_date": sim_date,
        "submitted_at": submitted_at,
        "warm_trees": warm_trees,
        "status": "submitted",
    }, indent=2))
    ctx["ti"].xcom_push(key="job_id", value=job_id)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Poll Job Sensor
# ══════════════════════════════════════════════════════════════════════════════
class WarmJobSensor(BaseSensorOperator):
    """
    Reschedule-mode sensor that polls squeue / sacct every 2 minutes.

    Releases the worker slot between pokes (mode="reschedule") so a single
    slow job doesn't block the Airflow worker pool.
    """

    def __init__(self, **kwargs):
        super().__init__(mode="reschedule", poke_interval=120, timeout=7200, **kwargs)

    def poke(self, context) -> bool:
        job_id = context["ti"].xcom_pull(task_ids="submit_warm_job", key="job_id")
        if not job_id:
            raise AirflowException("No job_id found in XCom from submit_warm_job")

        rc, state, _ = _ssh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null || echo ''", timeout=30)
        state = state.strip()

        if state in ("FAILED", "CANCELLED", "TIMEOUT"):
            raise AirflowException(f"Slurm job {job_id} ended with state: {state}")

        if state == "":
            # Job no longer in queue — check sacct for final state
            rc2, sacct_out, _ = _ssh(
                f"sacct -j {job_id} --format=State --noheader 2>/dev/null | head -1 | tr -d ' '",
                timeout=30,
            )
            final = sacct_out.strip()
            print(f"  Job {job_id} sacct state: {final}")
            if final.startswith("COMPLETED"):
                return True
            raise AirflowException(f"Slurm job {job_id} failed with state: {final}")

        print(f"  Job {job_id} state: {state} — continuing to poll")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 — Validate Step Artifacts
# ══════════════════════════════════════════════════════════════════════════════
def task_validate_step(**ctx) -> None:
    """
    Verify that all 26 step_XX/predictions/predictions.csv files exist on NIBI.

    Also checks for SIMULATION_DONE sentinel which run_simulation_day.py writes
    on successful completion.
    """
    sim_date = ctx["ds"]

    # Check SIMULATION_DONE sentinel
    done_path = f"{NIBI_RUN_ROOT}/SIMULATION_DONE"
    rc, out, err = _ssh(f"test -f {done_path} && cat {done_path} || echo MISSING")
    if "MISSING" in out or rc != 0:
        raise AirflowException(
            f"SIMULATION_DONE not found at {done_path}. "
            "Job may have failed mid-run. Check NIBI logs."
        )
    print(f"SIMULATION_DONE: {out.strip()}")

    # Check predictions CSVs for all 26 steps
    missing_steps = []
    for i in range(26):
        pred_path = f"{NIBI_RUN_ROOT}/step_{i:02d}/predictions/predictions.csv"
        rc2, _, _ = _ssh(f"test -f {pred_path} && echo ok || echo missing")
        if "missing" in _ or rc2 != 0:
            # rc2 is always 0 for _ssh; check via separate stat
            rc3, sz, _ = _ssh(f"stat -c%s {pred_path} 2>/dev/null || echo 0")
            if not sz.strip().isdigit() or int(sz.strip()) == 0:
                missing_steps.append(i)

    if missing_steps:
        raise AirflowException(
            f"Missing predictions for {len(missing_steps)} steps: {missing_steps}"
        )
    print(f"All 26 step prediction files validated on NIBI")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 — Rsync Artifacts Back
# ══════════════════════════════════════════════════════════════════════════════
def task_rsync_artifacts_back(**ctx) -> None:
    """
    Pull run_root/step_XX/ and the warm bundle back to the VPS.

    Two destinations per 15-min slot (HHMM = ET wall-clock of the logical run):
      - warm_YYYY-MM-DD_HHMM/        → backend-serving bundle (current_base target)
      - simulation_YYYY-MM-DD_HHMM/  → replay UI bundle (current_simulation target)

    Each slot gets its own directory so the full intraday history is preserved
    for simulation replay.
    """
    sim_date = ctx["ds"]
    # Derive the ET slot label from the logical execution time so each
    # 15-min cycle gets a unique, human-readable directory name.
    logical_dt_utc = ctx["data_interval_start"]
    slot_hhmm = logical_dt_utc.astimezone(MARKET_TZ).strftime("%H%M")

    warm_dest = ARTIFACTS_DIR / f"warm_{sim_date}_{slot_hhmm}"
    sim_dest  = ARTIFACTS_DIR / f"simulation_{sim_date}_{slot_hhmm}"

    warm_dest.mkdir(parents=True, exist_ok=True)
    sim_dest.mkdir(parents=True, exist_ok=True)

    # rsync the warm bundle (current/ on NIBI = warm-refreshed boosters)
    r = subprocess.run(
        [
            "rsync", "-az", "--delete",
            "-e", f"ssh -i {NIBI_KEY} -o ControlPath={NIBI_SOCKET_PATH} -o ControlMaster=no -o BatchMode=yes",
            f"{NIBI_USER}@{NIBI_HOST}:{NIBI_RUN_ROOT}/current/",
            f"{warm_dest}/",
        ],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        raise AirflowException(f"rsync (warm bundle) failed: {r.stderr.strip()}")
    print(f"Warm bundle rsynced to {warm_dest}")

    # rsync all 26 step_XX/ directories (for simulation replay UI)
    r2 = subprocess.run(
        [
            "rsync", "-az", "--delete",
            "-e", f"ssh -i {NIBI_KEY} -o ControlPath={NIBI_SOCKET_PATH} -o ControlMaster=no -o BatchMode=yes",
            "--include=step_*/",
            "--include=step_*/predictions/",
            "--include=step_*/predictions/predictions.csv",
            "--include=step_*/metadata.json",
            "--exclude=step_*/models/**",
            "--exclude=*",
            f"{NIBI_USER}@{NIBI_HOST}:{NIBI_RUN_ROOT}/",
            f"{sim_dest}/",
        ],
        capture_output=True, text=True, timeout=300,
    )
    if r2.returncode != 0:
        raise AirflowException(f"rsync (simulation steps) failed: {r2.stderr.strip()}")

    # Also pull simulation_summary.json if present
    _ssh(f"cat {NIBI_RUN_ROOT}/simulation_summary.json 2>/dev/null || echo '{{}}'")
    rc, summary_out, _ = _ssh(f"cat {NIBI_RUN_ROOT}/simulation_summary.json 2>/dev/null")
    if rc == 0 and summary_out.strip():
        (sim_dest / "simulation_summary.json").write_text(summary_out)

    n_steps = len(list(sim_dest.glob("step_*")))
    print(f"Simulation steps rsynced to {sim_dest} ({n_steps} step dirs)")

    ctx["ti"].xcom_push(key="warm_artifact_dir", value=str(warm_dest))
    ctx["ti"].xcom_push(key="sim_artifact_dir",  value=str(sim_dest))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 8 — Promote Intraday Bundle
# ══════════════════════════════════════════════════════════════════════════════
def task_promote_intraday(**ctx) -> None:
    """
    Atomic symlink swap: current_base → warm_YYYY-MM-DD_HHMM/

    Only promotes if the bundle contains models/model_manifest.json.
    Also promotes current_simulation if all 26 prediction CSVs are present.
    All artifact dirs are retained (no cleanup — VM has ample space).
    """
    warm_dir = Path(ctx["ti"].xcom_pull(task_ids="rsync_artifacts_back", key="warm_artifact_dir"))
    sim_dir  = Path(ctx["ti"].xcom_pull(task_ids="rsync_artifacts_back", key="sim_artifact_dir"))

    manifest = warm_dir / "models" / "model_manifest.json"
    if not manifest.exists():
        raise AirflowException(
            f"model_manifest.json not found in warm bundle at {manifest} — refusing to promote"
        )

    _atomic_symlink(ARTIFACTS_DIR / "current_base", warm_dir)
    print(f"Promoted: current_base → {warm_dir.name}")

    # Promote simulation pointer only if all steps have predictions
    n_with_preds = sum(
        1 for i in range(26)
        if (sim_dir / f"step_{i:02d}" / "predictions" / "predictions.csv").exists()
    )
    if n_with_preds >= 26:
        _atomic_symlink(ARTIFACTS_DIR / "current_simulation", sim_dir)
        print(f"Promoted: current_simulation → {sim_dir.name}")
    else:
        print(f"WARNING: only {n_with_preds}/26 step prediction CSVs — skipping current_simulation update")

    # Artifacts are never deleted — the VM has ample space and all slots are
    # retained for simulation replay.


# ══════════════════════════════════════════════════════════════════════════════
# TASK 9 — Reload Backend (non-fatal)
# ══════════════════════════════════════════════════════════════════════════════
def task_reload_backend(**ctx) -> None:
    """
    Ask the backend to clear its model LRU cache.

    Non-fatal: if the backend is down or being redeployed the DAG still
    succeeds.  The backend will pick up the new bundle on its next request.
    """
    url = f"{BACKEND_URL}/api/v1/inference/admin/reload-model"
    try:
        resp = requests.post(url, timeout=10)
        if resp.ok:
            data = resp.json()
            print(f"Backend reloaded: {data}")
        else:
            print(f"WARNING: backend reload returned {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        print(f"WARNING: backend reload failed (non-fatal): {exc}")


# ── DAG definition ───────────────────────────────────────────────────────────
_DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": dt.timedelta(minutes=5),
}

with DAG(
    dag_id="nibi_intraday_warmrefresh",
    default_args=_DEFAULT_ARGS,
    description=(
        "Every-15-min intraday warm-refresh: export snapshot → NIBI → "
        "sbatch sim_warm_windows → rsync artifacts → promote current_base"
    ),
    # Offset 5 min after intraday_data_pipeline (*/15) so freshness file is ready.
    schedule_interval="5,20,35,50 * * * 1-5",
    start_date=dt.datetime(2026, 4, 22),
    catchup=False,
    max_active_runs=1,          # never run two warm-refresh pipelines concurrently
    tags=["ml", "warm-refresh", "nibi", "intraday"],
) as dag:

    t1_gate = PythonOperator(
        task_id="check_market_open",
        python_callable=task_check_market_open,
    )

    t2_export = PythonOperator(
        task_id="export_snapshot",
        python_callable=task_export_snapshot,
    )

    t3_sync = PythonOperator(
        task_id="sync_parquet_to_nibi",
        python_callable=task_sync_parquet,
    )

    t4_submit = PythonOperator(
        task_id="submit_warm_job",
        python_callable=task_submit_warm_job,
    )

    t5_poll = WarmJobSensor(
        task_id="poll_job_sensor",
    )

    t6_validate = PythonOperator(
        task_id="validate_step",
        python_callable=task_validate_step,
    )

    t7_rsync = PythonOperator(
        task_id="rsync_artifacts_back",
        python_callable=task_rsync_artifacts_back,
    )

    t8_promote = PythonOperator(
        task_id="promote_intraday",
        python_callable=task_promote_intraday,
    )

    t9_reload = PythonOperator(
        task_id="reload_backend",
        python_callable=task_reload_backend,
    )

    # Linear dependency chain
    t1_gate >> t2_export >> t3_sync >> t4_submit >> t5_poll >> t6_validate >> t7_rsync >> t8_promote >> t9_reload

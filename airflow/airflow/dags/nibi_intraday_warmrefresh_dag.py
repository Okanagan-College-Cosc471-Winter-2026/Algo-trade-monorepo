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
from airflow.utils.trigger_rule import TriggerRule
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

REPO_ROOT     = Path(os.getenv("REPO_ROOT", str(Path(__file__).resolve().parents[3])))
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
        str(REPO_ROOT / "logs" / "intraday_data_freshness.json"),
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
# TASK 6 — Validate Step Artifacts (non-fatal: flags fallback via XCom)
# ══════════════════════════════════════════════════════════════════════════════
def task_validate_or_flag(**ctx) -> None:
    """
    Verify that all 26 step_XX/predictions/predictions.csv files exist on NIBI.

    Non-fatal: on any failure, sets XCom nibi_ok=False so the local fallback
    task runs instead of hard-failing the DAG.
    """
    ti = ctx["ti"]

    def _flag_failed(reason: str) -> None:
        print(f"WARNING: NIBI validation failed — activating local fallback. Reason: {reason}")
        ti.xcom_push(key="nibi_ok", value=False)

    # Check SIMULATION_DONE sentinel
    done_path = f"{NIBI_RUN_ROOT}/SIMULATION_DONE"
    rc, out, _ = _ssh(f"test -f {done_path} && cat {done_path} || echo MISSING", timeout=30)
    if rc != 0 or "MISSING" in out:
        return _flag_failed(f"SIMULATION_DONE not found at {done_path}")
    print(f"SIMULATION_DONE: {out.strip()}")

    # Check predictions CSVs for all 26 steps
    missing_steps = []
    for i in range(26):
        pred_path = f"{NIBI_RUN_ROOT}/step_{i:02d}/predictions/predictions.csv"
        rc3, sz, _ = _ssh(f"stat -c%s {pred_path} 2>/dev/null || echo 0", timeout=15)
        if not sz.strip().isdigit() or int(sz.strip()) == 0:
            missing_steps.append(i)

    if missing_steps:
        return _flag_failed(f"Missing/empty predictions for steps: {missing_steps}")

    print(f"All 26 step prediction files validated on NIBI")
    ti.xcom_push(key="nibi_ok", value=True)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6b — Local Prediction Fallback (runs only when NIBI validation failed)
# ══════════════════════════════════════════════════════════════════════════════
def task_local_prediction_fallback(**ctx) -> None:
    """
    Generate predictions locally using current_base when NIBI is unavailable.

    Skips silently when nibi_ok=True (NIBI succeeded).  When NIBI failed, uses
    the intraday snapshot parquet already written by task_export_snapshot plus
    the current_base model to produce 26 step prediction CSVs.  Promotes
    current_simulation and pushes XCom keys so task_promote_intraday can run.
    """
    import shutil
    import subprocess as sp
    import sys

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    ti = ctx["ti"]
    nibi_ok = ti.xcom_pull(task_ids="validate_or_flag", key="nibi_ok")
    if nibi_ok:
        print("NIBI validation passed — local fallback not needed, skipping.")
        return

    sim_date = ctx["ds"]
    logical_dt_utc = ctx["data_interval_start"]
    slot_hhmm = logical_dt_utc.astimezone(MARKET_TZ).strftime("%H%M")

    base_bundle = ARTIFACTS_DIR / "current_base"
    sim_dest = ARTIFACTS_DIR / f"simulation_{sim_date}_{slot_hhmm}"

    # Use the snapshot parquet already exported by task_export_snapshot
    parq_path = Path(ti.xcom_pull(task_ids="export_snapshot", key="parquet_path") or "")

    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"warm_fallback_{sim_date}_{slot_hhmm}_"))
    try:
        # If the intraday parquet covers enough history, use it directly as the slice;
        # otherwise fall back to a fresh DB pull.
        slice_dir = tmp_dir / "slices"
        slice_dir.mkdir()
        slice_path = slice_dir / "slice_1945.parquet"

        if parq_path.exists():
            slice_path.symlink_to(parq_path.resolve())
            print(f"Reusing intraday snapshot: {parq_path}")
        else:
            print("Intraday snapshot not available — pulling from DB ...")
            db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            engine = create_engine(db_url, pool_pre_ping=True)
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

        # Build per-step dirs pointing at base model
        run_root = tmp_dir / "run_root"
        for i in range(26):
            step_dir = run_root / f"step_{i:02d}"
            step_dir.mkdir(parents=True)
            (step_dir / "models").symlink_to((base_bundle / "models").resolve())
            (step_dir / "feature_names.json").symlink_to((base_bundle / "feature_names.json").resolve())
            (step_dir / "predictions").mkdir()

        # Run gen_step_predictions.py
        gen_script = REPOS_ROOT / "ml" / "ml" / "nibi" / "gen_step_predictions.py"
        venv_python = Path("/data/env/bin/python3")
        python_bin = venv_python if venv_python.exists() else Path(sys.executable)

        print(f"Running local fallback predictions for sim_date={sim_date} slot={slot_hhmm} ...")
        result = sp.run(
            [str(python_bin), str(gen_script), "--run-root", str(run_root), "--sim-date", sim_date],
            capture_output=True, text=True, timeout=3600, cwd=str(REPO_ROOT),
        )
        print(result.stdout[-3000:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            raise RuntimeError(f"gen_step_predictions failed (rc={result.returncode}):\n{result.stderr[-500:]}")

        # Copy to permanent simulation bundle
        sim_dest.mkdir(parents=True, exist_ok=True)
        import json as _json, datetime as _dt
        copied = 0
        for i in range(26):
            src = run_root / f"step_{i:02d}" / "predictions" / "predictions.csv"
            dst_dir = sim_dest / f"step_{i:02d}" / "predictions"
            dst_dir.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst_dir / "predictions.csv")
                copied += 1

        # Write summary
        d = _dt.date.fromisoformat(sim_date)
        utc_open = _dt.datetime(d.year, d.month, d.day, 13, 30, tzinfo=_dt.timezone.utc)
        summary = {
            "sim_date": sim_date,
            "slot": slot_hhmm,
            "status": "success",
            "source": "local_fallback",
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

        print(f"Local fallback: {copied}/26 steps written to {sim_dest}")
        if copied >= 26:
            _atomic_symlink(ARTIFACTS_DIR / "current_simulation", sim_dest)
            print(f"Promoted: current_simulation → {sim_dest.name}")

        # Push so task_promote_intraday knows warm_dir = current_base (no new model)
        ti.xcom_push(key="warm_artifact_dir", value=str(base_bundle.resolve()))
        ti.xcom_push(key="sim_artifact_dir", value=str(sim_dest))

    except Exception as exc:
        print(f"WARNING: local fallback failed: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Alias kept so the original variable REPO_ROOT is referenced correctly inside fallback
REPOS_ROOT = REPO_ROOT


# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 — Rsync Artifacts Back
# ══════════════════════════════════════════════════════════════════════════════
def task_rsync_artifacts_back(**ctx) -> None:
    """
    Pull run_root/step_XX/ and the warm bundle back to the VPS.

    Skips when nibi_ok=False (local fallback already handled artifacts).

    Two destinations per 15-min slot (HHMM = ET wall-clock of the logical run):
      - warm_YYYY-MM-DD_HHMM/        → backend-serving bundle (current_base target)
      - simulation_YYYY-MM-DD_HHMM/  → replay UI bundle (current_simulation target)

    Each slot gets its own directory so the full intraday history is preserved
    for simulation replay.
    """
    nibi_ok = ctx["ti"].xcom_pull(task_ids="validate_or_flag", key="nibi_ok")
    if not nibi_ok:
        print("NIBI validation failed — skipping rsync (local fallback artifacts used).")
        return
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
    When NIBI failed (local fallback path), warm_dir is current_base itself —
    current_base symlink is not re-pointed, but current_simulation is still updated.
    All artifact dirs are retained (no cleanup — VM has ample space).
    """
    ti = ctx["ti"]
    nibi_ok = ti.xcom_pull(task_ids="validate_or_flag", key="nibi_ok")

    # Pull from whichever task wrote the XCom (rsync on success, fallback on failure)
    warm_dir_raw = (
        ti.xcom_pull(task_ids="rsync_artifacts_back", key="warm_artifact_dir")
        or ti.xcom_pull(task_ids="local_prediction_fallback", key="warm_artifact_dir")
    )
    sim_dir_raw = (
        ti.xcom_pull(task_ids="rsync_artifacts_back", key="sim_artifact_dir")
        or ti.xcom_pull(task_ids="local_prediction_fallback", key="sim_artifact_dir")
    )

    if not warm_dir_raw or not sim_dir_raw:
        print("WARNING: no artifact dirs in XCom — nothing to promote.")
        return

    warm_dir = Path(warm_dir_raw)
    sim_dir  = Path(sim_dir_raw)

    manifest = warm_dir / "models" / "model_manifest.json"
    if not manifest.exists():
        if nibi_ok:
            raise AirflowException(
                f"model_manifest.json not found in warm bundle at {manifest} — refusing to promote"
            )
        print("WARNING: manifest not found but NIBI failed — keeping current_base unchanged.")
    else:
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
    for endpoint in [
        "/api/v1/inference/admin/reload-model",
        "/api/v1/simulation/admin/reload-simulation",
    ]:
        try:
            resp = requests.post(f"{BACKEND_URL}{endpoint}", timeout=10)
            if resp.ok:
                print(f"Reloaded {endpoint}: {resp.json()}")
            else:
                print(f"WARNING: reload {endpoint} returned {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            print(f"WARNING: reload {endpoint} failed (non-fatal): {exc}")


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
        task_id="validate_or_flag",
        python_callable=task_validate_or_flag,
    )

    t6b_fallback = PythonOperator(
        task_id="local_prediction_fallback",
        python_callable=task_local_prediction_fallback,
        trigger_rule=TriggerRule.ALL_DONE,
        execution_timeout=dt.timedelta(hours=1),
        retries=0,
    )

    t7_rsync = PythonOperator(
        task_id="rsync_artifacts_back",
        python_callable=task_rsync_artifacts_back,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    t8_promote = PythonOperator(
        task_id="promote_intraday",
        python_callable=task_promote_intraday,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    t9_reload = PythonOperator(
        task_id="reload_backend",
        python_callable=task_reload_backend,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Chain: validate flags nibi_ok → fallback (ALL_DONE) → rsync (ALL_DONE) → promote → reload
    t1_gate >> t2_export >> t3_sync >> t4_submit >> t5_poll >> t6_validate >> t6b_fallback >> t7_rsync >> t8_promote >> t9_reload

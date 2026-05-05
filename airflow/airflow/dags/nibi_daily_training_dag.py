"""
nibi_daily_training_dag.py
==========================

Daily NIBI warm-refresh pipeline.  Runs Mon–Fri at 10:00 UTC (06:00 ET),
after morning_login.sh has opened the SSH ControlMaster.

Full pipeline:
  1. ssh_health_check     — verify NIBI is reachable and Slurm is up
  2. export_parquet       — dump ml.market_data_15m → local parquet snapshot
  3. sync_code_to_nibi    — rsync ml/ code tree → NIBI test_simulation/ml/
  4. sync_parquet_to_nibi — SCP parquet → NIBI (skips if remote size matches)
  5. sync_base_model      — rsync base model → NIBI run_root/current/
  6. submit_slurm_job     — sbatch simulate_full_day.sbatch --skip-base
  7. poll_job_sensor      — Airflow Sensor: pokes squeue/sacct every 2 min
                            returns True when COMPLETED, raises on FAILED/CANCELLED
  8. validate_artifacts   — check SIMULATION_DONE sentinel + all 26 step_XX/ dirs
  9. rsync_artifacts_back — pull run_root/ → local model_artifacts/sim_<date>/
 10. promote_model        — atomic symlink swap  current_base → new bundle
 11. reload_backend       — POST /api/v1/admin/reload-model

──────────────────────────────────────────────────────────────────────────────
CONCEPTS EXPLAINED (read top-to-bottom, each section teaches something)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator

# ── Config ─────────────────────────────────────────────────────────────────
NIBI_ALIAS    = "nibi"                          # ~/.ssh/config Host alias
NIBI_USER     = os.getenv("NIBI_USER",    "harshsaw")
NIBI_SIM_DIR  = os.getenv("NIBI_SIM_DIR", "/home/harshsaw/projects/def-youry/test_simulation")

# Main application database — same instance the collector and backend use.
# Set POSTGRES_* in the Airflow environment (or Docker .env) to match docker-compose.
DB_HOST = os.getenv("POSTGRES_SERVER", os.getenv("OLD_DB_HOST",  "localhost"))
DB_PORT = int(os.getenv("POSTGRES_PORT",   os.getenv("OLD_DB_PORT", "5432")))
DB_NAME = os.getenv("POSTGRES_DB",     os.getenv("OLD_DB_NAME",  "market_data"))
DB_USER = os.getenv("POSTGRES_USER",   os.getenv("OLD_DB_USER",  "mluser"))
DB_PASS = os.getenv("POSTGRES_PASSWORD", os.getenv("OLD_DB_PASSWORD", ""))

REPO_ROOT      = Path(__file__).resolve().parents[3]
DATASETS_DIR   = REPO_ROOT / "datasets"
ARTIFACTS_DIR  = REPO_ROOT / "model_artifacts"
ML_SRC         = REPO_ROOT / "ml" / "ml"
COLLECTOR_SRC = REPO_ROOT / "services" / "collector" / "src"
PIPELINE_PYTHON_PATH = REPO_ROOT / "pipeline-venv" / "bin" / "python"
if PIPELINE_PYTHON_PATH.exists():
    # If it exists but is a broken symlink (common in Docker), subprocess.run will fail
    try:
        subprocess.run([str(PIPELINE_PYTHON_PATH), "--version"], capture_output=True, check=True)
        PIPELINE_PYTHON = PIPELINE_PYTHON_PATH
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
        PIPELINE_PYTHON = Path(sys.executable)
else:
    PIPELINE_PYTHON = Path(sys.executable)

NIBI_DATA_DIR = f"{NIBI_SIM_DIR}/data"

NIBI_RUN_ROOT   = f"{NIBI_SIM_DIR}/run_root"
NIBI_SBATCH     = f"{NIBI_SIM_DIR}/ml/ml/nibi/simulate_full_day.sbatch"

# BASE_MODEL_DIR: resolved at task runtime via Airflow Variable "nibi_base_model_dir".
# Falls back to env var BASE_MODEL_DIR, then to the current_base symlink so the
# pipeline is self-referential after the first successful run.
# Override via UI: Admin → Variables → nibi_base_model_dir = /path/to/explicit/bundle
_BASE_MODEL_DIR_DEFAULT = os.getenv(
    "BASE_MODEL_DIR",
    str(ARTIFACTS_DIR / "current_base"),
)
BASE_MODEL_DIR  = Path(_BASE_MODEL_DIR_DEFAULT)  # may be overridden at runtime — see task_sync_base_model

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

NIBI_HOST        = os.getenv("NIBI_HOST", "nibi.sharcnet.ca")
NIBI_KEY         = Path(os.getenv("NIBI_SSH_KEY", str(Path.home() / ".ssh" / "nibi_key")))
NIBI_SOCKET_DIR  = Path(os.getenv("NIBI_SOCKET_DIR", str(Path.home() / ".ssh" / "cm")))
NIBI_SOCKET_PATH = str(NIBI_SOCKET_DIR / f"nibi-{NIBI_USER}@{NIBI_HOST}:22")
NIBI_VENV        = os.getenv("NIBI_VENV", f"/home/{NIBI_USER}/ENV")

# Libraries that must be importable in NIBI venv before job submission.
# Update this list if new dependencies are added to the ML scripts.
REQUIRED_LIBS = [
    "xgboost", "seaborn", "pandas", "numpy",
    "sklearn", "matplotlib", "scipy", "joblib",
]

TRAINING_COVERAGE_LOOKBACK_TRADING_DAYS = int(os.getenv("TRAINING_COVERAGE_LOOKBACK_TRADING_DAYS", "5"))
TRAINING_COVERAGE_REFERENCE_CALENDAR_DAYS = int(os.getenv("TRAINING_COVERAGE_REFERENCE_CALENDAR_DAYS", "45"))
TRAINING_COVERAGE_MIN_SYMBOL_RATIO = float(os.getenv("TRAINING_COVERAGE_MIN_SYMBOL_RATIO", "0.98"))
TRAINING_RTH_BAR_COUNT = 26
TRAINING_BACKFILL_OPEN_ET = dt.time(4, 0)
TRAINING_BACKFILL_CLOSE_ET = dt.time(20, 0)
TRAINING_MARKET_TZ = ZoneInfo("America/New_York")

# Usage meter — append one JSON record per run to track GPU allocation hours.
USAGE_LOG = REPO_ROOT / "logs" / "nibi_usage_meter.jsonl"


# ── Usage meter ─────────────────────────────────────────────────────────────
def _record_usage(record: dict) -> None:
    """Append a usage record to the JSONL meter log (one JSON object per line)."""
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ── Shared SSH helper ───────────────────────────────────────────────────────
def _ssh(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """
    Run a command on NIBI via the existing ControlMaster socket.

    NIBI enforces two-factor auth (SSH key + Duo), so we cannot authenticate
    headlessly from scratch.  Instead we reuse the ControlMaster socket that
    was established when the user last approved a Duo push.  The socket is
    kept alive by a cron keepalive every 20 min (ControlPersist=24h).

    BatchMode=yes + ControlMaster=no means: use the socket if it exists,
    fail immediately (rc=255) if it doesn't — never hang waiting for input.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — SSH Health Check
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Fail-Fast Gate
#
# Every pipeline should start with the cheapest possible check that proves
# the environment is ready.  If NIBI is unreachable or Slurm is down, we want
# to know in 5 seconds — not after wasting 3 minutes exporting 427 MB of parquet
# only to fail at the SCP step.
#
# EDGE CASES HANDLED:
#   • SSH ControlMaster expired (MFA session timed out overnight)
#     → BatchMode=yes makes ssh return rc=255 immediately instead of hanging
#     → Clear error message tells operator to re-run morning_login.sh
#   • NIBI node is up but Slurm scheduler is down for maintenance
#     → We check both "echo pong" (SSH works) AND "sbatch --version" (Slurm works)
#     → Failing here prevents a job submission that would silently never run
# ══════════════════════════════════════════════════════════════════════════════
def task_ssh_health_check(**ctx):
    rc, out, err = _ssh("echo pong && sbatch --version | head -1", timeout=20)
    if rc != 0:
        raise AirflowException(
            f"NIBI SSH failed (rc={rc}): {err}\n"
            f"Check that the SSH key {NIBI_KEY} is registered in CCDB "
            "(Alliance Canada account → SSH Keys)."
        )
    print(f"NIBI OK: {out.replace(chr(10), ' | ')}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1b — Check NIBI Libraries
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs in parallel with the other post-health-check tasks.
# Activates the NIBI venv and tries to import every lib in REQUIRED_LIBS.
# Fails immediately with a clear list of what's missing — before we waste an
# hour waiting for a GPU slot only to crash in the first 6 seconds.
# ══════════════════════════════════════════════════════════════════════════════
def task_check_libraries(**ctx):
    check_cmds = " && ".join(
        f'python -c "import {lib}" 2>/dev/null || echo "MISSING:{lib}"'
        for lib in REQUIRED_LIBS
    )
    rc, out, err = _ssh(
        f"source {NIBI_VENV}/bin/activate && {check_cmds}",
        timeout=60,
    )
    if rc != 0:
        raise AirflowException(f"Library check SSH failed (rc={rc}): {err}")

    missing = [line.split("MISSING:")[1] for line in out.splitlines() if line.startswith("MISSING:")]
    if missing:
        raise AirflowException(
            f"Missing libraries in NIBI venv ({NIBI_VENV}):\n"
            + "\n".join(f"  pip install {lib}" for lib in missing)
            + f"\n\nFix: ssh nibi && source {NIBI_VENV}/bin/activate && pip install {' '.join(missing)}"
        )
    print(f"All {len(REQUIRED_LIBS)} required libraries present in {NIBI_VENV}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1c — Clean NIBI run_root
# ══════════════════════════════════════════════════════════════════════════════
#
# Wipes NIBI_RUN_ROOT before submitting so stale artifacts from a previous
# partial run don't confuse validate_artifacts (which checks for SIMULATION_DONE
# and 26 step_XX/ dirs — leftovers from a failed prior run would give false pass).
#
# Runs AFTER all sync tasks complete (data/code/model already on NIBI) but
# BEFORE job submission so we don't delete a live run's output.
# ══════════════════════════════════════════════════════════════════════════════
def task_clean_nibi_run_root(**ctx):
    # Safety: refuse if any algo_sim job is RUNNING or COMPLETING
    rc, running, _ = _ssh(
        f"squeue -u {NIBI_USER} -h -o '%j %T' 2>/dev/null | grep algo_sim | grep -E 'RUNNING|COMPLETING' || true",
        timeout=15,
    )
    if running.strip():
        raise AirflowException(
            f"A simulation job is RUNNING — refusing to clean run_root:\n{running.strip()}"
        )

    rc, _, err = _ssh(
        f"rm -rf {NIBI_RUN_ROOT} && mkdir -p {NIBI_RUN_ROOT}",
        timeout=60,
    )
    if rc != 0:
        raise AirflowException(f"Failed to clean run_root (rc={rc}): {err}")
    print(f"Cleaned: {NIBI_RUN_ROOT}")


def _expected_recent_trading_dates(end_date: dt.date, limit: int) -> list[dt.date]:
    import exchange_calendars as xcals

    xnys = xcals.get_calendar("XNYS")
    schedule = xnys.schedule.loc[: str(end_date)].tail(limit)
    return [ts.date() for ts in schedule.index]


def _load_training_coverage(conn, expected_dates: list[dt.date]) -> tuple[int, list[str], list[dict[str, int | dt.date]]]:
    if not expected_dates:
        return 0, [], []

    reference_start = expected_dates[0] - dt.timedelta(days=TRAINING_COVERAGE_REFERENCE_CALENDAR_DAYS)
    reference_end = expected_dates[-1]

    regular_hours_filter = """
        (window_ts AT TIME ZONE 'America/New_York')::time >= TIME '09:30'
        AND (window_ts AT TIME ZONE 'America/New_York')::time < TIME '16:00'
    """

    with conn.cursor() as cur:
        cur.execute(
            f"""
                WITH counts AS (
                    SELECT trade_date,
                           symbol,
                           COUNT(*) FILTER (WHERE {regular_hours_filter}) AS bars_rth
                    FROM ml.market_data_15m
                    WHERE trade_date BETWEEN %s AND %s
                    GROUP BY 1,2
                ),
                day_totals AS (
                    SELECT trade_date,
                           COUNT(*) FILTER (WHERE bars_rth >= %s) AS complete_symbols
                    FROM counts
                    GROUP BY 1
                )
                SELECT COALESCE(MAX(complete_symbols), 0)
                FROM day_totals
            """,
            (reference_start, reference_end, TRAINING_RTH_BAR_COUNT),
        )
        reference_symbol_count = int(cur.fetchone()[0] or 0)

        cur.execute(
            f"""
                WITH counts AS (
                    SELECT trade_date,
                           symbol,
                           COUNT(*) FILTER (WHERE {regular_hours_filter}) AS bars_rth
                    FROM ml.market_data_15m
                    WHERE trade_date BETWEEN %s AND %s
                    GROUP BY 1,2
                )
                SELECT DISTINCT symbol
                FROM counts
                WHERE bars_rth >= %s
                ORDER BY symbol
            """,
            (reference_start, reference_end, TRAINING_RTH_BAR_COUNT),
        )
        reference_symbols = [row[0] for row in cur.fetchall()]

        if not reference_symbols:
            cur.execute("SELECT symbol FROM market.stocks WHERE is_active = true ORDER BY symbol")
            reference_symbols = [row[0] for row in cur.fetchall()]

        cur.execute(
            f"""
                WITH expected_days AS (
                    SELECT unnest(%s::date[]) AS trade_date
                ),
                counts AS (
                    SELECT trade_date,
                           symbol,
                           COUNT(*) FILTER (WHERE {regular_hours_filter}) AS bars_rth
                    FROM ml.market_data_15m
                    WHERE trade_date = ANY(%s::date[])
                    GROUP BY 1,2
                )
                SELECT d.trade_date,
                       COALESCE(COUNT(c.symbol) FILTER (WHERE c.bars_rth > 0), 0) AS present_symbols,
                       COALESCE(COUNT(c.symbol) FILTER (WHERE c.bars_rth >= %s), 0) AS complete_symbols
                FROM expected_days d
                LEFT JOIN counts c
                  ON c.trade_date = d.trade_date
                GROUP BY d.trade_date
                ORDER BY d.trade_date
            """,
            (expected_dates, expected_dates, TRAINING_RTH_BAR_COUNT),
        )
        coverage_rows = [
            {
                "trade_date": row[0],
                "present_symbols": int(row[1] or 0),
                "complete_symbols": int(row[2] or 0),
            }
            for row in cur.fetchall()
        ]

    return reference_symbol_count, reference_symbols, coverage_rows


def _iter_backfill_slots_utc(trade_dates: list[dt.date]):
    for trade_date in sorted(set(trade_dates)):
        slot = dt.datetime.combine(
            trade_date,
            TRAINING_BACKFILL_OPEN_ET,
            tzinfo=TRAINING_MARKET_TZ,
        ).astimezone(dt.timezone.utc)
        close_utc = dt.datetime.combine(
            trade_date,
            TRAINING_BACKFILL_CLOSE_ET,
            tzinfo=TRAINING_MARKET_TZ,
        ).astimezone(dt.timezone.utc)
        while slot < close_utc:
            yield trade_date, slot
            slot += dt.timedelta(minutes=15)


def _collector_import_path() -> None:
    collector_src = str(COLLECTOR_SRC)
    if collector_src not in sys.path:
        sys.path.insert(0, collector_src)


def _export_staging_backfill_to_core() -> dict[str, int]:
    _collector_import_path()

    from model.orm_db import get_engine as collector_get_engine, get_session_factory
    from utils.scheduled_pipeline import export_staging_to_core

    engine = collector_get_engine(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS)
    try:
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            summary = export_staging_to_core(session)
            session.commit()
        return {
            "processed_rows": int(summary.processed_rows),
            "exported_rows": int(summary.exported_rows),
            "duplicate_rows": int(summary.duplicate_rows),
            "quality_error_rows": int(summary.quality_error_rows),
        }
    finally:
        engine.dispose()


def task_ensure_training_db_coverage(**ctx):
    import psycopg2
    from sqlalchemy import create_engine, text

    sim_date = dt.date.fromisoformat(ctx["ds"])
    
    # Only check coverage for dates strictly before today (local time) to avoid
    # gather_past_data.py restrictions, which refuses to fetch data for "today".
    # Today's data should be handled by the intraday pipeline.
    tz = ZoneInfo("America/New_York")
    now_local = dt.datetime.now(tz).date()
    end_check_date = min(sim_date, now_local - dt.timedelta(days=1))
    
    expected_dates = _expected_recent_trading_dates(end_check_date, TRAINING_COVERAGE_LOOKBACK_TRADING_DAYS)
    if not expected_dates:
        print(f"No expected trading dates found before {now_local} (sim_date={sim_date}). Skipping coverage check.")
        ctx["ti"].xcom_push(key="backfill_performed", value=False)
        ctx["ti"].xcom_push(key="gap_dates", value=[])
        return

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        connect_timeout=10,
    )
    try:
        reference_symbol_count, reference_symbols, coverage_rows = _load_training_coverage(conn, expected_dates)
    finally:
        conn.close()

    if reference_symbol_count <= 0:
        raise AirflowException("Could not determine a reference symbol universe from ml.market_data_15m")

    required_symbols = math.ceil(reference_symbol_count * TRAINING_COVERAGE_MIN_SYMBOL_RATIO)
    gap_dates: list[dt.date] = []

    print(
        f"Training DB coverage check: expected_dates={expected_dates[0]}..{expected_dates[-1]} "
        f"reference_symbols={reference_symbol_count} required_complete>={required_symbols}"
    )
    for row in coverage_rows:
        trade_date = row["trade_date"]
        present_symbols = int(row["present_symbols"])
        complete_symbols = int(row["complete_symbols"])
        print(
            f"  {trade_date}: present_symbols={present_symbols} "
            f"complete_regular_session_symbols={complete_symbols}"
        )
        if complete_symbols < required_symbols:
            gap_dates.append(trade_date)

    if not gap_dates:
        print("Training DB coverage OK — no backfill required.")
        ctx["ti"].xcom_push(key="backfill_performed", value=False)
        ctx["ti"].xcom_push(key="gap_dates", value=[])
        return

    if not reference_symbols:
        raise AirflowException("Coverage preflight found gaps but could not resolve a reference symbol list.")

    gap_dates = sorted(set(gap_dates))
    print(f"Coverage gaps detected for trade dates: {', '.join(d.isoformat() for d in gap_dates)}")

    fetch_env = os.environ.copy()
    fetch_env["PYTHONPATH"] = str(COLLECTOR_SRC)
    fetch_env.setdefault("DB_HOST", DB_HOST)
    fetch_env.setdefault("DB_PORT", str(DB_PORT))
    fetch_env.setdefault("DB_NAME", DB_NAME)
    fetch_env.setdefault("DB_USER", DB_USER)
    fetch_env.setdefault("DB_PASSWORD", DB_PASS)

    fetch_cmd = [
        str(PIPELINE_PYTHON),
        str(COLLECTOR_SRC / "gather_past_data.py"),
        "--from-date",
        gap_dates[0].isoformat(),
        "--to-date",
        gap_dates[-1].isoformat(),
        "--symbols",
        ",".join(reference_symbols),
    ]
    print(
        f"Running historical backfill for {len(reference_symbols)} symbols "
        f"from {gap_dates[0]} to {gap_dates[-1]} ..."
    )
    fetch_proc = subprocess.run(
        fetch_cmd,
        cwd=str(COLLECTOR_SRC),
        env=fetch_env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if fetch_proc.returncode != 0:
        raise AirflowException(
            "Historical gap backfill failed.\n"
            f"stdout:\n{fetch_proc.stdout[-4000:]}\n"
            f"stderr:\n{fetch_proc.stderr[-4000:]}"
        )
    print(fetch_proc.stdout[-2000:] or "Historical backfill completed.")

    export_summary = _export_staging_backfill_to_core()
    print(
        "Backfill export summary: "
        f"processed={export_summary['processed_rows']} "
        f"exported={export_summary['exported_rows']} "
        f"duplicates={export_summary['duplicate_rows']} "
        f"quality_errors={export_summary['quality_error_rows']}"
    )

    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    try:
        slot_count = 0
        with engine.begin() as sql_conn:
            for trade_date, slot in _iter_backfill_slots_utc(gap_dates):
                sql_conn.execute(text("CALL dw.process_15min_window(:window_ts)"), {"window_ts": slot})
                slot_count += 1
        print(f"Re-aggregated {slot_count} 15-minute windows across {len(gap_dates)} trade dates.")
    finally:
        engine.dispose()

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        connect_timeout=10,
    )
    try:
        refreshed_reference_count, _, refreshed_rows = _load_training_coverage(conn, expected_dates)
    finally:
        conn.close()

    refreshed_required = math.ceil(refreshed_reference_count * TRAINING_COVERAGE_MIN_SYMBOL_RATIO)
    remaining_gaps = [
        row["trade_date"]
        for row in refreshed_rows
        if int(row["complete_symbols"]) < refreshed_required
    ]
    if remaining_gaps:
        raise AirflowException(
            "Training DB coverage is still incomplete after backfill for trade dates: "
            + ", ".join(d.isoformat() for d in remaining_gaps)
        )

    print("Training DB coverage repaired successfully.")
    ctx["ti"].xcom_push(key="backfill_performed", value=True)
    ctx["ti"].xcom_push(key="gap_dates", value=[d.isoformat() for d in gap_dates])


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Export Parquet
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Idempotency
#
# A task is idempotent if running it twice produces the same result as running
# it once.  This matters because Airflow can re-run tasks on retry or manual
# re-trigger.  If we blindly re-export 427 MB every time, we waste 3 minutes
# and risk partial writes.
#
# Pattern: "check before doing" — if the expected output already exists and
# looks correct (right size), skip the work and move on.
#
# EDGE CASES HANDLED:
#   • Re-run / retry: parquet already exists → skip DB query (saves 3 min)
#   • DB is slow: chunked read with progress logging so the task doesn't
#     look frozen to Airflow's heartbeat monitor
#   • Partial write from a previous crash: we write to a .tmp file first,
#     then rename atomically — so a partial file never looks complete
# ══════════════════════════════════════════════════════════════════════════════
def task_export_parquet(**ctx):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    sim_date = ctx["ds"]                            # Airflow injects "2026-04-13"
    out_path = DATASETS_DIR / f"snapshot_{sim_date}.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    backfill_performed = bool(
        ctx["ti"].xcom_pull(task_ids="ensure_training_db_coverage", key="backfill_performed")
    )

    if out_path.exists():
        if backfill_performed:
            print("DB backfill repaired training data — rebuilding existing parquet snapshot.")
            out_path.unlink()
            if tmp_path.exists():
                tmp_path.unlink()
        else:
            mb = out_path.stat().st_size / 1e6
            print(f"Parquet already exists ({mb:.1f} MB) — skipping export")
            ctx["ti"].xcom_push(key="parquet_path", value=str(out_path))
            return

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    chunks = []
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM ml.market_data_15m")).scalar()
        print(f"Total rows: {total:,}")
        for chunk in pd.read_sql(
            text("SELECT * FROM ml.market_data_15m ORDER BY symbol, window_ts"),
            conn, chunksize=200_000,
        ):
            chunks.append(chunk)
            loaded = sum(len(c) for c in chunks)
            print(f"  Loaded {loaded:,} / {total:,} rows ...")
    engine.dispose()

    df = pd.concat(chunks, ignore_index=True)
    # Write to .tmp first, then rename — guarantees no partial file
    pq.write_table(pa.Table.from_pandas(df), tmp_path)
    tmp_path.rename(out_path)

    mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path}  ({mb:.1f} MB, {df['symbol'].nunique()} symbols)")
    ctx["ti"].xcom_push(key="parquet_path", value=str(out_path))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Sync ML Code to NIBI
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Code/Data Co-location
#
# The training script runs *on NIBI*, so the latest version of the code must
# be there.  We rsync (not scp) because rsync only transfers changed files —
# if nothing changed since yesterday, this finishes in under 1 second.
#
# EDGE CASES HANDLED:
#   • Remote directory doesn't exist yet (first run)
#     → mkdir -p before rsync, or rsync --mkpath (rsync 3.2+)
#   • Code changed but parquet didn't (common case)
#     → rsync detects by mtime+size, only transfers deltas
#   • Rsync partial failure mid-transfer
#     → rsync is restartable by design; the next task will re-attempt on retry
# ══════════════════════════════════════════════════════════════════════════════
def task_sync_code(**ctx):
    _ssh(f"mkdir -p {NIBI_SIM_DIR}/ml/ml")
    r = subprocess.run(
        ["rsync", "-az", "--delete",
         "-e", f"ssh -i {NIBI_KEY} -o ControlPath={NIBI_SOCKET_PATH} -o ControlMaster=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
         str(ML_SRC) + "/",
         f"{NIBI_USER}@{NIBI_HOST}:{NIBI_SIM_DIR}/ml/ml/"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise AirflowException(f"rsync ml code failed:\n{r.stderr}")
    print(f"ML code synced to NIBI")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — SCP Parquet to NIBI
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Bandwidth-Aware Idempotency
#
# The parquet is 427 MB.  SCP-ing it every day wastes ~30 seconds and burns
# Alliance Canada scratch quota.  We compare local vs remote file size first.
# If they match, skip.  (MD5 would be more correct but adds another SSH round-
# trip and 5 seconds of hashing — size is good enough for our use case.)
#
# EDGE CASES HANDLED:
#   • Remote file missing (first run, or NIBI scratch purged after 60 days)
#     → stat returns 0, we send unconditionally
#   • Remote file partially written (previous SCP crashed mid-transfer)
#     → sizes won't match → re-send
#   • Parquet path comes from XCom (Task 2 pushed it) — if Task 2 was skipped
#     on retry, XCom still has the value from the previous run
# ══════════════════════════════════════════════════════════════════════════════
def task_sync_parquet(**ctx):
    parquet_path = Path(ctx["ti"].xcom_pull(task_ids="export_parquet", key="parquet_path"))
    sim_date     = ctx["ds"]
    remote_path  = f"{NIBI_DATA_DIR}/snapshot_{sim_date}.parquet"

    _ssh(f"mkdir -p {NIBI_DATA_DIR}")

    local_size = parquet_path.stat().st_size
    rc, remote_size_str, _ = _ssh(f"stat -c%s {remote_path} 2>/dev/null || echo 0")
    if rc == 0 and remote_size_str.isdigit() and int(remote_size_str) == local_size:
        print(f"Remote parquet matches local ({local_size/1e6:.1f} MB) — skipping SCP")
        ctx["ti"].xcom_push(key="remote_parquet", value=remote_path)
        return

    print(f"Sending {local_size/1e6:.1f} MB ...")
    r = subprocess.run(
        ["scp", "-i", str(NIBI_KEY),
         "-o", f"ControlPath={NIBI_SOCKET_PATH}", "-o", "ControlMaster=no",
         "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         str(parquet_path), f"{NIBI_USER}@{NIBI_HOST}:{remote_path}"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise AirflowException(f"SCP failed:\n{r.stderr}")
    print(f"Parquet on NIBI: {remote_path}")
    ctx["ti"].xcom_push(key="remote_parquet", value=remote_path)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Sync Base Model to NIBI
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Pre-condition Validation
#
# Before we touch NIBI, verify the base model exists locally.  If it's missing
# (e.g. someone deleted it, or it was never trained), fail with a clear message
# rather than submitting a job that will crash on NIBI 30 minutes later with a
# cryptic "file not found" in a log we can't easily see.
#
# EDGE CASES HANDLED:
#   • Base model missing locally → fail immediately with actionable error
#   • Base model already on NIBI from yesterday → rsync transfers 0 bytes
#   • metadata.json missing from base model dir → warn but don't block
#     (metadata is informational, not required for training)
#   • A manually submitted job is currently RUNNING on NIBI and reading from
#     run_root/current/ mid-warm-refresh → writing to current/ would corrupt it.
#     We check squeue first and refuse to overwrite if any algo_sim job is active.
# ══════════════════════════════════════════════════════════════════════════════
def task_sync_base_model(**ctx):
    from airflow.models import Variable
    skip_base = Variable.get("nibi_skip_base", default_var="false").lower() == "true"
    if not skip_base:
        print("nibi_skip_base=false — base training is enabled, so sync_base_model is not required.")
        return

    base_dir = Path(Variable.get("nibi_base_model_dir", default_var=_BASE_MODEL_DIR_DEFAULT))

    if not base_dir.exists():
        raise AirflowException(
            f"Base model not found at {base_dir}\n"
            "Set Airflow Variable 'nibi_base_model_dir' or BASE_MODEL_DIR env var."
        )
    # Rebind module-level for the rest of this task
    global BASE_MODEL_DIR
    BASE_MODEL_DIR = base_dir

    # Safety guard: do not overwrite run_root/current/ if a simulation job
    # is actively running — it reads current/ during warm refresh.
    # This protects against manually submitted jobs overlapping with the DAG.
    rc, running_jobs, _ = _ssh(
        f"squeue -u {NIBI_USER} -h -o '%j %T' 2>/dev/null | grep algo_sim | grep -E 'RUNNING|COMPLETING' || true",
        timeout=15,
    )
    if running_jobs.strip():
        raise AirflowException(
            f"A simulation job is currently RUNNING on NIBI — refusing to overwrite run_root/current/:\n"
            f"  {running_jobs.strip()}\n"
            "Wait for it to finish or cancel it before re-triggering this DAG."
        )

    meta_path = BASE_MODEL_DIR / "metadata.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        print(f"Base model: {m.get('n_estimators','?')} trees, "
              f"cutoff={m.get('train_end_date','?')}")
    else:
        print("WARNING: metadata.json missing — continuing anyway")

    _ssh(f"mkdir -p {NIBI_RUN_ROOT}/current")
    r = subprocess.run(
        ["rsync", "-az",
         "-e", f"ssh -i {NIBI_KEY} -o ControlPath={NIBI_SOCKET_PATH} -o ControlMaster=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
         str(BASE_MODEL_DIR) + "/",
         f"{NIBI_USER}@{NIBI_HOST}:{NIBI_RUN_ROOT}/current/"],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        raise AirflowException(f"rsync base model failed:\n{r.stderr}")

    rc, count, _ = _ssh(f"find {NIBI_RUN_ROOT}/current -name '*.json' | wc -l")
    print(f"Base model on NIBI — {count.strip()} json files in current/")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 — Submit Slurm Job
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: XCom (Cross-Communication)
#
# Airflow tasks are isolated functions — they can't share variables directly.
# XCom is Airflow's key-value store for passing small values between tasks.
# Here we push the job_id so the sensor (Task 7) can pull it and poll.
#
# CONCEPT: Slurm Job Submission
#
# sbatch is non-blocking — it returns the job ID immediately and the job sits
# in the queue until a GPU node is free.  We never know how long the queue is.
# Submitting and waiting are two separate concerns — hence two separate tasks.
#
# EDGE CASES HANDLED:
#   • sbatch returns no job ID in output (malformed response)
#     → parse explicitly, raise if no numeric token found
#   • Job already submitted today (manual re-trigger after a DAG failure
#     at a later task) — we record the job ID in a JSON file so we can
#     detect and skip re-submission
#   • sbatch itself fails (quota exceeded, bad sbatch script syntax)
#     → rc != 0, stderr surfaced in the exception message
# ══════════════════════════════════════════════════════════════════════════════
def task_submit_job(**ctx):
    sim_date     = ctx["ds"]
    remote_parq  = ctx["ti"].xcom_pull(task_ids="sync_parquet_to_nibi", key="remote_parquet")
    job_record   = REPO_ROOT / "logs" / f"nibi_job_{sim_date}.json"

    # If a job was already submitted and is still in-flight (status != "completed"),
    # reuse the existing job_id rather than submitting a second job.
    # Do NOT reuse a completed job: clean_nibi_run_root already wiped run_root,
    # so reusing a completed ID would make the sensor return immediately and
    # leave run_root empty, causing validate_artifacts to fail.
    if job_record.exists():
        rec = json.loads(job_record.read_text())
        existing_id = rec.get("job_id")
        if existing_id and rec.get("status") != "completed":
            print(f"Job already submitted today: {existing_id} — reusing (delete "
                  f"{job_record} to force re-submit)")
            ctx["ti"].xcom_push(key="job_id", value=existing_id)
            return

    # skip_base=True  → reuse existing base model (fast, ~9 min)
    # skip_base=False → train new base model first (slower, but keeps current_base fresh)
    # Controlled by Airflow Variable "nibi_skip_base" (default False).
    # Override via UI: Admin → Variables → nibi_skip_base = true
    from airflow.models import Variable
    skip_base = Variable.get("nibi_skip_base", default_var="false").lower() == "true"

    cmd = (
        f"sbatch {NIBI_SBATCH} "
        f"--parquet {remote_parq} "
        f"--sim-date {sim_date} "
        + ("--skip-base" if skip_base else "")
    )
    print(f"Submitting: {cmd}")
    rc, out, err = _ssh(cmd, timeout=30)
    if rc != 0:
        raise AirflowException(f"sbatch failed (rc={rc}):\n{err}")

    job_id = next((tok for tok in out.split() if tok.isdigit()), None)
    if not job_id:
        raise AirflowException(f"Could not parse job ID from sbatch output: {out!r}")

    print(f"Job submitted: {job_id}")

    submitted_at = dt.datetime.utcnow().isoformat()

    # Persist so re-triggers can detect duplicate submission
    job_record.parent.mkdir(parents=True, exist_ok=True)
    job_record.write_text(json.dumps({
        "job_id": job_id,
        "sim_date": sim_date,
        "submitted_at": submitted_at,
        "status": "submitted",
    }, indent=2))

    # Usage meter: record submission event
    _record_usage({
        "event": "submitted",
        "job_id": job_id,
        "sim_date": sim_date,
        "submitted_at": submitted_at,
        "gpu_type": "h100",
        "account": "def-youry",
    })

    ctx["ti"].xcom_push(key="job_id", value=job_id)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 — Poll Job (Sensor)
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Sensors vs Operators — this is the most important design choice here
#
# The old DRAC DAG used a single SSHOperator that held a bash `while` loop for
# up to 3 hours.  That is fragile for two reasons:
#
#   1. SSH connection drops are common over hours.  One TCP timeout and the
#      operator fails — even if the job is still running fine on the cluster.
#
#   2. One Airflow worker slot is blocked for the entire wait.  If you have
#      multiple DAGs running, you can starve the worker pool.
#
# Airflow Sensors solve this differently:
#   - poke()  is called every poke_interval seconds
#   - each call is SHORT (open SSH, check state, close SSH) — maybe 3 seconds
#   - between pokes, the worker slot is RELEASED (mode="reschedule")
#   - if poke() returns True → done.  False → sleep and try again.
#   - if poke() raises AirflowException → task fails immediately
#
# CONCEPT: mode="reschedule" vs mode="poke"
#   - mode="poke" (default): worker holds the slot between pokes (wastes resources)
#   - mode="reschedule": worker releases the slot, re-acquires only for each poke
#   Use "reschedule" for long waits (hours).  Use "poke" only for sub-minute waits.
#
# CONCEPT: sacct vs squeue
#   - squeue  shows RUNNING/PENDING jobs.  Returns nothing once a job finishes.
#   - sacct   shows historical records — the only way to get COMPLETED/FAILED state
#             after a job exits the queue.
#   Always check both: if squeue is empty, fall through to sacct.
#
# EDGE CASES HANDLED:
#   • Job cancelled by Slurm admin or scheduler (happened to 12114084–12115062)
#     → sacct shows CANCELLED → raise immediately, don't wait out the full timeout
#   • Job failed on startup (happened to 12115066 — missing seaborn)
#     → sacct shows FAILED → raise, surface the .err log content in the message
#   • Job timed out on the cluster (--time=08:00:00 exceeded)
#     → sacct shows TIMEOUT → raise
#   • SSH blip during a poke (transient network issue)
#     → rc != 0 but we don't raise — just log a warning and retry next poke
#     → only raise after 3 consecutive SSH failures (avoids false alarms)
#   • Job ID from XCom is stale (re-triggered DAG, job long gone from sacct)
#     → sacct returns empty string → treat as unknown, keep polling briefly
#     → after timeout, raise with clear message
# ══════════════════════════════════════════════════════════════════════════════
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
FAILED_STATES   = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


class NibiJobSensor(BaseSensorOperator):
    """
    Polls a Slurm job on NIBI until it reaches a terminal state.

    poke_interval : seconds between SSH checks (120 = 2 min)
    timeout       : give up after this many seconds (32400 = 9 hours)
    mode          : "reschedule" releases the worker slot between pokes
    """

    def __init__(self, job_id_task: str, **kwargs):
        super().__init__(
            poke_interval=120,      # check every 2 minutes
            timeout=32_400,         # 9 hours max wait
            mode="reschedule",      # release worker between pokes — critical for long waits
            **kwargs,
        )
        self.job_id_task = job_id_task
        self._ssh_fail_streak = 0   # track consecutive SSH failures

    def poke(self, context) -> bool:
        # Pull sim_date from context — NOT from constructor.
        # Jinja "{{ ds }}" is only rendered for fields listed in template_fields.
        # Getting it from context["ds"] here is the correct pattern for sensors.
        sim_date = context["ds"]
        job_id   = context["ti"].xcom_pull(task_ids=self.job_id_task, key="job_id")
        if not job_id:
            raise AirflowException("No job_id in XCom — submit_job task may have failed.")

        # Step 1: check squeue (fast — only works while job is still queued/running)
        rc, squeue_out, _ = _ssh(
            f"squeue -j {job_id} -h -o '%T' 2>/dev/null || true",
            timeout=20,
        )

        if rc != 0:
            # SSH itself failed — transient network issue
            self._ssh_fail_streak += 1
            print(f"WARNING: SSH failed (streak={self._ssh_fail_streak}) — will retry")
            if self._ssh_fail_streak >= 3:
                raise AirflowException(
                    f"SSH to NIBI failed {self._ssh_fail_streak} times in a row. "
                    f"Check that {NIBI_KEY} is registered in CCDB and NIBI is reachable."
                )
            return False  # retry next poke

        self._ssh_fail_streak = 0  # reset streak on success
        state = squeue_out.strip()

        if state in ("RUNNING", "PENDING", "COMPLETING"):
            elapsed = context["ti"].duration or 0
            print(f"  [{state}] job {job_id} — {int(elapsed/60)}min elapsed")
            return False  # keep waiting

        # Step 2: squeue empty → job left the queue → ask sacct for final state
        _, sacct_out, _ = _ssh(
            f"sacct -j {job_id} --format=State --noheader 2>/dev/null | head -1",
            timeout=20,
        )
        final_state = sacct_out.strip().split()[0] if sacct_out.strip() else ""
        print(f"  squeue empty → sacct state: {final_state!r}")

        if not final_state:
            # sacct has no record yet (race condition — job just finished)
            # Give it one more poke cycle
            print("  sacct has no record yet — will retry next poke")
            return False

        if final_state.startswith("COMPLETED"):
            print(f"Job {job_id} COMPLETED.")
            completed_at = dt.datetime.utcnow().isoformat()

            # Update local job record
            record_path = REPO_ROOT / "logs" / f"nibi_job_{sim_date}.json"
            submitted_at = None
            if record_path.exists():
                rec = json.loads(record_path.read_text())
                rec["status"] = "completed"
                rec["completed_at"] = completed_at
                record_path.write_text(json.dumps(rec, indent=2))
                submitted_at = rec.get("submitted_at")

            # Compute elapsed GPU hours from sacct
            _, elapsed_raw, _ = _ssh(
                f"sacct -j {job_id} --format=Elapsed --noheader -X 2>/dev/null | head -1",
                timeout=20,
            )
            gpu_hours = None
            try:
                parts = elapsed_raw.strip().split(":")
                if len(parts) == 3:
                    h, m, s = parts
                    gpu_hours = round(int(h) + int(m) / 60 + int(s) / 3600, 3)
            except Exception:
                pass

            # Usage meter: record completion event
            _record_usage({
                "event": "completed",
                "job_id": job_id,
                "sim_date": sim_date,
                "submitted_at": submitted_at,
                "completed_at": completed_at,
                "gpu_hours": gpu_hours,
                "gpu_type": "h100",
                "account": "def-youry",
            })
            if gpu_hours is not None:
                print(f"GPU usage recorded: {gpu_hours:.3f} H100-hours")

            return True  # sensor done

        if any(final_state.startswith(s) for s in FAILED_STATES):
            # Pull the .err log from NIBI to surface the crash reason
            _, err_content, _ = _ssh(
                f"tail -30 {NIBI_SIM_DIR}/logs/sim_full_day_{job_id}.err 2>/dev/null || echo '(no err log)'",
                timeout=20,
            )
            _record_usage({
                "event": "failed",
                "job_id": job_id,
                "sim_date": sim_date,
                "failed_at": dt.datetime.utcnow().isoformat(),
                "final_state": final_state,
                "gpu_type": "h100",
                "account": "def-youry",
            })
            raise AirflowException(
                f"Job {job_id} ended with state: {final_state}\n"
                f"--- Last 30 lines of .err log ---\n{err_content}"
            )

        # Unknown state (e.g. "RESIZING") — keep polling
        print(f"  Unknown state {final_state!r} — continuing to poll")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TASK 8 — Validate Artifacts on NIBI
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Post-Condition Validation (Trust But Verify)
#
# The Slurm job reported COMPLETED — but does that mean the *training* succeeded?
# Slurm COMPLETED means the job script exited with code 0.  But the Python script
# inside could have trained 0 windows and written exit(0) after a partial run.
#
# We check two things:
#   1. SIMULATION_DONE sentinel file — written only after ALL windows finish
#   2. All 26 step_XX/ directories exist — proves every window trained
#
# If either check fails, we raise — don't rsync partial results and don't promote.
#
# EDGE CASES HANDLED:
#   • Some windows trained, others hit OOM errors — simulation_progress.json
#     shows per-window status, we surface the failed window numbers
#   • SIMULATION_DONE exists but some step_XX dirs are missing
#     → raise with list of missing steps
#   • SSH fails during validation → retry (task-level Airflow retry)
# ══════════════════════════════════════════════════════════════════════════════
def task_validate_artifacts(**ctx):
    rc, sentinel, _ = _ssh(
        f"cat {NIBI_RUN_ROOT}/SIMULATION_DONE 2>/dev/null || echo MISSING",
        timeout=20,
    )
    if rc != 0:
        raise AirflowException("SSH failed during artifact validation.")
    if sentinel.strip() == "MISSING":
        # Pull progress JSON to show how many windows actually completed
        _, progress_raw, _ = _ssh(
            f"cat {NIBI_RUN_ROOT}/simulation_progress.json 2>/dev/null || echo '{{}}'",
            timeout=20,
        )
        try:
            progress = json.loads(progress_raw)
            steps = progress.get("steps", [])
            ok    = [s["step"] for s in steps if s.get("status") == "ok"]
            errs  = [s for s in steps if s.get("status", "").startswith("error")]
            print(f"Progress: {len(ok)}/26 windows OK, {len(errs)} errors")
            for e in errs:
                print(f"  Window {e['step']} ({e.get('et_label','')}): {e.get('status')}")
        except Exception:
            pass
        raise AirflowException(
            f"SIMULATION_DONE not found at {NIBI_RUN_ROOT}/SIMULATION_DONE\n"
            "The simulation did not complete all 26 windows."
        )

    print(f"SIMULATION_DONE found:\n{sentinel}")

    # Verify all 26 step directories exist — one SSH call, not 26
    rc, out, _ = _ssh(
        f"for i in $(seq -w 0 25); do test -d {NIBI_RUN_ROOT}/step_$i || echo \"missing:step_$i\"; done",
        timeout=20,
    )
    missing = [line.replace("missing:", "") for line in out.splitlines() if line.startswith("missing:")]
    if missing:
        raise AirflowException(
            f"Simulation incomplete — {len(missing)} step dirs missing: {missing}"
        )

    print("All 26 step directories present. Artifacts look good.")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 9 — Rsync Artifacts Back
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Pull vs Push for Model Artifacts
#
# We rsync FROM NIBI TO local, not push from NIBI.  Why pull?
#   - The VPS initiated the SSH connection (authenticated with ControlMaster)
#   - NIBI doesn't have SSH access back to the VPS (firewall, no key setup)
#   - Pull from the VPS side reuses the existing authenticated session
#
# EDGE CASES HANDLED:
#   • run_root/ is large (26 × model snapshots, can be several GB)
#     → rsync with --compress saves bandwidth on slow Alliance Canada links
#   • Previous day's artifacts still in place
#     → we write to a dated subdirectory (sim_YYYY-MM-DD/) not a fixed path
#     → old artifacts are never overwritten, only new dir is created
# ══════════════════════════════════════════════════════════════════════════════
def task_rsync_artifacts_back(**ctx):
    sim_date  = ctx["ds"]
    warm_dest = ARTIFACTS_DIR / f"warm_{sim_date}"
    sim_dest  = ARTIFACTS_DIR / f"simulation_{sim_date}"

    if warm_dest.exists():
        shutil.rmtree(warm_dest)
    if sim_dest.exists():
        shutil.rmtree(sim_dest)

    warm_dest.mkdir(parents=True, exist_ok=True)
    sim_dest.mkdir(parents=True, exist_ok=True)

    SSH_E = (
        f"ssh -i {NIBI_KEY}"
        f" -o ControlPath={NIBI_SOCKET_PATH}"
        f" -o ControlMaster=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    )

    # 1. Pull warm model bundle: run_root/current/ → warm_{date}/
    print(f"Rsyncing {NIBI_RUN_ROOT}/current/ → {warm_dest}/")
    r = subprocess.run(
        ["rsync", "-az", "--compress",
         "-e", SSH_E,
         f"{NIBI_USER}@{NIBI_HOST}:{NIBI_RUN_ROOT}/current/",
         str(warm_dest) + "/"],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise AirflowException(f"rsync warm model failed:\n{r.stderr}")

    # 2. Pull step predictions + metadata → simulation_{date}/step_NN/
    print(f"Rsyncing step predictions → {sim_dest}/")
    for i in range(26):
        step = f"step_{i:02d}"
        step_dest = sim_dest / step
        (step_dest / "predictions").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-az", "--compress", "-e", SSH_E,
             f"{NIBI_USER}@{NIBI_HOST}:{NIBI_RUN_ROOT}/{step}/predictions/",
             str(step_dest / "predictions") + "/"],
            capture_output=True, text=True, timeout=300,
        )
        for fname in ("metadata.json", "feature_names.json"):
            subprocess.run(
                ["rsync", "-az", "-e", SSH_E,
                 f"{NIBI_USER}@{NIBI_HOST}:{NIBI_RUN_ROOT}/{step}/{fname}",
                 str(step_dest / fname)],
                capture_output=True, text=True, timeout=60,
            )

    # 3. Fetch simulation_progress.json → simulation_summary.json
    r2 = subprocess.run(
        ["ssh", "-i", str(NIBI_KEY),
         "-o", f"ControlPath={NIBI_SOCKET_PATH}",
         "-o", "ControlMaster=no", "-o", "BatchMode=yes",
         f"{NIBI_USER}@{NIBI_HOST}",
         f"cat {NIBI_RUN_ROOT}/simulation_progress.json"],
        capture_output=True, text=True, timeout=30,
    )
    if r2.returncode == 0 and r2.stdout.strip():
        (sim_dest / "simulation_summary.json").write_text(r2.stdout)
        print(f"Wrote simulation_summary.json to {sim_dest}")
    else:
        print(f"WARNING: could not fetch simulation_progress.json: {r2.stderr}")

    n_warm_files = len(list(warm_dest.rglob("*")))
    n_steps      = len(list(sim_dest.glob("step_*")))
    print(f"warm bundle: {warm_dest} ({n_warm_files} files)")
    print(f"sim bundle:  {sim_dest}  ({n_steps} step dirs)")

    ctx["ti"].xcom_push(key="warm_artifact_dir",  value=str(warm_dest))
    ctx["ti"].xcom_push(key="sim_artifact_dir",   value=str(sim_dest))
    ctx["ti"].xcom_push(key="local_artifact_dir", value=str(warm_dest))  # compat


# ══════════════════════════════════════════════════════════════════════════════
# TASK 10 — Promote Model (Atomic Symlink Swap)
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Atomic Deployment / Blue-Green for ML Models
#
# The backend reads the model from a symlink: model_artifacts/current_base → ???
# If we update the symlink with: rm current_base && ln -s new_dir current_base
# there's a brief moment where current_base doesn't exist → backend crash.
#
# The atomic pattern uses the OS rename syscall, which is atomic on Linux:
#   1. ln -sfn new_dir  current_base.new    (create new symlink with temp name)
#   2. mv -f current_base.new  current_base  (rename atomically into place)
# Between steps 1 and 2, current_base still points to the old model.
# After step 2, current_base atomically points to the new model.
# The backend is never in a state where the symlink doesn't exist.
#
# EDGE CASES HANDLED:
#   • current_base doesn't exist yet (first ever promotion)
#     → mv still works — creates current_base from current_base.new
#   • Promotion of a partial artifact (validation already caught this in Task 8,
#     but we double-check step count here as a safety net)
#   • Old artifact directory cleanup — we keep last 7 days, remove older ones
# ══════════════════════════════════════════════════════════════════════════════
def _atomic_symlink(symlink: Path, target: Path) -> None:
    """Atomically update a symlink via Linux rename() — uses relative path for Docker compat."""
    tmp = symlink.parent / (symlink.name + ".new")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    # Relative symlink: resolves correctly both on host and inside Docker container
    tmp.symlink_to(target.name)
    tmp.rename(symlink)


def task_promote_model(**ctx):
    warm_dir = Path(ctx["ti"].xcom_pull(task_ids="rsync_artifacts_back", key="warm_artifact_dir"))
    sim_dir  = Path(ctx["ti"].xcom_pull(task_ids="rsync_artifacts_back", key="sim_artifact_dir"))

    # Safety net: warm bundle must have model_manifest.json
    manifest = warm_dir / "models" / "model_manifest.json"
    if not manifest.exists():
        raise AirflowException(
            f"model_manifest.json not found in warm bundle at {manifest} — refusing to promote"
        )

    # Atomically promote current_base → warm_{date}
    _atomic_symlink(ARTIFACTS_DIR / "current_base", warm_dir)
    print(f"Promoted: current_base → {warm_dir}")

    # Also promote current_base_eod — static snapshot used for all-day base prediction
    _atomic_symlink(ARTIFACTS_DIR / "current_base_eod", warm_dir)
    print(f"Promoted: current_base_eod → {warm_dir}")

    # Promote current_simulation → simulation_{date} only if all 26 steps have predictions
    n_with_preds = sum(
        1 for i in range(26)
        if (sim_dir / f"step_{i:02d}" / "predictions" / "predictions.csv").exists()
    )
    if n_with_preds >= 26:
        _atomic_symlink(ARTIFACTS_DIR / "current_simulation", sim_dir)
        print(f"Promoted: current_simulation → {sim_dir}")
    else:
        print(f"WARNING: only {n_with_preds}/26 steps have predictions — skipping current_simulation update")

    # Cleanup: remove warm_* and simulation_* dirs older than 7 days
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=7)
    for pattern, fmt in [("warm_????-??-??", "warm_%Y-%m-%d"), ("simulation_????-??-??", "simulation_%Y-%m-%d")]:
        for old_dir in sorted(ARTIFACTS_DIR.glob(pattern)):
            try:
                dir_date = dt.datetime.strptime(old_dir.name, fmt)
                if dir_date < cutoff and old_dir not in (warm_dir, sim_dir):
                    shutil.rmtree(old_dir)
                    print(f"Cleaned up old artifact: {old_dir.name}")
            except (ValueError, OSError):
                pass


# ══════════════════════════════════════════════════════════════════════════════
# TASK 11 — Reload Backend
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: Non-Fatal Tail Task
#
# The entire value of the pipeline is complete once the model is promoted.
# Reloading the backend is a "nice to have" — if it fails (backend is down,
# being redeployed, etc.), we do NOT want the whole DAG to show as FAILED.
#
# Pattern: catch all exceptions, log as WARNING, don't re-raise.
# The operator still shows SUCCESS in the UI.
#
# EDGE CASES HANDLED:
#   • Backend is down / restarting → timeout → warn, don't fail
#   • Backend returns non-200 → warn with response body → don't fail
#   • Backend doesn't have the reload endpoint yet → warn → don't fail
#   • Backend picks up new model automatically on next request (lazy load)
#     → reload is just an optimization to warm the cache eagerly
# ══════════════════════════════════════════════════════════════════════════════
def task_reload_backend(**ctx):
    for endpoint, label in [
        ("/api/v1/inference/admin/reload-model",        "inference (warm)"),
        ("/api/v1/inference/admin/reload-base-model",   "inference (base EOD)"),
        ("/api/v1/simulation/admin/reload-simulation",  "simulation"),
    ]:
        try:
            resp = requests.post(f"{BACKEND_URL}{endpoint}", timeout=30)
            if resp.status_code == 200:
                print(f"Backend {label} reloaded: {resp.json()}")
            else:
                print(f"WARNING: {label} reload returned HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            # Non-fatal — model is promoted, backend will pick it up on next request
            print(f"WARNING: {label} reload failed (non-fatal): {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT: DAG-level defaults vs task-level overrides
#
# default_args apply to ALL operators unless overridden at the task level.
# retries=1 with retry_delay=10min means: on failure, wait 10 min and try once more.
# This handles transient failures (SSH blip, DB briefly unavailable) automatically.
#
# max_active_runs=1: only one instance of this DAG can run at a time.
# Important for NIBI because we only have one test_simulation/ directory — two
# concurrent runs would overwrite each other's run_root/.
#
# CONCEPT: schedule timing
#   "0 21 * * 1-5"  = 21:00 UTC, Monday–Friday = 17:00 ET
#   Fires 1 hour after market close (16:00 ET), after intraday data settles.
#   morning_login.sh ControlMaster must still be alive (ControlPersist=24h covers this).
#
# CONCEPT: catchup=False
#   If the DAG was paused for 3 days and you re-enable it, Airflow will NOT
#   try to back-fill the 3 missed runs.  We only want to train on today's data.
# ══════════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id="nibi_daily_warm_refresh",
    default_args={
        "owner":          "ml-team",
        "depends_on_past": False,
        "retries":         1,
        "retry_delay":     dt.timedelta(minutes=10),
        "email_on_failure": False,
    },
    description="Daily NIBI warm-refresh: export → sync → train → promote",
    schedule="0 21 * * 1-5",           # 21:00 UTC = 17:00 ET (1h after close), Mon–Fri
    start_date=dt.datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,                  # prevents concurrent run_root/ collisions
    tags=["ml", "training", "nibi"],
) as dag:

    t1_health = PythonOperator(
        task_id="ssh_health_check",
        python_callable=task_ssh_health_check,
        execution_timeout=dt.timedelta(minutes=2),
    )

    t1b_libs = PythonOperator(
        task_id="check_nibi_libraries",
        python_callable=task_check_libraries,
        execution_timeout=dt.timedelta(minutes=3),
        retries=0,          # missing libs won't fix themselves on retry
    )

    t1c_db = PythonOperator(
        task_id="ensure_training_db_coverage",
        python_callable=task_ensure_training_db_coverage,
        execution_timeout=dt.timedelta(minutes=60),
        retries=0,
    )

    t2_export = PythonOperator(
        task_id="export_parquet",
        python_callable=task_export_parquet,
        execution_timeout=dt.timedelta(minutes=15),
    )

    t3_code = PythonOperator(
        task_id="sync_code_to_nibi",
        python_callable=task_sync_code,
        execution_timeout=dt.timedelta(minutes=5),
    )

    t4_parquet = PythonOperator(
        task_id="sync_parquet_to_nibi",
        python_callable=task_sync_parquet,
        execution_timeout=dt.timedelta(minutes=15),
    )

    t5_model = PythonOperator(
        task_id="sync_base_model",
        python_callable=task_sync_base_model,
        execution_timeout=dt.timedelta(minutes=10),
    )

    t5b_clean = PythonOperator(
        task_id="clean_nibi_run_root",
        python_callable=task_clean_nibi_run_root,
        execution_timeout=dt.timedelta(minutes=3),
        retries=1,
    )

    t6_submit = PythonOperator(
        task_id="submit_slurm_job",
        python_callable=task_submit_job,
        execution_timeout=dt.timedelta(minutes=2),
        retries=2,          # sbatch can be flaky; give it 2 extra tries
    )

    t7_poll = NibiJobSensor(
        task_id="poll_job_until_done",
        job_id_task="submit_slurm_job",
        # sim_date is read from context["ds"] inside poke() — not a constructor arg.
        # Jinja templates only render in template_fields; passing "{{ ds }}" here
        # would give the literal string, not the date.
    )

    t8_validate = PythonOperator(
        task_id="validate_artifacts",
        python_callable=task_validate_artifacts,
        execution_timeout=dt.timedelta(minutes=5),
    )

    t9_rsync = PythonOperator(
        task_id="rsync_artifacts_back",
        python_callable=task_rsync_artifacts_back,
        execution_timeout=dt.timedelta(minutes=30),
    )

    t10_promote = PythonOperator(
        task_id="promote_model",
        python_callable=task_promote_model,
        execution_timeout=dt.timedelta(minutes=5),
        retries=0,          # promotion is idempotent but we don't retry automatically
                            # — a human should verify if this fails
    )

    t11_reload = PythonOperator(
        task_id="reload_backend",
        python_callable=task_reload_backend,
        execution_timeout=dt.timedelta(minutes=2),
        retries=0,          # non-fatal, no retry needed
    )

    # ── Dependency chain ─────────────────────────────────────────
    #
    # After health check, DB coverage repair (t1c), code sync (t3), base-model
    # sync (t5), and library checks (t1b) can run in parallel. Export parquet
    # waits for the DB coverage gate so it snapshots repaired data if needed.
    # All must pass before we SCP the parquet (t4).
    # After all syncs land, clean run_root (t5b) then submit (t6).
    #
    #   t1 ──┬── t1c ── t2 ───┐
    #        ├── t3 ──────────┤
    #        ├── t5 ──────────┤
    #        └── t1b (libs) ──┘
    #                          └── t4 ── t5b (clean) ── t6 ── t7 ── t8 ── t9 ── t10 ── t11

    t1_health >> [t1c_db, t3_code, t5_model, t1b_libs]
    t1c_db >> t2_export
    [t2_export, t3_code, t5_model, t1b_libs] >> t4_parquet
    t4_parquet >> t5b_clean >> t6_submit >> t7_poll >> t8_validate >> t9_rsync >> t10_promote >> t11_reload

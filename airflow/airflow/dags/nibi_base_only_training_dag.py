"""
nibi_base_only_training_dag.py
==============================

Base-only NIBI training pipeline.

Use this when the system needs a fresh `current_base` / `current_base_eod`
bundle before market open, without running same-day warm-refresh windows.
The base train uses the latest completed trading session as the data cutoff and
produces the static bundle that intraday warm refresh can build on later.
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

import requests

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator

NIBI_USER = os.getenv("NIBI_USER", "harshsaw")
NIBI_HOST = os.getenv("NIBI_HOST", "nibi.sharcnet.ca")
NIBI_SIM_DIR = os.getenv("NIBI_SIM_DIR", "/home/harshsaw/projects/def-youry/test_simulation")

DB_HOST = os.getenv("POSTGRES_SERVER", os.getenv("OLD_DB_HOST", "localhost"))
DB_PORT = int(os.getenv("POSTGRES_PORT", os.getenv("OLD_DB_PORT", "5432")))
DB_NAME = os.getenv("POSTGRES_DB", os.getenv("OLD_DB_NAME", "market_data"))
DB_USER = os.getenv("POSTGRES_USER", os.getenv("OLD_DB_USER", "mluser"))
DB_PASS = os.getenv("POSTGRES_PASSWORD", os.getenv("OLD_DB_PASSWORD", ""))

REPO_ROOT = Path(os.getenv("REPO_ROOT", str(Path(__file__).resolve().parents[3])))
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

REQUIRED_LIBS = [
    "xgboost", "seaborn", "pandas", "numpy",
    "sklearn", "matplotlib", "scipy", "joblib",
]

TRAINING_COVERAGE_LOOKBACK_TRADING_DAYS = int(os.getenv("TRAINING_COVERAGE_LOOKBACK_TRADING_DAYS", "5"))
TRAINING_COVERAGE_REFERENCE_CALENDAR_DAYS = int(os.getenv("TRAINING_COVERAGE_REFERENCE_CALENDAR_DAYS", "45"))
TRAINING_COVERAGE_MIN_SYMBOL_RATIO = float(os.getenv("TRAINING_COVERAGE_MIN_SYMBOL_RATIO", "0.98"))
TRAINING_RTH_BAR_COUNT = 26


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


def task_check_libraries(**ctx) -> None:
    check_cmds = " && ".join(
        f'python -c "import {lib}" 2>/dev/null || echo "MISSING:{lib}"'
        for lib in REQUIRED_LIBS
    )
    rc, out, err = _ssh(f"source {NIBI_VENV}/bin/activate && {check_cmds}", timeout=60)
    if rc != 0:
        raise AirflowException(f"Library check SSH failed (rc={rc}): {err}")
    missing = [line.split("MISSING:")[1] for line in out.splitlines() if line.startswith("MISSING:")]
    if missing:
        raise AirflowException(f"Missing libraries in NIBI venv ({NIBI_VENV}): {', '.join(missing)}")
    print(f"All {len(REQUIRED_LIBS)} required libraries present in {NIBI_VENV}")


def _expected_recent_trading_dates(end_date: dt.date, limit: int) -> list[dt.date]:
    import exchange_calendars as xcals

    xnys = xcals.get_calendar("XNYS")
    schedule = xnys.schedule.loc[: str(end_date)].tail(limit)
    return [ts.date() for ts in schedule.index]


def _next_trading_day(trade_date: dt.date) -> dt.date:
    import exchange_calendars as xcals

    xnys = xcals.get_calendar("XNYS")
    schedule = xnys.schedule.loc[str(trade_date + dt.timedelta(days=1)) :]
    if schedule.empty:
        raise AirflowException(f"No next trading day found after {trade_date}")
    return schedule.index[0].date()


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
                    SELECT trade_date, symbol,
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
                    SELECT trade_date, symbol,
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
                    SELECT trade_date, symbol,
                           COUNT(*) FILTER (WHERE {regular_hours_filter}) AS bars_rth
                    FROM ml.market_data_15m
                    WHERE trade_date = ANY(%s::date[])
                    GROUP BY 1,2
                )
                SELECT d.trade_date,
                       COALESCE(COUNT(c.symbol) FILTER (WHERE c.bars_rth > 0), 0) AS present_symbols,
                       COALESCE(COUNT(c.symbol) FILTER (WHERE c.bars_rth >= %s), 0) AS complete_symbols
                FROM expected_days d
                LEFT JOIN counts c ON c.trade_date = d.trade_date
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


def task_ensure_training_db_coverage(**ctx) -> None:
    import psycopg2

    sim_date = dt.date.fromisoformat(ctx["ds"])
    coverage_anchor = sim_date - dt.timedelta(days=1)
    expected_dates = _expected_recent_trading_dates(coverage_anchor, TRAINING_COVERAGE_LOOKBACK_TRADING_DAYS)
    if not expected_dates:
        raise AirflowException(f"No expected trading dates found for coverage_anchor={coverage_anchor}")

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        connect_timeout=10,
    )
    try:
        reference_symbol_count, _, coverage_rows = _load_training_coverage(conn, expected_dates)
    finally:
        conn.close()

    if reference_symbol_count <= 0:
        raise AirflowException("Could not determine a reference symbol universe from ml.market_data_15m")

    required_symbols = math.ceil(reference_symbol_count * TRAINING_COVERAGE_MIN_SYMBOL_RATIO)
    print(
        f"Base-only coverage check anchor={coverage_anchor} "
        f"expected_dates={expected_dates[0]}..{expected_dates[-1]} "
        f"reference_symbols={reference_symbol_count} required_complete>={required_symbols}"
    )
    for row in coverage_rows:
        print(
            f"  {row['trade_date']}: present_symbols={row['present_symbols']} "
            f"complete_regular_session_symbols={row['complete_symbols']}"
        )

    complete_dates = [
        row["trade_date"]
        for row in coverage_rows
        if int(row["complete_symbols"]) >= required_symbols
    ]
    if not complete_dates:
        raise AirflowException(
            "No fully covered trading day found in the recent coverage window. "
            "Cannot train a base model from the latest available data."
        )

    latest_complete_date = max(complete_dates)
    effective_sim_date = _next_trading_day(latest_complete_date).isoformat()
    degraded = latest_complete_date != expected_dates[-1]
    print(
        f"Selected latest complete trading day {latest_complete_date}; "
        f"effective_sim_date={effective_sim_date}; degraded={degraded}"
    )
    ctx["ti"].xcom_push(key="latest_complete_trade_date", value=latest_complete_date.isoformat())
    ctx["ti"].xcom_push(key="effective_sim_date", value=effective_sim_date)


def task_export_parquet(**ctx) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    sim_date = ctx["ds"]
    out_path = DATASETS_DIR / f"snapshot_{sim_date}.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        mb = out_path.stat().st_size / 1e6
        print(f"Parquet already exists ({mb:.1f} MB) — skipping export")
        ctx["ti"].xcom_push(key="parquet_path", value=str(out_path))
        return

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


def task_sync_code(**ctx) -> None:
    _ssh(f"mkdir -p {NIBI_SIM_DIR}/ml/ml")
    r = subprocess.run(
        [
            "rsync", "-az", "--delete",
            "-e", f"ssh -i {NIBI_KEY} -o ControlPath={NIBI_SOCKET_PATH} -o ControlMaster=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
            str(ML_SRC) + "/",
            f"{NIBI_USER}@{NIBI_HOST}:{NIBI_SIM_DIR}/ml/ml/",
        ],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise AirflowException(f"rsync ml code failed:\n{r.stderr}")
    print("ML code synced to NIBI")


def task_sync_parquet(**ctx) -> None:
    parquet_path = Path(ctx["ti"].xcom_pull(task_ids="export_parquet", key="parquet_path"))
    sim_date = ctx["ds"]
    remote_path = f"{NIBI_DATA_DIR}/snapshot_{sim_date}.parquet"

    _ssh(f"mkdir -p {NIBI_DATA_DIR}")
    local_size = parquet_path.stat().st_size
    rc, remote_size_str, _ = _ssh(f"stat -c%s {remote_path} 2>/dev/null || echo 0")
    if rc == 0 and remote_size_str.isdigit() and int(remote_size_str) == local_size:
        print(f"Remote parquet matches local ({local_size / 1e6:.1f} MB) — skipping SCP")
        ctx["ti"].xcom_push(key="remote_parquet", value=remote_path)
        return

    r = subprocess.run(
        [
            "scp", "-i", str(NIBI_KEY),
            "-o", f"ControlPath={NIBI_SOCKET_PATH}",
            "-o", "ControlMaster=no",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            str(parquet_path),
            f"{NIBI_USER}@{NIBI_HOST}:{remote_path}",
        ],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise AirflowException(f"SCP failed:\n{r.stderr}")
    print(f"Parquet on NIBI: {remote_path}")
    ctx["ti"].xcom_push(key="remote_parquet", value=remote_path)


def task_clean_nibi_run_root(**ctx) -> None:
    rc, running_jobs, _ = _ssh(
        f"squeue -u {NIBI_USER} -h -o '%j %T' 2>/dev/null | grep -E 'sim_base_train|algo_sim' | grep -E 'RUNNING|COMPLETING' || true",
        timeout=15,
    )
    if running_jobs.strip():
        raise AirflowException(f"A simulation/base-train job is currently RUNNING:\n{running_jobs.strip()}")
    rc, _, err = _ssh(f"rm -rf {NIBI_RUN_ROOT} && mkdir -p {NIBI_RUN_ROOT}", timeout=60)
    if rc != 0:
        raise AirflowException(f"Failed to clean run_root (rc={rc}): {err}")
    print(f"Cleaned: {NIBI_RUN_ROOT}")


def task_submit_base_job(**ctx) -> None:
    logical_date = ctx["ds"]
    effective_sim_date = ctx["ti"].xcom_pull(task_ids="ensure_training_db_coverage", key="effective_sim_date") or logical_date
    remote_parq = ctx["ti"].xcom_pull(task_ids="sync_parquet_to_nibi", key="remote_parquet")
    job_record = REPO_ROOT / "logs" / f"nibi_base_job_{logical_date}.json"

    if job_record.exists():
        rec = json.loads(job_record.read_text())
        existing_id = rec.get("job_id")
        if existing_id and rec.get("status") != "completed":
            print(f"Base job already submitted for {logical_date}: {existing_id} — reusing")
            ctx["ti"].xcom_push(key="job_id", value=existing_id)
            return

    cmd = f"sbatch {NIBI_SBATCH} --parquet {remote_parq} --sim-date {effective_sim_date} --base-only"
    print(f"Submitting: {cmd}")
    rc, out, err = _ssh(cmd, timeout=30)
    if rc != 0:
        raise AirflowException(f"sbatch failed (rc={rc}):\n{err}")

    job_id = next((tok for tok in out.split() if tok.isdigit()), None)
    if not job_id:
        raise AirflowException(f"Could not parse job ID from sbatch output: {out!r}")

    job_record.parent.mkdir(parents=True, exist_ok=True)
    job_record.write_text(json.dumps({
        "job_id": job_id,
        "logical_date": logical_date,
        "effective_sim_date": effective_sim_date,
        "submitted_at": dt.datetime.utcnow().isoformat(),
        "status": "submitted",
    }, indent=2))
    print(f"Base-only job submitted: {job_id}")
    ctx["ti"].xcom_push(key="job_id", value=job_id)


FAILED_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


class NibiBaseJobSensor(BaseSensorOperator):
    def __init__(self, job_id_task: str, **kwargs):
        super().__init__(poke_interval=120, timeout=14_400, mode="reschedule", **kwargs)
        self.job_id_task = job_id_task
        self._ssh_fail_streak = 0

    def poke(self, context) -> bool:
        sim_date = context["ds"]
        job_id = context["ti"].xcom_pull(task_ids=self.job_id_task, key="job_id")
        if not job_id:
            raise AirflowException("No job_id in XCom.")

        rc, squeue_out, _ = _ssh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null || true", timeout=20)
        if rc != 0:
            self._ssh_fail_streak += 1
            if self._ssh_fail_streak >= 3:
                raise AirflowException("SSH to NIBI failed 3 times in a row.")
            return False
        self._ssh_fail_streak = 0

        state = squeue_out.strip()
        if state in ("RUNNING", "PENDING", "COMPLETING"):
            print(f"  [{state}] base job {job_id}")
            return False

        _, sacct_out, _ = _ssh(f"sacct -j {job_id} --format=State --noheader 2>/dev/null | head -1", timeout=20)
        final_state = sacct_out.strip().split()[0] if sacct_out.strip() else ""
        if not final_state:
            return False
        if final_state.startswith("COMPLETED"):
            record_path = REPO_ROOT / "logs" / f"nibi_base_job_{sim_date}.json"
            if record_path.exists():
                rec = json.loads(record_path.read_text())
                rec["status"] = "completed"
                rec["completed_at"] = dt.datetime.utcnow().isoformat()
                record_path.write_text(json.dumps(rec, indent=2))
            print(f"Base-only job {job_id} COMPLETED.")
            return True
        if any(final_state.startswith(s) for s in FAILED_STATES):
            _, err_content, _ = _ssh(
                f"tail -30 {NIBI_SIM_DIR}/logs/sim_base_{job_id}.err 2>/dev/null || echo '(no err log)'",
                timeout=20,
            )
            raise AirflowException(f"Base-only job {job_id} ended with state: {final_state}\n{err_content}")
        return False


def task_validate_base_artifacts(**ctx) -> None:
    rc, out, err = _ssh(f"test -f {NIBI_RUN_ROOT}/current/models/model_manifest.json && echo ok || echo missing", timeout=20)
    if rc != 0 or out.strip() != "ok":
        raise AirflowException(f"Base artifacts missing on NIBI at {NIBI_RUN_ROOT}/current/models/model_manifest.json")
    print("Base artifacts validated on NIBI")


def task_rsync_base_artifacts_back(**ctx) -> None:
    sim_date = ctx["ds"]
    warm_dest = ARTIFACTS_DIR / f"warm_{sim_date}"
    if warm_dest.exists():
        shutil.rmtree(warm_dest)
    warm_dest.mkdir(parents=True, exist_ok=True)

    ssh_e = (
        f"ssh -i {NIBI_KEY}"
        f" -o ControlPath={NIBI_SOCKET_PATH}"
        f" -o ControlMaster=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    )
    r = subprocess.run(
        [
            "rsync", "-az", "--compress",
            "-e", ssh_e,
            f"{NIBI_USER}@{NIBI_HOST}:{NIBI_RUN_ROOT}/current/",
            str(warm_dest) + "/",
        ],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise AirflowException(f"rsync base model failed:\n{r.stderr}")
    print(f"Base bundle rsynced to {warm_dest}")
    ctx["ti"].xcom_push(key="warm_artifact_dir", value=str(warm_dest))


def _atomic_symlink(symlink: Path, target: Path) -> None:
    tmp = symlink.parent / (symlink.name + ".new")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target.name)
    tmp.rename(symlink)


def task_promote_base_model(**ctx) -> None:
    warm_dir = Path(ctx["ti"].xcom_pull(task_ids="rsync_artifacts_back", key="warm_artifact_dir"))
    manifest = warm_dir / "models" / "model_manifest.json"
    if not manifest.exists():
        raise AirflowException(f"model_manifest.json not found in warm bundle at {manifest}")
    _atomic_symlink(ARTIFACTS_DIR / "current_base", warm_dir)
    _atomic_symlink(ARTIFACTS_DIR / "current_base_eod", warm_dir)
    print(f"Promoted: current_base/current_base_eod → {warm_dir}")


def task_generate_base_predictions(**ctx) -> None:
    """
    Generate all 26 step prediction CSVs locally using the just-promoted base model.

    Runs on the VPS (no NIBI required) so predictions are ready before market open.
    Uses the same gen_step_predictions.py logic but invoked via subprocess with the
    /data/env Python that has xgboost installed.

    Non-fatal: a failure logs a warning but does not block the backend reload.
    """
    import shutil
    import subprocess as sp
    import tempfile

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    sim_date = ctx["ti"].xcom_pull(task_ids="ensure_training_db_coverage", key="effective_sim_date") or ctx["ds"]
    base_bundle = ARTIFACTS_DIR / "current_base"
    sim_dest = ARTIFACTS_DIR / f"simulation_{sim_date}"

    try:
        # ── 1. Export 15-day snapshot to a temp parquet ──────────────────────
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"base_pred_{sim_date}_"))
        slice_path = tmp_dir / "slices" / "slice_1945.parquet"
        slice_path.parent.mkdir(parents=True)

        db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url, pool_pre_ping=True)
        print(f"Pulling 15 trading days from DB for base predictions (sim_date={sim_date}) ...")
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
        print(f"  {len(df):,} rows, {df['symbol'].nunique()} symbols → {slice_path}")

        # ── 2. Build per-step dirs with symlinks to base model ───────────────
        run_root = tmp_dir / "run_root"
        for i in range(26):
            step_dir = run_root / f"step_{i:02d}"
            step_dir.mkdir(parents=True)
            models_link = step_dir / "models"
            if not models_link.exists():
                models_link.symlink_to((base_bundle / "models").resolve())
            fn_link = step_dir / "feature_names.json"
            if not fn_link.exists():
                fn_link.symlink_to((base_bundle / "feature_names.json").resolve())
            (step_dir / "predictions").mkdir(exist_ok=True)

        # ── 3. Run gen_step_predictions.py ───────────────────────────────────
        gen_script = REPO_ROOT / "ml" / "ml" / "nibi" / "gen_step_predictions.py"
        venv_python = Path("/data/env/bin/python3")
        python_bin = venv_python if venv_python.exists() else Path(sys.executable)

        print(f"Running gen_step_predictions.py for sim_date={sim_date} ...")
        result = sp.run(
            [str(python_bin), str(gen_script), "--run-root", str(run_root), "--sim-date", sim_date],
            capture_output=True, text=True, timeout=3600, cwd=str(REPO_ROOT),
        )
        print(result.stdout[-3000:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            raise RuntimeError(f"gen_step_predictions failed (rc={result.returncode}):\n{result.stderr[-1000:]}")

        # ── 4. Copy prediction CSVs to simulation bundle ─────────────────────
        sim_dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for i in range(26):
            src = run_root / f"step_{i:02d}" / "predictions" / "predictions.csv"
            dst_dir = sim_dest / f"step_{i:02d}" / "predictions"
            dst_dir.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst_dir / "predictions.csv")
                copied += 1
        print(f"Copied {copied}/26 prediction CSVs to {sim_dest}")

        if copied < 26:
            print(f"WARNING: only {copied}/26 steps have predictions — skipping current_simulation promote")
            return

        # ── 5. Write simulation_summary.json ─────────────────────────────────
        import datetime as _dt
        import json as _json

        base_meta_path = base_bundle / "metadata.json"
        base_meta = _json.loads(base_meta_path.read_text()) if base_meta_path.exists() else {}
        d = _dt.date.fromisoformat(sim_date)
        utc_open = _dt.datetime(d.year, d.month, d.day, 13, 30, tzinfo=_dt.timezone.utc)
        steps = [
            {
                "step": i,
                "as_of_ts": (_dt.datetime(d.year, d.month, d.day, 13, 30, tzinfo=_dt.timezone.utc)
                             + _dt.timedelta(minutes=15 * i)).isoformat(),
                "et_label": (utc_open + _dt.timedelta(minutes=15 * i - 4 * 60)).strftime("%H:%M"),
                "status": "ok",
            }
            for i in range(26)
        ]
        summary = {
            "sim_date": sim_date,
            "status": "success",
            "source": "base_model_local",
            "base_train_sec": base_meta.get("train_sec", 0),
            "steps": steps,
        }
        (sim_dest / "simulation_summary.json").write_text(_json.dumps(summary, indent=2))

        # ── 6. Promote current_simulation ─────────────────────────────────────
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
            resp = requests.post(f"{BACKEND_URL}{endpoint}", timeout=30)
            if resp.status_code == 200:
                print(f"Reloaded {endpoint}: {resp.json()}")
            else:
                print(f"WARNING: reload {endpoint} returned HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            print(f"WARNING: reload {endpoint} failed: {exc}")


with DAG(
    dag_id="nibi_base_only_training",
    default_args={
        "owner": "ml-team",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": dt.timedelta(minutes=10),
        "email_on_failure": False,
    },
    description="Base-only NIBI training: export → sync → base train → promote current_base/current_base_eod",
    schedule="0 2 * * 1-5",  # 02:00 UTC = 10 PM ET, safely after intraday warm refresh completes
    start_date=dt.datetime(2026, 5, 4),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "nibi", "base-only"],
) as dag:
    t1_health = PythonOperator(task_id="ssh_health_check", python_callable=task_ssh_health_check, execution_timeout=dt.timedelta(minutes=2))
    t1b_libs = PythonOperator(task_id="check_nibi_libraries", python_callable=task_check_libraries, execution_timeout=dt.timedelta(minutes=3), retries=0)
    t1c_db = PythonOperator(task_id="ensure_training_db_coverage", python_callable=task_ensure_training_db_coverage, execution_timeout=dt.timedelta(minutes=10), retries=0)
    t2_export = PythonOperator(task_id="export_parquet", python_callable=task_export_parquet, execution_timeout=dt.timedelta(minutes=15))
    t3_code = PythonOperator(task_id="sync_code_to_nibi", python_callable=task_sync_code, execution_timeout=dt.timedelta(minutes=5))
    t4_parquet = PythonOperator(task_id="sync_parquet_to_nibi", python_callable=task_sync_parquet, execution_timeout=dt.timedelta(minutes=15))
    t5_clean = PythonOperator(task_id="clean_nibi_run_root", python_callable=task_clean_nibi_run_root, execution_timeout=dt.timedelta(minutes=3))
    t6_submit = PythonOperator(task_id="submit_base_job", python_callable=task_submit_base_job, execution_timeout=dt.timedelta(minutes=2), retries=2)
    t7_poll = NibiBaseJobSensor(task_id="poll_job_until_done", job_id_task="submit_base_job")
    t8_validate = PythonOperator(task_id="validate_base_artifacts", python_callable=task_validate_base_artifacts, execution_timeout=dt.timedelta(minutes=5))
    t9_rsync = PythonOperator(task_id="rsync_artifacts_back", python_callable=task_rsync_base_artifacts_back, execution_timeout=dt.timedelta(minutes=30))
    t10_promote = PythonOperator(task_id="promote_base_model", python_callable=task_promote_base_model, execution_timeout=dt.timedelta(minutes=5), retries=0)
    t10b_gen_preds = PythonOperator(task_id="generate_base_predictions", python_callable=task_generate_base_predictions, execution_timeout=dt.timedelta(hours=1), retries=0)
    t11_reload = PythonOperator(task_id="reload_backend", python_callable=task_reload_backend, execution_timeout=dt.timedelta(minutes=2), retries=0)

    t1_health >> [t1b_libs, t1c_db, t3_code]
    t1c_db >> t2_export
    [t1b_libs, t2_export, t3_code] >> t4_parquet
    t4_parquet >> t5_clean >> t6_submit >> t7_poll >> t8_validate >> t9_rsync >> t10_promote >> t10b_gen_preds >> t11_reload

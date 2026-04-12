#!/usr/bin/env python3
"""
simulate_april7.py — End-to-end April 7 warm-refresh simulation orchestrator.

What this tests:
  Phase 0  NIBI health check (SSH, scratch dirs, sbatch)
  Phase 1  Setup NIBI (copy repo + base model)
  Phase 2  Extract April 1–7 data → parquet
  Phase 3  SCP parquet to NIBI
  Phase 4  Submit simulate_april7.sbatch
  Phase 5  Poll until COMPLETED (max 90 min)
  Phase 6  Rsync simulation artifacts back
  Phase 7  Promote latest warm model (symlink + report)

Logs every phase with start/end/elapsed time.
Prints a summary report at the end.

Usage (from repo root, after morning_login.sh):
    python tests/simulate_april7.py [--dry-run] [--skip-setup]

Env vars:
    NIBI_USER, NIBI_SCRATCH   (set in .env or exported)
    OLD_DB_HOST, OLD_DB_PORT, OLD_DB_NAME, OLD_DB_USER, OLD_DB_PASSWORD
        — defaults to the-project-maverick DB at localhost:5432
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
NIBI_ALIAS  = "nibi"   # matches ~/.ssh/config Host alias
NIBI_USER   = os.getenv("NIBI_USER", "harshsaw")
NIBI_HOST   = os.getenv("NIBI_HOST", "nibi.sharcnet.ca")
NIBI_SCRATCH = os.getenv("NIBI_SCRATCH", f"/scratch/{NIBI_USER}")

# Data source: use old maverick DB by default
OLD_DB_HOST = os.getenv("OLD_DB_HOST", "localhost")
OLD_DB_PORT = int(os.getenv("OLD_DB_PORT", "5432"))
OLD_DB_NAME = os.getenv("OLD_DB_NAME", "algotrade")
OLD_DB_USER = os.getenv("OLD_DB_USER", "cosc_admin")
OLD_DB_PASS = os.getenv("OLD_DB_PASSWORD", "")

# Paths
LOCAL_BASE_MODEL  = REPO_ROOT / "model_artifacts" / "current_base"
LOCAL_SIM_OUT     = REPO_ROOT / "model_artifacts" / "sim_2026-04-07"
LOCAL_DATASETS    = REPO_ROOT / "datasets"
SNAPSHOT_PARQUET  = LOCAL_DATASETS / "snapshot_2026-04-07.parquet"

# NIBI paths
NIBI_ALGO_DIR    = f"{NIBI_SCRATCH}/algo"
NIBI_RUN_ROOT    = f"{NIBI_SCRATCH}/ml/run_root"
NIBI_SIM_OUT     = f"{NIBI_SCRATCH}/ml/simulation_2026-04-07"
NIBI_DATA_DIR    = f"{NIBI_SCRATCH}/data"
NIBI_PARQUET     = f"{NIBI_DATA_DIR}/snapshot_2026-04-07.parquet"
NIBI_SBATCH      = f"{NIBI_ALGO_DIR}/ml/ml/nibi/simulate_april7.sbatch"
NIBI_DONE_FILE   = f"{NIBI_SIM_OUT}/DONE"

POLL_INTERVAL_SEC = 60
MAX_POLL_MIN      = 90


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = REPO_ROOT / "logs" / f"sim_april7_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_log_fh = open(LOG_FILE, "w")

def log(msg: str) -> None:
    ts = dt.datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _log_fh.write(line + "\n")
    _log_fh.flush()


# ── Timing ────────────────────────────────────────────────────────────────────
_timings: list[dict] = []

class Timer:
    def __init__(self, phase: str):
        self.phase = phase
        self.t0 = time.time()

    def done(self, note: str = "") -> float:
        elapsed = time.time() - self.t0
        _timings.append({"phase": self.phase, "elapsed_sec": round(elapsed, 1), "note": note})
        log(f"  ✓ {self.phase} — {elapsed:.1f}s{' | ' + note if note else ''}")
        return elapsed


# ── SSH helpers ───────────────────────────────────────────────────────────────
def ssh(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout=15", NIBI_ALIAS, cmd],
        capture_output=True, text=True, timeout=timeout + 5,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def scp_to(local: str, remote: str, timeout: int = 300) -> None:
    result = subprocess.run(
        ["scp", "-o", "BatchMode=yes", local, f"{NIBI_ALIAS}:{remote}"],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr.strip()}")


def rsync_from(remote_dir: str, local_dir: Path, timeout: int = 300) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["rsync", "-az", "--progress",
         "-e", "ssh -o BatchMode=yes",
         f"{NIBI_ALIAS}:{remote_dir}/", str(local_dir) + "/"],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rsync failed: {result.stderr.strip()}")


def rsync_to(local_dir: str | Path, remote_dir: str, timeout: int = 300) -> None:
    result = subprocess.run(
        ["rsync", "-az", "--progress",
         "-e", "ssh -o BatchMode=yes",
         str(local_dir) + "/", f"{NIBI_ALIAS}:{remote_dir}/"],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rsync failed: {result.stderr.strip()}")


# ── Phase helpers ─────────────────────────────────────────────────────────────

def phase0_health_check() -> None:
    log("\n── Phase 0: NIBI Health Check ──")
    t = Timer("phase0_ssh_check")
    rc, out, err = ssh("echo ok && whoami && sbatch --version | head -1")
    if rc != 0:
        raise RuntimeError(f"SSH check failed: {err}\nRun: bash ml/ml/nibi/morning_login.sh")
    t.done(f"user={out.splitlines()[1] if len(out.splitlines()) > 1 else '?'}")

    t = Timer("phase0_check_dirs")
    rc, _, err = ssh(f"mkdir -p {NIBI_DATA_DIR} {NIBI_SCRATCH}/ml/logs {NIBI_SCRATCH}/ml/run_root && echo ok")
    if rc != 0:
        raise RuntimeError(f"mkdir failed: {err}")
    t.done()


def phase1_setup_nibi(skip: bool = False) -> None:
    log("\n── Phase 1: Setup NIBI (repo + base model) ──")

    if skip:
        log("  [SKIP] --skip-setup flag set, assuming NIBI already configured")
        return

    # Copy ml code to NIBI
    t = Timer("phase1_rsync_repo")
    ml_src = REPO_ROOT / "ml" / "ml"
    rsync_to(ml_src, f"{NIBI_ALGO_DIR}/ml/ml")
    t.done(f"src={ml_src}")

    # Copy base model as run_root/current
    t = Timer("phase1_rsync_base_model")
    base_src = Path("/data/projects/the-project-maverick/model_artifacts/base_2026-04-07")
    if not base_src.exists():
        raise RuntimeError(f"Base model not found at {base_src}")
    rsync_to(base_src, f"{NIBI_RUN_ROOT}/current")
    t.done(f"trees=1157, date_range=..2026-04-06")

    # Verify model arrived
    t = Timer("phase1_verify_model")
    rc, out, err = ssh(
        f"ls {NIBI_RUN_ROOT}/current/models/horizon_00.json 2>/dev/null && echo FOUND || echo MISSING"
    )
    if "FOUND" not in out:
        raise RuntimeError("Base model not found on NIBI after rsync")
    model_count = ssh(f"ls {NIBI_RUN_ROOT}/current/models/ | wc -l")[1]
    t.done(f"horizon files={model_count.strip()}")


def phase2_extract_parquet() -> None:
    log("\n── Phase 2: Extract April 1–7 data → parquet ──")
    t = Timer("phase2_extract")

    LOCAL_DATASETS.mkdir(parents=True, exist_ok=True)

    if SNAPSHOT_PARQUET.exists():
        size_mb = SNAPSHOT_PARQUET.stat().st_size / 1e6
        log(f"  Parquet already exists ({size_mb:.1f} MB) — reusing")
        t.done(f"reused existing, {size_mb:.1f}MB")
        return

    # Extract from old DB
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    log(f"  Connecting to {OLD_DB_HOST}:{OLD_DB_PORT}/{OLD_DB_NAME} ...")
    engine = create_engine(
        f"postgresql+psycopg2://{OLD_DB_USER}:{OLD_DB_PASS}@{OLD_DB_HOST}:{OLD_DB_PORT}/{OLD_DB_NAME}"
    )

    sql = text("""
        SELECT * FROM ml.market_data_15m
        WHERE trade_date >= '2024-03-25'
          AND trade_date <= '2026-04-07'
        ORDER BY symbol, window_ts
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    engine.dispose()

    rows = len(df)
    syms = df["symbol"].nunique()
    log(f"  Extracted {rows:,} rows, {syms} symbols")

    pq.write_table(pa.Table.from_pandas(df), SNAPSHOT_PARQUET)
    size_mb = SNAPSHOT_PARQUET.stat().st_size / 1e6
    t.done(f"{rows:,} rows, {syms} symbols, {size_mb:.1f}MB")


def phase3_scp_parquet() -> None:
    log("\n── Phase 3: SCP parquet → NIBI ──")
    t = Timer("phase3_scp")

    # Check if already there and same size
    local_size = SNAPSHOT_PARQUET.stat().st_size
    rc, remote_size, _ = ssh(f"stat -c%s {NIBI_PARQUET} 2>/dev/null || echo 0")
    if rc == 0 and remote_size.strip().isdigit() and int(remote_size.strip()) == local_size:
        log(f"  Remote file matches local ({local_size / 1e6:.1f} MB) — skipping transfer")
        t.done("skipped, already up to date")
        return

    log(f"  Uploading {local_size / 1e6:.1f} MB ...")
    scp_to(str(SNAPSHOT_PARQUET), NIBI_PARQUET, timeout=600)
    t.done(f"{local_size / 1e6:.1f}MB")


def phase4_submit_job() -> str:
    log("\n── Phase 4: Submit Slurm job ──")
    t = Timer("phase4_submit")

    rc, out, err = ssh(f"sbatch {NIBI_SBATCH} --parquet {NIBI_PARQUET}")
    if rc != 0:
        raise RuntimeError(f"sbatch failed: {err}")

    job_id = None
    for token in out.split():
        if token.isdigit():
            job_id = token
            break
    if not job_id:
        raise RuntimeError(f"Could not parse job ID from: {out!r}")

    t.done(f"job_id={job_id}")
    return job_id


def phase5_poll_job(job_id: str) -> None:
    log(f"\n── Phase 5: Poll job {job_id} (max {MAX_POLL_MIN} min) ──")
    t = Timer("phase5_poll")

    deadline = time.time() + MAX_POLL_MIN * 60
    last_state = "PENDING"
    while time.time() < deadline:
        rc, state_out, _ = ssh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null || echo GONE")
        state = state_out.strip()

        if not state or state == "GONE":
            # Job left queue — check sacct for final state
            rc2, sacct_out, _ = ssh(
                f"sacct -j {job_id} --noheader --format=State | head -1"
            )
            final = sacct_out.strip().split()[0] if sacct_out.strip() else "COMPLETED"
            if final in ("COMPLETED", "COMPLETING"):
                t.done(f"job_id={job_id} state={final}")
                return
            else:
                raise RuntimeError(f"Job {job_id} ended in state: {final}")

        if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
            raise RuntimeError(f"Job {job_id} failed with state: {state}")

        if state != last_state:
            log(f"  [{state}] job {job_id} ...")
            last_state = state

        time.sleep(POLL_INTERVAL_SEC)

    raise RuntimeError(f"Job {job_id} did not complete within {MAX_POLL_MIN} minutes")


def phase6_rsync_artifacts() -> None:
    log("\n── Phase 6: Rsync simulation artifacts ← NIBI ──")
    t = Timer("phase6_rsync")

    # Verify DONE file
    rc, done_content, _ = ssh(f"cat {NIBI_DONE_FILE} 2>/dev/null || echo MISSING")
    if "MISSING" in done_content:
        raise RuntimeError(f"DONE file not found at {NIBI_DONE_FILE}")

    if LOCAL_SIM_OUT.exists():
        shutil.rmtree(LOCAL_SIM_OUT)

    rsync_from(NIBI_SIM_OUT, LOCAL_SIM_OUT, timeout=300)
    step_count = len(list(LOCAL_SIM_OUT.glob("step_*")))
    size_mb = sum(f.stat().st_size for f in LOCAL_SIM_OUT.rglob("*") if f.is_file()) / 1e6
    t.done(f"steps={step_count}, {size_mb:.1f}MB")


def phase7_promote_and_report() -> None:
    log("\n── Phase 7: Promote model + final report ──")
    t = Timer("phase7_promote")

    # Point current_base → sim_2026-04-07 (latest warm)
    current_link = REPO_ROOT / "model_artifacts" / "current_base"
    if current_link.is_symlink():
        current_link.unlink()
    elif current_link.exists():
        shutil.move(str(current_link), str(current_link) + ".bak")

    current_link.symlink_to(LOCAL_SIM_OUT)
    t.done(f"current_base → {LOCAL_SIM_OUT.name}")

    # Load simulation_summary.json if present
    summary_path = LOCAL_SIM_OUT / "simulation_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    # Print timing report
    log("\n" + "=" * 60)
    log(" SIMULATION REPORT — April 7 Warm-Refresh")
    log("=" * 60)
    log(f" Log file : {LOG_FILE}")
    log(f" Artifacts: {LOCAL_SIM_OUT}")
    log("")
    log(" Phase Timings:")
    total = 0.0
    for item in _timings:
        total += item["elapsed_sec"]
        note = f"  [{item['note']}]" if item["note"] else ""
        log(f"   {item['phase']:<30} {item['elapsed_sec']:>7.1f}s{note}")
    log(f"   {'TOTAL':<30} {total:>7.1f}s")
    log("")
    if summary:
        steps = summary.get("steps_completed", "?")
        syms  = summary.get("symbols", "?")
        log(f" Simulation steps    : {steps}")
        log(f" Symbols updated     : {syms}")
    log("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="April 7 simulation orchestrator")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Run health check only, do not submit job")
    parser.add_argument("--skip-setup",  action="store_true",
                        help="Skip Phase 1 (repo + model already on NIBI)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip Phase 2 (parquet already exists locally)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log(f"April 7 Simulation — {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log(f"NIBI: {NIBI_USER}@{NIBI_HOST}  scratch={NIBI_SCRATCH}")
    log(f"Log : {LOG_FILE}")

    try:
        phase0_health_check()

        if args.dry_run:
            log("\n[dry-run] Health check passed. Exiting.")
            return

        phase1_setup_nibi(skip=args.skip_setup)

        if not args.skip_extract:
            phase2_extract_parquet()
        else:
            log("\n── Phase 2: SKIPPED (--skip-extract) ──")

        phase3_scp_parquet()
        job_id = phase4_submit_job()
        phase5_poll_job(job_id)
        phase6_rsync_artifacts()
        phase7_promote_and_report()

    except KeyboardInterrupt:
        log("\n[INTERRUPTED] Ctrl+C received")
        sys.exit(1)
    except Exception as exc:
        log(f"\n[FAILED] {exc}")
        log(f"  Partial timings:")
        for item in _timings:
            log(f"    {item['phase']}: {item['elapsed_sec']}s")
        sys.exit(1)
    finally:
        _log_fh.close()


if __name__ == "__main__":
    main()

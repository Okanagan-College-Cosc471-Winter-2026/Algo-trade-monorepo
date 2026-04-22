#!/usr/bin/env python3
"""
simulate_market_day.py — Full pipeline simulation: base train → April 7 intraday warm-refresh.

Flow (single 8-hour GPU job — queue wait happens ONCE):
  Phase 0  NIBI health check (SSH + sbatch)
  Phase 1  Setup NIBI  — rsync ml code + base model to test_simulation/
  Phase 2  Extract parquet from DB → local datasets/
           SCP parquet → NIBI test_simulation/data/
  Phase 3  Submit simulate_full_day.sbatch (8h H100)
           Poll ONE job until COMPLETED
  Phase 4  Rsync run_root/ back → local model_artifacts/sim_2026-04-07/
  Phase 5  Print timing report from simulation_progress.json

Usage (from repo root, after: bash ml/ml/nibi/morning_login.sh):
    python tests/simulate_market_day.py              # full run
    python tests/simulate_market_day.py --fast       # 200-tree base (flow test)
    python tests/simulate_market_day.py --dry-run    # Phase 0 only
    python tests/simulate_market_day.py --skip-setup --skip-extract --skip-base
    python tests/simulate_market_day.py --start-window 5

Env vars (from .env):
    NIBI_USER, NIBI_HOST, NIBI_SIM_DIR
    OLD_DB_HOST, OLD_DB_PORT, OLD_DB_NAME, OLD_DB_USER, OLD_DB_PASSWORD
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
NIBI_ALIAS   = "nibi"
NIBI_USER    = os.getenv("NIBI_USER", "harshsaw")
NIBI_HOST    = os.getenv("NIBI_HOST", "nibi.sharcnet.ca")
NIBI_SIM_DIR = os.getenv("NIBI_SIM_DIR", "/home/harshsaw/projects/def-youry/test_simulation")

OLD_DB_HOST  = os.getenv("OLD_DB_HOST", "localhost")
OLD_DB_PORT  = int(os.getenv("OLD_DB_PORT", "5432"))
OLD_DB_NAME  = os.getenv("OLD_DB_NAME", "market_data")
OLD_DB_USER  = os.getenv("OLD_DB_USER", "mluser")
OLD_DB_PASS  = os.getenv("OLD_DB_PASSWORD", "mlpassword")

# Local paths
LOCAL_DATASETS    = REPO_ROOT / "datasets"
FULL_PARQUET      = LOCAL_DATASETS / "snapshot_2026-04-07.parquet"
LOCAL_MODEL_DIR   = REPO_ROOT / "model_artifacts" / "sim_2026-04-07"

# NIBI paths (all under test_simulation/)
NIBI_DATA_DIR     = f"{NIBI_SIM_DIR}/data"
NIBI_FULL_PARQUET = f"{NIBI_SIM_DIR}/data/snapshot_2026-04-07.parquet"
NIBI_RUN_ROOT     = f"{NIBI_SIM_DIR}/run_root"
NIBI_FULL_SBATCH  = f"{NIBI_SIM_DIR}/ml/ml/nibi/simulate_full_day.sbatch"

POLL_INTERVAL   = 60    # seconds between squeue checks
MAX_POLL_HOURS  = 8     # abort if job not done in 8h


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = REPO_ROOT / "logs" / f"sim_marketday_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_log_fh = open(LOG_FILE, "w", buffering=1)

def log(msg: str) -> None:
    ts = dt.datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _log_fh.write(line + "\n")


# ── Timing ────────────────────────────────────────────────────────────────────
class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def done(self, note: str = "") -> float:
        e = self.elapsed()
        log(f"  ✓ {self.label}: {e:.1f}s{' — ' + note if note else ''}")
        return e


# ── SSH / transfer helpers ────────────────────────────────────────────────────
def ssh(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", NIBI_ALIAS, cmd],
        capture_output=True, text=True, timeout=timeout + 5,
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def scp_to(local: str, remote: str, timeout: int = 300) -> None:
    r = subprocess.run(
        ["scp", "-o", "BatchMode=yes", local, f"{NIBI_ALIAS}:{remote}"],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"SCP failed: {r.stderr.strip()}")


def rsync_from(remote: str, local: Path, timeout: int = 600) -> None:
    local.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rsync", "-az", "-e", "ssh -o BatchMode=yes",
         f"{NIBI_ALIAS}:{remote}/", str(local) + "/"],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr.strip()}")


def rsync_to(local: Path, remote: str, timeout: int = 300) -> None:
    r = subprocess.run(
        ["rsync", "-az", "-e", "ssh -o BatchMode=yes",
         str(local) + "/", f"{NIBI_ALIAS}:{remote}/"],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr.strip()}")


# ── Phases ────────────────────────────────────────────────────────────────────

def phase0_health_check() -> None:
    log("\n══ Phase 0: NIBI Health Check ══")
    t = Timer("ssh + sbatch check")
    rc, out, err = ssh("echo pong && sbatch --version | head -1")
    if rc != 0:
        raise RuntimeError(f"SSH failed: {err}\nRun: bash ml/ml/nibi/morning_login.sh")
    log(f"  {out}")
    t.done()


def phase1_setup_nibi(skip: bool = False) -> None:
    log("\n══ Phase 1: Setup NIBI ══")

    if skip:
        log("  [skip] --skip-setup set")
        return

    # Ensure test_simulation dir structure exists
    ssh(f"mkdir -p {NIBI_SIM_DIR}/data {NIBI_SIM_DIR}/logs {NIBI_SIM_DIR}/run_root {NIBI_SIM_DIR}/ml/ml")

    # Rsync ml code → test_simulation/ml/ml/
    t = Timer("rsync ml code → NIBI")
    rsync_to(REPO_ROOT / "ml" / "ml", f"{NIBI_SIM_DIR}/ml/ml")
    t.done()

    # Rsync base model → test_simulation/run_root/current/
    base_src = Path("/data/projects/the-project-maverick/model_artifacts/base_2026-04-07")
    if not base_src.exists():
        raise RuntimeError(f"Base model not found: {base_src}")

    t = Timer("rsync base model → NIBI run_root/current/")
    ssh(f"mkdir -p {NIBI_RUN_ROOT}/current/models")
    rsync_to(base_src / "models", f"{NIBI_RUN_ROOT}/current/models")
    scp_to(str(base_src / "metadata.json"),      f"{NIBI_RUN_ROOT}/current/metadata.json")
    scp_to(str(base_src / "feature_names.json"), f"{NIBI_RUN_ROOT}/current/feature_names.json")
    rc, count, _ = ssh(f"ls {NIBI_RUN_ROOT}/current/models/*.json | wc -l")
    t.done(f"1157 trees, cutoff=2026-04-06 — {count} files on NIBI")


def phase2_extract_parquet() -> None:
    log("\n══ Phase 2a: Extract parquet from DB ══")
    LOCAL_DATASETS.mkdir(parents=True, exist_ok=True)

    if FULL_PARQUET.exists():
        mb = FULL_PARQUET.stat().st_size / 1e6
        log(f"  Parquet already exists ({mb:.1f} MB) — reusing")
        return

    t = Timer("extract ml.market_data_15m → parquet")
    import pandas as pd
    from sqlalchemy import create_engine, text
    import pyarrow as pa, pyarrow.parquet as pq

    engine = create_engine(
        f"postgresql+psycopg2://{OLD_DB_USER}:{OLD_DB_PASS}@"
        f"{OLD_DB_HOST}:{OLD_DB_PORT}/{OLD_DB_NAME}"
    )
    log(f"  Querying {OLD_DB_NAME} on {OLD_DB_HOST}:{OLD_DB_PORT} ...")
    CHUNK = 200_000
    chunks = []
    with engine.connect() as conn:
        total_rows = conn.execute(text("SELECT COUNT(*) FROM ml.market_data_15m")).scalar()
        log(f"  Total rows: {total_rows:,}")
        for chunk_df in pd.read_sql(
            text("SELECT * FROM ml.market_data_15m ORDER BY symbol, window_ts"),
            conn, chunksize=CHUNK,
        ):
            chunks.append(chunk_df)
            log(f"  Loaded {sum(len(c) for c in chunks):,}/{total_rows:,} ...")
    engine.dispose()

    df = pd.concat(chunks, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(df), FULL_PARQUET)
    mb = FULL_PARQUET.stat().st_size / 1e6
    t.done(f"{len(df):,} rows, {df['symbol'].nunique()} symbols, {mb:.1f}MB")


def phase2b_scp_parquet() -> None:
    log("\n══ Phase 2b: SCP parquet → NIBI ══")
    local_size = FULL_PARQUET.stat().st_size
    rc, remote_size, _ = ssh(f"stat -c%s {NIBI_FULL_PARQUET} 2>/dev/null || echo 0")
    if rc == 0 and remote_size.strip().isdigit() and int(remote_size.strip()) == local_size:
        log(f"  Remote matches local ({local_size/1e6:.1f}MB) — skipping")
        return
    t = Timer("scp full parquet → NIBI")
    ssh(f"mkdir -p {NIBI_DATA_DIR}")
    scp_to(str(FULL_PARQUET), NIBI_FULL_PARQUET, timeout=600)
    t.done(f"{local_size/1e6:.1f}MB")


def phase3_submit_and_poll(fast: bool, skip_base: bool, start_window: int) -> str:
    """Submit one 8-hour job, poll until done. Returns job_id."""
    log("\n══ Phase 3: Submit Full-Day Simulation Job (8h GPU) ══")

    fast_flag       = "--fast"         if fast       else ""
    skip_base_flag  = "--skip-base"    if skip_base  else ""
    window_flag     = f"--start-window {start_window}" if start_window > 0 else ""
    extra = " ".join(f for f in [fast_flag, skip_base_flag, window_flag] if f)

    cmd = (
        f"sbatch {NIBI_FULL_SBATCH} "
        f"--parquet {NIBI_FULL_PARQUET} "
        f"--sim-date 2026-04-07 "
        f"{extra}"
    ).strip()
    log(f"  {cmd}")

    t_submit = Timer("sbatch submit")
    rc, out, err = ssh(cmd, timeout=30)
    if rc != 0:
        raise RuntimeError(f"sbatch failed: {err}")
    job_id = next((tok for tok in out.split() if tok.isdigit()), None)
    if not job_id:
        raise RuntimeError(f"No job ID in: {out!r}")
    t_submit.done(f"job_id={job_id}")

    log(f"  Polling job {job_id} every {POLL_INTERVAL}s (max {MAX_POLL_HOURS}h) ...")
    deadline = time.time() + MAX_POLL_HOURS * 3600
    last_state = ""
    t_poll = Timer("job wall time")

    while time.time() < deadline:
        rc, state, _ = ssh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null || echo GONE")
        state = state.strip()

        if state != last_state:
            elapsed_min = t_poll.elapsed() / 60
            log(f"  [{state}] job {job_id} — {elapsed_min:.0f}min elapsed")
            last_state = state

        if not state or state == "GONE":
            rc2, sacct, _ = ssh(f"sacct -j {job_id} --noheader --format=State | head -1")
            final = sacct.strip().split()[0] if sacct.strip() else "COMPLETED"
            if final not in ("COMPLETED", "COMPLETING"):
                raise RuntimeError(f"Job {job_id} ended with state: {final}")
            break

        if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            raise RuntimeError(f"Job {job_id} failed: {state}")

        time.sleep(POLL_INTERVAL)
    else:
        raise RuntimeError(f"Timeout waiting for job {job_id} after {MAX_POLL_HOURS}h")

    t_poll.done(f"job {job_id} COMPLETED")
    return job_id


def phase4_rsync_results() -> None:
    log("\n══ Phase 4: Rsync Results ← NIBI ══")

    # Check SIMULATION_DONE sentinel
    rc, sentinel, _ = ssh(f"cat {NIBI_RUN_ROOT}/SIMULATION_DONE 2>/dev/null || echo MISSING")
    if "MISSING" in sentinel:
        log("  WARNING: SIMULATION_DONE sentinel not found — rsync anyway")
    else:
        log(f"  Sentinel: {sentinel.splitlines()[0]}")

    t = Timer("rsync run_root/ ← NIBI")
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rsync_from(NIBI_RUN_ROOT, LOCAL_MODEL_DIR, timeout=600)

    # Count step dirs
    steps = sorted(LOCAL_MODEL_DIR.glob("step_*"))
    t.done(f"{len(steps)} step dirs synced to {LOCAL_MODEL_DIR}")


def print_report() -> None:
    log("\n" + "=" * 60)
    log(" SIMULATION REPORT — April 7, 2026 Intraday Warm-Refresh")
    log("=" * 60)
    log(f" Log      : {LOG_FILE}")
    log(f" Artifacts: {LOCAL_MODEL_DIR}")

    progress_path = LOCAL_MODEL_DIR / "simulation_progress.json"
    if not progress_path.exists():
        log(" (no simulation_progress.json found)")
        log("=" * 60)
        return

    progress = json.loads(progress_path.read_text())
    log(f" Status   : {progress.get('status')}")
    log(f" Base train: {progress.get('base_train_sec', '?')}s")
    log("")
    log(f" {'Win':>3}  {'ET':>5}  {'Train':>7}  {'Total':>7}  Status")
    log(f" {'---':>3}  {'-----':>5}  {'-------':>7}  {'-------':>7}  ------")

    total_train = total_wall = 0.0
    steps = progress.get("steps", [])
    for s in steps:
        ok = "✓" if s["status"] == "ok" else "✗"
        train = s.get("train_sec") or 0
        total = s.get("total_sec") or 0
        log(f" {s['step']:>3}  {s['et_label']:>5}  {train:>6.0f}s  {total:>6.0f}s  {ok}")
        total_train += train
        total_wall  += total

    n = len(steps)
    if n > 0:
        log(f" {'---':>3}  {'-----':>5}  {'-------':>7}  {'-------':>7}")
        log(f" {'TOT':>3}  {'':>5}  {total_train:>6.0f}s  {total_wall:>6.0f}s")
        log(f" {'AVG':>3}  {'':>5}  {total_train/n:>6.0f}s  {total_wall/n:>6.0f}s")

    log("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",      action="store_true", help="Phase 0 only")
    p.add_argument("--skip-setup",   action="store_true", help="Skip Phase 1 (code + model already on NIBI)")
    p.add_argument("--skip-extract", action="store_true", help="Skip Phase 2 (parquet already local + on NIBI)")
    p.add_argument("--skip-base",    action="store_true", help="Skip base train inside NIBI job")
    p.add_argument("--fast",         action="store_true", help="200-tree base train (flow test)")
    p.add_argument("--start-window", type=int, default=0,  help="Resume warm refresh from window N (0-25)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log(f"Market Day Simulation — {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log(f"NIBI     : {NIBI_USER}@{NIBI_HOST}")
    log(f"SIM_DIR  : {NIBI_SIM_DIR}")
    log(f"Log      : {LOG_FILE}")

    try:
        phase0_health_check()
        if args.dry_run:
            log("\n[dry-run] Health check passed. Exiting.")
            return

        phase1_setup_nibi(skip=args.skip_setup)

        if not args.skip_extract:
            phase2_extract_parquet()
            phase2b_scp_parquet()
        else:
            log("\n══ Phase 2: SKIPPED (--skip-extract) ══")

        job_id = phase3_submit_and_poll(
            fast=args.fast,
            skip_base=args.skip_base,
            start_window=args.start_window,
        )

        phase4_rsync_results()
        print_report()

    except KeyboardInterrupt:
        log("\n[INTERRUPTED]")
        print_report()
        sys.exit(1)
    except Exception as exc:
        log(f"\n[FAILED] {exc}")
        print_report()
        sys.exit(1)
    finally:
        _log_fh.close()


if __name__ == "__main__":
    main()

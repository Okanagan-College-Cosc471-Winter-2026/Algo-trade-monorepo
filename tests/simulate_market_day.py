#!/usr/bin/env python3
"""
simulate_market_day.py — Full pipeline simulation: base train → April 7 intraday warm-refresh.

Flow
────
  Phase 0  NIBI health check (SSH + sbatch)
  Phase 1  Setup NIBI (rsync repo to scratch)
  Phase 2  Extract parquet from old DB (2024-03-25 → 2026-04-07)
  Phase 3  SCP parquet to NIBI
  Phase 4  BASE TRAIN on NIBI (data up to April 6 cutoff)
             sbatch train_base.sbatch --parquet ... [--fast]
             Polls until COMPLETED, rsyncs base model back
  Phase 5  For each of the 26 April 7 bars (09:30–15:45 ET):
             A  Slice parquet up to this bar's close time
             B  SCP slice to NIBI
             C  sbatch warm_refresh.sbatch (+30 trees on previous model)
             D  Poll squeue until COMPLETED
             E  rsync updated model back → local step_XX/
             F  Log timing for this window
  Phase 6  Print full timing report (per-window table + totals)

Usage (from repo root, after: bash ml/ml/nibi/morning_login.sh):
    python tests/simulate_market_day.py              # full run (base + 26 windows)
    python tests/simulate_market_day.py --fast       # 200-tree base (flow test, ~20 min)
    python tests/simulate_market_day.py --dry-run    # Phase 0 only
    python tests/simulate_market_day.py --skip-setup --skip-base  # resume: warm refresh only
    python tests/simulate_market_day.py --start-window 5          # resume warm refresh at window 5

Env vars:
    NIBI_USER, NIBI_SCRATCH  (from .env)
    OLD_DB_HOST, OLD_DB_PORT, OLD_DB_NAME, OLD_DB_USER, OLD_DB_PASSWORD
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
NIBI_ALIAS   = "nibi"
NIBI_USER    = os.getenv("NIBI_USER", "harshsaw")
NIBI_HOST    = os.getenv("NIBI_HOST", "nibi.sharcnet.ca")
NIBI_SCRATCH = os.getenv("NIBI_SCRATCH", "/scratch/harshsaw")

OLD_DB_HOST  = os.getenv("OLD_DB_HOST", "localhost")
OLD_DB_PORT  = int(os.getenv("OLD_DB_PORT", "5432"))
OLD_DB_NAME  = os.getenv("OLD_DB_NAME", "market_data")
OLD_DB_USER  = os.getenv("OLD_DB_USER", "mluser")
OLD_DB_PASS  = os.getenv("OLD_DB_PASSWORD", "mlpassword")

# Local paths
LOCAL_DATASETS   = REPO_ROOT / "datasets"
FULL_PARQUET     = LOCAL_DATASETS / "snapshot_2026-04-07.parquet"
SLICE_DIR        = LOCAL_DATASETS / "slices"
LOCAL_MODEL_DIR  = REPO_ROOT / "model_artifacts" / "sim_2026-04-07"
CURRENT_LINK     = REPO_ROOT / "model_artifacts" / "current_base"

# NIBI paths
NIBI_ALGO_DIR        = f"{NIBI_SCRATCH}/algo"
NIBI_RUN_ROOT        = f"{NIBI_SCRATCH}/ml/run_root"
NIBI_SLICE_DIR       = f"{NIBI_SCRATCH}/data/slices"
NIBI_FULL_PARQUET    = f"{NIBI_SCRATCH}/data/snapshot_2026-04-07.parquet"
NIBI_SBATCH          = f"{NIBI_ALGO_DIR}/ml/ml/nibi/warm_refresh.sbatch"
NIBI_BASE_SBATCH     = f"{NIBI_ALGO_DIR}/ml/ml/nibi/train_base.sbatch"

# April 7 windows: 09:30–15:45 ET = 13:30–19:45 UTC, every 15 min (26 slots)
_apr7_utc_start = dt.datetime(2026, 4, 7, 13, 30, 0, tzinfo=dt.timezone.utc)
WINDOWS = [_apr7_utc_start + dt.timedelta(minutes=15 * i) for i in range(26)]

POLL_INTERVAL   = 30    # seconds between squeue checks
MAX_POLL_MIN    = 60    # abort if job not done in 60 min


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
_window_timings: list[dict] = []
_phase_timings:  list[dict] = []

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


def rsync_from(remote: str, local: Path, timeout: int = 300) -> None:
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
    _phase_timings.append({"phase": "health_check", "sec": t.done()})


def phase1_setup_nibi(skip: bool = False) -> None:
    log("\n══ Phase 1: Setup NIBI ══")

    if skip:
        log("  [skip] --skip-setup set")
        return

    # Rsync ml code
    t = Timer("rsync ml code → NIBI")
    rsync_to(REPO_ROOT / "ml" / "ml", f"{NIBI_ALGO_DIR}/ml/ml")
    _phase_timings.append({"phase": "rsync_code", "sec": t.done()})

    # Rsync base model (trained up to April 6) as run_root/current/
    base_src = Path("/data/projects/the-project-maverick/model_artifacts/base_2026-04-07")
    if not base_src.exists():
        raise RuntimeError(f"Base model not found: {base_src}")

    t = Timer("rsync base model → NIBI run_root/current/")
    ssh(f"mkdir -p {NIBI_RUN_ROOT}/current/models")
    rsync_to(base_src / "models", f"{NIBI_RUN_ROOT}/current/models")
    scp_to(str(base_src / "metadata.json"), f"{NIBI_RUN_ROOT}/current/metadata.json")
    scp_to(str(base_src / "feature_names.json"), f"{NIBI_RUN_ROOT}/current/feature_names.json")
    sec = t.done("1157 trees, cutoff=2026-04-06")
    _phase_timings.append({"phase": "rsync_base_model", "sec": sec})

    # Verify
    rc, count, _ = ssh(f"ls {NIBI_RUN_ROOT}/current/models/*.json | wc -l")
    log(f"  Model files on NIBI: {count}")

    # Create slice dir
    ssh(f"mkdir -p {NIBI_SLICE_DIR}")


def phase2_scp_parquet() -> None:
    log("\n══ Phase 2b: SCP parquet → NIBI ══")
    local_size = FULL_PARQUET.stat().st_size
    rc, remote_size, _ = ssh(f"stat -c%s {NIBI_FULL_PARQUET} 2>/dev/null || echo 0")
    if rc == 0 and remote_size.strip().isdigit() and int(remote_size.strip()) == local_size:
        log(f"  Remote matches local ({local_size/1e6:.1f}MB) — skipping")
        _phase_timings.append({"phase": "scp_parquet", "sec": 0, "note": "skipped"})
        return
    t = Timer("scp full parquet")
    ssh(f"mkdir -p {NIBI_SCRATCH}/data")
    scp_to(str(FULL_PARQUET), NIBI_FULL_PARQUET, timeout=600)
    _phase_timings.append({"phase": "scp_parquet", "sec": t.done(f"{local_size/1e6:.1f}MB")})


def phase3_base_train(fast: bool = False) -> None:
    log(f"\n══ Phase 3: Base Train on NIBI ({'FAST 200 trees' if fast else 'FULL 1157 trees'}) ══")

    fast_flag = "--fast" if fast else ""
    t_submit = Timer("sbatch train_base")
    rc, out, err = ssh(
        f"sbatch {NIBI_BASE_SBATCH} --parquet {NIBI_FULL_PARQUET} {fast_flag}"
    )
    if rc != 0:
        raise RuntimeError(f"sbatch train_base failed: {err}")
    job_id = next((tok for tok in out.split() if tok.isdigit()), None)
    if not job_id:
        raise RuntimeError(f"No job ID from: {out!r}")
    t_submit.done(f"job_id={job_id}")

    # Poll — base train takes longer, check every 60s
    log(f"  Polling job {job_id} (this will take {'~20 min' if fast else '~2-3 hrs'})...")
    max_poll = 30 * 60 if fast else 240 * 60
    deadline = time.time() + max_poll
    last_state = ""
    t_poll = Timer("base train wall time")

    while time.time() < deadline:
        rc, state, _ = ssh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null || echo GONE")
        state = state.strip()
        if state != last_state:
            log(f"  [{state}] job {job_id} — elapsed {t_poll.elapsed():.0f}s")
            last_state = state
        if not state or state == "GONE":
            rc2, sacct, _ = ssh(f"sacct -j {job_id} --noheader --format=State | head -1")
            final = sacct.strip().split()[0] if sacct.strip() else "COMPLETED"
            if final not in ("COMPLETED", "COMPLETING"):
                raise RuntimeError(f"Base train job {job_id} ended: {final}")
            break
        if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            raise RuntimeError(f"Base train job {job_id} failed: {state}")
        time.sleep(60)
    else:
        raise RuntimeError(f"Base train timeout after {max_poll//60} min")

    wall_sec = t_poll.done(f"job {job_id} COMPLETED")

    # Rsync base model back
    t = Timer("rsync base model ← NIBI")
    base_local = REPO_ROOT / "model_artifacts" / "base_2026-04-06_nibi"
    rsync_from(f"{NIBI_RUN_ROOT}/current", base_local)
    horizon_count = len(list((base_local / "models").glob("horizon_*.json")))
    t.done(f"{horizon_count} horizons, wall={wall_sec:.0f}s")

    _phase_timings.append({"phase": "base_train", "sec": wall_sec,
                            "note": f"job={job_id} {'fast' if fast else 'full'}"})
    log(f"  Base model saved locally: {base_local}")


def phase2_extract_parquet() -> None:
    log("\n══ Phase 2: Extract parquet from DB ══")
    LOCAL_DATASETS.mkdir(parents=True, exist_ok=True)

    if FULL_PARQUET.exists():
        mb = FULL_PARQUET.stat().st_size / 1e6
        log(f"  Parquet already exists ({mb:.1f} MB) — reusing")
        _phase_timings.append({"phase": "extract_parquet", "sec": 0, "note": "reused"})
        return

    t = Timer("extract ml.market_data_15m → parquet")
    from sqlalchemy import create_engine, text
    import pyarrow as pa, pyarrow.parquet as pq

    engine = create_engine(
        f"postgresql+psycopg2://{OLD_DB_USER}:{OLD_DB_PASS}@"
        f"{OLD_DB_HOST}:{OLD_DB_PORT}/{OLD_DB_NAME}"
    )
    log(f"  Querying {OLD_DB_NAME} on {OLD_DB_HOST}:{OLD_DB_PORT} ...")
    with engine.connect() as conn:
        df = pd.read_sql(text(
            "SELECT * FROM ml.market_data_15m ORDER BY symbol, window_ts"
        ), conn)
    engine.dispose()

    pq.write_table(pa.Table.from_pandas(df), FULL_PARQUET)
    mb = FULL_PARQUET.stat().st_size / 1e6
    sec = t.done(f"{len(df):,} rows, {df['symbol'].nunique()} symbols, {mb:.1f}MB")
    _phase_timings.append({"phase": "extract_parquet", "sec": sec})


def make_slice(window_ts: dt.datetime, df_full: pd.DataFrame) -> Path:
    """Slice: all data up to and including this window's bar."""
    SLICE_DIR.mkdir(parents=True, exist_ok=True)
    label = window_ts.strftime("%H%M")
    path  = SLICE_DIR / f"slice_apr7_{label}.parquet"

    # Include all history up to April 6 PLUS April 7 bars up to this window
    cutoff = window_ts
    mask   = df_full["window_ts"] <= cutoff
    sliced = df_full[mask].copy()

    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(sliced), path)
    return path


def run_window(idx: int, window_ts: dt.datetime, df_full: pd.DataFrame) -> dict:
    """Execute one 15-min warm-refresh cycle. Returns timing dict."""
    et_time = (window_ts - dt.timedelta(hours=4)).strftime("%H:%M")   # approx ET
    label   = window_ts.strftime("%H%M")
    log(f"\n── Window {idx:02d}/25 — {et_time} ET ({window_ts.strftime('%H:%M')} UTC) ──")

    timing = {
        "window": idx,
        "window_ts": window_ts.isoformat(),
        "et_time": et_time,
        "scp_sec": 0.0, "queue_sec": 0.0,
        "train_sec": 0.0, "rsync_sec": 0.0,
        "total_sec": 0.0, "job_id": "",
        "status": "ok",
    }
    wall_t0 = time.time()

    try:
        # A — Slice parquet
        t = Timer(f"  slice")
        slice_path = make_slice(window_ts, df_full)
        slice_mb   = slice_path.stat().st_size / 1e6
        t.done(f"{slice_mb:.1f}MB, rows={len(pd.read_parquet(slice_path)):,}")

        # B — SCP slice
        t = Timer(f"  scp")
        remote_slice = f"{NIBI_SLICE_DIR}/slice_apr7_{label}.parquet"
        scp_to(str(slice_path), remote_slice, timeout=120)
        timing["scp_sec"] = t.done(f"{slice_mb:.1f}MB → NIBI")

        # C — Submit job
        t = Timer(f"  sbatch")
        rc, out, err = ssh(f"sbatch {NIBI_SBATCH} --parquet {remote_slice}")
        if rc != 0:
            raise RuntimeError(f"sbatch failed: {err}")
        job_id = next((tok for tok in out.split() if tok.isdigit()), None)
        if not job_id:
            raise RuntimeError(f"No job ID in: {out!r}")
        timing["job_id"] = job_id
        t.done(f"job_id={job_id}")

        # D — Poll
        t_queue = Timer(f"  queue_wait")
        t_train_start = None
        deadline = time.time() + MAX_POLL_MIN * 60
        last_state = ""

        while time.time() < deadline:
            rc, state_out, _ = ssh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null || echo GONE")
            state = state_out.strip()

            if state in ("RUNNING", "COMPLETING") and t_train_start is None:
                timing["queue_sec"] = t_queue.elapsed()
                log(f"  [RUNNING] job {job_id} — queue wait: {timing['queue_sec']:.0f}s")
                t_train_start = time.time()

            if not state or state == "GONE":
                rc2, sacct_out, _ = ssh(
                    f"sacct -j {job_id} --noheader --format=State | head -1"
                )
                final = sacct_out.strip().split()[0] if sacct_out.strip() else "COMPLETED"
                if final in ("COMPLETED", "COMPLETING"):
                    break
                raise RuntimeError(f"Job {job_id} ended: {final}")

            if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                raise RuntimeError(f"Job {job_id} failed: {state}")

            if state != last_state:
                log(f"  [{state}] job {job_id}...")
                last_state = state

            time.sleep(POLL_INTERVAL)
        else:
            raise RuntimeError(f"Timeout waiting for job {job_id}")

        if t_train_start:
            timing["train_sec"] = round(time.time() - t_train_start, 1)
            log(f"  ✓ GPU train: {timing['train_sec']:.1f}s")
        if timing["queue_sec"] == 0:          # job was instant (no RUNNING state seen)
            timing["queue_sec"] = t_queue.elapsed()

        # E — Rsync model back
        t = Timer(f"  rsync model")
        step_local = LOCAL_MODEL_DIR / f"step_{idx:02d}"
        rsync_from(f"{NIBI_RUN_ROOT}/current", step_local)
        horizon_count = len(list((step_local / "models").glob("horizon_*.json")))
        timing["rsync_sec"] = t.done(f"{horizon_count} horizon files → local")

        # Verify trees grew
        meta_path = step_local / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            trees = meta.get("base_trees") or meta.get("n_estimators") or "?"
            log(f"  total trees now: {trees}")

    except Exception as exc:
        log(f"  [ERROR] window {idx}: {exc}")
        timing["status"] = f"error: {exc}"

    timing["total_sec"] = round(time.time() - wall_t0, 1)
    _window_timings.append(timing)
    return timing


def print_report() -> None:
    log("\n" + "=" * 70)
    log(" SIMULATION REPORT — April 7, 2026 Intraday Warm-Refresh")
    log("=" * 70)
    log(f" Log file : {LOG_FILE}")
    log(f" Artifacts: {LOCAL_MODEL_DIR}")
    log("")
    log(f" {'Win':>3}  {'ET':>5}  {'SCP':>6}  {'Queue':>6}  {'Train':>6}  {'Rsync':>6}  {'Total':>7}  {'Job ID':>12}  Status")
    log(f" {'---':>3}  {'-----':>5}  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}  {'-------':>7}  {'--------':>12}  ------")

    total_scp = total_queue = total_train = total_rsync = total_wall = 0.0
    for t in _window_timings:
        status = "✓" if t["status"] == "ok" else "✗"
        log(
            f" {t['window']:>3}  {t['et_time']:>5}  "
            f"{t['scp_sec']:>5.0f}s  {t['queue_sec']:>5.0f}s  "
            f"{t['train_sec']:>5.0f}s  {t['rsync_sec']:>5.0f}s  "
            f"{t['total_sec']:>6.0f}s  {t['job_id']:>12}  {status}"
        )
        total_scp   += t["scp_sec"]
        total_queue += t["queue_sec"]
        total_train += t["train_sec"]
        total_rsync += t["rsync_sec"]
        total_wall  += t["total_sec"]

    n = len(_window_timings)
    if n > 0:
        log(f" {'':>3}  {'':>5}  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}  {'-------':>7}")
        log(
            f" {'TOT':>3}  {'':>5}  "
            f"{total_scp:>5.0f}s  {total_queue:>5.0f}s  "
            f"{total_train:>5.0f}s  {total_rsync:>5.0f}s  "
            f"{total_wall:>6.0f}s"
        )
        log(
            f" {'AVG':>3}  {'':>5}  "
            f"{total_scp/n:>5.0f}s  {total_queue/n:>5.0f}s  "
            f"{total_train/n:>5.0f}s  {total_rsync/n:>5.0f}s  "
            f"{total_wall/n:>6.0f}s"
        )

    log("")
    log(" Phase overhead (outside windows):")
    for p in _phase_timings:
        log(f"   {p['phase']:<30} {p['sec']:>6.1f}s")
    log("=" * 70)

    # Save report JSON
    report = {
        "replay_date": "2026-04-07",
        "windows": _window_timings,
        "phases": _phase_timings,
        "generated_at": dt.datetime.utcnow().isoformat(),
    }
    report_path = LOCAL_MODEL_DIR / "simulation_timing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    log(f" Report saved: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",      action="store_true", help="Phase 0 only")
    p.add_argument("--skip-setup",   action="store_true", help="Skip Phase 1 (repo already on NIBI)")
    p.add_argument("--skip-extract", action="store_true", help="Skip extract (parquet already local)")
    p.add_argument("--skip-base",    action="store_true", help="Skip Phase 3 base train (model already on NIBI)")
    p.add_argument("--fast",         action="store_true", help="Base train: 200 trees (flow test)")
    p.add_argument("--start-window", type=int, default=0,  help="Resume warm refresh at window index (0-25)")
    p.add_argument("--end-window",   type=int, default=25, help="Stop after window index (default: 25)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log(f"Market Day Simulation — {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log(f"NIBI  : {NIBI_USER}@{NIBI_HOST}  scratch={NIBI_SCRATCH}")
    log(f"Log   : {LOG_FILE}")
    log(f"Windows: {args.start_window}–{args.end_window} of 0–25")

    try:
        phase0_health_check()
        if args.dry_run:
            log("\n[dry-run] Health check passed. Exiting.")
            return

        phase1_setup_nibi(skip=args.skip_setup)

        if not args.skip_extract:
            phase2_extract_parquet()
            phase2_scp_parquet()
        else:
            log("\n══ Phase 2: SKIPPED ══")

        if not args.skip_base:
            phase3_base_train(fast=args.fast)
        else:
            log("\n══ Phase 3: Base train SKIPPED ══")

        # Load full parquet once into memory for slicing
        log(f"\nLoading full parquet into memory ...")
        t = Timer("load parquet")
        df_full = pd.read_parquet(FULL_PARQUET)
        df_full["window_ts"] = pd.to_datetime(df_full["window_ts"], utc=True)
        t.done(f"{len(df_full):,} rows")

        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Run each window
        windows_to_run = WINDOWS[args.start_window: args.end_window + 1]
        log(f"\nStarting {len(windows_to_run)} warm-refresh cycles...")

        for i, window_ts in enumerate(windows_to_run):
            global_idx = args.start_window + i
            run_window(global_idx, window_ts, df_full)

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

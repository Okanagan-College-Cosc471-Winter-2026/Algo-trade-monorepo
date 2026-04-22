#!/usr/bin/env python3
"""
nibi_orchestrator.py — Scheduled NIBI job orchestrator.

Runs from the scheduler container (or host) as part of the nightly pipeline:
  1. Extract ml.market_data_15m → local parquet snapshot
  2. SCP parquet → NIBI test_simulation/data/
  3. Rsync existing base model → NIBI run_root/current/
  4. Submit simulate_full_day.sbatch with --skip-base (warm refreshes only)
  5. Log job_id for monitoring

The base model (1157 trees) is trained separately and lives in
MODEL_ARTIFACTS_DIR/base_<date>/. The daily job only does warm refreshes.

Called by scheduler crontab at 06:00 ET (10:00 UTC) Mon–Fri.
SSH ControlMaster must already be open (morning_login.sh run before market open).

Env vars (from .env):
    NIBI_USER, NIBI_HOST, NIBI_SIM_DIR, NIBI_SSH_KEY
    OLD_DB_HOST, OLD_DB_PORT, OLD_DB_NAME, OLD_DB_USER, OLD_DB_PASSWORD
    DATASETS_DIR  (local dir to write parquet, default: ./datasets)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
NIBI_ALIAS    = "nibi"
NIBI_USER     = os.getenv("NIBI_USER",    "harshsaw")
NIBI_HOST     = os.getenv("NIBI_HOST",    "nibi.sharcnet.ca")
NIBI_SIM_DIR  = os.getenv("NIBI_SIM_DIR", "/home/harshsaw/projects/def-youry/test_simulation")

OLD_DB_HOST   = os.getenv("OLD_DB_HOST",     "localhost")
OLD_DB_PORT   = int(os.getenv("OLD_DB_PORT", "5432"))
OLD_DB_NAME   = os.getenv("OLD_DB_NAME",     "market_data")
OLD_DB_USER   = os.getenv("OLD_DB_USER",     "mluser")
OLD_DB_PASS   = os.getenv("OLD_DB_PASSWORD", "mlpassword")

DATASETS_DIR  = Path(os.getenv("DATASETS_DIR", "./datasets"))

NIBI_DATA_DIR     = f"{NIBI_SIM_DIR}/data"
NIBI_RUN_ROOT     = f"{NIBI_SIM_DIR}/run_root"
NIBI_FULL_SBATCH  = f"{NIBI_SIM_DIR}/ml/ml/nibi/simulate_full_day.sbatch"

# Local base model — trained separately, rsynced to NIBI before each daily job
BASE_MODEL_DIR    = Path(os.getenv(
    "BASE_MODEL_DIR",
    "/data/projects/the-project-maverick/model_artifacts/base_2026-04-07"
))

# ── Logging ───────────────────────────────────────────────────────
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"nibi_{dt.datetime.utcnow().strftime('%Y%m%d')}.log"),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────
def ssh(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", NIBI_ALIAS, cmd],
        capture_output=True, text=True, timeout=timeout + 5,
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def scp_to(local: Path, remote: str, timeout: int = 600) -> None:
    r = subprocess.run(
        ["scp", "-o", "BatchMode=yes", str(local), f"{NIBI_ALIAS}:{remote}"],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"SCP failed: {r.stderr.strip()}")


def rsync_to(local: Path, remote: str, timeout: int = 300) -> None:
    r = subprocess.run(
        ["rsync", "-az", "-e", "ssh -o BatchMode=yes",
         str(local) + "/", f"{NIBI_ALIAS}:{remote}/"],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr.strip()}")


# ── Steps ─────────────────────────────────────────────────────────
def check_nibi() -> None:
    log.info("=== Step 1: NIBI health check ===")
    rc, out, err = ssh("echo pong && sbatch --version | head -1")
    if rc != 0:
        raise RuntimeError(f"NIBI SSH failed: {err} — run morning_login.sh first")
    log.info("  NIBI OK: %s", out.replace("\n", " | "))


def export_parquet(sim_date: str) -> Path:
    log.info("=== Step 2: Export parquet from DB ===")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = DATASETS_DIR / f"snapshot_{sim_date}.parquet"

    if parquet_path.exists():
        mb = parquet_path.stat().st_size / 1e6
        log.info("  Parquet already exists (%.1f MB) — reusing", mb)
        return parquet_path

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    engine = create_engine(
        f"postgresql+psycopg2://{OLD_DB_USER}:{OLD_DB_PASS}@"
        f"{OLD_DB_HOST}:{OLD_DB_PORT}/{OLD_DB_NAME}"
    )
    log.info("  Querying ml.market_data_15m from %s:%d/%s ...", OLD_DB_HOST, OLD_DB_PORT, OLD_DB_NAME)

    chunks = []
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM ml.market_data_15m")).scalar()
        log.info("  Total rows: %s", f"{total:,}")
        for chunk in pd.read_sql(
            text("SELECT * FROM ml.market_data_15m ORDER BY symbol, window_ts"),
            conn, chunksize=200_000,
        ):
            chunks.append(chunk)
            log.info("  Loaded %s / %s rows ...", f"{sum(len(c) for c in chunks):,}", f"{total:,}")
    engine.dispose()

    df = pd.concat(chunks, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(df), parquet_path)
    mb = parquet_path.stat().st_size / 1e6
    log.info("  Saved: %s  (%.1f MB, %s symbols)", parquet_path, mb, df['symbol'].nunique())
    return parquet_path


def sync_base_model() -> None:
    """Rsync pre-trained base model to NIBI. Used when skipping base training."""
    log.info("=== Step 3: Rsync base model → NIBI ===")
    if not BASE_MODEL_DIR.exists():
        raise RuntimeError(f"Base model not found: {BASE_MODEL_DIR}")

    meta = BASE_MODEL_DIR / "metadata.json"
    if meta.exists():
        import json as _json
        m = _json.loads(meta.read_text())
        n_trees = m.get("n_estimators") or m.get("base_trees") or "?"
        log.info("  Base model: %s trees, cutoff=%s", n_trees, m.get("train_end_date", "?"))

    ssh(f"mkdir -p {NIBI_RUN_ROOT}/current/models")
    rsync_to(BASE_MODEL_DIR / "models", f"{NIBI_RUN_ROOT}/current/models")
    scp_to(BASE_MODEL_DIR / "metadata.json",      f"{NIBI_RUN_ROOT}/current/metadata.json")
    scp_to(BASE_MODEL_DIR / "feature_names.json", f"{NIBI_RUN_ROOT}/current/feature_names.json")
    rc, count, _ = ssh(f"ls {NIBI_RUN_ROOT}/current/models/*.json | wc -l")
    log.info("  Done — %s horizon files on NIBI", count.strip())


def send_to_nibi(parquet_path: Path) -> str:
    log.info("=== Step 2b: SCP parquet → NIBI ===")
    remote_path = f"{NIBI_DATA_DIR}/{parquet_path.name}"

    # Skip if sizes match
    local_size = parquet_path.stat().st_size
    rc, remote_size, _ = ssh(f"stat -c%s {remote_path} 2>/dev/null || echo 0")
    if rc == 0 and remote_size.strip().isdigit() and int(remote_size.strip()) == local_size:
        log.info("  Remote already matches (%.1f MB) — skipping", local_size / 1e6)
        return remote_path

    ssh(f"mkdir -p {NIBI_DATA_DIR}")
    log.info("  Sending %.1f MB ...", local_size / 1e6)
    scp_to(parquet_path, remote_path)
    log.info("  Done → %s", remote_path)
    return remote_path


def submit_job(remote_parquet: str, sim_date: str, skip_base: bool = False) -> str:
    log.info("=== Step 4: Submit NIBI job ===")
    skip_flag = "--skip-base" if skip_base else ""
    cmd = (
        f"sbatch {NIBI_FULL_SBATCH} "
        f"--parquet {remote_parquet} "
        f"--sim-date {sim_date} "
        f"{skip_flag}"
    ).strip()

    log.info("  %s", cmd)
    rc, out, err = ssh(cmd, timeout=30)
    if rc != 0:
        raise RuntimeError(f"sbatch failed: {err}")
    job_id = next((tok for tok in out.split() if tok.isdigit()), None)
    if not job_id:
        raise RuntimeError(f"No job ID in: {out!r}")
    log.info("  Job submitted: %s", job_id)
    return job_id


def save_job_record(job_id: str, sim_date: str, parquet_path: Path) -> None:
    record = {
        "job_id": job_id,
        "sim_date": sim_date,
        "parquet": str(parquet_path),
        "submitted_at": dt.datetime.utcnow().isoformat(),
        "status": "submitted",
    }
    record_path = LOG_DIR / f"nibi_job_{sim_date}.json"
    record_path.write_text(json.dumps(record, indent=2))
    log.info("  Job record: %s", record_path)


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sim-date",   default=None,
                   help="Simulation date YYYY-MM-DD (default: today)")
    p.add_argument("--skip-base", action="store_true",
                   help="Skip base train — rsync existing model and go straight to warm refreshes")
    args = p.parse_args()

    sim_date = args.sim_date or dt.date.today().isoformat()
    log.info("NIBI Orchestrator — sim_date=%s  base_model=%s", sim_date, BASE_MODEL_DIR)

    check_nibi()
    parquet_path   = export_parquet(sim_date)
    remote_parquet = send_to_nibi(parquet_path)
    if args.skip_base:
        sync_base_model()
    else:
        log.info("=== Step 3: Base train will run on NIBI (no pre-existing model) ===")
    job_id = submit_job(remote_parquet, sim_date, skip_base=args.skip_base)
    save_job_record(job_id, sim_date, parquet_path)

    log.info("=== Done — job %s queued on NIBI ===", job_id)
    log.info("Monitor: ssh nibi squeue -j %s", job_id)


if __name__ == "__main__":
    main()

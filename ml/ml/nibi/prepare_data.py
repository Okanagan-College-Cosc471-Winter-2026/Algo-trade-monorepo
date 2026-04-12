#!/usr/bin/env python3
"""
prepare_data.py — Extract ml.market_data_15m → Parquet, print scp command.

Run this locally (inside the Docker network) before manually submitting
the NIBI warm-refresh job.

Usage:
    python ml/ml/nibi/prepare_data.py [--days N] [--out-dir /path/to/datasets]

Env vars (from .env or Docker):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    NIBI_USER, NIBI_HOST, NIBI_SCRATCH

Output:
    /datasets/snapshot_YYYY-MM-DD.parquet
    Prints the scp command to copy the file to NIBI.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def build_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db   = os.getenv("DB_NAME", "algotrade")
    user = os.getenv("DB_USER", "appuser")
    pw   = os.getenv("DB_PASSWORD", "")
    return create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}")


EXTRACT_SQL = """
SELECT *
FROM ml.market_data_15m
WHERE trade_date >= :start_date
ORDER BY symbol, window_ts
"""


def extract(engine, days: int) -> pd.DataFrame:
    start_date = (date.today() - timedelta(days=days)).isoformat()
    print(f"[extract] Pulling ml.market_data_15m since {start_date} ...")
    with engine.connect() as conn:
        df = pd.read_sql(text(EXTRACT_SQL), conn, params={"start_date": start_date})
    print(f"[extract] {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df


def save_parquet(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_file = out_dir / f"snapshot_{today}.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_file)
    print(f"[extract] Saved: {out_file}  ({out_file.stat().st_size / 1e6:.1f} MB)")
    return out_file


def print_scp_command(local_path: Path) -> None:
    nibi_user    = os.getenv("NIBI_USER", "yournibiuser")
    nibi_host    = os.getenv("NIBI_HOST", "nibi.ok.ubc.ca")
    nibi_scratch = os.getenv("NIBI_SCRATCH", f"/scratch/{nibi_user}")
    nibi_key     = os.getenv("NIBI_SSH_KEY", "~/.ssh/nibi_key")

    remote_dir  = f"{nibi_scratch}/data/"
    remote_dest = f"{nibi_user}@{nibi_host}:{remote_dir}"

    print()
    print("=" * 60)
    print("NEXT STEP — copy parquet to NIBI then submit the job:")
    print()
    print(f"  scp -i {nibi_key} {local_path} {remote_dest}")
    print()
    print(f"  ssh -i {nibi_key} {nibi_user}@{nibi_host} \\")
    print(f"      \"mkdir -p {nibi_scratch}/data && \\")
    print(f"       sbatch {nibi_scratch}/algo/ml/nibi/warm_refresh.sbatch \\")
    print(f"           --parquet {remote_dir}{local_path.name}\"")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training data → Parquet for NIBI")
    parser.add_argument("--days",    type=int, default=365,
                        help="How many calendar days of history to extract (default: 365)")
    parser.add_argument("--out-dir", default=os.getenv("DATASETS_DIR", "./datasets"),
                        help="Output directory for parquet file")
    args = parser.parse_args()

    t0 = time.time()
    engine = build_engine()

    try:
        df = extract(engine, args.days)
    finally:
        engine.dispose()

    if df.empty:
        print("[extract] ERROR: No rows returned. Is the pipeline running?")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_file = save_parquet(df, out_dir)
    print(f"[extract] Elapsed: {time.time() - t0:.1f}s")

    print_scp_command(out_file)


if __name__ == "__main__":
    main()

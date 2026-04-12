#!/usr/bin/env python3
"""
run_simulation.py — NIBI wrapper for simulate_warm_refresh (April 7 replay).

Called by simulate_april7.sbatch. Runs all 26 intraday warm-refresh steps
for the replay date in a single GPU allocation. Much faster than submitting
26 separate Slurm jobs.

Usage:
    python run_simulation.py \
        --parquet /scratch/$USER/data/snapshot_2026-04-07.parquet \
        --run-root /scratch/$USER/ml/run_root \
        --output-dir /scratch/$USER/ml/simulation_2026-04-07 \
        --replay-date 2026-04-07 \
        [--warm-trees 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

try:
    from ml.XG_boost_3_multigpu_final import simulate_warm_refresh  # type: ignore
except ImportError as exc:
    print(f"ERROR: Cannot import simulate_warm_refresh — {exc}")
    print(f"  Expected XG_boost_3_multigpu_final.py at: {_repo_root / 'ml'}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIBI April 7 simulation wrapper")
    parser.add_argument("--parquet",      required=True, help="Input parquet snapshot path")
    parser.add_argument("--run-root",     required=True, help="Registry root (has current/ with base model)")
    parser.add_argument("--output-dir",   required=True, help="Where to write simulation step artifacts")
    parser.add_argument("--replay-date",  default="2026-04-07", help="Date to replay (default: 2026-04-07)")
    parser.add_argument("--warm-trees",   type=int, default=30, help="Trees to add per step (default: 30)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"ERROR: Parquet not found: {parquet_path}")
        sys.exit(1)

    run_root   = Path(args.run_root)
    output_dir = Path(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sim] parquet      = {parquet_path}")
    print(f"[sim] run_root     = {run_root}")
    print(f"[sim] output_dir   = {output_dir}")
    print(f"[sim] replay_date  = {args.replay_date}")
    print(f"[sim] warm_trees   = {args.warm_trees}")

    # Build namespace matching XG_boost_3_multigpu_final's argparse expectations
    train_args = argparse.Namespace(
        mode="simulate_warm_refresh",
        source_parquet=str(parquet_path),
        run_root=str(run_root),
        simulation_out=str(output_dir),
        replay_date=args.replay_date,
        warm_trees=args.warm_trees,
        warm_refresh_days=1,
        device="cuda",
        seed=42,
        verbose=True,
        log_level="INFO",
        symbols=None,           # use all symbols in parquet
        source_table=None,
        base_window_months=24,
        daily_windows=[5, 10, 20],
        slot_windows=[3, 5, 10],
        winsor_pct=0.01,
    )

    print("[sim] Starting simulate_warm_refresh ...")
    result = simulate_warm_refresh(train_args)

    if result is None or (isinstance(result, dict) and result.get("status") == "error"):
        print(f"[sim] FAILED: {result}")
        sys.exit(1)

    print(f"[sim] Done. Steps written to: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_warm_refresh.py — NIBI wrapper for production_warm_refresh().

Called by warm_refresh.sbatch. Loads the parquet snapshot, points
the main training script at it, and triggers a +30-tree warm-start
on yesterday's data on top of the current base bundle.

Usage:
    python run_warm_refresh.py \
        --parquet /scratch/$USER/data/snapshot_YYYY-MM-DD.parquet \
        --run-root /scratch/$USER/ml/warm_refresh \
        --output-dir /scratch/$USER/ml/warm_refresh/latest \
        [--warm-trees 30] [--warm-refresh-days 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Locate XG_boost_3_multigpu_final.py ───────────────────────────
# Layout on NIBI:
#   /scratch/$USER/algo/
#     ml/ml/nibi/run_warm_refresh.py   ← this file
#     ml/ml/XG_boost_3_multigpu_final.py
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

try:
    from ml.XG_boost_3_multigpu_final import production_warm_refresh  # type: ignore
except ImportError as exc:
    print(f"ERROR: Could not import production_warm_refresh — {exc}")
    print(f"  Expected XG_boost_3_multigpu_final.py at: {_repo_root / 'ml'}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIBI warm-refresh wrapper")
    parser.add_argument("--parquet",           required=True,
                        help="Path to input parquet snapshot")
    parser.add_argument("--run-root",          required=True,
                        help="Root dir for warm-refresh artefacts (run metadata written here)")
    parser.add_argument("--output-dir",        required=True,
                        help="Where to write the promoted bundle (replaces 'latest')")
    parser.add_argument("--warm-trees",        type=int, default=30,
                        help="Trees to add per horizon (default: 30)")
    parser.add_argument("--warm-refresh-days", type=int, default=1,
                        help="Days of recent data to train on (default: 1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"ERROR: Parquet file not found: {parquet_path}")
        sys.exit(1)

    run_root   = Path(args.run_root)
    output_dir = Path(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[warm_refresh] parquet       = {parquet_path}")
    print(f"[warm_refresh] run_root      = {run_root}")
    print(f"[warm_refresh] output_dir    = {output_dir}")
    print(f"[warm_refresh] warm_trees    = {args.warm_trees}")
    print(f"[warm_refresh] refresh_days  = {args.warm_refresh_days}")

    # Build a namespace that matches the argparse args of the main script
    # so production_warm_refresh() can be called directly.
    train_args = argparse.Namespace(
        mode="warm_refresh",
        parquet=str(parquet_path),
        run_root=str(run_root),
        output_dir=str(output_dir),
        warm_trees=args.warm_trees,
        warm_refresh_days=args.warm_refresh_days,
        # Defaults expected by production_warm_refresh:
        device="cuda",
        seed=42,
        verbose=True,
    )

    print("[warm_refresh] Calling production_warm_refresh() ...")
    result = production_warm_refresh(train_args)

    if result is None or (isinstance(result, dict) and result.get("status") == "error"):
        print(f"[warm_refresh] FAILED: {result}")
        sys.exit(1)

    print(f"[warm_refresh] Done. Bundle written to: {output_dir}")


if __name__ == "__main__":
    main()

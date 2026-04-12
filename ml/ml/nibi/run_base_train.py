#!/usr/bin/env python3
"""
run_base_train.py — NIBI wrapper for production_bootstrap().

Trains the base XGBoost model on data up to a cutoff date.
Output lands in run_root/current/ (26 horizon_XX.json files).

Usage:
    python run_base_train.py \
        --parquet /scratch/$USER/data/snapshot_2026-04-06.parquet \
        --run-root /scratch/$USER/ml/run_root \
        --cutoff 2026-04-06 \
        [--fast]   # 200 trees for flow test instead of 1157
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

try:
    from ml.XG_boost_3_multigpu_final import (   # type: ignore
        production_bootstrap,
        BEST_FIXED_VARIANT,
        FAST_REFRESH_VARIANT,
    )
except ImportError as exc:
    print(f"ERROR: Cannot import production_bootstrap — {exc}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NIBI base training wrapper")
    p.add_argument("--parquet",   required=True, help="Input parquet path")
    p.add_argument("--run-root",  required=True, help="Registry root")
    p.add_argument("--cutoff",    default="2026-04-06",
                   help="Training cutoff date — rows after this are excluded")
    p.add_argument("--fast",      action="store_true",
                   help="Use 200 trees (fast flow test) instead of 1157")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"ERROR: Parquet not found: {parquet_path}")
        sys.exit(1)

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    n_trees = 200 if args.fast else BEST_FIXED_VARIANT["n_estimators"]
    print(f"[base_train] parquet   = {parquet_path}")
    print(f"[base_train] run_root  = {run_root}")
    print(f"[base_train] cutoff    = {args.cutoff}")
    print(f"[base_train] n_trees   = {n_trees} ({'FAST' if args.fast else 'FULL'})")

    train_args = argparse.Namespace(
        mode="bootstrap",
        source_parquet=str(parquet_path),
        run_root=str(run_root),
        # Filter data to cutoff
        train_end_date=args.cutoff,
        # Override tree count if fast mode
        n_estimators=n_trees,
        device="cuda",
        seed=42,
        verbose=True,
        log_level="INFO",
        symbols=None,
        source_table=None,
        base_window_months=24,
        daily_windows=[5, 10, 20],
        slot_windows=[3, 5, 10],
        winsor_pct=0.01,
        warm_trees=30,
        warm_refresh_days=1,
    )

    print("[base_train] Starting production_bootstrap() ...")
    result = production_bootstrap(train_args)
    print(f"[base_train] Done — model_id={result.get('model_id')}")


if __name__ == "__main__":
    main()

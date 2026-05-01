#!/usr/bin/env python3
"""
run_simulation_day.py — Full-day simulation runner (executes inside one SLURM job).

Runs on NIBI with a single GPU allocation:
  1. Base train on data up to (sim_date - 1 day)          [skippable]
  2. 26 warm-refresh windows for sim_date (09:30–15:45 ET)
  3. Each window's model snapshot saved to run_root/step_XX/
  4. Progress written to run_root/simulation_progress.json  (live updates)
  5. Sentinel file run_root/SIMULATION_DONE written on success

Layout expected on NIBI after rsync from VM:
  test_simulation/
    ml/ml/
      nibi/run_simulation_day.py   ← this file
      XG_boost_3_multigpu_final.py (two dirs up → test_simulation/ml/)
    data/
      snapshot_2026-04-07.parquet
    run_root/
      current/                     ← live model (base → warm-refreshed each step)
      step_00/ … step_25/          ← snapshots per window

Usage:
    python run_simulation_day.py \\
        --parquet  /home/$USER/projects/def-youry/test_simulation/data/snapshot_2026-04-07.parquet \\
        --run-root /home/$USER/projects/def-youry/test_simulation/run_root \\
        --sim-date 2026-04-07 \\
        [--fast]           # 200-tree base (flow test, ~20 min)
        [--skip-base]      # skip base train — use existing current/
        [--start-window N] # resume warm refresh from window N (0-25)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Locate XG_boost_3_multigpu_final.py ───────────────────────────────────────
# File is at: test_simulation/ml/ml/nibi/run_simulation_day.py
# parents[2]  →  test_simulation/ml/
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

try:
    from ml.XG_boost_3_multigpu_final import (   # type: ignore
        production_bootstrap,
        production_warm_refresh,
        BEST_FIXED_VARIANT,
    )
except ImportError as exc:
    print(f"ERROR: Cannot import XG_boost_3_multigpu_final — {exc}")
    print(f"  Expected at: {_repo_root / 'ml' / 'XG_boost_3_multigpu_final.py'}")
    sys.exit(1)


# ── April 7 windows: 09:30–15:45 ET = 13:30–19:45 UTC, every 15 min ──────────
def build_windows(sim_date: str) -> list[dt.datetime]:
    d = dt.date.fromisoformat(sim_date)
    utc_open = dt.datetime(d.year, d.month, d.day, 13, 30, 0, tzinfo=dt.timezone.utc)
    return [utc_open + dt.timedelta(minutes=15 * i) for i in range(26)]


def slice_parquet(df_full: pd.DataFrame, cutoff_ts: dt.datetime, out_path: Path) -> None:
    mask = df_full["window_ts"] <= cutoff_ts
    sliced = df_full[mask].copy()
    pq.write_table(pa.Table.from_pandas(sliced), out_path)


def snapshot_current(run_root: Path, step_idx: int) -> None:
    """Copy run_root/current/ → run_root/step_XX/ for retrieval by VM."""
    src = run_root / "current"
    dst = run_root / f"step_{step_idx:02d}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_progress(run_root: Path, progress: dict) -> None:
    path = run_root / "simulation_progress.json"
    path.write_text(json.dumps(progress, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",       required=True, help="Full parquet path on NIBI")
    p.add_argument("--run-root",      required=True, help="Model registry root")
    p.add_argument("--sim-date",      default="2026-04-07", help="Simulation date (YYYY-MM-DD)")
    p.add_argument("--fast",          action="store_true", help="200-tree base train (flow test)")
    p.add_argument("--skip-base",     action="store_true", help="Skip base train, use existing current/")
    p.add_argument("--base-only",     action="store_true", help="Run base train only, skip all warm-refresh windows")
    p.add_argument("--start-window",  type=int, default=0, help="Resume at window index (0-25)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    run_root   = Path(args.run_root)
    parquet    = Path(args.parquet)
    slice_dir  = run_root / "slices"
    run_root.mkdir(parents=True, exist_ok=True)
    slice_dir.mkdir(parents=True, exist_ok=True)

    windows = build_windows(args.sim_date)
    cutoff_date = (dt.date.fromisoformat(args.sim_date) - dt.timedelta(days=1)).isoformat()

    progress = {
        "sim_date": args.sim_date,
        "status": "running",
        "base_train_sec": None,
        "steps": [],
        "started_at": dt.datetime.utcnow().isoformat(),
        "finished_at": None,
    }
    write_progress(run_root, progress)

    print(f"{'='*60}")
    print(f" run_simulation_day.py")
    print(f" sim_date   : {args.sim_date}")
    print(f" parquet    : {parquet}")
    print(f" run_root   : {run_root}")
    print(f" windows    : {args.start_window}–25  ({len(windows)} total)")
    print(f" fast mode  : {args.fast}")
    print(f" skip_base  : {args.skip_base}")
    print(f"{'='*60}")

    # ── 1. Base train ──────────────────────────────────────────────
    if not args.skip_base:
        print(f"\n[1/2] Base train (cutoff={cutoff_date}, {'FAST 200 trees' if args.fast else 'FULL'}) ...")
        t0 = time.time()
        n_trees = 200 if args.fast else BEST_FIXED_VARIANT["n_estimators"]
        base_args = argparse.Namespace(
            mode="bootstrap",
            source_parquet=str(parquet),
            run_root=str(run_root),
            output_dir=None,
            run_dir=None,
            train_end_date=cutoff_date,
            n_estimators=n_trees,
            device="cuda",
            random_state=42,
            verbose=True,
            log_level="INFO",
            symbols=None,
            source_table=None,
            base_window_months=24,
            windows_months=[24],
            daily_windows=[5, 10, 20],
            slot_windows=[3, 5, 10],
            winsor_pct=0.01,
            warm_trees=30,
            warm_refresh_days=1,
            as_of_ts=None,
            as_of_date=None,
            start_date=None,
            end_date=None,
            replay_date=None,
            truth_date=None,
            simulation_out=None,
            fast_refresh_days=60,
            parallel_horizons=min(8, __import__("os").cpu_count() or 1),
            n_folds=3,
            test_block_days=5,
            min_train_rows=500,
            report_days_back=6,
            train_profile="base",
            train_policy="simple",
            refresh_budget_sec=780,
            write_reports=False,
            visual_symbol_count=10,
            optuna_trials=None,
            optuna_timeout_min=None,
        )
        result = production_bootstrap(base_args)
        base_sec = round(time.time() - t0, 1)
        print(f"[1/2] Base train done — {base_sec:.0f}s  model_id={result.get('model_id')}")
        progress["base_train_sec"] = base_sec
        write_progress(run_root, progress)
    else:
        print("\n[1/2] Skipping base train — using existing current/")
        progress["base_train_sec"] = 0
        write_progress(run_root, progress)

    # ── 2. Load full parquet once ──────────────────────────────────
    print(f"\n[2/2] Loading parquet into memory ...")
    t0 = time.time()
    df_full = pd.read_parquet(parquet)
    df_full["window_ts"] = pd.to_datetime(df_full["window_ts"], utc=True)
    print(f"  Loaded {len(df_full):,} rows, {df_full['symbol'].nunique()} symbols in {time.time()-t0:.1f}s")

    # ── 3. Warm-refresh loop ───────────────────────────────────────
    if args.base_only:
        print("\n[2/2] --base-only set — skipping warm-refresh windows.")
        progress["status"] = "base_done"
        progress["finished_at"] = dt.datetime.utcnow().isoformat()
        write_progress(run_root, progress)
        print(f"\nBase train complete. Run sim_warm_windows.sbatch to continue.")
        return

    print(f"\nStarting {26 - args.start_window} warm-refresh windows ...\n")

    for idx, window_ts in enumerate(windows):
        if idx < args.start_window:
            continue

        et_label = (window_ts - dt.timedelta(hours=4)).strftime("%H:%M")
        print(f"── Window {idx:02d}/25  {et_label} ET ({window_ts.strftime('%H:%M')} UTC) ──")
        step_t0 = time.time()

        step_info = {
            "step": idx,
            "as_of_ts": window_ts.isoformat(),
            "et_label": et_label,
            "status": "running",
            "train_sec": None,
            "total_sec": None,
        }

        try:
            # Slice parquet up to this window
            slice_path = slice_dir / f"slice_{window_ts.strftime('%H%M')}.parquet"
            slice_t0 = time.time()
            slice_parquet(df_full, window_ts, slice_path)
            slice_mb = slice_path.stat().st_size / 1e6
            print(f"  slice: {slice_mb:.1f}MB  ({time.time()-slice_t0:.1f}s)")

            # Warm refresh
            train_t0 = time.time()
            warm_args = argparse.Namespace(
                mode="warm_refresh",
                source_parquet=str(slice_path),
                run_root=str(run_root),
                output_dir=str(run_root / "current"),
                run_dir=None,
                warm_trees=30,
                warm_refresh_days=1,
                device="cuda",
                random_state=42,
                verbose=True,
                log_level="INFO",
                symbols=None,
                source_table=None,
                base_window_months=24,
                windows_months=[24],
                daily_windows=[5, 10, 20],
                slot_windows=[3, 5, 10],
                winsor_pct=0.01,
                n_estimators=None,
                train_end_date=None,
                as_of_ts=window_ts.isoformat(),
                as_of_date=None,
                start_date=None,
                end_date=None,
                replay_date=None,
                truth_date=None,
                simulation_out=None,
                fast_refresh_days=60,
                parallel_horizons=min(8, __import__("os").cpu_count() or 1),
                n_folds=3,
                test_block_days=5,
                min_train_rows=500,
                report_days_back=6,
                train_profile="warm_refresh",
                train_policy="simple",
                refresh_budget_sec=780,
                write_reports=False,
                visual_symbol_count=10,
                optuna_trials=None,
                optuna_timeout_min=None,
            )
            result = production_warm_refresh(warm_args)
            train_sec = round(time.time() - train_t0, 1)
            print(f"  warm_refresh: {train_sec:.1f}s")

            # Snapshot current/ → step_XX/
            snapshot_current(run_root, idx)
            print(f"  snapshot → step_{idx:02d}/")

            step_info["train_sec"]  = train_sec
            step_info["status"]     = "ok"

        except Exception as exc:
            print(f"  [ERROR] window {idx}: {exc}")
            step_info["status"] = f"error: {exc}"

        step_info["total_sec"] = round(time.time() - step_t0, 1)
        progress["steps"].append(step_info)
        write_progress(run_root, progress)

        ok = "✓" if step_info["status"] == "ok" else "✗"
        print(f"  {ok}  total: {step_info['total_sec']:.1f}s\n")

    # ── Done ───────────────────────────────────────────────────────
    error_steps = [s for s in progress["steps"] if str(s.get("status", "")).startswith("error")]
    ok_steps = [s for s in progress["steps"] if s.get("status") == "ok"]

    progress["finished_at"] = dt.datetime.utcnow().isoformat()
    progress["status"] = "failed" if error_steps else "success"
    write_progress(run_root, progress)

    print(f"\n{'='*60}")
    print(
        f" Simulation {'failed' if error_steps else 'complete'}"
        f" — ok={len(ok_steps)} error={len(error_steps)} total={len(progress['steps'])}"
    )
    print(f" Artifacts: {run_root}/step_XX/")
    print(f" Progress : {run_root}/simulation_progress.json")
    print(f"{'='*60}")

    if error_steps:
        raise SystemExit(1)

    sentinel = run_root / "SIMULATION_DONE"
    sentinel.write_text(
        f"finished_at={progress['finished_at']}\n"
        f"steps={len(progress['steps'])}\n"
        f"status=success\n"
    )


if __name__ == "__main__":
    main()

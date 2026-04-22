#!/usr/bin/env python3
"""
gen_step_predictions.py — Generate per-step prediction CSVs for simulation display.

Strategy: build features ONCE from the final slice (slice_1945), then for each
of the 26 warm-refresh steps filter to the appropriate origin bar and run
inference with that step's model. This is ~26x faster than rebuilding features
per step.

Output: run_root/step_XX/predictions/predictions.csv for all 26 steps.

Run via Slurm (recommended):
    sbatch ml/ml/nibi/gen_step_predictions.sbatch
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

try:
    from ml.XG_boost_3_multigpu_final import (
        align_features_for_inference,
        build_symbol_day_dataset,
        predicted_direction_label,
        predict_path_matrix,
    )
except ImportError as exc:
    print(f"ERROR: Cannot import XG_boost_3_multigpu_final — {exc}")
    sys.exit(1)

# 26 intraday windows: 09:30–15:45 ET = 13:30–19:45 UTC
def build_windows(sim_date: str) -> list[tuple[int, pd.Timestamp, str]]:
    d = dt.date.fromisoformat(sim_date)
    utc_open = dt.datetime(d.year, d.month, d.day, 13, 30, tzinfo=dt.timezone.utc)
    return [
        (i, pd.Timestamp(utc_open + dt.timedelta(minutes=15 * i)), f"slice_{int(13*60+30+15*i):04d}.parquet")
        for i in range(26)
    ]


def save_predictions(step_dir: Path, as_of_rows: pd.DataFrame, preds: np.ndarray) -> int:
    base_close = as_of_rows["close"].to_numpy(dtype=float)
    predicted_close = np.exp(preds) * base_close[:, None]
    predicted_full_day_return = preds[:, -1]

    rows = []
    for i, (_, row) in enumerate(as_of_rows.iterrows()):
        entry: dict = {
            "symbol": str(row["symbol"]),
            "forecast_origin_ts": str(row["bar_ts"]),
            "predicted_full_day_return": float(predicted_full_day_return[i]),
            "predicted_direction": predicted_direction_label(
                np.array([predicted_full_day_return[i]])
            )[0],
        }
        for h in range(26):
            entry[f"pred_log_return_h{h:02d}"] = float(preds[i, h])
            entry[f"pred_close_h{h:02d}"] = float(predicted_close[i, h])
        rows.append(entry)

    pred_df = (
        pd.DataFrame(rows)
        .sort_values("predicted_full_day_return", ascending=False)
        .reset_index(drop=True)
    )
    pred_csv = step_dir / "predictions" / "predictions.csv"
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_csv, index=False)
    return len(pred_df)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root",    required=True)
    p.add_argument("--sim-date",    default="2026-04-15")
    p.add_argument("--start-step",  type=int, default=0)
    p.add_argument("--lookback-days", type=int, default=10,
                   help="Days of history to load for feature engineering (default: 10, matches warm_refresh)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    windows  = build_windows(args.sim_date)

    print(f"{'='*60}")
    print(f" gen_step_predictions.py  (build-features-once strategy)")
    print(f" sim_date     : {args.sim_date}")
    print(f" run_root     : {run_root}")
    print(f" lookback_days: {args.lookback_days}")
    print(f" steps        : {args.start_step}–25")
    print(f"{'='*60}\n")

    # ── Step 1: build features ONCE from the final slice ──────────
    # Use slice_1945 (all data up to 15:45) so all 26 origin bars exist
    final_slice = run_root / "slices" / "slice_1945.parquet"
    if not final_slice.exists():
        print(f"ERROR: final slice not found: {final_slice}")
        sys.exit(1)

    cutoff_ts = windows[-1][1]  # 2026-04-15 19:45 UTC
    cutoff_date = (cutoff_ts - pd.Timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")

    print(f"[1/2] Loading {final_slice.name} (filtering to {args.lookback_days} days from {cutoff_date}) ...")
    t0 = time.time()
    df = pd.read_parquet(final_slice)
    df["window_ts"] = pd.to_datetime(df["window_ts"], utc=True)
    df = df[df["window_ts"] >= cutoff_date].rename(columns={"window_ts": "bar_ts"})
    print(f"      {len(df):,} rows  {df['symbol'].nunique()} symbols  ({time.time()-t0:.1f}s)")

    print(f"[2/2] Building features ...")
    t1 = time.time()
    dataset, _ = build_symbol_day_dataset(
        df, daily_windows=[5, 10, 20], slot_windows=[3, 5, 10], winsor_pct=0.01
    )
    print(f"      dataset: {len(dataset):,} rows  ({time.time()-t1:.1f}s)\n")

    # ── Step 2: for each step, filter to origin bar + run inference ─
    ok = 0
    for step_idx, as_of_ts, _ in windows:
        if step_idx < args.start_step:
            continue

        step_dir = run_root / f"step_{step_idx:02d}"
        pred_csv = step_dir / "predictions" / "predictions.csv"
        et_label = (as_of_ts - pd.Timedelta(hours=4)).strftime("%H:%M")

        if pred_csv.exists():
            print(f"  step_{step_idx:02d} ({et_label} ET): already done — skip")
            ok += 1
            continue

        if not step_dir.exists():
            print(f"  step_{step_idx:02d}: missing step dir — skip")
            continue

        print(f"  step_{step_idx:02d} ({et_label} ET) ...", end="", flush=True)
        t2 = time.time()
        try:
            feat = json.loads((step_dir / "feature_names.json").read_text())

            # Origin bar = latest bar on or before as_of_ts per symbol
            candidates = dataset[dataset["bar_ts"] <= as_of_ts]
            if candidates.empty:
                print(f" no data before {as_of_ts} — skip")
                continue
            origin_bar_ts = candidates["bar_ts"].max()
            as_of_rows = candidates[candidates["bar_ts"] == origin_bar_ts].copy()

            X = align_features_for_inference(as_of_rows, feat)
            preds = predict_path_matrix(step_dir, X)
            n = save_predictions(step_dir, as_of_rows, preds)
            print(f" {n} symbols  {time.time()-t2:.1f}s  ✓")
            ok += 1
        except Exception as exc:
            print(f" ERROR: {exc}")

    print(f"\nDone — {ok}/26 steps complete")


if __name__ == "__main__":
    main()

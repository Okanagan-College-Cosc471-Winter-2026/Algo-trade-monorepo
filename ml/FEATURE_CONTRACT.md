# ML Feature Contract

This document is the authoritative reference for the feature set used
across training (NIBI), serving (backend inference), and the DB pipeline.
Any change to this contract requires updating all three layers simultaneously.

---

## Contract Version

| Field | Value |
|-------|-------|
| Active bundle | `model_artifacts/current_base` |
| Features file | `model_artifacts/current_base/feature_names.json` |
| Feature count | 156 |
| Model horizons | 26 (target\_h00 … target\_h25, one per 15-min bar 09:30–15:45 ET) |
| Target definition | `log(next_regular_session_close_h / current_bar_close)` |
| Base trees | 1 157 (BEST\_FIXED\_VARIANT, tuned via Optuna) |
| Warm-refresh trees | 30 per step (FAST\_REFRESH\_VARIANT) |

---

## Feature Sources

Features come from **two layers** that must stay in sync:

### Layer 1 — DB Stored Columns (`dw.market_data_15m`)

Written by `dw.process_15min_window()` (see `db/init/05_procedures.sql`)
and exposed as `ml.market_data_15m` via the bridge view
(`db/init/06_bridge_views.sql`).

These columns are what the backend reads in `market/crud.py::get_recent_inference_bars()`.

| Column | Description |
|--------|-------------|
| `open`, `high`, `low`, `close`, `volume` | OHLCV for the 15-min window |
| `slot_count` | Number of 5-min bars aggregated (3 = complete) |
| `lag_close_1/5/10` | Close price lagged by 1/5/10 windows for this symbol |
| `close_diff_1/5` | Absolute close difference vs lag |
| `pct_change_1/5` | Percentage change vs lag |
| `log_return_1` | `ln(close / lag_close_1)` |
| `sma_close_5/10/20` | Simple moving average of close over 5/10/20 windows |
| `sma_volume_5` | Simple moving average of volume over 5 windows |
| `day_of_week`, `hour_of_day`, `month_of_year` | Calendar features |
| `day_monday … day_friday` | One-hot weekday indicators |
| `quarter_1 … quarter_4` | One-hot quarter indicators |
| `hour_early_morning … hour_late_afternoon` | One-hot time-of-day indicators |
| `previous_close` | Prior day's close (from `market.daily_prices`) |
| `overnight_gap_pct` | `(today_open − prev_close) / prev_close` |
| `overnight_log_return` | `ln(today_open / prev_close)` |
| `is_gap_up`, `is_gap_down` | Binary gap direction flags |

> **Gap features require `market.daily_prices` to be populated.**  
> These columns remain `NULL` for symbols not in `market.daily_prices`.  
> The bootstrap (`initial_data.py`) seeds synthetic daily prices for all
> tracked symbols so these features are always non-null on startup.

### Layer 2 — Derived Features (Python, `prepare_production_features`)

Computed in `services/backend/app/modules/inference/features.py::derive_market_features()`
from the DB columns above.  These are the additional features the model
was actually trained on.

| Derived feature | Formula / source |
|-----------------|-----------------|
| `slot_idx` | `cumcount()` within `(symbol, trade_date)` group = 0–25 |
| `bars_seen` | `slot_idx + 1` |
| `bars_remaining` | `26 - bars_seen` |
| `minutes_seen` | `bars_seen × 15` |
| `intraday_progress` | `bars_seen / 26` |
| `cutoff_close/high_seen/low_seen/volume_seen` | Cumulative daily high/low/volume to this bar |
| `cutoff_slot_idx` | Alias of `slot_idx` (int) |
| `close_open_log_return` | `ln(close / open)` for this bar |
| `high_low_log_range` | `ln(high / low)` for this bar |
| `close_vs_lag_close_1/5/10` | `ln(close / lag_close_N)` |
| `close_vs_sma_close_5/10/20` | `ln(close / sma_close_N)` |
| `cutoff_return_from_open` | `ln(close / day_open)` |
| `cutoff_range_seen` | `ln(cutoff_high_seen / cutoff_low_seen)` |
| `cutoff_price_vs_prev_close` | `ln(close / previous_close)` |
| `log_volume` | `log1p(volume)` |
| `log_slot_count` | `log1p(slot_count)` |
| `log_cutoff_volume_seen` | `log1p(cutoff_volume_seen)` |
| `volume_vs_sma_volume_5` | `log1p(volume) − log1p(sma_volume_5)` |

#### Rolling features (grouped by `(symbol, slot_idx)`)

Windows: 5d, 10d, 20d, 60d — suffix `_{mean,std}_{N}d`

Applied to: `close_open_log_return`, `high_low_log_range`, `pct_change_1`,
`pct_change_5`, `log_return_1`, `cutoff_return_from_open`, `cutoff_range_seen`,
`cutoff_price_vs_prev_close`, `cutoff_volume_seen`, `overnight_gap_pct`,
`overnight_log_return`

#### Cutoff-relative rolling features (grouped by `(symbol, slot_idx)`)

Windows: 5cut, 10cut, 20cut (see `add_group_rolling_features_by_keys`) — suffix `_{mean,std}_{N}cut`

Applied to: `cutoff_return_from_open`, `cutoff_range_seen`,
`cutoff_volume_seen`, `pct_change_1`

---

## Missing-Column Policy

The backend's `prepare_production_features()` fills any feature listed in
`feature_names.json` that is absent from the DB rows with **`0.0`** (see
`features.py` line ~419).  This is safe only for features whose absence
is rare and whose zero-imputation is economically neutral.

Features that **cannot be safely zero-imputed** (their absence means bad data):
- `lag_close_1`, `lag_close_5`, `lag_close_10` — if these are NULL the
  bar is too early in the history; the endpoint returns a 400.
- `overnight_gap_pct`, `overnight_log_return` — NULL when `market.daily_prices`
  is missing; imputed as 0.0 (no-gap assumption) until real prices are available.

---

## Training Alignment Rules

1. **NIBI training uses the parquet exported from `ml.market_data_15m`**, not
   raw 5-min bars.  `XG_boost_3_multigpu_final.py` calls `load_bars()` which
   reads the same columns listed in Layer 1 above.

2. **`feature_names.json` in the bundle must only reference columns computable
   from `ml.market_data_15m`**.  Any new feature added to Python must also be
   available in `derive_market_features()` or zeroed by the missing-column
   policy.

3. **`build_symbol_day_dataset()` in the training script builds features using
   the same formulas as `derive_market_features()`** (log-ratios, rolling
   windows, slot indices).  If you change one, change the other.

4. **The bundle's `feature_names.json` is the single source of truth** for which
   features the active model expects.  The backend loads this file at startup
   and passes it to `prepare_production_features()`.

5. **Warm-refresh uses the same feature set as base training**.  The
   `simulate_warm_refresh()` function passes the same `feature_names` through
   `align_features_for_inference()` before each step.

---

## Validation Steps

After any schema or feature change:

```bash
# 1. Confirm DB columns
psql -h localhost -p 5433 -U $POSTGRES_USER -d $POSTGRES_DB \
  -c "\d dw.market_data_15m"

# 2. Confirm feature_names.json matches DB + derived columns
python - <<'EOF'
import json, pathlib
bundle = pathlib.Path("model_artifacts/current_base")
features = json.loads((bundle / "feature_names.json").read_text())
print(f"Feature count: {len(features)}")
print("First 10:", features[:10])
EOF

# 3. End-to-end inference smoke test
curl -s http://localhost:8000/api/v1/inference/predict/AAPL | python -m json.tool | head -30
```

---

## Bundle Directory Layout (Required)

```
model_artifacts/current_base/          ← symlink, points to actual bundle dir
  metadata.json                        ← training provenance (date, trees, etc.)
  feature_names.json                   ← ORDERED list of 156 feature names
  models/
    model_manifest.json                ← {"models": {"target_h00": "horizon_00.json", …}}
    horizon_00.json … horizon_25.json  ← XGBoost native booster files
```

The backend's `NextDayPathBundle` and the Airflow promotion tasks both
assert that `models/model_manifest.json` exists before completing.  A bundle
missing this file will never be promoted.

---

## Adding New Features — Checklist

- [ ] Add the column to `dw.market_data_15m` DDL (`db/init/03_dw_tables.sql`)
- [ ] Compute and write the column in `dw.process_15min_window()`
      (`db/init/05_procedures.sql`)
- [ ] Add the derivation to `derive_market_features()` or
      `prepare_production_features()` in `services/backend/app/modules/inference/features.py`
- [ ] Retrain the base model on NIBI so the new bundle's `feature_names.json`
      includes the new feature
- [ ] Promote the new bundle and reload the backend
- [ ] Update this document

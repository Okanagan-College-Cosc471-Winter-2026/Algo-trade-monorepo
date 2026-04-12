# April 7 Simulation — Plan & Progress

**Branch:** `ft-april7-simulation`
**Created:** 2026-04-08
**Goal:** Retrain the XGBoost model on clean backfilled data (cutoff April 6), run a full warm-refresh intraday simulation for April 7, and wire the new artifacts into the simulation backend.

---

## Context

The existing simulation in the backend (`/simulation/*` endpoints) was built on a replay of **2026-03-23**.
Artifacts live in `model_artifacts/warm_refresh_replay_2026-03-23/`.

We now have clean, quality-checked 15-min data in `ml.market_data_15m` for 484 symbols spanning **2024-03-25 → 2026-04-07** (backfilled April 2026 via `refetch_market_data_15m_quality.py`).

The idea:
- **Train** on data up to and including **2026-04-06** (hold out April 7 entirely)
- **Base model** predicts the full next-day 26-bar path for **2026-04-07**
- **Warm refresh** simulates intraday updates: as each real April 7 bar closes, add 30 trees and re-predict
- **April 7 actual data** exists in the DB — it is used only for warm refresh steps, never seen during base training
- The result is a like-for-like replay of what the model would have done live on April 7

---

## Architecture (How XG_boost_3.py Works)

The main pipeline is `ml/notebooks/XG_boost_3.py`.

### Model structure
- **26 independent XGBoost boosters** — one per 15-min bar slot (09:30 → 15:45)
- Each booster predicts: `log(next_day_close_h / current_day_last_close)`
- Base training: ~1,157 trees, `learning_rate=0.015`, `max_depth=4–9`, `tree_method=hist` (GPU)
- ~176 production features (rolling technicals, slot features, cutoff stats, time encoding)

### Warm refresh mechanism
- Loads existing booster, trains 30 additional trees on the most recent N days
- Does NOT retrain from scratch — just appends trees (XGBoost native booster continuation)
- Each step takes ~6–10 sec on GPU
- At step K: total trees = 1,157 + (K+1) × 30

### Artifact layout (per simulation date)
```
model_artifacts/warm_refresh_replay_<DATE>/
├── simulation_summary.json
└── step_00/ … step_25/
    ├── feature_names.json
    ├── metadata.json
    ├── models/
    │   ├── model_manifest.json
    │   ├── horizon_00.json … horizon_25.json
    └── predictions/
        └── predictions.csv   (symbol, step, as_of_ts, pred_log_return_h00…h25)
```

---

## Plan

### Phase 1 — Retrain base model (cutoff 2026-04-06)

Run `XG_boost_3.py` in bootstrap/train mode with a hard cutoff of `2026-04-06`.
April 7 rows must be excluded from the training query.

```bash
conda activate cosc471
cd ml/notebooks
python XG_boost_3.py \
  --mode bootstrap \
  --split-date 2026-04-06 \
  --output-dir ../../model_artifacts/base_2026-04-06
```

Expected output: `model_artifacts/base_2026-04-06/` with 26 boosters + feature_names.json

### Phase 2 — Warm refresh simulation replay (2026-04-07)

Run the intraday replay mode, stepping through April 7 bar by bar.
At each step: load the previous booster, add 30 warm trees trained on recent data
(still excluding April 7 from the warm window — only the arriving bar is "new").

```bash
python XG_boost_3.py \
  --mode replay_intraday \
  --replay-date 2026-04-07 \
  --base-model-dir ../../model_artifacts/base_2026-04-06 \
  --output-dir ../../model_artifacts/warm_refresh_replay_2026-04-07 \
  --warm-trees-per-step 30
```

Expected output: 26 step directories, each with updated boosters + `predictions.csv`

### Phase 3 — Wire into simulation backend

Update `backend/app/modules/simulation/loader.py` to load from `warm_refresh_replay_2026-04-07`.
Keep `2026-03-23` simulation available (no breaking change — just add the new date or make it configurable).

The existing API endpoints require no schema changes:
- `GET /simulation/session` — returns step count, labels, tree counts
- `GET /simulation/base/{symbol}` — base model prediction for April 7
- `GET /simulation/step/{symbol}/{step}` — warm-refreshed prediction at step N

---

## Clean Training Symbol List

File: `ml/data/clean_symbols_training.json`

**498 symbols** selected from 503 S&P 500 cache. Excluded:

| Symbol | Reason |
|--------|--------|
| Q | New listing 2025-11-06 — only 102 trading days |
| SNDK | Spinoff/relist 2025-02-13 — only 286 trading days |
| EXE | New listing 2024-10-02 — only 377 trading days |
| ERIE | Thin intraday — avg 24.0 bars/day (illiquid, structural) |
| TPL | Thin intraday — avg 24.5 bars/day (illiquid, structural) |

**AZO retained** (avg 24.96 bars/day — borderline but kept as clean S&P 500 representative).

All 503 current S&P 500 stocks are present in the DB. The "15 missing days" for most symbols are confirmed market holidays — not data gaps.

---

## Progress

- [x] Data backfilled: 505 symbols, 2024-03-25 → 2026-04-07 in `ml.market_data_15m`
- [x] Branch created: `ft-april7-simulation`
- [x] Sanity check complete — clean symbol list generated (498 symbols)
- [ ] Phase 1: Retrain base model (cutoff 2026-04-06)
- [ ] Phase 2: Warm refresh replay for 2026-04-07 (26 steps)
- [ ] Phase 3: Wire new artifacts into simulation backend

---

## Key Files

| File | Purpose |
|------|---------|
| `ml/notebooks/XG_boost_3.py` | Main training + replay pipeline |
| `ml/features/technical_indicators.py` | Feature engineering (~176 features) |
| `backend/app/modules/simulation/loader.py` | Loads simulation artifacts for API |
| `backend/app/modules/simulation/service.py` | Serves base + step predictions |
| `backend/app/modules/inference/service.py` | Production live inference (separate, unchanged) |
| `model_artifacts/warm_refresh_replay_2026-03-23/` | Existing simulation artifacts (reference) |
| `model_artifacts/warm_refresh_replay_2026-04-07/` | New artifacts (to be created) |

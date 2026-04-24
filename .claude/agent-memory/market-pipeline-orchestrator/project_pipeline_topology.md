---
name: Pipeline topology and data flow
description: End-to-end pipeline architecture — data sources, stores, model artifacts, API endpoints, and key thresholds discovered 2026-04-24
type: project
---

## Data Storage

- **Primary market DB**: `algotrade` database in Postgres (port 5433 external). Only `alembic_version` table exists — market data is NOT stored in SQL tables.
- **Intraday store**: flat-file parquet snapshots in `/data/projects/Algo-trade-monorepo/datasets/` — this is the real data store.
- **Snapshot schema**: columns include `agg_id`, `symbol`, `window_ts`, `trade_date`, `open/high/low/close/volume`, `slot_count`, `status`, plus feature columns (lag_close_1/5/10, close_diff_1/5, pct_change_1/5, log_return_1, etc.). 45 columns total.
- **Status values**: `confirmed` (majority), `complete`, `provisional`

## Data Freshness (as of 2026-04-24 pre-market)

- Latest snapshot file: `snapshot_2026-04-23.parquet` (written 2026-04-23 05:51, 423MB)
- Max `window_ts` in that file: `2026-04-22 23:45:00 UTC`
- Max `trade_date`: `2026-04-22` (Tuesday)
- Coverage API shows 502/505 symbols have `data_to = 2026-04-21`; only 2 symbols reach 2026-04-23
- **Gap**: most symbols are missing 2026-04-22 full-day data and all of 2026-04-23 (Thursday) in the coverage table
- HOLX is the one symbol with gap_days > 3

## Collector

- Runs as a restarting container (`algo-trade-monorepo-collector-1`) — restart loop is expected outside market hours
- Last successful run: `2026-04-24T00:45:56 UTC`, stage=`aggregate_15m`, window=`2026-04-24T00:30:00 UTC`
- Intraday data last window: `2026-04-23T23:45:00 UTC` (staleness ~341 min at check time — expected idle)
- Freshness endpoint: `GET /api/v1/ops/data/freshness`

## Backend API (port 8000)

- Base path: `/api/v1`
- Key endpoints:
  - `GET /api/v1/utils/health-check/` — returns `true`
  - `GET /api/v1/ops/status` — system status, collector state, model info, disk, CPU
  - `GET /api/v1/ops/data/freshness` — data recency, market state, staleness
  - `GET /api/v1/market/coverage` — per-symbol data_from/data_to/rows/gap_days for all 505 symbols
  - `GET /api/v1/market/stocks` — 505 symbols list
  - `GET /api/v1/training/status` — training job state (idle/running)
  - `GET /api/v1/data/snapshots` — lists parquet snapshots in /datasets

## Model Artifacts

- Active base model: `/data/projects/Algo-trade-monorepo/model_artifacts/nibi_2026-04-16_job12292965/current/`
- Symlinks: `current_base` and `backup_base` both point to `nibi_2026-04-16_job12292965/current`
- Model type: XGBoost multi-horizon (26 horizons: h00–h25)
- Target: `log(next_regular_session_close_h / current_bar_close)`
- Promoted at: `2026-04-22T03:18:04 UTC`
- Simulation training data: as_of sim_date `2026-04-15`, trained on NIBI HPC (job12292965), finished `2026-04-16T21:14:20 UTC`
- Training duration: base train ~2227s, each of 26 intraday steps ~450s
- A second set of simulation artifacts exists at `/model_artifacts/model_artifacts/simulation_2026-04-07/` (older, step_25 only, Apr 18 timestamps)

## Infrastructure

- Root disk: `/dev/vda1` 29GB total, 27GB used (94%) — CRITICAL, very little headroom
- Data disk: `/dev/vdb` 49TB, 266GB used (1%) — ample space for datasets
- Datasets dir: 3.5GB total across 8 parquet files (Apr 7 through Apr 23)
- Model artifacts dir: 1.1GB
- RAM: 25.2GB total, 5.0GB used (20%) — healthy
- CPU: 16 cores (Intel Xeon SapphireRapids), load avg 0.47/0.63/0.50 — healthy

**Why:** Documenting exact paths and thresholds so future checks don't need re-discovery.
**How to apply:** Use `GET /api/v1/ops/data/freshness` and `GET /api/v1/market/coverage` as the primary data recency signals. Root disk at 94% needs monitoring — old snapshots or logs may need pruning.

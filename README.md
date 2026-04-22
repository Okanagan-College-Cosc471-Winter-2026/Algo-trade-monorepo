# Algo-Trade Monorepo

End-to-end algorithmic trading system: live market data collection, ML model training on HPC, and prediction serving via a REST API and Streamlit dashboard.

---

## System Overview

```
                         ┌─────────────────────────────────────────────────────┐
                         │              VPS  (this repo + the-project-maverick) │
                         │                                                      │
  FMP API  ──15min──►  collector ──► stg_raw ──ETL──► ml.market_data_15m      │
                         │                                   │                  │
                         │                           (nightly export)          │
                         │                                   │                  │
                         │              ┌────────────────────▼──────────────┐  │
                         │              │   Airflow DAG  nibi_daily_warm_refresh  │
                         │              │   (scheduler in the-project-maverick)   │
                         │              │   export → sync → submit → poll    │  │
                         │              │   validate → rsync back → promote  │  │
                         │              └────────────────────┬──────────────┘  │
                         │                                   │                  │
                         └──────────────────────────────────┼──────────────────┘
                                                             │ SSH / rsync
                                                             │ (ControlMaster socket,
                                                             │  Duo MFA reuse)
                                               ┌────────────▼────────────────┐
                                               │    NIBI  (Alliance Canada)  │
                                               │    H100 GPU — Slurm queue   │
                                               │    26-horizon XGBoost       │
                                               │    warm-refresh training    │
                                               └─────────────────────────────┘
                                                             │ artifacts rsynced back
                         ┌───────────────────────────────────▼─────────────────┐
                         │  model_artifacts/current_base  (symlink)             │
                         │         │                                             │
                         │         ├──► backend (FastAPI :8000)                 │
                         │         │       └──► predictions, simulation API      │
                         │         └──► frontend (Streamlit :8501)              │
                         └─────────────────────────────────────────────────────┘
```

---

## Services

| Service | Port | Role |
|---------|------|------|
| `db` (Postgres 16) | 5433 | Single database, all schemas |
| `collector` | — | Legacy standalone collector service (disabled by default) |
| `scheduler` | — | Legacy cron scheduler (disabled by default) |
| `backend` (FastAPI) | 8000 | Serves predictions and simulation artifacts |
| `frontend` (Streamlit) | 8501 | Dashboard — talks to backend |
| `dw-api` (FastAPI) | 8001 | Read-only data warehouse API |
| `adminer` | 8082 | Postgres admin UI |

Airflow (webserver :8081, scheduler) runs inside the **the-project-maverick** Docker stack and mounts this repo's DAGs read-only.

---

## Database Schema

```
stg_raw          ← raw FMP ingest, deduplicated on (symbol, ts)
stg_transform    ← intermediate cleaning
core_dbms        ← 5-minute candles, canonical market data
ml               ← 15-minute windows, 176 engineered features — ML training source
dw               ← data warehouse for analytics queries
market           ← market-facing tables (stocks, daily prices)
operation_logs   ← pipeline audit trail
```

Data flows left to right: `stg_raw → core_dbms → ml`. Airflow owns the 15-minute orchestration and applies calendar-aware runtime gating.

---

## ML Training Pipeline

The core ML model is an XGBoost ensemble of 26 independent boosters — one per 15-minute bar slot from 09:30 to 15:45 ET. Each booster predicts the log return path for the next trading day across all ~505 S&P 500 symbols simultaneously (global model, not per-symbol).

Training happens on NIBI (Alliance Canada HPC, H100 GPU) because the VPS does not have a GPU. The daily workflow:

1. Export `ml.market_data_15m` → local parquet snapshot (~427 MB, 505 symbols)
2. Rsync code + parquet + base model → NIBI
3. Submit an 8-hour Slurm job: 26 warm-refresh windows, one per intraday bar
4. Each window adds 30 trees to the existing boosters (incremental, not from scratch)
5. Rsync the 26 `step_XX/` artifact snapshots back to the VPS
6. Atomically promote `model_artifacts/current_base` to point to the new bundle
7. Reload the backend inference cache

This pipeline is orchestrated by Airflow. See the [DAG documentation](airflow/airflow/dags/NIBI_DAG_README.md) for the full design.

---

## Scheduling Architecture

Airflow is the authoritative control plane for intraday data + model workflows.

### Airflow (hosted in `the-project-maverick`)

Used for intraday data, warm-refresh, and daily training/promotion:
- Long polling (up to 9 hours waiting on a Slurm queue)
- Multi-step dependencies with data passing between tasks
- Automatic retry with backoff
- Artifact validation before promotion
- Visibility into which step failed and why

```
*/15 * * * 1-5   intraday_data_pipeline DAG    collect → export → aggregate
*/15 * * * 1-5   nibi_intraday_warmrefresh DAG single-window warm-refresh (freshness gated)
0 10 * * 1-5   nibi_daily_warm_refresh DAG   (10:00 UTC = 06:00 ET)
```

All intraday gates are calendar-aware via `exchange_calendars` (`XNYS`) in `America/New_York`.

### Intraday Warm-Refresh Modes

- Normal mode (performance): one trigger updates one expected closed 15-minute window.
  - Airflow passes `--single-window --as-of-ts <window_ts_utc>` into `sim_warm_windows.sbatch`.
  - Data export is bounded lookback + expected window, not a full-day replay.
- Recovery mode (fallback): keep multi-window replay path available for backfill or repair.
  - Use the existing multi-window flow in `run_simulation_day.py` when single-window flags are not provided.

### Legacy Cron Services

`collector` and `scheduler` are now profile-gated under `legacy-orchestrator` in `docker-compose.yml` and should stay disabled in production to avoid duplicate ownership.

DAG files live in this repo under `airflow/airflow/dags/`. The Maverick Airflow stack mounts them read-only:

```yaml
volumes:
  - /data/projects/Algo-trade-monorepo/airflow/airflow/dags:/opt/airflow/dags/algo_trade:ro
```

See [NIBI DAG README](airflow/airflow/dags/NIBI_DAG_README.md) for the complete pipeline, concepts, and failure runbook.

---

## Same-Day Commissioning Checklist

Use this flow when enabling Airflow intraday ownership in a fresh environment:

1. Keep `collector` and `scheduler` services disabled (no `legacy-orchestrator` profile).
2. Validate Airflow DAG parsing for `intraday_data_pipeline` and `nibi_intraday_warmrefresh`.
3. Verify DB connectivity and write access from Airflow workers.
4. Trigger one dry run and confirm skip logs show explicit gate reasons (`holiday`, `outside_rth`, etc.).
5. At market hours, enable both intraday DAGs and monitor first two windows:
   - `intraday_data_pipeline` writes `logs/intraday_data_freshness.json`
   - `nibi_intraday_warmrefresh` runs only after freshness window is current
6. Roll back immediately if freshness stalls or duplicate writes are detected:
   - pause intraday DAGs
   - fix root cause
   - replay missed window with manual DAG run params

---

## NIBI HPC Access

SSH access uses a **ControlMaster socket** to reuse a single Duo-MFA-authenticated session across all pipeline SSH calls. The socket lives at `~/.ssh/cm/nibi-harshsaw@nibi.sharcnet.ca:22` on the VPS host.

Establish the socket once (requires Duo approval):

```bash
ssh -M -o "ControlPath=~/.ssh/cm/nibi-harshsaw@nibi.sharcnet.ca:22" \
    -o "ControlPersist=8h" \
    -i ~/.ssh/nibi_key \
    harshsaw@nibi.sharcnet.ca
```

Verify it works:

```bash
ssh -o "ControlPath=~/.ssh/cm/nibi-harshsaw@nibi.sharcnet.ca:22" \
    -o "ControlMaster=no" -o "BatchMode=yes" \
    -i ~/.ssh/nibi_key \
    harshsaw@nibi.sharcnet.ca "echo ok"
```

The Airflow containers run as **uid 1000** (matching the host `ubuntu` user) so they can authenticate to the ControlMaster socket via `SO_PEERCRED`. A cron job keeps socket permissions correct:

```
*/1 * * * * chmod 660 /home/ubuntu/.ssh/cm/nibi-* 2>/dev/null || true
*/20 * * * * ssh -O check nibi >> logs/ssh_keepalive.log 2>&1 || true
```

---

## Key Airflow Concepts Used (Quick Reference)

**Sensors vs Operators**
An Operator runs a task once. A Sensor calls `poke()` repeatedly until a condition is met. Use sensors when waiting for an external system (Slurm queue). With `mode="reschedule"`, the worker slot is released between pokes — the sensor uses essentially zero resources while waiting.

**XCom**
Airflow tasks are isolated functions. XCom (Cross-Communication) is the key-value store for passing small values (job IDs, file paths, counts) between tasks. Values persist across retries. Do not use XCom for large data.

**Idempotency**
A task is idempotent if running it twice produces the same result. Critical because Airflow can re-run tasks on retry or manual trigger.

**Fail-Fast**
Start pipelines with a cheap health check. If SSH is down or NIBI is unreachable, fail immediately rather than wasting time on work that cannot complete.

**Atomic Deployment**
When swapping the active model, use `ln -sfn new_path symlink.new && mv -f symlink.new symlink`. The `mv` is atomic — no moment where the symlink is missing.

**Non-Fatal Tail Tasks**
If a tail task is "nice to have" (cache reload, notification), catch all exceptions and log as warnings. Don't fail the DAG over a backend reload.

**Runtime Variables**

| Variable | Default | Effect |
|----------|---------|--------|
| `nibi_skip_base` | `false` | Skip base model training; use existing artifact |
| `nibi_base_model_dir` | `current_base` | Override which base model is synced to NIBI |

Full explanations with edge cases: [NIBI DAG README](airflow/airflow/dags/NIBI_DAG_README.md).

---

## Directory Structure

```
Algo-trade-monorepo/
├── airflow/
│   └── airflow/
│       └── dags/
│           ├── intraday_data_pipeline_dag.py ← intraday collect/export/aggregate DAG
│           ├── nibi_intraday_warmrefresh_dag.py ← intraday warm-refresh DAG
│           ├── nibi_daily_training_dag.py   ← daily training/promotion DAG
│           ├── NIBI_DAG_README.md           ← DAG concepts + edge cases + runbook
│           ├── daily_drac_training_dag.py   ← legacy (DRAC cluster, unused)
│           ├── daily_dataset_snapshot_dag.py
│           └── retrain_model.py
├── datasets/                                ← local parquet snapshots
├── db/init/                                 ← Postgres schema SQL (runs on first start)
├── docker/scheduler/                        ← legacy cron scheduler (disabled by default)
├── docker-compose.yml
├── logs/                                    ← simulation + pipeline logs
│   └── nibi_usage_meter.jsonl              ← GPU job usage log (H100 hours)
├── ml/ml/                                   ← ML code (rsynced to NIBI for training)
│   ├── nibi/                                ← NIBI-specific scripts + sbatch files
│   │   ├── simulate_full_day.sbatch        ← Slurm GPU job definition
│   │   └── run_simulation.py               ← per-window warm-refresh runner
│   ├── training/                            ← train_lgbm.py
│   ├── features/                            ← feature engineering
│   └── XG_boost_3_multigpu_final.py        ← main XGBoost training pipeline
├── model_artifacts/
│   ├── current_base  →  sim_YYYY-MM-DD/    ← symlink, updated by promote_model task
│   └── sim_YYYY-MM-DD/                     ← dated artifact bundles
│       ├── step_00/ … step_25/             ← one per warm-refresh window (26 total)
│       └── SIMULATION_DONE
├── services/
│   ├── backend/                             ← FastAPI (predictions + simulation)
│   ├── collector/                           ← collector runtime scripts
│   ├── dw-api/                              ← data warehouse read API
│   ├── frontend/                            ← Streamlit dashboard
│   └── scheduler/                           ← legacy cron container
└── tests/
    ├── simulate_market_day.py               ← manual simulation runner (CLI)
    └── test_*.py
```

---

## Quick Start

```bash
# Start all services
docker compose up -d

# Check everything is healthy
docker compose ps

# Establish NIBI ControlMaster session (requires Duo MFA once)
ssh -M -o "ControlPath=~/.ssh/cm/nibi-harshsaw@nibi.sharcnet.ca:22" \
    -o "ControlPersist=8h" -i ~/.ssh/nibi_key harshsaw@nibi.sharcnet.ca

# Trigger a training run via Airflow (in the-project-maverick stack)
docker compose -f ../the-project-maverick/docker-compose.yml exec airflow-scheduler \
  airflow dags trigger nibi_daily_warm_refresh --conf '{"trade_date": "2026-04-08"}'

# Fetch a new trading date into the feature store
python ml/ml/scripts/fetch_new_date.py --date 2026-04-09

# Check simulation logs
tail -f logs/sim_marketday_$(date +%Y%m%d)_*.log
```

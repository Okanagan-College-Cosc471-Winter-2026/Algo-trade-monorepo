---
name: Airflow deployment status
description: Airflow is live on port 8081 with LocalExecutor, 8 DAGs registered, admin user created — became operational 2026-04-24
type: project
---

Airflow is fully operational as of 2026-04-24. Updated status as of 2026-04-27.

- Image: apache/airflow:2.9.3
- Webserver: http://localhost:8081 (internal container port 8080) — Up 2 days (healthy)
- Scheduler: Up 2 days (healthy)
- Executor: LocalExecutor
- Metadata DB: postgresql+psycopg2://appuser@db:5432/airflow (named volume `airflow` DB created by db/init/00_airflow_db.sql)
- Admin credentials: username=admin / password=admin / email=admin@algo.local
- DAG folder: ./airflow/airflow/dags/ (mounted :ro into /opt/airflow/dags)
- Logs: named Docker volume `airflow-logs` (NOT bind-mounted — see infra quirks)
- Providers pre-installed: apache-airflow-providers-ssh, psycopg2-binary

DAG statuses as of 2026-04-27 (3 active / unpaused, 5 paused):
1. build_daily_dataset_snapshot — PAUSED
2. daily_drac_xgb_training — PAUSED
3. hello_airflow — PAUSED
4. intraday_data_pipeline — ACTIVE (unpaused), schedule */15 Mon-Fri; running successfully every 15 min
5. nibi_daily_warm_refresh — ACTIVE (unpaused), schedule 0 21 * * 1-5 (Mon-Fri 21:00 UTC); last SUCCESS 2026-04-24 manual run; scheduled runs for Apr 23 & Apr 24 FAILED (ssh_health_check task); Apr 25/26 missed (no scheduled run IDs in history — DAG may have been paused then)
6. nibi_intraday_warmrefresh — ACTIVE (unpaused), schedule */15 Mon-Fri; running successfully every 15 min today
7. retrain_model — PAUSED
8. seed_market_data — PAUSED

CRITICAL: Airflow REST API returns empty DAG list (0 dags) for admin/admin GET /api/v1/dags — only the Airflow CLI (docker exec airflow dags ...) works reliably.

nibi_daily_warm_refresh failure pattern: All scheduled/failed runs fail at ssh_health_check (NIBI SSH reachability). SSH socket at ~/.ssh/cm/nibi-harshsaw@nibi.sharcnet.ca:22 is currently active (tested working 2026-04-27). Next scheduled run for Apr 28 market: execution_date = 2026-04-27T21:00:00 UTC.

**Why:** Airflow replaces the legacy-orchestrator profile for all pipeline scheduling.
**How to apply:** When diagnosing pipeline scheduling issues, use docker exec CLI. The 3 active DAGs are the intraday and daily refresh pipelines. nibi_daily_warm_refresh failing historically at ssh_health_check — verify SSH socket freshness at ~09:00 UTC before tonight's 21:00 scheduled run.

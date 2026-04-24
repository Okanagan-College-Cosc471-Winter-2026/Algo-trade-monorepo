---
name: Airflow deployment status
description: Airflow is live on port 8081 with LocalExecutor, 8 DAGs registered, admin user created — became operational 2026-04-24
type: project
---

Airflow is fully operational as of 2026-04-24.

- Image: apache/airflow:2.9.3
- Webserver: http://localhost:8081 (internal container port 8080)
- Scheduler: running, healthy, SchedulerJob alive
- Executor: LocalExecutor
- Metadata DB: postgresql+psycopg2://appuser@db:5432/airflow (named volume `airflow` DB created by db/init/00_airflow_db.sql)
- Admin credentials: username=admin / password=admin / email=admin@algo.local
- DAG folder: ./airflow/airflow/dags/ (mounted :ro into /opt/airflow/dags)
- Logs: named Docker volume `airflow-logs` (NOT bind-mounted — see infra quirks)
- Providers pre-installed: apache-airflow-providers-ssh, psycopg2-binary

DAGs registered (all paused by default — ZERO runs ever executed as of 2026-04-24):
1. build_daily_dataset_snapshot — schedule @daily, start 2025-01-01
2. daily_drac_xgb_training — schedule 02:30 UTC daily, start 2026-03-01
3. hello_airflow — test DAG
4. intraday_data_pipeline — schedule */15 Mon-Fri, start 2026-04-22
5. nibi_daily_warm_refresh — schedule 21:00 UTC Mon-Fri (17:00 ET, 1h after close), start 2026-04-01
6. nibi_intraday_warmrefresh — schedule */15 Mon-Fri, start 2026-04-22
7. retrain_model — schedule Sundays 02:00 UTC, start 2024-01-01
8. seed_market_data

CRITICAL: Airflow REST API returns 403 for admin/admin — only the Airflow CLI works.
All DAGs are paused and have no run history. The pipeline relies entirely on manual triggers or legacy services.

**Why:** Airflow replaces the legacy-orchestrator profile (collector + scheduler services) for all pipeline scheduling.
**How to apply:** When diagnosing pipeline scheduling issues, check Airflow first. All DAGs need to be manually unpaused and triggered until automation is confirmed. Use `airflow dags list-runs -d <dag_id> -s <date>` syntax (no --limit flag in 2.9.3). REST API auth is broken for admin user.

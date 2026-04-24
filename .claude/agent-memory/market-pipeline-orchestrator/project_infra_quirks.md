---
name: Known infrastructure quirks
description: Recurring known issues and their fixes for the Algo-trade-monorepo Docker stack
type: project
---

## 1. Airflow logs volume — must be a named volume, not a bind mount

The host `./logs/` directory is owned by uid 1000 (ubuntu). The Airflow container runs as uid 50000 (airflow). Binding `./logs` to `/opt/airflow/logs` causes a PermissionError on the `scheduler/` subdirectory at startup, preventing Airflow from even loading its config.

**Fix:** Use named Docker volume `airflow-logs` for Airflow logs in docker-compose.yml. Both `x-airflow-common` volumes block and the `volumes:` top-level section must declare `airflow-logs`.

## 2. Airflow init command — use single-line bash -c, not YAML > block scalar

The `airflow users create` command has many `--flag value` arguments. When written multiline under a `>` YAML block scalar, it folds correctly (newlines become spaces), but the `|| true` on a trailing line caused an exit code 2 from `docker compose run`. Use a single-line `bash -c "..."` string instead.

**Fix:** `command: bash -c "airflow db migrate && (airflow users create --username admin ... || true)"`

## 3. Collector restart loop — expected behavior, not a failure

The legacy `collector` service (profile: legacy-orchestrator) exits with code 0 when outside market hours (04:00–21:00 ET). Because its restart policy is `always`, Docker immediately restarts it. This causes it to show "Restarting" in `docker compose ps` during off-hours. It is not crashing — this is intentional poll-and-sleep behavior.

**Why it matters:** Do not alert on the collector's Restarting state. Check logs for "falls outside market hours" to confirm it's benign.

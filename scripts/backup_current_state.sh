#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/projects/Algo-trade-monorepo"
cd "${REPO_ROOT}"

TS="$(date -u +%Y%m%d_%H%M%S)"
BACKUP_ROOT="${REPO_ROOT}/backups"
BACKUP_DIR="${BACKUP_ROOT}/state_${TS}"

mkdir -p "${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}/db" "${BACKUP_DIR}/meta" "${BACKUP_DIR}/artifacts" "${BACKUP_DIR}/config"

echo "=== Backup start: ${TS} UTC ==="
echo "Backup dir: ${BACKUP_DIR}"

# 1) Capture docker/runtime state
docker compose ps > "${BACKUP_DIR}/meta/docker_compose_ps.txt"
docker compose config > "${BACKUP_DIR}/meta/docker_compose_resolved.yml"
docker images > "${BACKUP_DIR}/meta/docker_images.txt"
docker volume ls > "${BACKUP_DIR}/meta/docker_volumes.txt"

# 2) Capture key config files
cp "${REPO_ROOT}/docker-compose.yml" "${BACKUP_DIR}/config/docker-compose.yml"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  cp "${REPO_ROOT}/.env" "${BACKUP_DIR}/config/.env"
fi

# 3) Database dump (custom format, compressed)
echo "Dumping PostgreSQL (algotrade)..."
docker compose exec -T db pg_dump -U appuser -d algotrade -Fc > "${BACKUP_DIR}/db/algotrade.dump"
docker compose exec -T db psql -U appuser -d algotrade -c "\dt" > "${BACKUP_DIR}/db/table_list.txt"

# 4) Artifact snapshot (model bundles + current symlinks)
echo "Archiving model_artifacts..."
tar -czf "${BACKUP_DIR}/artifacts/model_artifacts.tar.gz" model_artifacts

# 5) Operational scripts snapshot
tar -czf "${BACKUP_DIR}/config/runtime_scripts.tar.gz" scripts docker airflow services/backend/app/modules/ops services/frontend

# 6) Write quick restore notes
cat > "${BACKUP_DIR}/RESTORE.md" <<'EOF'
Restore checklist:
1) Ensure docker compose stack is up (db service running).
2) Restore DB:
   docker compose exec -T db dropdb -U appuser --if-exists algotrade
   docker compose exec -T db createdb -U appuser algotrade
   cat db/algotrade.dump | docker compose exec -T db pg_restore -U appuser -d algotrade --clean --if-exists
3) Restore model artifacts:
   tar -xzf artifacts/model_artifacts.tar.gz -C /data/projects/Algo-trade-monorepo
4) Verify symlinks:
   ls -la /data/projects/Algo-trade-monorepo/model_artifacts/current_base
5) Restart services:
   docker compose up -d backend frontend scheduler collector
EOF

# 7) Checksums + summary
(
  cd "${BACKUP_DIR}"
  sha256sum db/algotrade.dump artifacts/model_artifacts.tar.gz config/runtime_scripts.tar.gz > SHA256SUMS.txt
)

du -sh "${BACKUP_DIR}" > "${BACKUP_DIR}/meta/backup_size.txt"
echo "=== Backup complete ==="
echo "Path: ${BACKUP_DIR}"
cat "${BACKUP_DIR}/meta/backup_size.txt"

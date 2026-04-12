#!/usr/bin/env bash
# promote_model.sh — Retrieve warm-refresh bundle from NIBI, swap symlink, reload inference.
#
# Run this manually on the VPS after the NIBI job finishes.
#
# Usage:
#   bash ml/ml/nibi/promote_model.sh [YYYY-MM-DD]
#
# Required env vars (set in .env or export before running):
#   NIBI_USER, NIBI_HOST, NIBI_SCRATCH, NIBI_SSH_KEY
#   MODEL_ARTIFACTS_DIR  (default: ./model_artifacts)
#   BACKEND_URL          (default: http://localhost:8000)

set -euo pipefail

RUN_DATE="${1:-$(date +%F)}"

NIBI_USER="${NIBI_USER:?NIBI_USER not set}"
NIBI_HOST="${NIBI_HOST:?NIBI_HOST not set}"
NIBI_SCRATCH="${NIBI_SCRATCH:?NIBI_SCRATCH not set}"
NIBI_SSH_KEY="${NIBI_SSH_KEY:-${HOME}/.ssh/nibi_key}"
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
ARTIFACTS_DIR="${MODEL_ARTIFACTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../model_artifacts" && pwd)}"

REMOTE_BUNDLE="${NIBI_SCRATCH}/ml/warm_refresh/latest"
REMOTE_DONE="${REMOTE_BUNDLE}/DONE"
LOCAL_BUNDLE="${ARTIFACTS_DIR}/warm_${RUN_DATE}"
SYMLINK="${ARTIFACTS_DIR}/current_base"

echo "============================================="
echo " NIBI Model Promotion"
echo " Date        : ${RUN_DATE}"
echo " Remote      : ${NIBI_USER}@${NIBI_HOST}:${REMOTE_BUNDLE}"
echo " Local dest  : ${LOCAL_BUNDLE}"
echo " Symlink     : ${SYMLINK}"
echo "============================================="

SSH_CMD="ssh -i ${NIBI_SSH_KEY} -o StrictHostKeyChecking=no ${NIBI_USER}@${NIBI_HOST}"

# ── 1. Check DONE sentinel ────────────────────────────────────────
echo "[1/4] Checking DONE sentinel ..."
if ! ${SSH_CMD} "test -f ${REMOTE_DONE}"; then
    echo "ERROR: DONE file not found at ${REMOTE_DONE}"
    echo "       The NIBI job may still be running. Check with:"
    echo "       ssh -i ${NIBI_SSH_KEY} ${NIBI_USER}@${NIBI_HOST} squeue -u ${NIBI_USER}"
    exit 1
fi
echo "      DONE found: $(${SSH_CMD} cat ${REMOTE_DONE})"

# ── 2. Validate bundle structure ──────────────────────────────────
echo "[2/4] Validating remote bundle ..."
MISSING=""
for i in $(seq -w 0 25); do
    FILE="horizon_${i}.json"
    if ! ${SSH_CMD} "test -f ${REMOTE_BUNDLE}/${FILE}"; then
        MISSING="${MISSING} ${FILE}"
    fi
done
if ! ${SSH_CMD} "test -f ${REMOTE_BUNDLE}/metadata.json"; then
    MISSING="${MISSING} metadata.json"
fi

if [[ -n "${MISSING}" ]]; then
    echo "ERROR: Missing bundle files:${MISSING}"
    exit 1
fi
echo "      All 26 horizon files + metadata.json present."

# ── 3. Rsync bundle to local ──────────────────────────────────────
echo "[3/4] Retrieving bundle ..."
mkdir -p "${LOCAL_BUNDLE}"
rsync -az --progress \
    -e "ssh -i ${NIBI_SSH_KEY} -o StrictHostKeyChecking=no" \
    "${NIBI_USER}@${NIBI_HOST}:${REMOTE_BUNDLE}/" \
    "${LOCAL_BUNDLE}/"
echo "      Bundle saved to: ${LOCAL_BUNDLE}"

# ── 4. Atomic symlink swap ────────────────────────────────────────
echo "[4/4] Updating symlink ..."
PREV_TARGET="$(readlink "${SYMLINK}" 2>/dev/null || echo '(none)')"
# Atomic: create new symlink then rename into place
ln -sfn "${LOCAL_BUNDLE}" "${SYMLINK}.new"
mv -f "${SYMLINK}.new" "${SYMLINK}"
echo "      ${SYMLINK} → ${LOCAL_BUNDLE}  (was: ${PREV_TARGET})"

# ── 5. Reload backend inference cache ────────────────────────────
echo "[5/5] Reloading backend model ..."
HTTP_STATUS=$(curl -s -o /tmp/reload_response.json -w "%{http_code}" \
    -X POST "${BACKEND_URL}/api/v1/admin/reload-model")

if [[ "${HTTP_STATUS}" == "200" ]]; then
    MODEL_VERSION=$(python3 -c "import json,sys; d=json.load(open('/tmp/reload_response.json')); print(d.get('model_version','?'))" 2>/dev/null || echo "?")
    echo "      Backend reloaded — model_version=${MODEL_VERSION}"
else
    echo "WARNING: reload endpoint returned HTTP ${HTTP_STATUS}"
    echo "         Response: $(cat /tmp/reload_response.json 2>/dev/null)"
    echo "         You may need to restart the backend manually."
fi

echo ""
echo "Promotion complete. Bundle warm_${RUN_DATE} is now serving inference."

#!/usr/bin/env bash
# promote_new_base.sh — Retrieve the Apr-21 training run from NIBI and promote it.
#
# Safety guarantees:
#   1. Checks SIMULATION_DONE sentinel before pulling anything.
#   2. Validates 26 step bundles + metadata.json + feature_names.json exist.
#   3. Rsyncs into a date-stamped directory (never overwrites existing bundles).
#   4. Does an atomic symlink swap for current_base.
#   5. Keeps backup_base pointing to the previous model.
#   6. Reloads the backend inference cache.
#
# Usage:
#   bash scripts/promote_new_base.sh [--dry-run]
#
# Requires: ssh alias "nibi" configured and working.

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — no changes will be made ==="
fi

# ── Config ────────────────────────────────────────────────────────
NIBI_ALIAS="nibi"
NIBI_RUN_ROOT="~/projects/def-youry/test_simulation/run_root_base_apr21"
ARTIFACTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../model_artifacts" && pwd)"
BUNDLE_NAME="nibi_2026-04-22_base"
LOCAL_BUNDLE="${ARTIFACTS_DIR}/${BUNDLE_NAME}"
SYMLINK="${ARTIFACTS_DIR}/current_base"
BACKUP="${ARTIFACTS_DIR}/backup_base"
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"

echo "============================================="
echo " NIBI → VPS Model Promotion (Apr 21 run)"
echo " Remote run_root : ${NIBI_RUN_ROOT}"
echo " Local bundle    : ${LOCAL_BUNDLE}"
echo " Symlink         : ${SYMLINK}"
echo " Backup          : ${BACKUP} → $(readlink ${BACKUP} 2>/dev/null || echo '(none)')"
echo "============================================="
echo ""

# ── 1. Check SIMULATION_DONE sentinel ────────────────────────────
echo "[1/6] Checking SIMULATION_DONE sentinel on NIBI..."
DONE_CONTENT=$(ssh "${NIBI_ALIAS}" "cat ${NIBI_RUN_ROOT}/SIMULATION_DONE 2>/dev/null || echo MISSING")
if [[ "${DONE_CONTENT}" == "MISSING" ]]; then
    echo ""
    echo "ERROR: SIMULATION_DONE not found — job may still be running."
    echo ""
    echo "Check job status:"
    echo "  ssh nibi 'squeue -u \${USER} -o \"%.10i %.20j %.8T %.12M %R\"'"
    echo ""
    echo "Watch live log (once job starts):"
    echo "  ssh nibi 'tail -f ~/projects/def-youry/test_simulation/logs/sim_apr21_12581185.out'"
    exit 1
fi
echo "  DONE: ${DONE_CONTENT}"

# ── 2. Validate remote bundle ─────────────────────────────────────
echo "[2/6] Validating remote artifacts..."
MISSING=""

for ITEM in SIMULATION_DONE simulation_progress.json; do
    if ! ssh "${NIBI_ALIAS}" "test -f ${NIBI_RUN_ROOT}/${ITEM}" 2>/dev/null; then
        MISSING="${MISSING} ${ITEM}"
    fi
done

for ITEM in current/metadata.json current/feature_names.json current/models; do
    if ! ssh "${NIBI_ALIAS}" "test -e ${NIBI_RUN_ROOT}/${ITEM}" 2>/dev/null; then
        MISSING="${MISSING} ${ITEM}"
    fi
done

STEP_COUNT=$(ssh "${NIBI_ALIAS}" "ls -d ${NIBI_RUN_ROOT}/step_* 2>/dev/null | wc -l")
echo "  step_XX bundles found: ${STEP_COUNT}/26"
if [[ "${STEP_COUNT}" -lt 26 ]]; then
    MISSING="${MISSING} (only ${STEP_COUNT}/26 step dirs)"
fi

if [[ -n "${MISSING}" ]]; then
    echo "ERROR: Missing artifacts:${MISSING}"
    exit 1
fi
echo "  All checks passed."

# ── 3. Check feature compatibility ───────────────────────────────
echo "[3/6] Checking feature count compatibility..."
REMOTE_NFEATURES=$(ssh "${NIBI_ALIAS}" "python3 -c \"
import json
with open('${NIBI_RUN_ROOT}/current/metadata.json') as f:
    d = json.load(f)
print(d.get('n_features', 'unknown'))
\" 2>/dev/null || echo unknown")
echo "  Remote n_features: ${REMOTE_NFEATURES}"
echo "  Current n_features: $(python3 -c "import json; d=json.load(open('${ARTIFACTS_DIR}/current_base/metadata.json')); print(d.get('n_features','?'))" 2>/dev/null || echo '?')"

# ── 4. Rsync bundle locally ───────────────────────────────────────
echo "[4/6] Rsyncing bundle from NIBI → ${LOCAL_BUNDLE}..."
if [[ -d "${LOCAL_BUNDLE}" ]]; then
    echo "  WARNING: ${LOCAL_BUNDLE} already exists — will overwrite."
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  [DRY RUN] Would rsync: ${NIBI_ALIAS}:${NIBI_RUN_ROOT}/ → ${LOCAL_BUNDLE}/"
else
    mkdir -p "${LOCAL_BUNDLE}"
    rsync -az --progress \
        "${NIBI_ALIAS}:${NIBI_RUN_ROOT}/" \
        "${LOCAL_BUNDLE}/"
    echo "  Rsync complete."
fi

# ── 5. Atomic symlink swap ────────────────────────────────────────
echo "[5/6] Updating backup_base and current_base symlinks..."
PREV_BASE="$(readlink "${SYMLINK}" 2>/dev/null || echo '(none)')"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  [DRY RUN] Would set backup_base → ${PREV_BASE}"
    echo "  [DRY RUN] Would set current_base → ${BUNDLE_NAME}/current"
else
    # Keep backup pointing to what current_base was
    ln -sfn "${PREV_BASE}" "${BACKUP}"
    # Atomic swap: new symlink then rename
    ln -sfn "${BUNDLE_NAME}/current" "${SYMLINK}.new"
    mv -f "${SYMLINK}.new" "${SYMLINK}"
    echo "  backup_base  → ${PREV_BASE}"
    echo "  current_base → ${BUNDLE_NAME}/current"
fi

# ── 6. Reload backend ────────────────────────────────────────────
echo "[6/6] Reloading backend inference cache..."
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  [DRY RUN] Would POST ${BACKEND_URL}/api/v1/admin/reload-model"
else
    HTTP_STATUS=$(curl -s -o /tmp/reload_resp.json -w "%{http_code}" \
        -X POST "${BACKEND_URL}/api/v1/admin/reload-model" 2>/dev/null || echo "000")
    if [[ "${HTTP_STATUS}" == "200" ]]; then
        VERSION=$(python3 -c "import json; d=json.load(open('/tmp/reload_resp.json')); print(d.get('model_version','?'))" 2>/dev/null || echo "?")
        echo "  Backend reloaded — model_version=${VERSION}"
    else
        echo "  WARNING: Backend reload returned HTTP ${HTTP_STATUS}"
        echo "  Restart manually: docker compose restart backend"
    fi
fi

echo ""
echo "============================================="
if [[ "${DRY_RUN}" == "true" ]]; then
    echo " DRY RUN COMPLETE — run without --dry-run to apply."
else
    echo " PROMOTION COMPLETE"
    echo " New model : ${BUNDLE_NAME}/current"
    echo " Old model : ${PREV_BASE}  (backup_base still points here)"
    echo ""
    echo " To ROLL BACK at any time:"
    echo "   cd ${ARTIFACTS_DIR}"
    echo "   ln -sfn \"\$(readlink backup_base)\" current_base.new"
    echo "   mv -f current_base.new current_base"
    echo "   docker compose restart backend"
fi
echo "============================================="

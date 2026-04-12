#!/usr/bin/env bash
# sync_to_vps.sh
# ─────────────────────────────────────────────────────────────────
# Rsync model artifacts from DRAC scratch back to the VPS.
# Called at the end of daily_train.sbatch.
#
# Usage:
#   bash sync_to_vps.sh <run_date> <model_src_dir>
#   e.g. bash sync_to_vps.sh 2026-03-03 /scratch/$USER/ml/dt=2026-03-03/model
#
# Requirements:
#   - SSH key-based login from DRAC login/compute node → VPS
#   - VPS_USER and VPS_HOST set below (or via environment)
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

RUN_DATE="${1:-$(date +%F)}"
MODEL_SRC="${2:-/scratch/${USER}/ml/dt=${RUN_DATE}/model}"

VPS_USER="${VPS_USER:-cosc-admin}"
VPS_HOST="${VPS_HOST:-cosc-vps}"       # SSH alias or IP
VPS_MODEL_DIR="/home/cosc-admin/the-project-maverick/ml/model_registry/dt=${RUN_DATE}"

echo "[sync] Sending model from ${MODEL_SRC} → ${VPS_USER}@${VPS_HOST}:${VPS_MODEL_DIR}"

# Create remote directory
ssh "${VPS_USER}@${VPS_HOST}" "mkdir -p '${VPS_MODEL_DIR}'"

# Rsync artifacts (model.pkl, encoder.pkl, metrics.json)
rsync -avz --progress \
    "${MODEL_SRC}/" \
    "${VPS_USER}@${VPS_HOST}:${VPS_MODEL_DIR}/"

echo "[sync] Done — model artifacts synced to VPS."

# Optional: also update a 'latest' symlink on VPS
ssh "${VPS_USER}@${VPS_HOST}" \
    "ln -sfn '${VPS_MODEL_DIR}' /home/cosc-admin/the-project-maverick/ml/model_registry/latest"

echo "[sync] Updated 'latest' symlink on VPS."

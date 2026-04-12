#!/usr/bin/env bash
#SBATCH --job-name=daily_xgb_train
#SBATCH --output=/home/%u/logs/train-%j.out
#SBATCH --error=/home/%u/logs/train-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --account=def-yourpi        # ← change to your DRAC allocation account

# ─────────────────────────────────────────────────────────────────
# daily_train.sbatch
# Submitted daily by Airflow on the VPS via:
#   ssh drac_user@drac_login "sbatch ~/the-project-maverick/ml/drac/daily_train.sbatch"
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

RUN_DATE=$(date +%F)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_DATA="/scratch/${USER}/ml/dt=${RUN_DATE}"
PROJECT_MODELS="/project/${USER}/ml/models/dt=${RUN_DATE}"
LOG_DIR="${HOME}/logs"

mkdir -p "${SCRATCH_DATA}" "${PROJECT_MODELS}" "${LOG_DIR}"

echo "============================================="
echo " DRAC Daily XGBoost Training Job"
echo " Date      : ${RUN_DATE}"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : ${HOSTNAME}"
echo " SCRATCH   : ${SCRATCH_DATA}"
echo " MODELS    : ${PROJECT_MODELS}"
echo "============================================="

# ── 1. Load modules ───────────────────────────────────────────────
echo "[step 1/4] Loading modules ..."
module load gcc arrow/17.0.0   # adjust version as needed (module spider arrow)
module load cuda/12.2          # for XGBoost GPU (optional, remove if CPU only)

# ── 2. Activate virtual environment ──────────────────────────────
VENV_PATH="${HOME}/ENV"        # ← adjust to your venv path
if [ -d "${VENV_PATH}" ]; then
    echo "[step 2/4] Activating venv at ${VENV_PATH} ..."
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: venv not found at ${VENV_PATH}"
    exit 1
fi

# ── 3. Extract Parquet snapshot from VPS Postgres ─────────────────
echo "[step 3/4] Extracting data from Postgres ..."
python "${SCRIPT_DIR}/extract_and_save_parquet.py" \
    --out_dir "${SCRATCH_DATA}" \
    --host "${POSTGRES_HOST:-$(cat ${HOME}/.drac_pg_host 2>/dev/null || echo localhost)}" \
    --db   "${POSTGRES_DB:-app}" \
    --user "${POSTGRES_USER:-postgres}" \
    --password "${POSTGRES_PASSWORD:-changethis}"

# ── 4. Train model ────────────────────────────────────────────────
echo "[step 4/4] Training model ..."
python "${SCRIPT_DIR}/train_model.py" \
    --data_dir      "${SCRATCH_DATA}" \
    --model_out     "${SCRATCH_DATA}/model" \
    --horizon       78 \
    --optuna_trials 50 \
    --device        cuda \
    --filter_market_hours

# ── 5. Copy to persistent project storage ─────────────────────────
echo "[post] Copying model to project storage ..."
cp -r "${SCRATCH_DATA}/model/." "${PROJECT_MODELS}/"

# ── 6. Sync model back to VPS ─────────────────────────────────────
echo "[post] Syncing model to VPS ..."
bash "${SCRIPT_DIR}/sync_to_vps.sh" "${RUN_DATE}" "${SCRATCH_DATA}/model"

echo ""
echo "✅ Daily training job complete — ${RUN_DATE}"
echo "   Artifacts in: ${PROJECT_MODELS}"

#!/usr/bin/env bash
#SBATCH --job-name=algo_warm_refresh
#SBATCH --output=/scratch/%u/ml/logs/warm_refresh_%j.out
#SBATCH --error=/scratch/%u/ml/logs/warm_refresh_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=def-youry

# ─────────────────────────────────────────────────────────────────
# warm_refresh.sbatch
# Submitted manually (or by nibi_orchestrator.py once SSH is confirmed).
#
# Usage:
#   sbatch warm_refresh.sbatch --parquet /scratch/$USER/data/snapshot_YYYY-MM-DD.parquet
#
# Output bundle written to: /scratch/$USER/ml/warm_refresh/latest/
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Parse --parquet argument passed after sbatch flags ────────────
PARQUET_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet) PARQUET_FILE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "${PARQUET_FILE}" ]]; then
    echo "ERROR: --parquet <path> required"
    exit 1
fi

RUN_DATE=$(date +%F)
SCRATCH="${SCRATCH:-/scratch/${USER}}"
WARM_ROOT="${SCRATCH}/ml/warm_refresh"
OUT_DIR="${WARM_ROOT}/latest"
LOG_DIR="${SCRATCH}/ml/logs"

mkdir -p "${WARM_ROOT}" "${OUT_DIR}" "${LOG_DIR}"

echo "============================================="
echo " NIBI Warm-Refresh Job"
echo " Date        : ${RUN_DATE}"
echo " Job ID      : ${SLURM_JOB_ID}"
echo " Node        : ${HOSTNAME}"
echo " Parquet     : ${PARQUET_FILE}"
echo " Output dir  : ${OUT_DIR}"
echo "============================================="

# ── 1. Load modules ───────────────────────────────────────────────
echo "[1/4] Loading modules ..."
module load gcc arrow/17.0.0
# GPU module loaded automatically via --gres=gpu:1 on NIBI

# ── 2. Activate virtual environment ──────────────────────────────
VENV_PATH="${HOME}/ENV"
if [ -d "${VENV_PATH}" ]; then
    echo "[2/4] Activating venv at ${VENV_PATH} ..."
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: venv not found at ${VENV_PATH}. Create it with:"
    echo "  python -m venv ~/ENV && pip install xgboost pandas sqlalchemy pyarrow"
    exit 1
fi

# ── 3. Run warm refresh ───────────────────────────────────────────
echo "[3/4] Running warm refresh ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/run_warm_refresh.py" \
    --parquet        "${PARQUET_FILE}" \
    --run-root       "${WARM_ROOT}" \
    --output-dir     "${OUT_DIR}" \
    --warm-trees     30 \
    --warm-refresh-days 1

# ── 4. Signal completion ─────────────────────────────────────────
echo "[4/4] Writing DONE sentinel ..."
echo "job_id=${SLURM_JOB_ID} date=${RUN_DATE}" > "${OUT_DIR}/DONE"

echo ""
echo "Warm refresh complete — bundle at: ${OUT_DIR}"
echo ""
echo "To retrieve and promote the bundle, run on your VPS:"
echo "  bash promote_model.sh ${RUN_DATE}"

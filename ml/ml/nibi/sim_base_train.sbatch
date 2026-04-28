#!/usr/bin/env bash
#SBATCH --job-name=sim_base_train
#SBATCH --output=/home/%u/projects/def-youry/test_simulation/logs/sim_base_%j.out
#SBATCH --error=/home/%u/projects/def-youry/test_simulation/logs/sim_base_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-youry

# ─────────────────────────────────────────────────────────────────
# sim_base_train.sbatch
#
# Step 1 of 2: Base train only (data up to sim_date - 1 day).
# Outputs current/ model to run_root — then run sim_warm_windows.sbatch
# for the 26 intraday warm-refresh windows.
#
# Args:
#   --parquet <path>    Full parquet on NIBI
#   --sim-date <date>   Simulation date (default: 2026-04-07)
#   --fast              200-tree base train (flow test, ~10 min)
#
# Usage:
#   sbatch sim_base_train.sbatch \
#       --parquet /home/$USER/projects/def-youry/test_simulation/data/snapshot_2026-04-14.parquet
#
#   # After it completes:
#   sbatch sim_warm_windows.sbatch \
#       --parquet /home/$USER/projects/def-youry/test_simulation/data/snapshot_2026-04-14.parquet
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

PARQUET_FILE=""
SIM_DATE="2026-04-07"
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet)  PARQUET_FILE="$2"; shift 2 ;;
        --sim-date) SIM_DATE="$2"; shift 2 ;;
        --fast)     EXTRA_FLAGS="${EXTRA_FLAGS} --fast"; shift ;;
        --base-only) EXTRA_FLAGS="${EXTRA_FLAGS} --base-only"; shift ;;  # default behaviour, kept for explicitness
        *) shift ;;
    esac
done

if [[ -z "${PARQUET_FILE}" ]]; then
    echo "ERROR: --parquet <path> required"; exit 1
fi

SIM_DIR="/home/${USER}/projects/def-youry/test_simulation"
RUN_ROOT="${SIM_DIR}/run_root"
LOG_DIR="${SIM_DIR}/logs"
SCRIPT="${SIM_DIR}/ml/ml/nibi/run_simulation_day.py"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

echo "============================================="
echo " Base Train Job"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : ${HOSTNAME}"
echo " Sim date  : ${SIM_DATE}"
echo " Parquet   : ${PARQUET_FILE}"
echo " Run root  : ${RUN_ROOT}"
echo " GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================="

echo "[1/3] Loading modules ..."
module load gcc arrow/17.0.0

VENV_PATH="${HOME}/myenv"
if [[ -d "${VENV_PATH}" ]]; then
    echo "[2/3] Activating venv at ${VENV_PATH} ..."
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: venv not found at ${VENV_PATH}"; exit 1
fi

echo "[3/3] Running base train only ..."
python "${SCRIPT}" \
    --parquet   "${PARQUET_FILE}" \
    --run-root  "${RUN_ROOT}" \
    --sim-date  "${SIM_DATE}" \
    --base-only \
    ${EXTRA_FLAGS}

echo ""
echo "Base train complete — model at ${RUN_ROOT}/current/"
echo "Next: sbatch sim_warm_windows.sbatch --parquet ${PARQUET_FILE} --sim-date ${SIM_DATE}"

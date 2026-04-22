#!/usr/bin/env bash
#SBATCH --job-name=sim_warm_windows
#SBATCH --output=/home/%u/projects/def-youry/test_simulation/logs/sim_warm_%j.out
#SBATCH --error=/home/%u/projects/def-youry/test_simulation/logs/sim_warm_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --account=def-youry

# ─────────────────────────────────────────────────────────────────
# sim_warm_windows.sbatch
#
# Step 2 of 2: 26 intraday warm-refresh windows (09:30–15:45 ET).
# Requires sim_base_train.sbatch to have completed first so that
# run_root/current/ exists.
#
# Uses the smallest H100 MIG slice (1g.10gb = 10GB VRAM) —
# warm_trees=30 per window fits easily, near-zero queue wait.
#
# Args:
#   --parquet <path>      Full parquet on NIBI
#   --sim-date <date>     Simulation date (default: 2026-04-07)
#   --start-window N      Resume from window N if a previous run failed (0-25)
#
# Usage:
#   sbatch sim_warm_windows.sbatch \
#       --parquet /home/$USER/projects/def-youry/test_simulation/data/snapshot_2026-04-14.parquet
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

PARQUET_FILE=""
SIM_DATE="2026-04-07"
EXTRA_FLAGS="--skip-base"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet)      PARQUET_FILE="$2"; shift 2 ;;
        --sim-date)     SIM_DATE="$2"; shift 2 ;;
        --start-window) EXTRA_FLAGS="${EXTRA_FLAGS} --start-window $2"; shift 2 ;;
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

# Guard: refuse to run if base model doesn't exist
if [[ ! -d "${RUN_ROOT}/current" ]]; then
    echo "ERROR: ${RUN_ROOT}/current/ not found."
    echo "  Run sim_base_train.sbatch first."
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "============================================="
echo " Warm-Refresh Windows Job"
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

echo "[3/3] Running 26 warm-refresh windows (--skip-base) ..."
python "${SCRIPT}" \
    --parquet   "${PARQUET_FILE}" \
    --run-root  "${RUN_ROOT}" \
    --sim-date  "${SIM_DATE}" \
    ${EXTRA_FLAGS}

echo ""
echo "Warm-refresh complete — snapshots at ${RUN_ROOT}/step_00/ … step_25/"

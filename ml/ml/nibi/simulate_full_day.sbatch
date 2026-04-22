#!/usr/bin/env bash
#SBATCH --job-name=algo_sim_full_day
#SBATCH --output=/home/%u/projects/def-youry/test_simulation/logs/sim_full_day_%j.out
#SBATCH --error=/home/%u/projects/def-youry/test_simulation/logs/sim_full_day_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-youry

# ─────────────────────────────────────────────────────────────────
# simulate_full_day.sbatch
#
# Holds an 8-hour H100 GPU allocation and runs the complete
# April 7 intraday simulation in one shot:
#   1. Base train (data up to April 6)
#   2. 26 warm-refresh windows (09:30–15:45 ET)
#
# Queue wait happens ONCE. Everything runs inside this job.
# Outputs land in: test_simulation/run_root/step_XX/
#
# Args (passed after sbatch flags):
#   --parquet <path>    Full parquet on NIBI
#   --sim-date <date>   Simulation date (default: 2026-04-07)
#   --fast              200-tree base train (flow test, ~30 min total)
#   --skip-base         Skip base train, use existing current/
#   --start-window N    Resume warm refresh from window N
#
# Usage:
#   sbatch simulate_full_day.sbatch \
#       --parquet /home/$USER/projects/def-youry/test_simulation/data/snapshot_2026-04-07.parquet
#
#   # Fast flow test:
#   sbatch simulate_full_day.sbatch \
#       --parquet /home/$USER/projects/def-youry/test_simulation/data/snapshot_2026-04-07.parquet \
#       --fast
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────
PARQUET_FILE=""
SIM_DATE="2026-04-07"
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet)      PARQUET_FILE="$2"; shift 2 ;;
        --sim-date)     SIM_DATE="$2"; shift 2 ;;
        --fast)         EXTRA_FLAGS="${EXTRA_FLAGS} --fast"; shift ;;
        --skip-base)    EXTRA_FLAGS="${EXTRA_FLAGS} --skip-base"; shift ;;
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

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

echo "============================================="
echo " Full-Day Simulation Job"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : ${HOSTNAME}"
echo " Sim date  : ${SIM_DATE}"
echo " Parquet   : ${PARQUET_FILE}"
echo " Run root  : ${RUN_ROOT}"
echo " Extra     : ${EXTRA_FLAGS}"
echo "============================================="

# ── 1. Load modules ───────────────────────────────────────────────
echo "[1/3] Loading modules ..."
module load gcc arrow/17.0.0

# ── 2. Activate venv ──────────────────────────────────────────────
VENV_PATH="${HOME}/myenv"
if [[ -d "${VENV_PATH}" ]]; then
    echo "[2/3] Activating venv at ${VENV_PATH} ..."
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: venv not found at ${VENV_PATH}"
    exit 1
fi

# ── 3. Run full-day simulation ────────────────────────────────────
echo "[3/3] Starting run_simulation_day.py ..."
python "${SCRIPT}" \
    --parquet   "${PARQUET_FILE}" \
    --run-root  "${RUN_ROOT}" \
    --sim-date  "${SIM_DATE}" \
    ${EXTRA_FLAGS}

echo ""
echo "Job ${SLURM_JOB_ID} complete — artifacts at ${RUN_ROOT}/step_XX/"

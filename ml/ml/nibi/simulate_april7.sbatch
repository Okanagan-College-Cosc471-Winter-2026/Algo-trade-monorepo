#!/usr/bin/env bash
#SBATCH --job-name=algo_sim_apr7
#SBATCH --output=/scratch/%u/ml/logs/sim_apr7_%j.out
#SBATCH --error=/scratch/%u/ml/logs/sim_apr7_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=def-youry

# ─────────────────────────────────────────────────────────────────
# simulate_april7.sbatch
# Runs simulate_warm_refresh for the full April 7 intraday session:
#   - 26 steps (09:30–15:45 ET), one per 15-min bar
#   - Each step loads previous horizon models + adds 30 warm trees
#   - Outputs step artifacts to $SCRATCH/ml/simulation_2026-04-07/
#
# Submitted by tests/simulate_april7.py on the VPS.
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

PARQUET_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet) PARQUET_FILE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "${PARQUET_FILE}" ]]; then
    echo "ERROR: --parquet <path> is required"; exit 1
fi

SCRATCH="${SCRATCH:-/scratch/${USER}}"
RUN_ROOT="${SCRATCH}/ml/run_root"
SIM_OUT="${SCRATCH}/ml/simulation_2026-04-07"
LOG_DIR="${SCRATCH}/ml/logs"
REPO="${SCRATCH}/algo"

mkdir -p "${RUN_ROOT}" "${SIM_OUT}" "${LOG_DIR}"

echo "============================================="
echo " NIBI April 7 Warm-Refresh Simulation"
echo " Job ID  : ${SLURM_JOB_ID}"
echo " Node    : ${HOSTNAME}"
echo " Parquet : ${PARQUET_FILE}"
echo " Run root: ${RUN_ROOT}"
echo " Sim out : ${SIM_OUT}"
echo "============================================="

# ── 1. Load modules ───────────────────────────────────────────────
echo "[1/4] Loading modules ..."
module load gcc arrow/17.0.0

# ── 2. Activate venv ─────────────────────────────────────────────
VENV_PATH="${HOME}/ENV"
if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating venv at ${VENV_PATH} ..."
    python -m venv "${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
    pip install --quiet xgboost pandas pyarrow sqlalchemy psycopg2-binary scikit-learn
else
    source "${VENV_PATH}/bin/activate"
fi

# ── 3. Run simulation ─────────────────────────────────────────────
echo "[3/4] Running simulate_warm_refresh for 2026-04-07 ..."
SCRIPT_DIR="${REPO}/ml/ml/nibi"

python "${SCRIPT_DIR}/run_simulation.py" \
    --parquet        "${PARQUET_FILE}" \
    --run-root       "${RUN_ROOT}" \
    --output-dir     "${SIM_OUT}" \
    --replay-date    "2026-04-07" \
    --warm-trees     30

# ── 4. Signal completion ─────────────────────────────────────────
echo "[4/4] Writing DONE sentinel ..."
echo "job_id=${SLURM_JOB_ID} completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${SIM_OUT}/DONE"
echo ""
echo "Simulation complete — artifacts at: ${SIM_OUT}"

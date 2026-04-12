#!/usr/bin/env bash
#SBATCH --job-name=algo_base_train
#SBATCH --output=/scratch/%u/ml/logs/base_train_%j.out
#SBATCH --error=/scratch/%u/ml/logs/base_train_%j.err
#SBATCH --time=03:30:00
# NOTE: use --fast flag for 200-tree flow test (~20 min actual GPU time)
#       remove --fast for full 1157-tree production run (~2-3 hrs GPU time)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=def-youry

# ─────────────────────────────────────────────────────────────────
# train_base.sbatch
# Trains the base XGBoost model on data up to April 6, 2026.
# Output: $NIBI_SCRATCH/ml/run_root/current/ (26 horizon files)
#
# Args:
#   --parquet <path>     Input parquet (all history up to Apr 6)
#   --fast               Use 200 trees instead of 1157 (for flow test)
#
# Usage:
#   sbatch train_base.sbatch --parquet /scratch/$USER/data/snapshot_2026-04-06.parquet
#   sbatch train_base.sbatch --parquet /scratch/$USER/data/snapshot_2026-04-06.parquet --fast
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

PARQUET_FILE=""
FAST_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet) PARQUET_FILE="$2"; shift 2 ;;
        --fast)    FAST_MODE=true; shift ;;
        *) shift ;;
    esac
done

if [[ -z "${PARQUET_FILE}" ]]; then
    echo "ERROR: --parquet <path> required"; exit 1
fi

SCRATCH="${SCRATCH:-/scratch/${USER}}"
RUN_ROOT="${SCRATCH}/ml/run_root"
LOG_DIR="${SCRATCH}/ml/logs"
REPO="${SCRATCH}/algo"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

echo "============================================="
echo " NIBI Base Model Training"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : ${HOSTNAME}"
echo " Parquet   : ${PARQUET_FILE}"
echo " Fast mode : ${FAST_MODE}"
echo " Run root  : ${RUN_ROOT}"
echo "============================================="

# ── 1. Load modules ───────────────────────────────────────────────
echo "[1/3] Loading modules ..."
module load gcc arrow/17.0.0

# ── 2. Activate venv ─────────────────────────────────────────────
source "${HOME}/ENV/bin/activate"

# ── 3. Train ─────────────────────────────────────────────────────
echo "[3/3] Running base training (cutoff=2026-04-06) ..."

FAST_FLAG=""
if [[ "${FAST_MODE}" == "true" ]]; then
    FAST_FLAG="--fast"
    echo "  [FAST MODE] Using 200 trees for flow test"
fi

python "${REPO}/ml/ml/nibi/run_base_train.py" \
    --parquet    "${PARQUET_FILE}" \
    --run-root   "${RUN_ROOT}" \
    --cutoff     "2026-04-06" \
    ${FAST_FLAG}

echo "Base training complete — model at: ${RUN_ROOT}/current/"
echo "job_id=${SLURM_JOB_ID} completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${RUN_ROOT}/current/TRAIN_DONE"

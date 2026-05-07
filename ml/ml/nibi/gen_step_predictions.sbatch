#!/usr/bin/env bash
#SBATCH --job-name=algo_gen_predictions
#SBATCH --output=/home/%u/projects/def-youry/test_simulation/logs/gen_predictions_%j.out
#SBATCH --error=/home/%u/projects/def-youry/test_simulation/logs/gen_predictions_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-youry

set -euo pipefail

SIM_DIR="/home/${USER}/projects/def-youry/test_simulation"
RUN_ROOT="${SIM_DIR}/run_root"

SIM_DATE="${SIM_DATE:-2026-05-04}"

echo "============================================="
echo " Generate Step Prediction CSVs"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Node     : ${HOSTNAME}"
echo " Run root : ${RUN_ROOT}"
echo " Sim date : ${SIM_DATE}"
echo "============================================="

echo "[1/3] Loading modules ..."
module load gcc arrow/17.0.0

echo "[2/3] Activating venv ..."
source ~/myenv/bin/activate

echo "[3/3] Running gen_step_predictions.py ..."
python "${SIM_DIR}/ml/ml/nibi/gen_step_predictions.py" \
    --run-root "${RUN_ROOT}" \
    --sim-date "${SIM_DATE}"

echo "Job ${SLURM_JOB_ID} complete."

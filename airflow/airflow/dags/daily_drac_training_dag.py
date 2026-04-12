"""
daily_drac_training_dag.py
Airflow DAG — daily at 2:30 AM UTC.
SSH into DRAC → sbatch → poll → verify model synced to VPS.

Airflow connection required:
  Conn ID   : drac_login
  Conn Type : SSH
  Host      : <drac login node>
  Username  : <your drac username>
  Private Key File : /home/cosc-admin/.ssh/id_rsa
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.ssh.operators.ssh import SSHOperator

DRAC_SSH_CONN_ID   = "drac_login"
SBATCH_SCRIPT      = "~/the-project-maverick/ml/drac/daily_train.sbatch"
VPS_MODEL_REGISTRY = "/home/cosc-admin/the-project-maverick/ml/model_registry"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

with DAG(
    dag_id="daily_drac_xgb_training",
    default_args=DEFAULT_ARGS,
    description="Submit daily XGBoost training job on DRAC via Slurm",
    schedule_interval="30 2 * * *",
    start_date=datetime(2026, 3, 1),
    catchup=False,
    tags=["ml", "training", "drac", "xgboost"],
) as dag:

    submit_slurm_job = SSHOperator(
        task_id="submit_slurm_job",
        ssh_conn_id=DRAC_SSH_CONN_ID,
        command=f"""
            set -e
            JOB_ID=$(sbatch {SBATCH_SCRIPT} | awk '{{print $NF}}')
            echo "Submitted Slurm job ID: $JOB_ID"
            echo "$JOB_ID" > ~/last_train_job_id.txt
        """,
        cmd_timeout=60,
        conn_timeout=30,
    )

    poll_slurm_job = SSHOperator(
        task_id="poll_until_complete",
        ssh_conn_id=DRAC_SSH_CONN_ID,
        command="""
            set -e
            JOB_ID=$(cat ~/last_train_job_id.txt)
            echo "Polling Slurm job $JOB_ID ..."
            MAX_WAIT=10800
            ELAPSED=0
            INTERVAL=120
            while true; do
                STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "")
                echo "  [$ELAPSED s] state: $STATE"
                if [ "$STATE" = "" ]; then
                    RESULT=$(sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
                    echo "  Final state: $RESULT"
                    [[ "$RESULT" == COMPLETED* ]] && { echo "Job done."; exit 0; } || { echo "Job failed: $RESULT"; exit 1; }
                fi
                [[ "$STATE" == "FAILED" || "$STATE" == "CANCELLED" || "$STATE" == "TIMEOUT" ]] && { echo "Job failed: $STATE"; exit 1; }
                sleep "$INTERVAL"; ELAPSED=$((ELAPSED + INTERVAL))
                [ "$ELAPSED" -ge "$MAX_WAIT" ] && { echo "Timed out"; exit 1; }
            done
        """,
        cmd_timeout=10860,
        conn_timeout=30,
    )

    def verify_model_synced(**context):
        run_date  = context["ds"]
        model_dir = Path(VPS_MODEL_REGISTRY) / f"dt={run_date}"
        required  = ["model.pkl", "encoder.pkl", "metrics.json"]
        missing   = [f for f in required if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Sync incomplete for {run_date}. Missing: {missing}")
        with open(model_dir / "metrics.json") as mf:
            m = json.load(mf)
        print(f"Model verified for {run_date}")
        print(f"  RMSE={m.get('rmse','N/A'):.6f}  DirAcc={m.get('dir_accuracy','N/A'):.4f}  Features={m.get('n_features','N/A')}")
        latest = Path(VPS_MODEL_REGISTRY) / "latest"
        if latest.is_symlink(): latest.unlink()
        latest.symlink_to(model_dir)

    verify_model = PythonOperator(
        task_id="verify_model_synced",
        python_callable=verify_model_synced,
    )

    submit_slurm_job >> poll_slurm_job >> verify_model

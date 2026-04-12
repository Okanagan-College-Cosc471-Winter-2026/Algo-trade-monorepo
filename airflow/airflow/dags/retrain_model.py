"""
ML Model Retraining DAG
========================

Retrains the global XGBoost stock prediction model weekly.

Task flow:
    validate_data_freshness  →  train_model  →  evaluate_model  →  backup_artifacts

What each task does:
    1. validate_data_freshness  — ensure market data is recent and complete
    2. train_model              — run ml/scripts/train_global_model.py
    3. evaluate_model           — load the saved model and verify it works
    4. backup_artifacts         — copy model files to a timestamped directory

Schedule:  Every Sunday at 2:00 AM  (cron: 0 2 * * 0)
           Runs 1 hour after seed_market_data finishes.

Industry concepts demonstrated:
    - Data validation gate:  don't train on stale/bad data
    - Artifact versioning:   timestamped backups for rollback
    - Execution timeout:     prevents runaway training from blocking the scheduler
    - Separation of concerns: validation, training, evaluation are independent steps
"""

import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

CONN_ID = "postgres_app"

# Paths inside the Airflow container (from docker-compose volume mounts)
ML_DIR = "/opt/airflow/ml"
ARTIFACTS_DIR = f"{ML_DIR}/model_artifacts_global"

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,              # training is expensive — only retry once
    "retry_delay": timedelta(minutes=15),
}


# ===========================================================================
# TASK FUNCTIONS
# ===========================================================================

def validate_data_freshness(**context):
    """
    Task 1: Gate check — is the data good enough to train on?

    This is a best practice in ML pipelines.  Training on stale or incomplete
    data produces a bad model that looks fine until predictions go wrong.

    Checks:
        - At least 5 active stocks
        - Data is no more than 7 days old
        - Every stock has >= 400 rows (enough for feature engineering windows)
    """
    hook = PostgresHook(postgres_conn_id=CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()

    try:
        # Check 1: active stock count
        cur.execute("SELECT COUNT(*) FROM market.stocks WHERE is_active = true;")
        n_stocks = cur.fetchone()[0]
        if n_stocks < 5:
            raise AirflowException(
                f"Only {n_stocks} active stocks (need >= 5). "
                "Run seed_market_data DAG first."
            )
        print(f"Active stocks: {n_stocks}")

        # Check 2: data recency
        cur.execute(
            "SELECT MAX(date), CURRENT_DATE - MAX(date) "
            "FROM market.daily_prices;"
        )
        latest_date, days_old = cur.fetchone()
        if days_old is not None and days_old > 7:
            raise AirflowException(
                f"Data is {days_old} days old (latest: {latest_date}). "
                "Run seed_market_data DAG first."
            )
        print(f"Latest data: {latest_date} ({days_old} days old)")

        # Check 3: sufficient rows per stock
        cur.execute(
            "SELECT symbol, COUNT(*) AS n FROM market.daily_prices "
            "GROUP BY symbol HAVING COUNT(*) < 400;"
        )
        sparse = cur.fetchall()
        if sparse:
            symbols = [row[0] for row in sparse]
            raise AirflowException(f"Insufficient data for: {symbols}")

        # Check 4: total volume
        cur.execute("SELECT COUNT(*) FROM market.daily_prices;")
        total = cur.fetchone()[0]
        print(f"Total daily_prices rows: {total}")

        print("Data validation PASSED — ready for training.")

    finally:
        cur.close()
        conn.close()


def train_model(**context):
    """
    Task 2: Run the existing XGBoost training script.

    The script lives at ml/scripts/train_global_model.py and:
        1. Connects to Postgres and loads all OHLC data
        2. Engineers 12 features (returns, SMA, RSI, volatility, etc)
        3. Trains an XGBoost regressor with TimeSeriesSplit
        4. Saves model.json + ticker_encoder.joblib to model_artifacts_global/

    We import and call main() directly — no shell wrapping needed.
    """
    scripts_path = f"{ML_DIR}/scripts"
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    # The training script hardcodes DB_CONFIG with localhost.
    # Inside Docker, we need to override the host to 'db'.
    # We patch the module-level dict after import.
    os.environ["POSTGRES_SERVER"] = "db"

    try:
        import train_global_model

        # Override the hardcoded DB config to use Docker networking
        train_global_model.DB_CONFIG["host"] = "db"
        train_global_model.DB_CONFIG["port"] = int(
            os.getenv("POSTGRES_PORT", "5432")
        )
        train_global_model.DB_CONFIG["database"] = os.getenv("POSTGRES_DB", "app")
        train_global_model.DB_CONFIG["user"] = os.getenv(
            "POSTGRES_USER", "postgres"
        )
        train_global_model.DB_CONFIG["password"] = os.getenv(
            "POSTGRES_PASSWORD", "changethis"
        )

        # Also fix the output directory to the mounted path
        train_global_model.OUTPUT_DIR = ARTIFACTS_DIR
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # Reduce CPU usage in container (host might not have 28 cores)
        train_global_model.N_JOBS = min(train_global_model.N_JOBS, 4)

        train_global_model.main()
        print("Training completed successfully.")

        # Verify key files exist
        for name in ("global_model.json", "ticker_encoder.joblib"):
            path = Path(ARTIFACTS_DIR) / name
            if not path.exists():
                raise AirflowException(f"Expected artifact missing: {path}")
            print(f"  Artifact: {path} ({path.stat().st_size:,} bytes)")

    except AirflowException:
        raise
    except Exception as exc:
        raise AirflowException(f"Training failed: {exc}") from exc


def evaluate_model(**context):
    """
    Task 3: Smoke-test the trained model.

    In a production ML system, this step would:
        - Run inference on a holdout test set
        - Compare metrics (RMSE, MAE) to the previous model
        - Decide whether the new model is "better" before deploying

    For now, we just verify the model + encoder load without errors.
    """
    try:
        import joblib
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(f"{ARTIFACTS_DIR}/global_model.json")
        print(f"Model loaded — {model.n_features_in_} features")

        encoder = joblib.load(f"{ARTIFACTS_DIR}/ticker_encoder.joblib")
        print(f"Encoder loaded — supports {len(encoder.classes_)} stocks: "
              f"{list(encoder.classes_)}")

        print("Evaluation PASSED.")

    except Exception as exc:
        raise AirflowException(f"Model evaluation failed: {exc}") from exc


def backup_artifacts(**context):
    """
    Task 4: Create a timestamped backup of model files.

    Why?
        If next week's retrained model performs worse, you can roll back to
        this version instantly.  This is called "model versioning".

    In production, you'd push artifacts to S3/GCS or a model registry
    like MLflow.  For now, a local timestamped copy works fine.
    """
    ts = context["execution_date"].strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"{ML_DIR}/model_backups/{ts}")

    source = Path(ARTIFACTS_DIR)
    if not source.exists():
        print("WARNING: No artifacts to back up.")
        return

    try:
        shutil.copytree(source, backup_dir)
        files = [f.name for f in backup_dir.iterdir()]
        print(f"Backup created: {backup_dir}")
        print(f"Files: {files}")
    except Exception as exc:
        # Backup failure should NOT fail the whole DAG —
        # the model was already trained and evaluated successfully.
        print(f"WARNING: Backup failed (non-fatal): {exc}")


# ===========================================================================
# DAG
# ===========================================================================
with DAG(
    dag_id="retrain_model",
    default_args=default_args,
    description="Weekly ML model retraining pipeline",
    # ---------------------------------------------------------------------------
    # Cron: 0 2 * * 0
    #   minute=0  hour=2  day=*  month=*  weekday=0 (Sunday)
    #   → runs at 2:00 AM every Sunday
    #
    # This is 1 hour after the daily seed DAG (0 1 * * *), giving it time
    # to finish before training starts.
    # ---------------------------------------------------------------------------
    schedule="0 2 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["production", "ml", "training"],
    max_active_runs=1,
) as dag:

    t_validate = PythonOperator(
        task_id="validate_data_freshness",
        python_callable=validate_data_freshness,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        # Training can take a while — fail if it exceeds 30 minutes
        execution_timeout=timedelta(minutes=30),
    )

    t_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t_backup = PythonOperator(
        task_id="backup_artifacts",
        python_callable=backup_artifacts,
    )

    t_validate >> t_train >> t_evaluate >> t_backup

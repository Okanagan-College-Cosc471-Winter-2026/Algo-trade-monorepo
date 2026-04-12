import os
from datetime import datetime, timedelta
import json
from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator

# --------------------------------------------------------------------------------
# DAG: Daily Dataset Snapshot Builder
#
# Description: This DAG hits the FastAPI Backend nightly to re-build 
#              the master Parquet dataset (containing all 29 tickers).
# --------------------------------------------------------------------------------

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='build_daily_dataset_snapshot',
    default_args=default_args,
    description='Triggers the backend API to compile the nightly All Tickers Parquet + CSV snapshot.',
    schedule_interval='@daily',
    catchup=False,
    tags=['data_pipeline', 'snapshots'],
) as dag:

    # Hit the local docker Backend container's exposed API route
    # Note: `http_conn_id="backend_api"` must be set in Airflow UI, OR 
    # we can pass the explicit endpoint (http://backend:8000) directly if needed.
    # Since this is a simple local backend, we'll configure the endpoint url via native airflow HTTP.

    trigger_snapshot = SimpleHttpOperator(
        task_id='trigger_fastapi_snapshot_build',
        http_conn_id='backend_api',  # We'll need an Airflow connection mapping to http://backend:8000
        endpoint='/api/v1/data/build-snapshot',
        method='POST',
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "ticker": "ALL",
            "format": "both"
        }),
        response_check=lambda response: response.status_code == 200 and "success" in response.text,
        log_response=True,
        # Allow it up to 10 minutes to finish the 4 Million row build
        execution_timeout=timedelta(minutes=10)
    )

    trigger_snapshot

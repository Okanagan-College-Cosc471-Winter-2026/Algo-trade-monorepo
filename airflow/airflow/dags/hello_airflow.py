"""
Hello Airflow — Educational Starter DAG
========================================

This DAG teaches you core Airflow concepts by running 4 simple tasks:

    say_hello  -->  print_date  -->  read_xcom  -->  summary

Key concepts demonstrated:
    - DAG:       A collection of tasks with defined execution order.
    - Operator:  A template for what a task does (Python function, bash cmd, etc).
    - Task:      A single unit of work — one instance of an operator.
    - XCom:      "Cross-communication" — lets tasks pass small data to each other.
    - Context:   Runtime metadata Airflow injects into your functions.

How to use:
    1. Open Airflow UI at http://localhost:8081 (admin / admin)
    2. Find "hello_airflow" in the DAG list
    3. Toggle the switch ON (unpause the DAG)
    4. Click the play button to trigger a manual run
    5. Click on each task -> "Log" tab to see the print() output
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# DEFAULT ARGS — applied to every task in this DAG unless overridden
# ---------------------------------------------------------------------------
# Think of this as a "template" for task settings.
# Real-world DAGs set retries, alerting emails, timeouts, etc. here.
default_args = {
    "owner": "airflow",             # Shows up in the UI — who owns this DAG
    "depends_on_past": False,       # Each run is independent of previous runs
    "email_on_failure": False,      # No email alerts (SMTP not configured)
    "retries": 1,                   # If a task fails, retry it once
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# DAG DEFINITION
# ---------------------------------------------------------------------------
# The `with DAG(...) as dag:` block defines the DAG and all tasks inside it.
#
# Key parameters:
#   dag_id          — unique name shown in the UI
#   schedule        — when to run: '@daily', '@hourly', cron string, or None
#   start_date      — the DAG won't schedule runs before this date
#   catchup=False   — DON'T backfill all dates between start_date and today
#   tags            — organize DAGs in the UI sidebar
with DAG(
    dag_id="hello_airflow",
    default_args=default_args,
    description="Educational DAG — learn Airflow concepts here",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["learning", "example"],
) as dag:

    # -------------------------------------------------------------------
    # TASK 1: PythonOperator — runs a Python function
    # -------------------------------------------------------------------
    def greet(**context):
        """
        A simple Python function executed as an Airflow task.

        The **context dict contains runtime metadata injected by Airflow:
            context['execution_date']  — logical date of this DAG run
            context['task_instance']   — the TaskInstance object (for XCom)
            context['dag_run']         — info about the current DAG run

        XCom push:  save a small value that other tasks can retrieve.
        """
        exec_date = context["execution_date"]
        print(f"Hello from Airflow!  Execution date: {exec_date}")

        # Push a value into XCom (key-value store between tasks)
        # Other tasks can pull this value using xcom_pull()
        context["task_instance"].xcom_push(key="greeting", value="Hello World")

    task_hello = PythonOperator(
        task_id="say_hello",          # unique ID within this DAG
        python_callable=greet,        # the function to run
    )

    # -------------------------------------------------------------------
    # TASK 2: BashOperator — runs a shell command
    # -------------------------------------------------------------------
    # Great for quick system commands, curl calls, or running scripts.
    task_date = BashOperator(
        task_id="print_date",
        bash_command="echo 'Current time:' && date",
    )

    # -------------------------------------------------------------------
    # TASK 3: Read XCom value from Task 1
    # -------------------------------------------------------------------
    def read_xcom(**context):
        """
        Pull the value that task_hello pushed into XCom.

        xcom_pull() parameters:
            task_ids — which task pushed the value
            key      — which key to look up (default is 'return_value')
        """
        greeting = context["task_instance"].xcom_pull(
            task_ids="say_hello",
            key="greeting",
        )
        print(f"Retrieved from XCom: {greeting}")

    task_xcom = PythonOperator(
        task_id="read_xcom",
        python_callable=read_xcom,
    )

    # -------------------------------------------------------------------
    # TASK 4: Summary — access DAG run metadata
    # -------------------------------------------------------------------
    def summarize(**context):
        """Show metadata about this DAG run."""
        dag_run = context["dag_run"]
        print(f"DAG Run ID:     {dag_run.run_id}")
        print(f"DAG ID:         {dag_run.dag_id}")
        print(f"Logical Date:   {dag_run.logical_date}")
        print("All tasks completed successfully!")

    task_summary = PythonOperator(
        task_id="summary",
        python_callable=summarize,
    )

    # -------------------------------------------------------------------
    # DEPENDENCIES — define execution order
    # -------------------------------------------------------------------
    # The >> operator means "runs before".
    # This creates a linear chain:
    #
    #   say_hello  →  print_date  →  read_xcom  →  summary
    #
    # If say_hello fails, none of the downstream tasks will run.
    task_hello >> task_date >> task_xcom >> task_summary

"""Airflow DAG template for Voyage Analytics model pipeline.

This DAG orchestrates the model validation, registration, and deployment workflow.
It does NOT train the model - that remains in Google Colab.

DAG Flow:
    validate_artifact
           ↓
    register_model
           ↓
    validate_service
           ↓
    deployment_ready
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# Default arguments for the DAG
default_args = {
    "owner": "Voyage Analytics",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG configuration
dag = DAG(
    "voyage_model_pipeline",
    default_args=default_args,
    description="Orchestrate model validation, registration, and deployment for Voyage Analytics",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["voyage-analytics", "mlops"],
)

# Model paths (configure via environment variables)
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/flight_price_pipeline.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "artifacts/model_metadata.json")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "voyage-flight-price")

# Start marker
start = EmptyOperator(
    task_id="start",
    dag=dag,
)

# Task 1: Validate artifact exists and is loadable
validate_artifact = BashOperator(
    task_id="validate_artifact",
    bash_command="""
        python scripts/validate_artifact.py \
            --artifact {{ var.value.MODEL_PATH|default('artifacts/flight_price_pipeline.joblib') }} \
            --metadata {{ var.value.METADATA_PATH|default('artifacts/model_metadata.json') }}
    """,
    env={
        "PYTHONPATH": "/opt/airflow/dags/voyage-analytics",
    },
    dag=dag,
)

# Task 2: Register model with MLflow
register_model = PythonOperator(
    task_id="register_model",
    python_callable=lambda: __import__("mlflow_tracking.tracking", fromlist=["ModelTracker"]).ModelTracker().log_model("{{ var.value.MODEL_PATH|default('artifacts/flight_price_pipeline.joblib') }}", registered=True),
    dag=dag,
)

# Task 3: Validate service endpoints
validate_service = BashOperator(
    task_id="validate_service",
    bash_command="""
        # Start API and test endpoints
        python -m api.app --host 0.0.0.0 --port 5000 &
        sleep 3
        curl -f http://localhost:5000/api/health || exit 1
        curl -f http://localhost:5000/api/model-info || exit 1
        curl -f -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d '{}' || exit 1
        pkill -f "python -m api.app"
    """,
    dag=dag,
)

# Task 4: Signal deployment ready
deployment_ready = BashOperator(
    task_id="deployment_ready",
    bash_command="echo 'Model is ready for deployment!'",
    dag=dag,
)

# End marker
end = EmptyOperator(
    task_id="end",
    dag=dag,
)

# Define DAG dependencies
start >> validate_artifact >> register_model >> validate_service >> deployment_ready >> end

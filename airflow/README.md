# Airflow DAG template for Voyage Analytics

This directory contains Airflow DAG templates for orchestrating the Voyage Analytics pipeline.

## Purpose

The DAG orchestrates the model validation, registration, and deployment workflow. It does NOT train the model - that remains in Google Colab.

## DAG Flow

```
validate_artifact
       ↓
register_model
       ↓
validate_service
       ↓
deployment_ready
```

## Configuration

| Task | Purpose |
|------|---------|
| `validate_artifact` | Check model artifact exists and is valid |
| `register_model` | Register model with MLflow |
| `validate_service` | Test API endpoints with model |
| `deployment_ready` | Signal that deployment can proceed |

## Usage

1. Copy the DAG to your Airflow `dags/` directory
2. Configure environment variables for MLflow and model paths
3. Trigger the DAG from Airflow UI or CLI

## Example

```bash
# Trigger DAG manually
airflow dags trigger voyage_model_pipeline

# Check DAG status
airflow dags status voyage_model_pipeline
```

## Extending the DAG

Add additional tasks for:

- CI/CD integration
- Slack notifications
- Rollback procedures
- Canary deployments
- A/B testing support

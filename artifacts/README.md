# Model artifact handoff directory

This directory receives the trained model artifacts from the ML team's Google Colab notebook.

## Expected Structure

```text
artifacts/
├── flight_price_pipeline.joblib    # Main model artifact (REQUIRED)
├── metrics.json                     # Training metrics (OPTIONAL)
├── feature_schema.json              # Feature definitions (OPTIONAL)
└── model_metadata.json              # Model metadata (OPTIONAL)
```

## How to Copy Artifacts from Colab

After training in Google Colab, copy the following files to this directory:

```python
# In Colab notebook (after training)
from google.colab import files

# Save model
joblib.dump(pipeline, 'flight_price_pipeline.joblib')

# Save metadata
import json
metadata = {
    "model_name": "flight_price_regression",
    "model_version": "1.0",
    "algorithm": "...",
    "training_date": "2024-...",
    "metrics": {...}
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Download files (run locally after Colab export)
files.download('flight_price_pipeline.joblib')
files.download('model_metadata.json')
files.download('metrics.json')
files.download('feature_schema.json')
```

## Copy to this project

Move downloaded files to:

```bash
cp flight_price_pipeline.joblib ../voyage-analytics/artifacts/
cp model_metadata.json ../voyage-analytics/artifacts/
cp metrics.json ../voyage-analytics/artifacts/
cp feature_schema.json ../voyage-analytics/artifacts/
```

## Using the Production Code

Once artifacts are in place, run:

```bash
# Ensure MODEL_PATH points to the artifact
export MODEL_PATH=artifacts/flight_price_pipeline.joblib

# Run the API
python -m api.app
```

The production code will automatically load the real model artifact.

## Development Mode

For local development, a dummy model is provided at:

```text
tests/fixtures/dummy_model.joblib
```

This is used when no real artifact is available. **Do not commit real model artifacts to git.**

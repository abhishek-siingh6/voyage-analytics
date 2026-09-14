# Features

Shared, deterministic feature engineering used identically by training and
serving to avoid train/serve skew.

| File | Purpose |
|------|---------|
| `gender_features.py` | Builds the gender classifier's feature row (`first_name`, `age`) |

The gender feature builder derives the lower-cased **first name** from the full
user name (the dominant signal for gender) plus the numeric age. It is used by
both `scripts/train_gender_model.py` and `src/services/gender_service.py` so
the deployed model receives the exact features it was trained on.

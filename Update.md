# Update.md — Known Issues & Open Requirements

This file records items that do not belong to a single folder's README — they
are cross-cutting concerns, environment caveats, and open requirements that
should be resolved as the project moves toward production.

---

## 1. MLflow registry is not live yet

The tracking integration is implemented and verified (scripts log metrics,
params and artifacts to a store). However:

- No MLflow **server is currently running**; the bundled compose starts one on
  port 5001 (`docker compose -f docker/docker-compose.mlflow.yml up -d`).
- **No model has been registered** in a live registry yet. Run
  `python scripts/track_models.py` against a running server to populate
  `voyage_flight_price`, `voyage_gender` and `voyage_recommendation`.

## 2. Model-flavour logging depends on a healthy Python environment

MLflow's `mlflow.sklearn` / `mlflow.xgboost` flavour code imports `torch`. On
the current Windows / Python 3.13 machine, torch is a broken native build and
fails with `OSError: [WinError 1114] ... c10.dll`. The tracking script handles
this gracefully (logs the raw artifact + metrics + metadata and warns), but
**model-flavour logging and registry serving require a working environment**
(e.g. the Linux CI runner or the Docker MLflow image).

## 3. Flight-price model artifact is git-ignored

`artifacts/flight_price_pipeline.joblib` is intentionally not committed (large
ML-team handoff). Consequences:

- A fresh clone / CI cannot run `/api/predict` until the model is placed there
  (the flight-price tests are skipped when it is absent).
- The Docker image built in CI will not include it.
- **Decision required**: commit the artifact for the deliverable, or keep the
  handoff convention and document how to inject it at build/deploy time.

## 4. Raw datasets are bundled but large

`artifacts/data/` contains the three CSVs (~27MB total). They are committed as
requested, but this grows the repository. If the repo becomes too large,
consider moving the data to an external store / object storage and fetching it
in the build pipeline instead.

## 5. Gender model accuracy is limited by the data

The gender classifier reaches ~0.52 accuracy. Analysis shows the dataset has
genuine label noise (the same first names appear across `male`, `female` and
`none`), not a pipeline error. To improve:

- Reframe to a 2-class task (`male`/`female`), and/or
- Use an external name-to-gender resource, and/or
- Engineer additional features from the `users.csv` profile.

## 6. Kubernetes is configured but not deployed

Manifests are updated and valid, but they have **not been applied to a real
cluster**. Remaining to demonstrate:

- `kubectl apply -f kubernetes/`
- An Ingress for external access
- An HPA / autoscaling policy (the scalability objective)
- A live `mlflow-service` reference inside the cluster

## 7. Project objectives 5 & 6 are undefined

The project brief (from the provided PPT) omits objectives 5 and 6. Confirm
what these were intended to be before they can be completed.

## 8. Airflow DAG is a template, not scheduled

`airflow/dags/voyage_model_pipeline.py` exists as an orchestration template but
is not wired into a running Airflow instance. If scheduled retraining /
re-registration is required, the DAG needs to be connected to real training
steps and an MLflow server.

## 9. Observability is minimal

There is no metrics endpoint (e.g. Prometheus), model-drift monitoring, or
alerting. Adding a `/metrics` endpoint and per-model health/quality checks
would complete the production monitoring story.

---

## Conventions

- Per-component documentation lives **inside that component's folder**
  (`api/README.md`, `scripts/README.md`, `docker/README.md`, …).
- This root `Update.md` holds cross-cutting notes only.

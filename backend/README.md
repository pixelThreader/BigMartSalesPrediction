BigMart Backend

Overview

- This backend provides model training, prediction, performance reporting, and dataset browsing APIs for the BigMart sales project.
- The app is built with FastAPI and runs on localhost:8000.
- The model is a scikit-learn pipeline using mean imputation plus linear regression.

Tech Stack

- Python 3.13+
- FastAPI
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

Project Structure

- main.py: FastAPI app and API endpoints
- model/model_service.py: model training, loading, prediction, report and chart generation
- dataset/BigMartSalesProd.csv: default dataset
- model/bin/: saved model artifacts
- model/bin/reports/: per-run metrics and charts
- model/bin/temp_exports/: frontend-friendly temporary chart copies

Run Locally

1. Create and activate a virtual environment.
2. Install dependencies from pyproject.toml.
3. Start the server:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

Base URL

- http://localhost:8000

Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

API Endpoints

1. GET /

- Purpose: Basic server welcome message.
- Response:
  - message: BigMart backend is running

2. GET /health

- Purpose: Health check.
- Response:
  - status: ok

3. POST /api/v1/train

- Purpose: Trigger new training, save timestamped model, save metrics and charts, and return frontend-ready assets.
- Request body:
  - dataset_path: string, optional
  - model_path: string, optional
  - test_size: float in (0, 1), optional, default 0.2
  - random_state: integer, optional, default 42
- Response includes:
  - message
  - run_id
  - stats: mae, mse, rmse, r2, mape
  - performance.metrics: mae, mse, rmse, r2, mape
  - performance.graphs: list of graph descriptors with id, file_name, path, url
  - model_path
  - report_dir
  - metrics_path
  - plots and plot_urls
  - temp_plot_dir
  - temp_plots and temp_plot_urls

4. POST /api/v1/predict

- Purpose: Run inference for one record or multiple records.
- Request body:
  - records: object or list of objects
  - model_path: optional string path to a specific `.joblib` model artifact
- Required model features in each record:
  - Item_Weight
  - Item_Visibility
  - Item_MRP
  - Outlet_Establishment_Year
- Response:
  - model: details for the resolved model artifact
  - model_path
  - required_features
  - predictions

5. POST /api/v1/models/predict

- Purpose: Same as `/api/v1/predict`, but explicit about using a selected model artifact.
- Request body:
  - records: object or list of objects
  - model_path: required when you want a custom model, optional otherwise
- Response:
  - model: details for the resolved model artifact
  - model_path
  - required_features
  - predictions

6. GET /api/v1/models

- Purpose: List all discovered `.joblib` model artifacts in `model/bin`.
- Response:
  - default_model_path
  - model_count
  - models: each item includes model_path, model_name, model_stem, size_bytes, timestamps, report metadata, and metrics when available

7. GET /api/v1/models/details

- Purpose: Inspect one specific model artifact.
- Query params:
  - model_path: optional string path to a model file; defaults to `model/bin/bigmart_model.joblib`
- Response:
  - model_path
  - model_name
  - model_stem
  - size_bytes
  - created_at
  - modified_at
  - report_dir
  - metrics_path
  - has_report
  - dataset_path
  - test_size
  - random_state
  - metrics
  - plots
  - feature_names
  - target

8. POST /api/v1/synthetic-data

- Purpose: Generate prediction-ready sample rows by randomly sampling existing rows from the dataset.
- Request body:
  - count: integer, required, how many rows to generate
  - dataset_path: optional string
  - random_state: optional integer, default 42
  - include_target: optional boolean, default false
- Response:
  - dataset_path
  - requested_count
  - actual_count
  - sample_with_replacement
  - required_features
  - records: prediction-ready feature rows
  - source_rows: sampled original rows
  - targets: included only when `include_target` is true

9. POST /api/v1/models/compare

- Purpose: Compare multiple model artifacts in a single response.
- Request body:
  - model_paths: list of model paths
  - dataset_path: optional string for evaluation fallback
  - sample_size: optional integer for evaluation fallback
  - random_state: optional integer, default 42
  - ranking_metric: optional string, default `r2`
- Response:
  - dataset_path
  - sample_size
  - random_state
  - ranking_metric
  - model_count
  - models: raw per-model comparison entries
  - ranking: models sorted by the ranking metric
  - best_model

10. GET /api/v1/reports/latest

- Purpose: Return the latest saved training report.
- Response:
  - metrics_file
  - model_path
  - dataset_path
  - test_size
  - random_state
  - metrics
  - plots
  - plot_urls

6. GET /api/v1/dataset

- Purpose: Fetch dataset rows in paginated form for frontend tables.
- Query params:
  - page: integer >= 1, default 1
  - page_size: integer 1 to 500, default 20
  - dataset_path: string, optional
- Response:
  - dataset_path
  - columns
  - pagination: page, page_size, total_records, total_pages, has_next, has_prev
  - data: list of row objects

Artifacts and File Serving

- Static route /artifacts serves files from model/bin.
- Example chart URL pattern:
  - /artifacts/temp_exports/{run_id}/actual-vs-predicted.png
- This allows frontend apps on the same machine to directly render saved graph images.

Training Outputs

- Saved model file pattern:
  - model/bin/bigmart_model_YYYYMMDD_HHMMSS.joblib
- Saved report files:
  - model/bin/reports/{run_id}/metrics.json
  - model/bin/reports/{run_id}/\*.png
- Temporary frontend chart bundle:
  - model/bin/temp_exports/{run_id}/\*.png

Important Behavior Notes

- If a default model file is missing, prediction loading falls back to the latest timestamped model.
- Temporary chart folders are automatically cleaned to keep only recent runs.
- CORS is enabled for simple same-machine frontend integration.

Common Error Cases

- 400 on train/predict: invalid payload or missing required columns/features.
- 404 on latest report: no training report exists yet.
- 404 on dataset endpoint: provided dataset_path does not exist.

Quick Frontend Integration Flow

1. Call GET /health at app startup.
2. Use POST /api/v1/train when user clicks Train.
3. Read stats and performance.graphs from train response.
4. Render graph image URLs directly.
5. Use GET /api/v1/models to populate a model picker.
6. Use GET /api/v1/models/details when the user selects a specific artifact.
7. Use POST /api/v1/synthetic-data to generate prediction-ready rows for the UI.
8. Use POST /api/v1/predict or POST /api/v1/models/predict for inference.
9. Use POST /api/v1/models/compare to render side-by-side model comparisons.
10. Use GET /api/v1/dataset with page and page_size for table UI.

Model Module Documentation

Overview

- This module contains the full ML workflow used by the backend:
  - Data loading and validation
  - Model training and retraining
  - Timestamped model saving
  - Metrics computation
  - Performance chart generation
  - Model loading and inference

Primary File

- core.py

Compatibility Wrapper

- model_service.py re-exports the public API from core.py for existing imports.

Module Split

- training_logic.py
  - Training and prediction helpers only
  - load_training_data
  - train_and_save_model
  - retrain_model
  - load_model
  - predict_dataframe
  - predict_records

- data_logic.py
  - Non-training helpers
  - get_model_details
  - list_available_models
  - generate_synthetic_dataset
  - get_trainable_columns
  - evaluate_model_on_dataset
  - compare_models

Model Type

- Pipeline:
  - SimpleImputer with mean strategy
  - LinearRegression

Dataset

- Default dataset path:
  - ../dataset/BigMartSalesProd.csv

Features and Target

- Input features:
  - Item_Weight
  - Item_Visibility
  - Item_MRP
  - Outlet_Establishment_Year
- Target:
  - Item_Outlet_Sales

Saved Artifact Locations

- Model directory:
  - model/bin
- Report directory:
  - model/bin/reports

Model Save Pattern

- Each training run saves a new model file with timestamp:
  - bigmart_model_YYYYMMDD_HHMMSS.joblib

Generated Reports Per Run

- metrics.json
- actual_vs_predicted.png
- residuals_vs_predicted.png
- residual_distribution.png
- actual_vs_predicted_distribution.png
- feature_coefficients.png

Core Functions

1. load_training_data(dataset_path)

- Reads CSV and validates required columns.
- Returns x_data and y_data.

2. train_and_save_model(dataset_path, model_path, test_size, random_state)

- Splits data into train and validation.
- Trains pipeline.
- Computes metrics:
  - mae
  - mse
  - rmse
  - r2
  - mape
- Saves timestamped model file.
- Saves report metrics.json and charts.
- Returns metrics and artifact paths.

3. retrain_model(dataset_path, model_path)

- Convenience wrapper over train_and_save_model.

4. load_model(model_path)

- Loads requested model if present.
- Falls back to latest timestamped model if default path is missing.

5. predict_dataframe(df, model)

- Validates feature columns.
- Returns model predictions as numpy array.

6. predict_records(records, model)

- Accepts one record or list of records.
- Converts to DataFrame and returns float predictions.

Internal Helper Methods

- \_validate_columns
- \_build_pipeline
- \_build_timestamped_model_path
- \_find_latest_versioned_model
- \_compute_regression_metrics
- \_save_performance_plots

Integration with API Layer

- The backend API imports and uses the public functions exposed by model_service.py.
- The actual implementation lives in core.py, which is the file to show when explaining the ML logic.
- The API also serves generated files under static artifacts for frontend graph rendering.

How to Trigger Training from CLI

- From backend root, run:
  - python -m model.model_service

Expected Training Console Output

- Validation metric values
- Saved model path
- Saved report directory
- Saved metrics.json path

Validation and Error Behavior

- Missing required training columns raises ValueError.
- Empty prediction records raise ValueError.
- Missing model file raises FileNotFoundError unless fallback model exists.

Notes

- NUMERIC_FEATURES and CATEGORICAL_FEATURES are present for extensibility.
- Current implementation uses only numeric features with linear regression.

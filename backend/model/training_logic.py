from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_DIR = Path(__file__).resolve().parent / "bin"
REPORTS_DIR = MODEL_DIR / "reports"
DEFAULT_MODEL_PATH = MODEL_DIR / "bigmart_model.joblib"
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "dataset" / "BigMartSalesProd.csv"
)
FEATURES = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]
TARGET = "Item_Outlet_Sales"
NUMERIC_FEATURES = FEATURES.copy()
CATEGORICAL_FEATURES: list[str] = []


def _read_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")
    return pd.read_csv(path)


def _pick_features(
    df: pd.DataFrame, feature_columns: Sequence[str] | None
) -> list[str]:
    selected = list(feature_columns) if feature_columns is not None else list(FEATURES)
    if TARGET in selected:
        raise ValueError(f"{TARGET} cannot be included in feature_columns")
    missing = [column for column in selected if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not all(pd.api.types.is_numeric_dtype(df[column]) for column in selected):
        raise ValueError("Only numeric feature columns are supported")
    return selected


def load_training_data(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    return load_training_data_with_columns(
        dataset_path=dataset_path, feature_columns=FEATURES
    )


def load_training_data_with_columns(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    feature_columns: Sequence[str] | None = None,
    target_column: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    df = _read_dataset(dataset_path)
    selected_features = _pick_features(df, feature_columns)
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    return df[selected_features].copy(), df[target_column].copy()


def _build_model() -> Pipeline:
    return Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("regressor", LinearRegression())]
    )


def _save_performance_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred
    plot_paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18)
    min_axis = float(min(np.min(y_true), np.min(y_pred)))
    max_axis = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([min_axis, max_axis], [min_axis, max_axis], "r--", linewidth=1.5)
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.grid(alpha=0.3)
    actual_vs_pred_path = output_dir / "actual_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(actual_vs_pred_path, dpi=150)
    plt.close(fig)
    plot_paths["actual_vs_predicted"] = str(actual_vs_pred_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, s=18)
    ax.axhline(0.0, color="r", linestyle="--", linewidth=1.5)
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted Sales")
    ax.set_ylabel("Residuals")
    ax.grid(alpha=0.3)
    residuals_path = output_dir / "residuals_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(residuals_path, dpi=150)
    plt.close(fig)
    plot_paths["residuals_vs_predicted"] = str(residuals_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=40, alpha=0.75, color="#1f77b4")
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    residual_hist_path = output_dir / "residual_distribution.png"
    fig.tight_layout()
    fig.savefig(residual_hist_path, dpi=150)
    plt.close(fig)
    plot_paths["residual_distribution"] = str(residual_hist_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(y_true, bins=40, alpha=0.6, label="Actual")
    ax.hist(y_pred, bins=40, alpha=0.6, label="Predicted")
    ax.set_title("Actual vs Predicted Distribution")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    dist_path = output_dir / "actual_vs_predicted_distribution.png"
    fig.tight_layout()
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    plot_paths["actual_vs_predicted_distribution"] = str(dist_path)

    return plot_paths


def train_and_save_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_columns: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    x_data, y_data = load_training_data_with_columns(dataset_path, feature_columns)
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=test_size, random_state=random_state
    )
    model = _build_model().fit(x_train, y_train)
    predictions = model.predict(x_val)
    output_path = Path(model_path).with_name(
        f"{Path(model_path).stem}_{pd.Timestamp.now():%Y%m%d_%H%M%S}{Path(model_path).suffix}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    report_dir = REPORTS_DIR / output_path.stem
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "metrics.json"
    y_true = y_val.to_numpy()
    plot_paths = _save_performance_plots(y_true, predictions, report_dir)
    metrics = {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "mse": float(mean_squared_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "mape": float(
            np.mean(np.abs((y_true - predictions) / np.maximum(np.abs(y_true), 1e-8)))
            * 100
        ),
    }
    payload = {
        "model_path": str(output_path),
        "dataset_path": str(dataset_path),
        "test_size": test_size,
        "random_state": random_state,
        "target_column": TARGET,
        "feature_columns": list(x_data.columns),
        "plots": plot_paths,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "model_path": str(output_path),
        "report_dir": str(report_dir),
        "metrics_path": str(metrics_path),
        "target_column": TARGET,
        "feature_columns": list(x_data.columns),
        **metrics,
        "plots": plot_paths,
    }


def retrain_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    return train_and_save_model(
        dataset_path=dataset_path,
        model_path=model_path,
        feature_columns=feature_columns,
    )


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


def predict_dataframe(
    df: pd.DataFrame,
    model: Pipeline | None = None,
    feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    selected_features = (
        list(feature_columns) if feature_columns is not None else list(FEATURES)
    )
    missing = [column for column in selected_features if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    inference_model = model if model is not None else load_model()
    return np.asarray(inference_model.predict(df[selected_features]))


def predict_records(
    records: dict[str, Any] | list[dict[str, Any]],
    model: Pipeline | None = None,
    feature_columns: Sequence[str] | None = None,
) -> list[float]:
    if isinstance(records, dict):
        records = [records]
    if not records:
        raise ValueError("records must contain at least one item")
    return [
        float(value)
        for value in predict_dataframe(
            pd.DataFrame(records), model=model, feature_columns=feature_columns
        )
    ]

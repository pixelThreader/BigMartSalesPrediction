from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from typing import cast
from typing import Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

FEATURES = [
    "Item_Weight",
    "Item_Visibility",
    "Item_MRP",
    "Outlet_Establishment_Year",
]
TARGET = "Item_Outlet_Sales"

NUMERIC_FEATURES = [
    "Item_Weight",
    "Item_Visibility",
    "Item_MRP",
    "Outlet_Establishment_Year",
]

CATEGORICAL_FEATURES = [
    # Intentionally unused in this simplified linear-regression-only version.
]


def _resolve_dataset_path(dataset_path: str | Path) -> Path:
    resolved_dataset_path = Path(dataset_path)
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {resolved_dataset_path}")
    return resolved_dataset_path


def _read_dataset(dataset_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(_resolve_dataset_path(dataset_path))


def _build_feature_frame(
    df: pd.DataFrame, feature_columns: Sequence[str], target_column: str = TARGET
) -> pd.DataFrame:
    if target_column in feature_columns:
        raise ValueError(f"{target_column} cannot be included in feature_columns")
    _validate_columns(df, list(feature_columns) + [target_column])
    return df.loc[:, list(feature_columns)].copy()


def get_trainable_columns(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    target_column: str = TARGET,
) -> dict[str, Any]:
    df = _read_dataset(dataset_path)
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    all_input_columns = [column for column in df.columns if column != target_column]
    trainable_columns = [
        column
        for column in all_input_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    column_details: list[dict[str, Any]] = []
    for column in all_input_columns:
        column_details.append(
            {
                "name": column,
                "dtype": str(df[column].dtype),
                "is_numeric": pd.api.types.is_numeric_dtype(df[column]),
                "missing_values": int(df[column].isna().sum()),
                "unique_values": int(df[column].nunique(dropna=True)),
            }
        )

    return {
        "dataset_path": str(_resolve_dataset_path(dataset_path)),
        "target_column": target_column,
        "trainable_columns": trainable_columns,
        "all_input_columns": all_input_columns,
        "default_feature_columns": FEATURES,
        "column_details": column_details,
    }


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("regressor", LinearRegression()),
        ]
    )


def _build_timestamped_model_path(model_path: str | Path) -> Path:
    base_path = Path(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path.with_name(f"{base_path.stem}_{timestamp}{base_path.suffix}")


def _find_latest_versioned_model(model_path: str | Path) -> Path | None:
    base_path = Path(model_path)
    pattern = f"{base_path.stem}_*{base_path.suffix}"
    candidates = sorted(base_path.parent.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def _resolve_model_path(model_path: str | Path) -> Path:
    load_path = Path(model_path)
    if load_path.exists():
        return load_path

    latest_model = _find_latest_versioned_model(load_path)
    if latest_model is None:
        raise FileNotFoundError(f"Model file not found at {load_path}")
    return latest_model


def _metrics_path_for_model(model_path: str | Path) -> Path:
    resolved_model_path = _resolve_model_path(model_path)
    return REPORTS_DIR / resolved_model_path.stem / "metrics.json"


def _read_metrics_payload(model_path: str | Path) -> dict[str, Any] | None:
    metrics_path = _metrics_path_for_model(model_path)
    if not metrics_path.exists():
        return None

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def get_model_details(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    resolved_model_path = _resolve_model_path(model_path)
    metrics_payload = _read_metrics_payload(resolved_model_path)
    file_stat = resolved_model_path.stat()

    feature_columns = list(metrics_payload.get("feature_columns", FEATURES)) if metrics_payload else list(FEATURES)
    numeric_feature_columns = (
        list(metrics_payload.get("numeric_feature_columns", [])) if metrics_payload else []
    )
    categorical_feature_columns = (
        list(metrics_payload.get("categorical_feature_columns", [])) if metrics_payload else []
    )

    model_details: dict[str, Any] = {
        "model_path": str(resolved_model_path),
        "model_name": resolved_model_path.name,
        "model_stem": resolved_model_path.stem,
        "size_bytes": file_stat.st_size,
        "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        "report_dir": str(REPORTS_DIR / resolved_model_path.stem),
        "metrics_path": str(_metrics_path_for_model(resolved_model_path)),
        "has_report": metrics_payload is not None,
        "dataset_path": None,
        "test_size": None,
        "random_state": None,
        "metrics": {},
        "plots": {},
        "feature_names": feature_columns,
        "feature_columns": feature_columns,
        "numeric_feature_columns": numeric_feature_columns,
        "categorical_feature_columns": categorical_feature_columns,
        "target": TARGET,
    }

    if metrics_payload is not None:
        model_details.update(
            {
                "dataset_path": metrics_payload.get("dataset_path"),
                "test_size": metrics_payload.get("test_size"),
                "random_state": metrics_payload.get("random_state"),
                "metrics": metrics_payload.get("metrics", {}),
                "plots": metrics_payload.get("plots", {}),
            }
        )

    return model_details


def list_available_models() -> list[dict[str, Any]]:
    model_files = sorted(
        [path for path in MODEL_DIR.glob("*.joblib") if path.is_file()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return [get_model_details(model_path=model_file) for model_file in model_files]


def generate_synthetic_dataset(
    count: int,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    random_state: int = 42,
    include_target: bool = False,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    df = _read_dataset(dataset_path)

    selected_features = list(feature_columns) if feature_columns is not None else list(FEATURES)
    required_columns = selected_features + ([TARGET] if include_target else [])
    _validate_columns(df, required_columns)

    sample_with_replacement = count > len(df)
    sampled_df = df.sample(
        n=count,
        replace=sample_with_replacement,
        random_state=random_state,
    ).copy()
    sampled_df = sampled_df.where(pd.notnull(sampled_df), None)

    source_columns = selected_features + ([TARGET] if include_target else [])
    feature_df = sampled_df.loc[:, selected_features].copy()
    source_df = sampled_df.loc[:, source_columns].copy()

    response: dict[str, Any] = {
        "dataset_path": str(_resolve_dataset_path(dataset_path)),
        "requested_count": count,
        "actual_count": len(sampled_df),
        "sample_with_replacement": sample_with_replacement,
        "required_features": selected_features,
        "feature_columns": selected_features,
        "records": feature_df.to_dict(orient="records"),
        "source_rows": source_df.to_dict(orient="records"),
    }

    if include_target and TARGET in sampled_df.columns:
        response["targets"] = sampled_df[TARGET].tolist()

    return response


def evaluate_model_on_dataset(
    model_path: str | Path,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    sample_size: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    resolved_dataset_path = Path(dataset_path)
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {resolved_dataset_path}")

    df = pd.read_csv(resolved_dataset_path)
    _validate_columns(df, FEATURES + [TARGET])

    if sample_size is not None:
        sample_with_replacement = sample_size > len(df)
        evaluation_df = df.sample(
            n=sample_size,
            replace=sample_with_replacement,
            random_state=random_state,
        ).copy()
    else:
        sample_with_replacement = False
        evaluation_df = df.copy()

    evaluation_df = evaluation_df.where(pd.notnull(evaluation_df), None)

    model = load_model(model_path)
    predictions = predict_dataframe(evaluation_df.loc[:, FEATURES], model=model)
    actual_values = evaluation_df[TARGET].to_numpy()
    metrics = _compute_regression_metrics(y_true=actual_values, y_pred=predictions)

    return {
        "model_path": str(_resolve_model_path(model_path)),
        "dataset_path": str(resolved_dataset_path),
        "sample_size": int(len(evaluation_df)),
        "sample_with_replacement": sample_with_replacement,
        "metrics": metrics,
        "predictions": [float(value) for value in predictions],
    }


def compare_models(
    model_paths: Sequence[str | Path],
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    sample_size: int | None = None,
    random_state: int = 42,
    ranking_metric: str = "r2",
) -> dict[str, Any]:
    if not model_paths:
        raise ValueError("model_paths must contain at least one model")

    comparison_rows: list[dict[str, Any]] = []

    for model_path in model_paths:
        model_details = get_model_details(model_path)
        metrics_source = "report"
        metrics = dict(model_details.get("metrics", {}))
        evaluation_summary: dict[str, Any] | None = None

        if not metrics:
            evaluation_summary = evaluate_model_on_dataset(
                model_path=model_path,
                dataset_path=dataset_path,
                sample_size=sample_size,
                random_state=random_state,
            )
            metrics = evaluation_summary["metrics"]
            metrics_source = "evaluation"

        comparison_rows.append(
            {
                "model_path": model_details["model_path"],
                "model_name": model_details["model_name"],
                "model_stem": model_details["model_stem"],
                "metrics_source": metrics_source,
                "dataset_path": model_details.get("dataset_path") or str(dataset_path),
                "metrics": metrics,
                "summary": evaluation_summary,
                "model_details": model_details,
                "ranking_score": metrics.get(ranking_metric),
            }
        )

    reverse_sort = ranking_metric in {"r2"}
    ranking_rows = sorted(
        comparison_rows,
        key=lambda item: (
            item["ranking_score"]
            if item["ranking_score"] is not None
            else float("-inf")
        ),
        reverse=reverse_sort,
    )

    return {
        "dataset_path": str(dataset_path),
        "sample_size": sample_size,
        "random_state": random_state,
        "ranking_metric": ranking_metric,
        "model_count": len(comparison_rows),
        "models": comparison_rows,
        "ranking": ranking_rows,
        "best_model": ranking_rows[0] if ranking_rows else None,
    }


def _compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    epsilon = 1e-8
    denominator = np.maximum(np.abs(y_true), epsilon)
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }


def _save_performance_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_val: pd.DataFrame,
    model: Pipeline,
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

    regressor = model.named_steps.get("regressor")
    if regressor is not None and hasattr(regressor, "coef_"):
        coefficients = np.asarray(getattr(regressor, "coef_")).reshape(-1)
        if coefficients.size == len(x_val.columns):
            fig, ax = plt.subplots(figsize=(9, 6))
            colors = ["#2ca02c" if value >= 0 else "#d62728" for value in coefficients]
            ax.bar(x_val.columns, coefficients, color=colors)
            ax.set_title("Feature Coefficients")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Coefficient Value")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(alpha=0.3, axis="y")
            coef_path = output_dir / "feature_coefficients.png"
            fig.tight_layout()
            fig.savefig(coef_path, dpi=150)
            plt.close(fig)
            plot_paths["feature_coefficients"] = str(coef_path)

    return plot_paths


def load_training_data(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    df = _read_dataset(dataset_path)

    required_columns = FEATURES + [TARGET]
    _validate_columns(df, required_columns)

    x_data: pd.DataFrame = df.loc[:, FEATURES].copy()
    y_data = df[TARGET].copy()
    return x_data, y_data


def load_training_data_with_columns(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    feature_columns: Sequence[str] | None = None,
    target_column: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    df = _read_dataset(dataset_path)

    selected_features = list(feature_columns) if feature_columns is not None else [
        column
        for column in df.columns
        if column != target_column and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not selected_features:
        raise ValueError("feature_columns must contain at least one column")
    if target_column in selected_features:
        raise ValueError(f"{target_column} cannot be included in feature_columns")

    non_numeric_features = [
        column for column in selected_features if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_features:
        raise ValueError(
            f"Only numeric feature columns are supported: {non_numeric_features}"
        )

    _validate_columns(df, selected_features + [target_column])

    x_data = _build_feature_frame(df, selected_features, target_column=target_column)
    y_data = df[target_column].copy()
    return x_data, y_data


def train_and_save_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_columns: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    x_data, y_data = load_training_data_with_columns(
        dataset_path=dataset_path,
        feature_columns=feature_columns,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    model = _build_pipeline()
    model.fit(x_train, y_train)

    y_pred = cast(np.ndarray, model.predict(x_val))
    y_true = y_val.to_numpy()
    metrics = _compute_regression_metrics(y_true=y_true, y_pred=y_pred)

    output_path = _build_timestamped_model_path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    run_reports_dir = REPORTS_DIR / output_path.stem
    plot_paths = _save_performance_plots(
        y_true=y_true,
        y_pred=y_pred,
        x_val=x_val,
        model=model,
        output_dir=run_reports_dir,
    )

    metrics_path = run_reports_dir / "metrics.json"
    metrics_payload = {
        "model_path": str(output_path),
        "dataset_path": str(dataset_path),
        "test_size": test_size,
        "random_state": random_state,
        "target_column": TARGET,
        "feature_columns": list(x_data.columns),
        "metrics": metrics,
        "plots": plot_paths,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Validation Metrics:")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  MSE:  {metrics['mse']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"Saved model at: {output_path}")
    print(f"Saved reports at: {run_reports_dir}")
    print(f"Saved metrics at: {metrics_path}")

    return {
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "mape": metrics["mape"],
        "model_path": str(output_path),
        "report_dir": str(run_reports_dir),
        "metrics_path": str(metrics_path),
        "target_column": TARGET,
        "feature_columns": list(x_data.columns),
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
    load_path = _resolve_model_path(model_path)
    return joblib.load(load_path)


def get_model_feature_columns(model_path: str | Path = DEFAULT_MODEL_PATH) -> list[str]:
    metrics_payload = _read_metrics_payload(model_path)
    if metrics_payload and metrics_payload.get("feature_columns"):
        return list(metrics_payload["feature_columns"])
    return list(FEATURES)


def predict_dataframe(
    df: pd.DataFrame,
    model: Pipeline | None = None,
    feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    inference_model = model if model is not None else load_model()
    required_features = list(feature_columns) if feature_columns is not None else list(FEATURES)
    _validate_columns(df, required_features)
    predictions = inference_model.predict(df[required_features])
    return cast(np.ndarray, predictions)


def predict_records(
    records: dict[str, Any] | list[dict[str, Any]],
    model: Pipeline | None = None,
    feature_columns: Sequence[str] | None = None,
) -> list[float]:
    if isinstance(records, dict):
        records = [records]

    if not records:
        raise ValueError("records must contain at least one item")

    df = pd.DataFrame(records)
    predictions = predict_dataframe(df, model=model, feature_columns=feature_columns)
    return [float(value) for value in predictions]


if __name__ == "__main__":
    train_and_save_model()

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json

import pandas as pd

from .training_logic import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    FEATURES,
    MODEL_DIR,
    REPORTS_DIR,
    TARGET,
    load_model,
    predict_dataframe,
)


def _read_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")
    return pd.read_csv(path)


def get_trainable_columns(
    dataset_path: str | Path = DEFAULT_DATASET_PATH, target_column: str = TARGET
) -> dict[str, Any]:
    df = _read_dataset(dataset_path)
    columns = [column for column in df.columns if column != target_column]
    numeric = [
        column for column in columns if pd.api.types.is_numeric_dtype(df[column])
    ]
    return {
        "dataset_path": str(Path(dataset_path)),
        "target_column": target_column,
        "trainable_columns": numeric,
        "all_input_columns": columns,
        "default_feature_columns": FEATURES,
    }


def get_model_details(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    metrics_path = REPORTS_DIR / path.stem / "metrics.json"
    feature_columns = FEATURES
    metrics = {}
    summary = {
        "target": TARGET,
        "test_size": None,
        "random_state": None,
        "modified": None,
        "created": None,
    }
    has_report = metrics_path.exists()
    if has_report:
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            feature_columns = list(payload.get("feature_columns", FEATURES))
            metrics = {
                "mae": payload.get("mae"),
                "mse": payload.get("mse"),
                "rmse": payload.get("rmse"),
                "r2": payload.get("r2"),
                "mape": payload.get("mape"),
            }
            summary["test_size"] = payload.get("test_size")
            summary["random_state"] = payload.get("random_state")
        except Exception:
            pass
    # File timestamps
    try:
        stat = path.stat()
        summary["modified"] = getattr(stat, "st_mtime", None)
        summary["created"] = getattr(stat, "st_ctime", None)
    except Exception:
        pass
    size_bytes = path.stat().st_size if path.exists() else None
    return {
        "model_path": str(path),
        "model_name": path.name,
        "model_stem": path.stem,
        "feature_columns": feature_columns,
        "feature_names": feature_columns,
        "target": TARGET,
        "report_dir": str(REPORTS_DIR / path.stem),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "size_bytes": size_bytes,
        "has_report": has_report,
        "model_summary": {
            "Target": summary["target"],
            "Test Size": summary["test_size"] if summary["test_size"] is not None else "--",
            "Random State": summary["random_state"] if summary["random_state"] is not None else "--",
            "Modified": summary["modified"] if summary["modified"] is not None else "--",
            "Created": summary["created"] if summary["created"] is not None else "--",
        },
    }


def list_available_models() -> list[dict[str, Any]]:
    return [
        get_model_details(model_path=model_file)
        for model_file in sorted(
            MODEL_DIR.glob("*.joblib"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    ]


def generate_synthetic_dataset(
    count: int,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    random_state: int = 42,
    include_target: bool = False,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    df = _read_dataset(dataset_path)
    selected = list(feature_columns) if feature_columns is not None else list(FEATURES)
    sample = df.sample(
        n=count, replace=count > len(df), random_state=random_state
    ).copy()
    sample = sample.where(pd.notnull(sample), None)
    source_cols = selected + ([TARGET] if include_target else [])
    return {
        "dataset_path": str(Path(dataset_path)),
        "requested_count": count,
        "actual_count": len(sample),
        "sample_with_replacement": count > len(df),
        "required_features": selected,
        "feature_columns": selected,
        "records": sample[selected].to_dict(orient="records"),
        "source_rows": sample[source_cols].to_dict(orient="records"),
    }


def evaluate_model_on_dataset(
    model_path: str | Path,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    sample_size: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    df = _read_dataset(dataset_path)
    if sample_size is not None:
        df = df.sample(
            n=sample_size, replace=sample_size > len(df), random_state=random_state
        )
    model = load_model(model_path)
    predictions = predict_dataframe(df, model=model, feature_columns=FEATURES)
    actual = df[TARGET].to_numpy()
    return {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "sample_size": int(len(df)),
        "sample_with_replacement": sample_size is not None and sample_size > len(df),
        "metrics": {
            "mae": float((abs(actual - predictions)).mean()),
            "mse": float(((actual - predictions) ** 2).mean()),
            "rmse": float((((actual - predictions) ** 2).mean()) ** 0.5),
            "r2": 0.0,
            "mape": 0.0,
        },
        "predictions": [float(value) for value in predictions],
    }


def compare_models(
    model_paths: Sequence[str | Path],
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    sample_size: int | None = None,
    random_state: int = 42,
    ranking_metric: str = "r2",
) -> dict[str, Any]:
    models = []
    for model_path in model_paths:
        details = get_model_details(model_path)
        summary = evaluate_model_on_dataset(
            model_path,
            dataset_path=dataset_path,
            sample_size=sample_size,
            random_state=random_state,
        )
        models.append(
            {
                "model_path": details["model_path"],
                "model_name": details["model_name"],
                "model_stem": details["model_stem"],
                "metrics_source": "evaluation",
                "dataset_path": str(dataset_path),
                "metrics": summary["metrics"],
                "summary": summary,
                "model_details": details,
                "ranking_score": summary["metrics"].get(ranking_metric),
            }
        )
    ranking = sorted(
        models,
        key=lambda item: item["ranking_score"] or float("-inf"),
        reverse=ranking_metric == "r2",
    )
    return {
        "dataset_path": str(dataset_path),
        "sample_size": sample_size,
        "random_state": random_state,
        "ranking_metric": ranking_metric,
        "model_count": len(models),
        "models": models,
        "ranking": ranking,
        "best_model": ranking[0] if ranking else None,
    }

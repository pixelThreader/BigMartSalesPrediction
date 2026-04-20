from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Annotated
from typing import Any
import uvicorn
import pandas as pd

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import Field

from model.model_service import DEFAULT_DATASET_PATH
from model.model_service import DEFAULT_MODEL_PATH
from model.model_service import FEATURES
from model.model_service import MODEL_DIR
from model.model_service import REPORTS_DIR
from model.model_service import compare_models
from model.model_service import generate_synthetic_dataset
from model.model_service import get_model_details
from model.model_service import get_trainable_columns
from model.model_service import list_available_models
from model.model_service import load_model
from model.model_service import predict_records
from model.model_service import train_and_save_model


TEMP_EXPORTS_DIR = MODEL_DIR / "temp_exports"
MAX_TEMP_RUN_FOLDERS = 20

PLOT_RENAME_MAP = {
    "actual_vs_predicted": "actual-vs-predicted.png",
    "residuals_vs_predicted": "residuals-vs-predicted.png",
    "residual_distribution": "residual-distribution.png",
    "actual_vs_predicted_distribution": "actual-vs-predicted-distribution.png",
    "feature_coefficients": "feature-coefficients.png",
}


class TrainRequest(BaseModel):
    dataset_path: str | None = None
    model_path: str | None = None
    feature_columns: list[str] | None = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 42


class PredictRequest(BaseModel):
    records: dict[str, Any] | list[dict[str, Any]]
    model_path: str | None = None


class SyntheticDataRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=5000)
    dataset_path: str | None = None
    model_path: str | None = None
    feature_columns: list[str] | None = None
    random_state: int = 42
    include_target: bool = False


class ModelCompareRequest(BaseModel):
    model_paths: list[str] = Field(min_length=1)
    dataset_path: str | None = None
    sample_size: int | None = Field(default=None, ge=1, le=5000)
    random_state: int = 42
    ranking_metric: str = Field(default="r2")


def _to_artifact_url(path_value: str | Path) -> str:
    resolved_path = Path(path_value).resolve()
    resolved_model_dir = MODEL_DIR.resolve()
    relative_path = resolved_path.relative_to(resolved_model_dir)
    return f"/artifacts/{relative_path.as_posix()}"


def _latest_metrics_path() -> Path:
    candidates = sorted(REPORTS_DIR.glob("bigmart_model_*/metrics.json"))
    if not candidates:
        raise FileNotFoundError(
            "No report metrics found. Train the model first using /api/v1/train."
        )
    return candidates[-1]


def _predict_with_model(
    records: dict[str, Any] | list[dict[str, Any]],
    model_path: str | None = None,
) -> dict[str, Any]:
    resolved_model_path = model_path or str(DEFAULT_MODEL_PATH)
    model = load_model(resolved_model_path)
    model_details = get_model_details(resolved_model_path)
    required_features = model_details.get("feature_columns", FEATURES)
    predictions = predict_records(
        records,
        model=model,
        feature_columns=required_features,
    )

    return {
        "model": model_details,
        "model_path": str(model_details["model_path"]),
        "required_features": required_features,
        "predictions": predictions,
    }


def _cleanup_temp_exports(max_folders: int = MAX_TEMP_RUN_FOLDERS) -> None:
    TEMP_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        [path for path in TEMP_EXPORTS_DIR.iterdir() if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for old_dir in run_dirs[max_folders:]:
        shutil.rmtree(old_dir, ignore_errors=True)


def _build_temp_plot_bundle(
    run_id: str,
    plot_paths: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], list[dict[str, str]]]:
    run_temp_dir = TEMP_EXPORTS_DIR / run_id
    if run_temp_dir.exists():
        shutil.rmtree(run_temp_dir, ignore_errors=True)
    run_temp_dir.mkdir(parents=True, exist_ok=True)

    renamed_plot_paths: dict[str, str] = {}
    renamed_plot_urls: dict[str, str] = {}
    graph_assets: list[dict[str, str]] = []

    for plot_key, source_path in plot_paths.items():
        source = Path(source_path)
        extension = source.suffix or ".png"
        frontend_name = PLOT_RENAME_MAP.get(
            plot_key, f"{plot_key.replace('_', '-')}{extension}"
        )
        destination = run_temp_dir / frontend_name
        shutil.copy2(source, destination)

        destination_str = str(destination)
        destination_url = _to_artifact_url(destination)

        renamed_plot_paths[plot_key] = destination_str
        renamed_plot_urls[plot_key] = destination_url
        graph_assets.append(
            {
                "id": plot_key,
                "file_name": frontend_name,
                "path": destination_str,
                "url": destination_url,
            }
        )

    return renamed_plot_paths, renamed_plot_urls, graph_assets


app = FastAPI(
    title="BigMart Model API",
    version="1.0.0",
    description="Train and serve BigMart sales predictions with model performance reports.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/artifacts", StaticFiles(directory=str(MODEL_DIR)), name="artifacts")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "BigMart backend is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/train")
def train_model(payload: TrainRequest) -> dict[str, Any]:
    dataset_path = payload.dataset_path or str(DEFAULT_DATASET_PATH)
    model_path = payload.model_path or str(DEFAULT_MODEL_PATH)

    try:
        result = train_and_save_model(
            dataset_path=dataset_path,
            model_path=model_path,
            feature_columns=payload.feature_columns,
            test_size=payload.test_size,
            random_state=payload.random_state,
        )
    except Exception as exc:  # pragma: no cover - surfaces runtime errors cleanly
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = Path(str(result["model_path"]))
    run_id = run_id.stem

    original_plot_paths = result.get("plots", {})
    if not isinstance(original_plot_paths, dict):
        raise HTTPException(status_code=500, detail="Unexpected plot artifact format")

    temp_plot_paths, temp_plot_urls, graph_assets = _build_temp_plot_bundle(
        run_id=run_id,
        plot_paths=original_plot_paths,
    )
    _cleanup_temp_exports()

    plot_urls = {
        key: _to_artifact_url(value) for key, value in original_plot_paths.items()
    }

    return {
        "message": "Model trained successfully",
        "run_id": run_id,
        "stats": {
            "mae": result["mae"],
            "mse": result["mse"],
            "rmse": result["rmse"],
            "r2": result["r2"],
            "mape": result["mape"],
        },
        "performance": {
            "metrics": {
                "mae": result["mae"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "r2": result["r2"],
                "mape": result["mape"],
            },
            "graphs": graph_assets,
        },
        "model_path": result["model_path"],
        "report_dir": result["report_dir"],
        "metrics_path": result["metrics_path"],
        "plots": original_plot_paths,
        "plot_urls": plot_urls,
        "temp_plot_dir": str(TEMP_EXPORTS_DIR / run_id),
        "temp_plots": temp_plot_paths,
        "temp_plot_urls": temp_plot_urls,
    }


@app.get("/api/v1/trainable-columns")
def get_trainable_columns_api(
    dataset_path: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    try:
        return get_trainable_columns(dataset_path or DEFAULT_DATASET_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected filesystem or parse errors
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/predict")
def predict_sales(payload: PredictRequest) -> dict[str, Any]:
    try:
        return _predict_with_model(payload.records, model_path=payload.model_path)
    except Exception as exc:  # pragma: no cover - surfaces validation errors cleanly
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/models/predict")
def predict_sales_with_custom_model(payload: PredictRequest) -> dict[str, Any]:
    try:
        return _predict_with_model(payload.records, model_path=payload.model_path)
    except Exception as exc:  # pragma: no cover - surfaces validation errors cleanly
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/models")
def get_models() -> dict[str, Any]:
    try:
        models = list_available_models()
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "default_model_path": str(DEFAULT_MODEL_PATH),
        "model_count": len(models),
        "models": models,
    }


@app.get("/api/v1/models/details")
def get_model_by_path(
    model_path: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    try:
        return get_model_details(model_path or DEFAULT_MODEL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/synthetic-data")
def create_synthetic_data(payload: SyntheticDataRequest) -> dict[str, Any]:
    try:
        model_details = None
        if payload.model_path:
            model_details = get_model_details(payload.model_path)
        feature_columns = payload.feature_columns or (
            model_details.get("feature_columns") if model_details else None
        )
        return generate_synthetic_dataset(
            count=payload.count,
            dataset_path=payload.dataset_path or DEFAULT_DATASET_PATH,
            random_state=payload.random_state,
            include_target=payload.include_target,
            feature_columns=feature_columns,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        Exception
    ) as exc:  # pragma: no cover - surfaces data/validation errors cleanly
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/models/compare")
def compare_selected_models(payload: ModelCompareRequest) -> dict[str, Any]:
    try:
        return compare_models(
            model_paths=payload.model_paths,
            dataset_path=payload.dataset_path or DEFAULT_DATASET_PATH,
            sample_size=payload.sample_size,
            random_state=payload.random_state,
            ranking_metric=payload.ranking_metric,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        Exception
    ) as exc:  # pragma: no cover - surfaces data/validation errors cleanly
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/reports/latest")
def get_latest_report() -> dict[str, Any]:
    try:
        metrics_path = _latest_metrics_path()
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected read/parse errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    plot_urls = {
        key: _to_artifact_url(value) for key, value in payload.get("plots", {}).items()
    }

    return {
        "metrics_file": str(metrics_path),
        "model_path": payload.get("model_path"),
        "dataset_path": payload.get("dataset_path"),
        "test_size": payload.get("test_size"),
        "random_state": payload.get("random_state"),
        "metrics": payload.get("metrics", {}),
        "plots": payload.get("plots", {}),
        "plot_urls": plot_urls,
    }


@app.get("/api/v1/dataset")
def get_dataset_paginated(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 20,
    dataset_path: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    resolved_dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    if not resolved_dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found at {resolved_dataset_path}",
        )

    try:
        df = pd.read_csv(resolved_dataset_path)
    except Exception as exc:  # pragma: no cover - file parse/read errors
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    total_records = len(df)
    total_pages = max(1, (total_records + page_size - 1) // page_size)
    if page > total_pages:
        raise HTTPException(
            status_code=400,
            detail=f"Requested page {page} exceeds total pages {total_pages}",
        )

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = df.iloc[start_idx:end_idx]
    page_df = page_df.where(pd.notnull(page_df), None)

    return {
        "dataset_path": str(resolved_dataset_path),
        "columns": df.columns.tolist(),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
        "data": page_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""FastAPI router for ML experiment management endpoints."""

import dataclasses
import threading
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response

from app.ml.experiments import experiment_engine

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/dataset-info")
def get_dataset_info():
    """Return dataset info (n_samples, n_features, target_distribution, feature_names)."""
    try:
        info = experiment_engine.get_dataset_info()
        return info
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Failed to retrieve dataset info: {e}"})


@router.post("/run")
def run_experiment():
    """Start a new experiment run in a background thread.

    Returns the experiment ID immediately while the experiment runs asynchronously.
    """
    experiment_id = str(uuid.uuid4())

    def _run():
        try:
            experiment_engine.run_all_experiments(experiment_id)
        except Exception:
            # Errors are captured inside the engine and stored in ExperimentResults.error_message
            pass

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "experiment_id": experiment_id,
        "status": "running",
        "message": "Experiment started",
    }


@router.get("/status/{experiment_id}")
def get_experiment_status(experiment_id: str):
    """Return the current status and basic info of an experiment."""
    results = experiment_engine.get_results(experiment_id)
    if results is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Experiment '{experiment_id}' not found", "status": "not_found"},
        )

    return {
        "experiment_id": results.experiment_id,
        "status": results.status,
        "started_at": results.started_at,
        "completed_at": results.completed_at,
        "error_message": results.error_message,
    }


@router.get("/results/{experiment_id}")
def get_experiment_results(experiment_id: str):
    """Return full results as JSON.

    If the experiment is still running, returns {"status": "running"}.
    If the experiment is not found, returns {"status": "not_found"}.
    """
    results = experiment_engine.get_results(experiment_id)
    if results is None:
        return {"status": "not_found"}

    if results.status != "completed":
        return {"status": results.status}

    return dataclasses.asdict(results)


@router.get("/results/{experiment_id}/export/json")
def export_results_json(experiment_id: str):
    """Return full results as a downloadable JSON file."""
    results = experiment_engine.get_results(experiment_id)
    if results is None:
        return JSONResponse(status_code=404, content={"detail": f"Experiment '{experiment_id}' not found"})

    if results.status != "completed":
        return JSONResponse(
            status_code=400,
            content={"detail": f"Experiment '{experiment_id}' is not completed (status: {results.status})"},
        )

    data = experiment_engine.export_results_json(experiment_id)
    return JSONResponse(
        content=data,
        headers={
            "Content-Disposition": f'attachment; filename="experiment_{experiment_id}.json"'
        },
    )


@router.get("/results/{experiment_id}/export/csv")
def export_results_csv(experiment_id: str):
    """Return comparison table as a downloadable CSV file."""
    results = experiment_engine.get_results(experiment_id)
    if results is None:
        return JSONResponse(status_code=404, content={"detail": f"Experiment '{experiment_id}' not found"})

    if results.status != "completed":
        return JSONResponse(
            status_code=400,
            content={"detail": f"Experiment '{experiment_id}' is not completed (status: {results.status})"},
        )

    csv_content = experiment_engine.export_results_csv(experiment_id)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="experiment_{experiment_id}.csv"'
        },
    )


@router.get("/list")
def list_experiments():
    """Return list of all experiment IDs with their status."""
    experiment_ids = experiment_engine.get_all_experiment_ids()
    experiments = []
    for eid in experiment_ids:
        results = experiment_engine.get_results(eid)
        experiments.append(
            {
                "experiment_id": eid,
                "status": results.status if results else "unknown",
                "started_at": results.started_at if results else None,
                "completed_at": results.completed_at if results else None,
            }
        )
    return {"experiments": experiments}


@router.get("/latest")
def get_latest_experiment():
    """Return the latest completed experiment results, or 404 if none."""
    experiment_ids = experiment_engine.get_all_experiment_ids()

    latest_result = None
    latest_completed_at = None

    for eid in experiment_ids:
        results = experiment_engine.get_results(eid)
        if results is not None and results.status == "completed" and results.completed_at is not None:
            if latest_completed_at is None or results.completed_at > latest_completed_at:
                latest_completed_at = results.completed_at
                latest_result = results

    if latest_result is None:
        return JSONResponse(status_code=404, content={"detail": "No completed experiments found"})

    return dataclasses.asdict(latest_result)

import os

from fastapi import APIRouter, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

router = APIRouter()   


class PromoteRequest(BaseModel):
    version: int
    alias: str = "meilleur"
    max_mae: float = 1.0
    min_r2: float = 0.85


@router.post("")
def promote(req: PromoteRequest):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name = os.getenv("MODEL_NAME", "student_success_model")

    c = MlflowClient(tracking_uri=tracking_uri)

    mv = c.get_model_version(model_name, str(req.version))
    run = c.get_run(mv.run_id)
    metrics = run.data.metrics

    mae = metrics.get("test_mae")
    r2 = metrics.get("test_r2")

    if mae is None or r2 is None:
        raise HTTPException(400, "Metrics manquantes sur ce run (test_mae/test_r2).")

    if mae > req.max_mae or r2 < req.min_r2:
        raise HTTPException(
            400, f"Refus promotion: mae={mae:.3f} (<= {req.max_mae}), r2={r2:.3f} (>= {req.min_r2})"
        )

    c.set_registered_model_alias(model_name, req.alias, str(req.version))

    # reload immédiat en mémoire
    from model_loader import model_handle

    model_handle.load()

    return {
        "status": "promoted",
        "model": model_name,
        "alias": req.alias,
        "version": req.version,
        "mae": mae,
        "r2": r2,
    }

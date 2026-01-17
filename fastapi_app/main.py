import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram
from sqlalchemy import text
from sqlalchemy.orm import Session

from crud import create_log, read_history
from db import SessionLocal, init_db
from model_loader import model_handle
from routes.promote import router as promote_router
from routes.train import router as train_router
from schemas import HistoryItem, PredictionResponse, StudentFeatures

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ModuleNotFoundError:
    Instrumentator = None

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Student Success Prediction API", version="1.0.0")
app.include_router(train_router, prefix="/train")
app.include_router(promote_router, prefix="/promote")

# Instrumentation HTTP (req/s, latence, codes…)
if Instrumentator is not None:
    Instrumentator().instrument(app).expose(app)


# MÉTRIQUES ML (Dashboard "ML Serving")

ml_predictions_total = Counter(
    "ml_predictions_total",
    "Nombre de prédictions réalisées",
    ["decision", "model_version"],
)

ml_predict_errors_total = Counter(
    "ml_predict_errors_total",
    "Nombre d'erreurs en prédiction",
    ["type"],
)

ml_predict_duration_seconds = Histogram(
    "ml_predict_duration_seconds",
    "Durée de l'inférence (secondes)",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

ml_model_info = Gauge(
    "ml_model_info",
    "Modèle courant (valeur = 1)",
    ["model_version"],
)

# Les features attendues = celles du schéma (mêmes colonnes en entrée)
EXPECTED_FEATURES = list(StudentFeatures.model_fields.keys())


def get_db():
    """Ouvre une session DB, la ferme proprement."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# Robustesse : middleware request_id + mesure de latence

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    start = time.time()

    try:
        response = await call_next(request)
        return response
    finally:
        latency_ms = int((time.time() - start) * 1000)
        logger.info(
            "request_done",
            extra={
                "request_id": request.state.request_id,
                "path": request.url.path,
                "method": request.method,
                "latency_ms": latency_ms,
            },
        )


# Gestion d’erreurs globale

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "request_id": req_id,
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", None)
    logger.exception("Unhandled error request_id=%s", req_id)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "request_id": req_id,
        },
    )


# Startup : init DB + tentative chargement modèle

@app.on_event("startup")
def startup():
    init_db()

    try:
        model_handle.load()
        logger.info("Model loaded at startup.")
    except Exception as e:
        # Ne bloque pas l’API : /health dira model_loaded=false.
        logger.warning("Model non chargé au démarrage. Raison: %s", str(e))



# Health check : DB + modèle

@app.get("/health")
def health(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    status = "ok" if db_ok else "degraded"

    return {
        "status": status,
        "db_ok": db_ok,
        "model_loaded": model_handle.model is not None,
        "model_version": model_handle.model_version,
    }


# Endpoint utile pour debug/présentation : infos modèle + features

@app.get("/model")
def model_info():
    return {
        "model_loaded": model_handle.model is not None,
        "model_version": model_handle.model_version,
        "expected_features": model_handle.expected_features or EXPECTED_FEATURES,
    }


# Prédiction

@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: StudentFeatures,
    db: Session = Depends(get_db),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    session_id: Optional[str] = None,
):
    """
    - on envoie au Pipeline sklearn exactement les mêmes colonnes
    - décision = réussite si G3 >= 10
    Monitoring ML :
    - nb prédictions, latence inference, erreurs, version modèle
    """
    # 1) Assurer que le modèle est disponible
    if model_handle.model is None:
        try:
            model_handle.load()
        except Exception as e:
            ml_predict_errors_total.labels(type="model_unavailable").inc()
            logger.warning("Model unavailable on predict. Raison: %s", str(e))
            raise HTTPException(status_code=503, detail="Modèle indisponible") from e

    # 2) session_id (header prioritaire) ou généré
    sid = x_session_id or session_id or str(uuid.uuid4())

    # 3) Marquer la version modèle (pour panel “version courante”)
    current_version = str(model_handle.model_version)
    ml_model_info.labels(model_version=current_version).set(1)

    # 4) Inférence (mesurée) + cohérence des features
    with ml_predict_duration_seconds.time():
        X = pd.DataFrame([payload.model_dump()])

        # si le loader fournit une liste attendue, on l’utilise ; sinon schéma Pydantic
        expected = model_handle.expected_features or EXPECTED_FEATURES
        X = X.reindex(columns=expected)

        y_pred = model_handle.model.predict(X)

    pred = float(np.array(y_pred).ravel()[0])
    pred_clamped = float(np.clip(pred, 0.0, 20.0))
    decision = "reussite" if pred_clamped >= 10.0 else "echec"

    # 5) métriques ML
    ml_predictions_total.labels(decision=decision, model_version=current_version).inc()

    # 6) Log DB (inputs/outputs/date/session/version)
    create_log(
        db=db,
        session_id=sid,
        inputs_json=payload.model_dump(),
        predicted_g3=pred_clamped,
        decision=decision,
        model_version=model_handle.model_version,
    )

    # 7) Réponse
    return PredictionResponse(
        session_id=sid,
        predicted_G3=pred_clamped,
        decision=decision,
        threshold=10.0,
        mlflow_model_version=model_handle.model_version,
        timestamp=datetime.now(UTC),
    )

# Historique

@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = 50, session_id: str | None = None, db: Session = Depends(get_db)):
    rows = read_history(db=db, limit=limit, session_id=session_id)
    return [
        HistoryItem(
            id=r.id,
            session_id=r.session_id,
            timestamp=r.timestamp,
            inputs_json=r.inputs_json,
            predicted_G3=r.predicted_G3,
            decision=r.decision,
            model_version=r.model_version,
        )
        for r in rows
    ]

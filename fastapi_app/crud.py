from datetime import UTC, datetime

from db import InferenceLog
from sqlalchemy.orm import Session


def create_log(
    db: Session,
    session_id: str,
    inputs_json: dict,
    predicted_g3: float,
    decision: str,
    model_version: str | None,
):
    row = InferenceLog(
        session_id=session_id,
        timestamp=datetime.now(UTC),
        inputs_json=inputs_json,
        predicted_G3=float(predicted_g3),
        decision=decision,
        model_version=model_version,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def read_history(db: Session, limit: int = 50, session_id: str | None = None):
    q = db.query(InferenceLog).order_by(InferenceLog.timestamp.desc())
    if session_id:
        q = q.filter(InferenceLog.session_id == session_id)
    return q.limit(limit).all()

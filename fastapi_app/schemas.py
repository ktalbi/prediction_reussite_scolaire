from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

School = Literal["GP", "MS"]
Reason = Literal["course", "home", "other", "reputation"]
YesNo = Literal["yes", "no"]

# Contraintes numériques d'après le sujet

Age = Annotated[int, Field(ge=15, le=22)]
Note = Annotated[float, Field(ge=0, le=20)]

Scale1to4 = Annotated[int, Field(ge=1, le=4)]
Scale0to4 = Annotated[int, Field(ge=0, le=4)]
Scale1to5 = Annotated[int, Field(ge=1, le=5)]
Absences = Annotated[int, Field(ge=0)]

class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")  # refuse les champs inattendus

    school: School
    age: Age
    reason: Reason
    nursery: YesNo

    traveltime: Scale1to4
    studytime: Scale1to4
    failures: Scale0to4

    schoolsup: YesNo
    famsup: YesNo
    paid: YesNo
    activities: YesNo
    higher: YesNo

    freetime: Scale1to5
    goout: Scale1to5

    absences: Absences

    G1: Note
    G2: Note


class PredictionResponse(BaseModel):
    session_id: str
    predicted_G3: float
    decision: Literal["reussite", "echec"]
    threshold: float = 10.0
    mlflow_model_version: str | None = None
    timestamp: datetime


class HistoryItem(BaseModel):
    id: int
    session_id: str
    timestamp: datetime
    inputs_json: dict
    predicted_G3: float
    decision: str
    model_version: str | None = None

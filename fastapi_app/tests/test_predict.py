import numpy as np
import importlib


def _import_main_module():
    # On patch le module qui contient model_handle ET create_log réellement utilisé par l'app
    try:
        return importlib.import_module("main")
    except Exception:
        return importlib.import_module("fastapi_app.main")


class DummyModel:
    def predict(self, X):
        return np.array([13.0])


def test_predict_200_with_mock_model(client, monkeypatch):
    main_mod = _import_main_module()

    # Mock modèle + version
    monkeypatch.setattr(main_mod.model_handle, "model", DummyModel())
    monkeypatch.setattr(main_mod.model_handle, "model_version", "test:v1")

    # Mock DB logging (évite dépendance SQLite/migrations/tables en CI)
    monkeypatch.setattr(main_mod, "create_log", lambda **kwargs: None)

    payload = {
        "school": "GP",
        "age": 17,
        "reason": "course",
        "nursery": "yes",
        "traveltime": 2,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "no",
        "higher": "yes",
        "freetime": 3,
        "goout": 3,
        "absences": 2,
        "G1": 10.0,
        "G2": 9.8,
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["predicted_G3"] == 13.0
    assert data["decision"] == "reussite"
    assert data["mlflow_model_version"] == "test:v1"


def test_predict_422_validation(client):
    # age hors bornes -> 422 (validation Pydantic)
    payload = {
        "school": "GP",
        "age": 50,
        "reason": "course",
        "nursery": "yes",
        "traveltime": 2,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "no",
        "higher": "yes",
        "freetime": 3,
        "goout": 3,
        "absences": 2,
        "G1": 10.0,
        "G2": 9.8,
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 422

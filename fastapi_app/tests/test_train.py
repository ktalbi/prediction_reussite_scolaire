import importlib

import numpy as np
import pandas as pd


def _import_train_module():
    # Module où RandomizedSearchCV est importé/utilisé (routes/train.py)
    try:
        return importlib.import_module("fastapi_app.routes.train")
    except Exception:
        return importlib.import_module("routes.train")


class DummyBestPipeline:
    def predict(self, X):
        return np.ones(len(X)) * 10.0


class DummySearch:
    def __init__(self, *args, **kwargs):
        self.best_estimator_ = DummyBestPipeline()
        self.best_params_ = {"model__n_estimators": 200}

    def fit(self, X, y):
        return self


def test_train_success(client, monkeypatch, tmp_path):
    train_mod = _import_train_module()

    # Mock RandomizedSearchCV pour éviter un vrai tuning (rapide/déterministe)
    monkeypatch.setattr(train_mod, "RandomizedSearchCV", DummySearch)

    # Dataset minimal cohérent (FEATURES_FORM + G3)
    df = pd.DataFrame(
        [{
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
            "G3": 11.0,
        }] * 20
    )

    dataset_path = tmp_path / "final.csv"
    df.to_csv(dataset_path, index=False)

    # Variables d'env utilisées par /train
    monkeypatch.setenv("DATA_PATH", str(dataset_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_exp")
    monkeypatch.setenv("MODEL_NAME", "student_success_model")
    seed_path = tmp_path / "best_model_rf_seed.joblib"
    monkeypatch.setenv("SEED_MODEL_PATH", str(seed_path))

    # Mock MLflow  
    class DummyRunInfo:
        run_id = "run_test_123"

    class DummyRun:
        info = DummyRunInfo()

    class DummyCtx:
        def __enter__(self): return DummyRun()
        def __exit__(self, *args): return False

    monkeypatch.setattr(train_mod.mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(train_mod.mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(train_mod.mlflow, "start_run", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(train_mod.mlflow, "log_param", lambda *a, **k: None)
    monkeypatch.setattr(train_mod.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(train_mod.mlflow.sklearn, "log_model", lambda *a, **k: None)

    # Mock joblib.dump
    monkeypatch.setattr(train_mod.joblib, "dump", lambda model, path: None)

    # Mock MLflow Model Registry client pour renvoyer une version liée au run_id
    class DummyVersion:
        def __init__(self, version, run_id):
            self.version = str(version)
            self.run_id = run_id

    class DummyClient:
        def __init__(self, *a, **k): pass
        def search_model_versions(self, *a, **k):
            return [DummyVersion(3, "run_test_123")]

    monkeypatch.setattr(train_mod, "MlflowClient", DummyClient)

    r = client.post("/train")
    assert r.status_code == 200
    data = r.json()

    # Compatibilité avec les 2 formats ("ok" ou "success")
    assert data["status"] in ("ok", "success")
    assert "run_id" in data
    assert ("version" in data) or ("model_version" in data)

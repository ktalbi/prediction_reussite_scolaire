import os
import time
import traceback
from datetime import UTC, datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge, Histogram
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from transforms import safe_log1p_absences



router = APIRouter()


# Prometheus (training)

ml_train_runs_total = Counter(
    "ml_train_runs_total",
    "Nombre d'entrainements déclenchés",
    ["status"],
)

ml_train_duration_seconds = Histogram(
    "ml_train_duration_seconds",
    "Durée d'entrainement (secondes)",
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300),
)

ml_train_last_success_timestamp = Gauge(
    "ml_train_last_success_timestamp",
    "Timestamp (epoch) du dernier entrainement réussi",
)

ml_train_last_mae = Gauge("ml_train_last_mae", "Dernière MAE (test) après entrainement")
ml_train_last_r2 = Gauge("ml_train_last_r2", "Dernier R2 (test) après entrainement")


@router.post("")
def train_scenario2_from_final_csv():
    """
    Entraîne un modèle de régression pour prédire G3 depuis les champs du formulaire.
    -> Tuning RandomForest via RandomizedSearchCV.
    -> Log dans MLflow (nouvelle version candidate).
    -> Ne promeut pas l'alias (promotion via /promote).
    -> Expose métriques Prometheus (training monitoré).

    NOTE (cohérence notebook/API) :
    - Transformation log1p appliquée UNIQUEMENT à la feature "absences" via le pipeline preprocess.
    """
    start = time.time()

    try:
        print("== TRAIN START", flush=True)

        # Dataset
        dataset_path = os.getenv("DATA_PATH", "/data/final.csv")
        print(f"[TRAIN] dataset_path = {dataset_path}", flush=True)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

        df = pd.read_csv(dataset_path)
        print(f"[TRAIN] dataset loaded shape={df.shape}", flush=True)

        TARGET = "G3"
        FEATURES_FORM = [
            "school", "age", "reason", "nursery",
            "traveltime", "studytime", "failures",
            "schoolsup", "famsup", "paid", "activities", "higher",
            "freetime", "goout", "absences", "G1", "G2",
        ]

        missing_cols = [c for c in FEATURES_FORM + [TARGET] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans le dataset: {missing_cols}")

        df = df[FEATURES_FORM + [TARGET]].copy()
        X = df[FEATURES_FORM]
        y = df[TARGET].astype(float)

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"[TRAIN] X_train={X_train.shape}, X_test={X_test.shape}", flush=True)

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
        print(f"[TRAIN] categorical_cols={categorical_cols}", flush=True)
        print(f"[TRAIN] numeric_cols={numeric_cols}", flush=True)

        # log1p UNIQUEMENT sur "absences" (dans le pipeline preprocess)
        ABS_COL = "absences"

        # Séparer absences des autres numériques
        num_cols_no_abs = [c for c in numeric_cols if c != ABS_COL]

        # Si jamais "absences" n'est pas détectée comme numérique (cas rare), on force le check
        if ABS_COL not in FEATURES_FORM:
            raise ValueError("La feature 'absences' doit exister dans FEATURES_FORM.")

        abs_pipe = Pipeline(steps=[
            ("log1p", FunctionTransformer(safe_log1p_absences, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ])
        

        num_pipe = Pipeline(steps=[
            ("scaler", StandardScaler()),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols_no_abs),
                ("absences", abs_pipe, [ABS_COL]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ],
            remainder="drop",
        )

        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", rf_base),
            ]
        )

        # Paramètres de tuning
        rf_param_dist = {
            "model__n_estimators": [200, 400, 700, 1000],
            "model__max_depth": [None, 5, 10, 20, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # MLflow settings
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "student_success_training")
        mlflow.set_experiment(experiment_name)

        model_name = os.getenv("MODEL_NAME", "student_success_model")

        # Seed fallback : on sauvegarde aussi le best pipeline localement
        seed_path = os.getenv("SEED_MODEL_PATH", "/app/models/best_model_rf.joblib")

        with ml_train_duration_seconds.time():
            with mlflow.start_run(run_name="scenario_2_rf_randomizedsearch") as run:
                run_id = run.info.run_id
                print(f"[TRAIN] MLflow run started run_id={run_id}", flush=True)

                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=rf_param_dist,
                    n_iter=25,
                    scoring="neg_mean_absolute_error",
                    cv=cv,
                    n_jobs=-1,
                    verbose=0,
                    random_state=42,
                )

                search.fit(X_train, y_train)
                best_pipeline = search.best_estimator_
                best_params = search.best_params_

                print(f"[TRAIN] RF best params: {best_params}", flush=True)

                # Test eval
                y_pred = best_pipeline.predict(X_test)
                mae_test = float(mean_absolute_error(y_test, y_pred))
                r2_test = float(r2_score(y_test, y_pred))
                print(f"[TRAIN] Test MAE={mae_test:.4f} R2={r2_test:.4f}", flush=True)

                # Log params/metrics MLflow
                mlflow.log_param("model_type", "RandomForestRegressor")
                mlflow.log_param("scoring", "neg_mean_absolute_error")
                mlflow.log_param("cv_splits", 5)
                mlflow.log_param("n_iter", 25)
                mlflow.log_param("n_features", len(FEATURES_FORM))
                mlflow.log_param("features", ",".join(FEATURES_FORM))
                mlflow.log_param("numeric_cols", ",".join(numeric_cols))
                mlflow.log_param("categorical_cols", ",".join(categorical_cols))
                mlflow.log_param("absences_transform", "log1p+scaler")

                for k, v in best_params.items():
                    mlflow.log_param(k, v)

                mlflow.log_metric("test_mae", mae_test)
                mlflow.log_metric("test_r2", r2_test)

                # Log model (candidate) in MLflow Registry
                print(f"[TRAIN] logging model to MLflow (registered_model_name={model_name})", flush=True)
                mlflow.sklearn.log_model(
                    best_pipeline,
                    artifact_path="model",
                    registered_model_name=model_name,
                )

                # Sauvegarde seed pour fallback API
                os.makedirs(os.path.dirname(seed_path), exist_ok=True)
                joblib.dump(best_pipeline, seed_path)
                print(f"[TRAIN] seed saved: {seed_path}", flush=True)

                # Récupérer la version créée pour ce run
                client = MlflowClient(tracking_uri=tracking_uri)
                created_version = None
                last_err = None
                for _ in range(20):
                    try:
                        versions = client.search_model_versions(f"name='{model_name}'")
                        matching = [v for v in versions if v.run_id == run_id]
                        if matching:
                            created_version = max(matching, key=lambda v: int(v.version))
                            break
                    except Exception as e:
                        last_err = e
                    time.sleep(0.5)

                if created_version is None:
                    try:
                        versions = client.search_model_versions(f"name='{model_name}'")
                        if versions:
                            created_version = max(versions, key=lambda v: int(v.version))
                            print(
                                f"[TRAIN] WARNING: version non trouvée via run_id; fallback dernière version v{created_version.version}",
                                flush=True,
                            )
                    except Exception as e:
                        last_err = e

                if created_version is None:
                    raise RuntimeError(
                        f"Impossible de déterminer la version du modèle créée. Dernière erreur: {last_err}"
                    )

                # Prometheus : last success + métriques
                ml_train_last_mae.set(mae_test)
                ml_train_last_r2.set(r2_test)
                ml_train_last_success_timestamp.set(int(datetime.now(UTC).timestamp()))
                ml_train_runs_total.labels(status="success").inc()

                duration_s = float(time.time() - start)

                print(f"[TRAIN] DONE candidate version=v{created_version.version}", flush=True)

                return {
                    "status": "ok",
                    "best_model": "RandomForestRegressor(RandomizedSearchCV)",
                    "best_params": best_params,
                    "test_mae": mae_test,
                    "test_r2": r2_test,
                    "run_id": run_id,
                    "model_name": model_name,
                    "version": int(created_version.version),
                    "seed_path": seed_path,
                    "duration_seconds": duration_s,
                    "note": "Candidate enregistrée. Utilise /promote pour basculer l'alias 'meilleur'.",
                }

    except Exception as e:
        ml_train_runs_total.labels(status="failed").inc()
        print("== TRAIN ERROR", flush=True)
        print(str(e), flush=True)
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__},
        ) from e

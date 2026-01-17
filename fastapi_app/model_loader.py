import os
import time
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import __main__

def _safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))



class ModelHandle:
    """
    Charge un modèle sklearn  
    depuis MLflow (alias) avec fallback seed joblib.

    Expose :
    - model : objet sklearn  
    - model_version : str (traceable, MLflow ou seed)
    - expected_features : liste de features attendues en entrée (optionnel)
    """

    def __init__(self):
        self.model = None
        self.model_version = None
        self.expected_features = None

    def _load_expected_features(self):
        """
        Permet de forcer l’ordre des features si besoin, via variable d’env.
        Exemple :
          EXPECTED_FEATURES="school,age,reason,nursery,traveltime,...,G1,G2"

        Si non fourni, l’API peut dériver la liste depuis le schéma Pydantic.
        """
        raw = os.getenv("EXPECTED_FEATURES", "").strip()
        if raw:
            self.expected_features = [c.strip() for c in raw.split(",") if c.strip()]

    def load(self):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)

        model_name = os.getenv("MODEL_NAME", "student_success_model")
        model_alias = os.getenv("MODEL_ALIAS", "meilleur")
        model_uri = f"models:/{model_name}@{model_alias}"

        seed_path = os.getenv("SEED_MODEL_PATH", "/app/models/best_model_rf.joblib")

        # Charger éventuellement la liste attendue des features (optionnel)
        self._load_expected_features()

        # 1) Essayer MLflow très brièvement
        last_err = None
        for i in range(2):
            try:
                print(f"[MODEL] Try MLflow load ({i+1}/2): {model_uri}", flush=True)
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = f"mlflow:{model_uri}"
                print(f"[MODEL] Loaded from MLflow: {self.model_version}", flush=True)
                return
            except Exception as e:
                last_err = e
                time.sleep(1)

        # 2) Fallback seed immédiat
        if os.path.exists(seed_path):
            print(
                f"[MODEL] MLflow pas pret, fallback to seed: {seed_path}. Last err: {last_err}",
                flush=True,
            )
            setattr(__main__, "_safe_log1p", _safe_log1p)
            self.model = joblib.load(seed_path)
            self.model_version = f"seed:{seed_path}"
            print(f"[MODEL] modèle seed chargé: {self.model_version}", flush=True)
            return

        raise RuntimeError(
            f"Impossible de charger le modèle.\n"
            f"- MLflow: {model_uri} (dernier err: {last_err})\n"
            f"- Seed absent: {seed_path}"
        )


model_handle = ModelHandle()

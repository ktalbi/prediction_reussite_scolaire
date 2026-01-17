# Prédiction de la réussite scolaire

## Objectif

Déployer une application complète de **Machine Learning en production** permettant de :
- prédire la **note finale G3 (0–20)** d’un élève,
- déterminer la **réussite (≥10) ou l’échec (<10)**,
- assurer un **workflow MLOps contrôlé** (train, validation, promotion, rollback),
- monitorer l’API et le ML (Prometheus/Grafana).

---

## Architecture

- **Notebook Jupyter** : exploration + sélection du meilleur modèle
- **FastAPI** :
  - `POST /predict` → prédiction (G3 + décision)
  - `POST /train` → entraînement d’un modèle *candidate* (monitoré)
  - `POST /promote` → promotion contrôlée (alias MLflow)
  - `GET /history` → historique des prédictions (SQLite)
  - `GET /health` → santé API + DB + modèle
  - `GET /metrics` → métriques Prometheus
- **MLflow** : tracking, registry, versioning (alias `meilleur`)
- **Streamlit** : interface utilisateur
- **Docker & Docker Compose** : orchestration
- **Prometheus + Grafana** : monitoring temps réel

---

## Données

- Dataset : `final.csv`
- Variable cible : **G3**
- Features utilisées (alignées formulaire + API) :
```
school, age, reason, nursery,
traveltime, studytime, failures,
schoolsup, famsup, paid, activities, higher,
freetime, goout, absences, G1, G2
```

---

## Modèle (cohérent Notebook → API)

**Pipeline scikit-learn** :
- `ColumnTransformer`
  - Numériques → `StandardScaler`
  - Catégorielles → `OneHotEncoder(handle_unknown="ignore")`
- Modèle : `RandomForestRegressor`
- Sélection : **RandomizedSearchCV** (scoring MAE)

---

## Chargement du modèle (API)
Ordre de chargement :
1. MLflow : modèle `models:/<MODEL_NAME>@<MODEL_ALIAS>` (alias par défaut `meilleur`)
2. Fallback automatique sur le **seed joblib** (`best_model_rf_seed.joblib`)

➡️ L’API reste utilisable même si MLflow n’est pas prêt.

---

## Robustesse API (validation + erreurs + traçabilité)
- **Validation stricte** des données entrantes via Pydantic (bornes, types, `extra="forbid"`).
- **Gestion centralisée des erreurs** :
  - 422 : validation
  - 503 : modèle indisponible
  - 500 : erreur interne (message générique, détails dans logs)
- **Traçabilité** :
  - `X-Session-Id` (header) ou session auto-générée
  - log DB des entrées/sorties (SQLite `inference_logs`)
  - `request_id` + latence dans les logs

---

## Endpoints

### `GET /health`
```bash
curl http://localhost:8000/health
```

### `POST /predict`
Exemple :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: demo-001" \
  -d '{
    "school":"GP","age":17,"reason":"course","nursery":"yes",
    "traveltime":2,"studytime":2,"failures":0,
    "schoolsup":"no","famsup":"yes","paid":"no","activities":"no","higher":"yes",
    "freetime":3,"goout":3,"absences":2,"G1":10.0,"G2":9.8
  }'
```

### `GET /history`
```bash
curl "http://localhost:8000/history?limit=50&session_id=demo-001"
```

### `POST /train` (entraînement monitoré)
- Entraîne un **RandomForestRegressor** avec **RandomizedSearchCV**
- Log dans MLflow (run + métriques + modèle) + enregistre une **version candidate**
- Sauvegarde aussi un joblib local (fallback)
- Ne modifie pas automatiquement l’alias `meilleur` (promotion via `/promote`)

```bash
curl -X POST http://localhost:8000/train
```

### `POST /promote` (promotion contrôlée)
```bash
curl -X POST http://localhost:8000/promote \
  -H "Content-Type: application/json" \
  -d '{"version":2,"alias":"meilleur","max_mae":1.0,"min_r2":0.85}'
```

---

## Monitoring (Prometheus / Grafana)

### Accès
- Prometheus : http://localhost:9090
- Grafana : http://localhost:3000 (admin / admin)
- API metrics : http://localhost:8000/metrics

---

## Lancer le stack (Docker Compose)

```bash
docker compose up -d --build
docker compose logs -f api
```

---

## Tests & CI/CD

### Lancer les tests en local
Depuis la racine du repo :
```bash
PYTHONPATH=. pytest -q fastapi_app/tests
```

### Remarques
- Les tests mockent MLflow, le tuning et l’écriture DB pour rester **rapides et déterministes** (compatibles CI).
- Le workflow GitHub Actions exécute : lint + tests, puis entraînement “seed”, puis build & push.

---

## Services accessibles
- **FastAPI** : http://localhost:8000/docs  
- **Streamlit** : http://localhost:8501
- **MLflow** : http://localhost:5000  
- **Prometheus** : http://localhost:9090  
- **Grafana** : http://localhost:3000 (admin / admin)
- **Uptime kuma** : http://localhost:3001

---

## Exemple `.env`
```env
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
MLFLOW_PORT=5000

MODEL_NAME=student_success_model
MODEL_ALIAS=meilleur
MLFLOW_EXPERIMENT_NAME=student_success_training

DATA_PATH=/data/final.csv
SEED_MODEL_PATH=/app/models/best_model_rf_seed.joblib

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
```
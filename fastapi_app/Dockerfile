FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY models/best_model_rf_seed.joblib /app/models/best_model_rf_seed.joblib


ENV PYTHONPATH=/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


"""
This module implements the Heart Disease Prediction API.

It calls the local model server in src/serving/serve.py over plain HTTP
on localhost.
"""

import logging
import os
import sys
from io import StringIO
from typing import List

import pandas as pd
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reach the repo-root "src" package from either layout this file runs in:
# locally as frontend-app/backend/api.py (repo root two levels up), or as
# /app/api.py in the Docker image, which has src/ copied alongside it.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common.features import engineer_features_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This URL points at the local model server started by `python -m src.serving.serve`.
# See the repo root README for details.
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8080")

logger.info(f"API initialized using local model server: {MODEL_SERVER_URL}")


class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class PredictionRequest(BaseModel):
    patients: List[PatientData]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    interpretation: str


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    elif probability < 0.8:
        return "High"
    else:
        return "Very High"


def get_interpretation(prediction: int, probability: float) -> str:
    if prediction == 0:
        return f"No heart disease detected. Risk score: {probability:.1%}"
    risk = get_risk_level(probability)
    return f"Heart disease detected. {risk} risk ({probability:.1%}). Consult a cardiologist."


def invoke_model(features: List[float]) -> dict:
    response = requests.post(
        f"{MODEL_SERVER_URL}/predict",
        json={"features": features},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


@app.get("/")
async def root():
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "model_server": MODEL_SERVER_URL,
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "predict_csv": "/predict/csv",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Check whether the local model server is reachable."""
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/health", timeout=5)
        response.raise_for_status()
        return {"status": "healthy", "model_server": MODEL_SERVER_URL}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model server unavailable: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(patient: PatientData):
    """Predict heart disease for a single patient."""
    try:
        features = engineer_features_dict(patient.dict())
        result = invoke_model(features)

        prediction = result["predictions"][0]
        probabilities = result["probabilities"][0]
        disease_prob = probabilities[1]

        return PredictionResponse(
            prediction=prediction,
            probability=disease_prob,
            risk_level=get_risk_level(disease_prob),
            interpretation=get_interpretation(prediction, disease_prob),
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: PredictionRequest):
    """Predict heart disease for multiple patients."""
    try:
        results = []
        for patient in request.patients:
            features = engineer_features_dict(patient.dict())
            result = invoke_model(features)

            prediction = result["predictions"][0]
            probabilities = result["probabilities"][0]
            disease_prob = probabilities[1]

            results.append({
                "patient_data": patient.dict(),
                "prediction": prediction,
                "probability": disease_prob,
                "risk_level": get_risk_level(disease_prob),
                "interpretation": get_interpretation(prediction, disease_prob),
            })

        return {"predictions": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Predict heart disease for every patient listed in an uploaded CSV file."""
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be CSV format")

        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        required_columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                             "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

        null_mask = df[required_columns].isnull()
        if null_mask.values.any():
            bad_rows = [
                {"row": int(idx), "missing_columns": [c for c in required_columns if null_mask.loc[idx, c]]}
                for idx in df.index[null_mask.any(axis=1)]
            ]
            raise HTTPException(
                status_code=400,
                detail=f"CSV has missing values in required columns: {bad_rows}",
            )

        results = []
        for idx, row in df.iterrows():
            patient_data = row[required_columns].to_dict()
            features = engineer_features_dict(patient_data)
            result = invoke_model(features)

            prediction = result["predictions"][0]
            probabilities = result["probabilities"][0]
            disease_prob = probabilities[1]

            results.append({
                "row": int(idx),
                "patient_data": patient_data,
                "prediction": prediction,
                "probability": disease_prob,
                "risk_level": get_risk_level(disease_prob),
                "interpretation": get_interpretation(prediction, disease_prob),
            })

        return {
            "total_patients": len(results),
            "predictions": results,
            "summary": {
                "disease_detected": sum(1 for r in results if r["prediction"] == 1),
                "no_disease": sum(1 for r in results if r["prediction"] == 0),
                "high_risk": sum(1 for r in results if r["risk_level"] in ["High", "Very High"]),
            },
        }
    except HTTPException:
        raise
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

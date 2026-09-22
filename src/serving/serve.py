"""
This module runs the local model server.

It runs entirely on your own machine as a small FastAPI process. The
process loads the currently deployed model from models/registry/latest/
and serves predictions over plain HTTP on localhost.
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Response

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.paths import MODELS_REGISTRY_LATEST_DIR, MONITORING_DIR, ensure_dirs
from src.inference.inference import format_response, load_model, parse_request, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_model = None
_scaler = None
PREDICTIONS_LOG = os.path.join(MONITORING_DIR, "predictions_log.jsonl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler
    ensure_dirs()
    model_path = os.path.join(MODELS_REGISTRY_LATEST_DIR, "model.pth")
    if not os.path.exists(model_path):
        logger.warning(
            f"No model found at {model_path}. Run the full pipeline, download, then "
            f"preprocess, then train, then deploy, before serving predictions."
        )
    else:
        _model, _scaler = load_model(MODELS_REGISTRY_LATEST_DIR)
        logger.info(f"Model loaded from {MODELS_REGISTRY_LATEST_DIR}")
    yield


app = FastAPI(title="Heart Disease Local Model Server", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Check whether a model is loaded and ready to serve predictions."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Deploy a model first.")
    return {"status": "healthy"}


@app.post("/predict")
async def predict_endpoint(request: Request):
    """Run a prediction. The request body must contain a features list of 19 floats."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Deploy a model first.")

    body = await request.body()
    try:
        input_data = parse_request(body, "application/json")
        result = predict(input_data, _model, _scaler)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    _log_prediction(json.loads(body.decode()), result)

    return Response(content=format_response(result, "application/json"), media_type="application/json")


def _log_prediction(request_payload, result):
    """Append each prediction to a local log file. The monitoring script reads this
    log later to check for feature drift."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": request_payload.get("features"),
        "prediction": result["predictions"][0],
        "probability": max(result["probabilities"][0]),
    }
    os.makedirs(MONITORING_DIR, exist_ok=True)
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

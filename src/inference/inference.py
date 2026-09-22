"""
This module holds the inference logic for the heart disease model.

It loads a trained model from disk, parses an incoming request, runs a
prediction, and formats the result. The local serving process in
src/serving/serve.py imports these functions directly.
"""

import json
import os
import pickle

import numpy as np
import torch

from src.common.model_def import HeartDiseaseNN


def load_model(model_dir):
    """Load the model and its fitted scaler from `model_dir`.

    This directory must contain model.pth and scaler.pkl. The model was
    trained on StandardScaler-normalized features, so the scaler has to
    travel with the model and be applied to every input at inference time.
    """
    device = torch.device("cpu")
    model = HeartDiseaseNN(input_size=19).to(device)

    model_path = os.path.join(model_dir, "model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def parse_request(request_body, content_type):
    """Parse a raw request body into a feature array."""
    if content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["features"], dtype=np.float32)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict(input_data, model, scaler):
    """Run a forward pass and return predictions with their probabilities.

    `input_data` must be the raw (unscaled) 19-feature vector. It is
    normalized with `scaler` before being handed to the model, matching
    the normalization applied to the training data in data_preprocess.py.
    """
    device = torch.device("cpu")

    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)

    scaled_input = scaler.transform(input_data)
    input_tensor = torch.FloatTensor(scaled_input).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

    return {
        "predictions": predictions.cpu().numpy().tolist(),
        "probabilities": probabilities.cpu().numpy().tolist(),
    }


def format_response(result, content_type):
    """Format a prediction result for the response body."""
    if content_type == "application/json":
        return json.dumps(result)
    raise ValueError(f"Unsupported content type: {content_type}")

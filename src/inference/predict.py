"""
This script is a command-line client for the local model server.

It posts a feature vector to the FastAPI server that src/serving/serve.py
starts, and prints the resulting prediction.
"""

import argparse
import json
import logging

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://localhost:8080"


def predict_single(server_url, features):
    response = requests.post(
        f"{server_url}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"features": features}),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def predict_batch(server_url, features_list):
    return [predict_single(server_url, features) for features in features_list]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--features", type=str, help="JSON list of 19 features")
    parser.add_argument("--input-file", help="JSON file with features (single record or list)")
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            results = predict_batch(args.server_url, data)
        else:
            results = predict_single(args.server_url, data["features"])
    elif args.features:
        results = predict_single(args.server_url, json.loads(args.features))
    else:
        # This example patient's feature vector already includes the 6 engineered features.
        example_features = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 9.135, 3.698, 2.381, 0, 1, 0]
        results = predict_single(args.server_url, example_features)

    print(json.dumps(results, indent=2))

    if isinstance(results, dict):
        prediction = results["predictions"][0]
        probability = results["probabilities"][0]
        logger.info(f"Prediction: {'Disease' if prediction == 1 else 'No Disease'}")
        logger.info(f"Confidence: {max(probability):.2%}")


if __name__ == "__main__":
    main()

import json

import numpy as np
import pytest

from src.common.model_def import HeartDiseaseNN
from src.inference.inference import format_response, parse_request, predict


def test_parse_request_reads_a_json_feature_list():
    body = json.dumps({"features": list(range(19))}).encode("utf-8")

    parsed = parse_request(body, "application/json")

    assert parsed.tolist() == list(range(19))


def test_parse_request_rejects_unsupported_content_type():
    with pytest.raises(ValueError):
        parse_request(b"not json", "text/plain")


def test_format_response_encodes_a_result_as_json():
    result = {"predictions": [1], "probabilities": [[0.1, 0.9]]}

    encoded = format_response(result, "application/json")

    assert json.loads(encoded) == result


def test_format_response_rejects_unsupported_content_type():
    with pytest.raises(ValueError):
        format_response({}, "text/plain")


class _IdentityScaler:
    """A stand-in for a fitted StandardScaler that leaves values unchanged."""

    def transform(self, X):
        return X


def test_predict_returns_one_prediction_and_probability_pair_per_sample():
    model = HeartDiseaseNN(input_size=19)
    model.eval()
    features = np.zeros(19, dtype=np.float32)

    result = predict(features, model, _IdentityScaler())

    assert len(result["predictions"]) == 1
    assert len(result["probabilities"]) == 1
    assert len(result["probabilities"][0]) == 2
    assert abs(sum(result["probabilities"][0]) - 1.0) < 1e-5


def test_predict_scales_input_through_the_provided_scaler():
    """predict() must normalize raw features before the forward pass, since the
    model is trained on StandardScaler-normalized data. Feeding it unscaled
    values silently produces meaningless predictions with no crash, so this
    test asserts the scaler is actually invoked rather than bypassed."""
    model = HeartDiseaseNN(input_size=19)
    model.eval()
    features = np.full(19, 1000.0, dtype=np.float32)

    class _RecordingScaler:
        def __init__(self):
            self.received = None

        def transform(self, X):
            self.received = X
            return np.zeros_like(X)

    scaler = _RecordingScaler()
    predict(features, model, scaler)

    assert scaler.received is not None
    assert scaler.received.tolist() == [features.tolist()]

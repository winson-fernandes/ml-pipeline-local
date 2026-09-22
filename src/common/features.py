"""
This module holds the feature engineering shared by the preprocessing
stage and the serving stage.

This logic must stay identical between training and serving. If the two
copies drift apart, the model silently receives different features than
it was trained on. data_preprocess.py and frontend-app/backend/api.py both
import this single copy instead of duplicating the transform.
"""

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# These are the 13 raw clinical features used as model inputs. Every column except target belongs here.
RAW_FEATURE_COLUMNS = [c for c in RAW_COLUMNS if c != "target"]

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# The model receives these 19 features in this exact order.
FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + [
    "age_bp_interaction", "chol_age_ratio", "heart_rate_age_ratio",
    "high_chol", "high_bp", "low_heart_rate",
]


def engineer_features_df(df):
    """Add the engineered columns to a pandas DataFrame (used during preprocessing)."""
    df["age_bp_interaction"] = df["age"] * df["trestbps"] / 1000
    df["chol_age_ratio"] = df["chol"] / df["age"]
    df["heart_rate_age_ratio"] = df["thalach"] / df["age"]
    df["high_chol"] = (df["chol"] > 240).astype(int)
    df["high_bp"] = (df["trestbps"] > 140).astype(int)
    df["low_heart_rate"] = (df["thalach"] < 100).astype(int)
    return df


def engineer_features_dict(data: dict):
    """Build the 19-value feature vector from a single raw patient record (used at inference time)."""
    age_bp_interaction = data["age"] * data["trestbps"] / 1000
    chol_age_ratio = data["chol"] / data["age"]
    heart_rate_age_ratio = data["thalach"] / data["age"]
    high_chol = 1 if data["chol"] > 240 else 0
    high_bp = 1 if data["trestbps"] > 140 else 0
    low_heart_rate = 1 if data["thalach"] < 100 else 0

    return [
        data["age"], data["sex"], data["cp"], data["trestbps"], data["chol"],
        data["fbs"], data["restecg"], data["thalach"], data["exang"],
        data["oldpeak"], data["slope"], data["ca"], data["thal"],
        age_bp_interaction, chol_age_ratio, heart_rate_age_ratio,
        high_chol, high_bp, low_heart_rate,
    ]

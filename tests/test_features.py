import pandas as pd

from src.common.features import (
    FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    engineer_features_df,
    engineer_features_dict,
)

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


def test_raw_feature_columns_has_thirteen_entries_and_excludes_target():
    assert len(RAW_FEATURE_COLUMNS) == 13
    assert "target" not in RAW_FEATURE_COLUMNS


def test_feature_columns_has_nineteen_entries():
    assert len(FEATURE_COLUMNS) == 19


def test_engineer_features_dict_returns_nineteen_values_in_order():
    features = engineer_features_dict(SAMPLE_PATIENT)

    assert len(features) == 19
    assert features[:13] == [
        SAMPLE_PATIENT["age"], SAMPLE_PATIENT["sex"], SAMPLE_PATIENT["cp"],
        SAMPLE_PATIENT["trestbps"], SAMPLE_PATIENT["chol"], SAMPLE_PATIENT["fbs"],
        SAMPLE_PATIENT["restecg"], SAMPLE_PATIENT["thalach"], SAMPLE_PATIENT["exang"],
        SAMPLE_PATIENT["oldpeak"], SAMPLE_PATIENT["slope"], SAMPLE_PATIENT["ca"],
        SAMPLE_PATIENT["thal"],
    ]
    high_chol, high_bp, low_heart_rate = features[16:]
    assert (high_chol, high_bp, low_heart_rate) == (0, 1, 0)


def test_engineer_features_df_matches_engineer_features_dict():
    df = pd.DataFrame([SAMPLE_PATIENT])
    df = engineer_features_df(df)

    dict_features = engineer_features_dict(SAMPLE_PATIENT)
    df_features = df[FEATURE_COLUMNS].iloc[0].tolist()

    assert df_features == dict_features

"""
This module centralizes every filesystem path the pipeline uses.

Every stage stores its files on local disk under PROJECT_ROOT. Keeping
every path in one module means you can retarget the whole pipeline to a
different drive by changing a single line.
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODELS_OUTPUT_DIR = os.path.join(MODELS_DIR, "output")
MODELS_SCALERS_DIR = os.path.join(MODELS_DIR, "scalers")
MODELS_REGISTRY_DIR = os.path.join(MODELS_DIR, "registry")
MODELS_REGISTRY_LATEST_DIR = os.path.join(MODELS_REGISTRY_DIR, "latest")

EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
MONITORING_DIR = os.path.join(PROJECT_ROOT, "monitoring")


def ensure_dirs():
    for d in [
        DATA_RAW_DIR, DATA_PROCESSED_DIR,
        MODELS_OUTPUT_DIR, MODELS_SCALERS_DIR, MODELS_REGISTRY_LATEST_DIR,
        EVALUATION_DIR, MONITORING_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def latest_file(directory, suffix):
    """Return the most recently modified file in `directory` ending with `suffix`."""
    if not os.path.isdir(directory):
        return None
    candidates = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def latest_run_dir(base_dir=MODELS_OUTPUT_DIR):
    """Return the most recently created training-run directory under models/output/."""
    if not os.path.isdir(base_dir):
        return None
    candidates = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

"""
inference_sentiment.py
──────────────────────
SageMaker inference handler for HW6 Option 1 (Regression).
Predicts AAPL daily stock return (float).

The model tarball contains:
    finalized_sentiment_model.joblib   <- fitted sklearn Pipeline
    Custom_Classes.py                  <- FeatureEngineer, PairFeatureEngineer, etc.

SageMaker extracts the tarball to SM_MODEL_DIR and sets that as the working
directory, so we import Custom_Classes directly (no src/ prefix needed).
"""

import os
import sys
import json
from io import BytesIO, StringIO

import numpy  as np
import pandas as pd
import joblib

# SageMaker sets SM_MODEL_DIR to where the tarball was extracted
model_dir = os.environ.get("SM_MODEL_DIR", ".")

# Add model_dir to path so Custom_Classes.py (bundled in the tarball) is found
for path in [model_dir, os.path.join(model_dir, "code")]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import custom transformer classes so joblib can deserialise pipelines
# that contain them. Non-fatal if pipeline uses only standard sklearn steps.
try:
    from Custom_Classes import (
        FeatureEngineer,
        AutoPowerTransformer,
        FeatureSelector,
        PairFeatureEngineer,
        Word2VecTransformer,
    )
    print("Custom_Classes imported successfully.")
except ImportError as e:
    print(f"WARNING: Could not import Custom_Classes ({e}).")


def model_fn(model_dir: str):
    """Load the sklearn pipeline from the model directory."""
    for fname in (
        "finalized_sentiment_model.joblib",
        "finalized_sentiment_model.pkl",
    ):
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"Model loaded from {path}")
            return model
    raise FileNotFoundError(
        f"No model file found in {model_dir}. "
        f"Contents: {os.listdir(model_dir)}"
    )


def input_fn(request_body, request_content_type: str):
    """Deserialise the incoming request into a pandas DataFrame."""
    print(f"Received content type: {request_content_type}")

    if request_content_type == "application/x-npy":
        arr = np.load(BytesIO(request_body), allow_pickle=True)
        return pd.DataFrame(arr)

    elif request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)

    elif request_content_type in ("text/csv", "text/plain"):
        return pd.read_csv(StringIO(request_body), header=None)

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_df: pd.DataFrame, model):
    """Run the sklearn pipeline and return predictions."""
    print(f"Running prediction on input shape: {input_df.shape}")
    return model.predict(input_df)


def output_fn(prediction, content_type: str):
    """Serialise predictions back to JSON."""
    print("Formatting output...")
    if isinstance(prediction, (np.ndarray, np.generic)):
        result = prediction.tolist()
    else:
        result = list(prediction)
    return json.dumps(result), "application/json"

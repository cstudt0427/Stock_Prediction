import joblib
import os
import pandas as pd
import json
import numpy as np
import sys
import imblearn
from io import BytesIO, StringIO


model_dir = os.environ.get('SM_MODEL_DIR')

if model_dir not in sys.path:
    sys.path.append(model_dir)

# If you have a Custom_Classes module in src/, keep this import.
# Otherwise, comment it out.
# from src.Custom_Classes import FeatureEngineer


def model_fn(model_dir):
    """Load the fraud detection model from the specified directory."""
    # ---------------------------------------------------------------
    # UPDATE: filename matches your saved pipeline artifact
    # ---------------------------------------------------------------
    path = os.path.join(model_dir, 'finalized_fraud_model.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")

    model = joblib.load(path)
    print("Model loaded successfully.")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request into a pandas DataFrame.

    Supported content types:
      - application/json   (preferred – preserves column names)
      - text/csv
      - application/x-npy
    """
    print(f"Receiving data of type: {request_content_type}")

    # ---------------------------------------------------------------
    # UPDATE (teacher rec #3): JSON is now the primary/preferred path
    # because it keeps column names intact for the sklearn pipeline.
    # ---------------------------------------------------------------
    if request_content_type == 'application/json':
        # request_body may arrive as bytes or str
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode('utf-8')

        payload = json.loads(request_body)

        # Accept both a plain dict (single row) and a list of dicts
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        elif isinstance(payload, list):
            return pd.DataFrame(payload)
        else:
            raise ValueError("JSON payload must be a dict or list of dicts.")

    elif request_content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body))

    elif request_content_type == 'application/x-npy':
        data = np.load(BytesIO(request_body), allow_pickle=True)
        return pd.DataFrame(data)

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_df, model):
    """Run the loaded pipeline on the deserialized DataFrame."""
    print("Running prediction pipeline...")
    return model.predict(input_df)


def output_fn(prediction, content_type):
    """
    Serialize predictions back to the caller as JSON.
    Returns a list so the Streamlit app (NumpyDeserializer / JSONDeserializer)
    can parse it consistently.
    """
    print("Formatting output...")
    if isinstance(prediction, (np.ndarray, np.generic)):
        res = prediction.tolist()
    else:
        res = prediction
    return json.dumps(res), "application/json"

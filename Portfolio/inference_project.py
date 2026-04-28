import joblib
import os
import pandas as pd
import json
import numpy as np
import sys
from io import BytesIO, StringIO


model_dir = os.environ.get('SM_MODEL_DIR', '.')

if model_dir not in sys.path:
    sys.path.append(model_dir)


def model_fn(model_dir):
    """Load the loan default model from the specified directory."""
    # Filename must match what was saved in the notebook (Section 13)
    path = os.path.join(model_dir, 'finalized_loan_model.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path)
    print("Model loaded successfully.")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request into a pandas DataFrame.
    rec #3: JSON is the preferred content type — it preserves column names
    so the sklearn ColumnTransformer can match numeric vs categorical features.
    """
    print(f"Receiving data of type: {request_content_type}")

    if request_content_type == 'application/json':
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode('utf-8')
        payload = json.loads(request_body)
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
    """Run the full ImbPipeline on the deserialized DataFrame."""
    print("Running prediction pipeline...")
    return model.predict(input_df)


def output_fn(prediction, content_type):
    """Serialize predictions back to the caller as JSON."""
    print("Formatting output...")
    if isinstance(prediction, (np.ndarray, np.generic)):
        res = prediction.tolist()
    else:
        res = prediction
    return json.dumps(res), "application/json"

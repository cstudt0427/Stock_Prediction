import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

warnings.simplefilter("ignore")

st.set_page_config(page_title="Bitcoin Signal Predictor", layout="wide")
st.title("₿ Bitcoin Buy / Hold / Sell (SageMaker Endpoint)")

# ---------------------------
# Make sure repo root is on path so we can import src/*
# (Streamlit Cloud runs from repo root, but this makes it robust)
# ---------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Import the SAME FeatureEngineer used in the notebook training
# Ensure your repo has: src/Custom_Classes.py
from src.Custom_Classes import FeatureEngineer

# ---------------------------
# Secrets (Streamlit Cloud)
# ---------------------------
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"].get("AWS_SESSION_TOKEN", None)
aws_region = st.secrets["aws_credentials"].get("AWS_REGION", "us-east-1")
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ---------------------------
# AWS Session
# ---------------------------
@st.cache_resource
def get_sm_session(_aws_id, _aws_secret, _aws_token, _aws_region):
    session = boto3.Session(
        aws_access_key_id=_aws_id,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        region_name=_aws_region,
    )
    return sagemaker.Session(boto_session=session)

sm_session = get_sm_session(aws_id, aws_secret, aws_token, aws_region)

@st.cache_resource
def get_predictor(endpoint_name: str):
    # CSV is most reliable with typical SageMaker input_fn parsing
    return Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sm_session,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer()
    )

predictor = get_predictor(aws_endpoint)

# ---------------------------
# UI
# ---------------------------
with st.form("pred_form"):
    st.subheader("Inputs")
    col1, col2 = st.columns(2)

    with col1:
        close = st.number_input(
            "Close",
            min_value=0.0,
            value=50000.0,
            step=10.0,
            help="BTC Close price used to compute technical features."
        )

    with col2:
        st.caption("This app computes the same engineered features used during training.")
        show_debug = st.checkbox("Show debug tables", value=True)

    submitted = st.form_submit_button("Run Prediction")

# ---------------------------
# Prediction helpers
# ---------------------------
LABEL_MAP = {-1: "Sell", 0: "Hold", 1: "Buy"}

def normalize_prediction(raw):
    """
    Handles common SageMaker output formats:
      - {"predictions":[...]} or {"prediction": ...}
      - [x] or [[x]]
      - "0" or "0.0"
    Returns an int label if possible; otherwise None.
    """
    if isinstance(raw, dict):
        if "predictions" in raw:
            raw = raw["predictions"]
        elif "prediction" in raw:
            raw = raw["prediction"]
        else:
            raw = list(raw.values())[0]

    if isinstance(raw, list):
        if len(raw) == 0:
            return None
        first = raw[0]
        if isinstance(first, list) and len(first) > 0:
            raw = first[0]
        else:
            raw = first

    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")

    try:
        return int(round(float(raw)))
    except Exception:
        return None

# ---------------------------
# Run prediction
# ---------------------------
if submitted:
    try:
        # 1) Create raw input row
        raw_df = pd.DataFrame({"Close": [float(close)]})

        # 2) Compute engineered features exactly like notebook training
        fe = FeatureEngineer(windows=[5])
        features_np = fe.transform(raw_df[["Close"]])

        # 3) Convert to DataFrame with stable names
        input_df = pd.DataFrame(
            features_np,
            columns=[f"feat_{i}" for i in range(features_np.shape[1])]
        )

        # Optional debug views
        if show_debug:
            st.subheader("Debug: Feature row sent to endpoint")
            st.dataframe(input_df)

        # 4) Call endpoint
        raw_pred = predictor.predict(input_df)
        pred_label = normalize_prediction(raw_pred)

        if pred_label is None:
            st.error(f"Could not parse prediction output: {raw_pred}")
        else:
            st.success(f"Prediction: **{LABEL_MAP.get(pred_label, str(pred_label))}** (raw={pred_label})")

    except Exception as e:
        st.error(f"Endpoint invocation failed: {e}")
        st.info("Most common causes: endpoint expects different feature columns, or FeatureEngineer differs from training.")

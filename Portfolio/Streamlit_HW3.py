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
    # Use CSV to match SageMaker input_fn path: text/csv -> pd.read_csv(...)
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
            help="BTC close price used by the model input."
        )

    with col2:
        st.caption("If your trained pipeline expects ONLY Close, leave the other fields blank.")
        include_ohlcv = st.checkbox("Send OHLCV (only if your model expects it)", value=False)

    open_p = high = low = vol_btc = None
    if include_ohlcv:
        c3, c4 = st.columns(2)
        with c3:
            open_p = st.number_input("Open", min_value=0.0, value=float(close), step=10.0)
            high = st.number_input("High", min_value=0.0, value=float(close), step=10.0)
        with c4:
            low = st.number_input("Low", min_value=0.0, value=float(close), step=10.0)
            vol_btc = st.number_input("Volume_(BTC)", min_value=0.0, value=0.0, step=0.01)

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
    # dict -> take predictions field or first value
    if isinstance(raw, dict):
        if "predictions" in raw:
            raw = raw["predictions"]
        elif "prediction" in raw:
            raw = raw["prediction"]
        else:
            raw = list(raw.values())[0]

    # list -> scalar
    if isinstance(raw, list):
        if len(raw) == 0:
            return None
        first = raw[0]
        if isinstance(first, list) and len(first) > 0:
            raw = first[0]
        else:
            raw = first

    # bytes -> decode
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")

    # string -> float -> int
    try:
        return int(round(float(raw)))
    except Exception:
        return None

# ---------------------------
# Run prediction
# ---------------------------
if submitted:
    # Build a one-row dataframe with expected column names.
    payload = {"Close": float(close)}
    if include_ohlcv:
        payload.update({
            "Open": float(open_p),
            "High": float(high),
            "Low": float(low),
            "Volume_(BTC)": float(vol_btc)
        })

    input_df = pd.DataFrame([payload])

    with st.expander("Debug: Payload sent to endpoint"):
        st.dataframe(input_df)

    try:
        # With CSVSerializer, sending a DataFrame yields a CSV body with header.
        raw_pred = predictor.predict(input_df)
        pred_label = normalize_prediction(raw_pred)

        if pred_label is None:
            st.error(f"Could not parse prediction output: {raw_pred}")
        else:
            st.success(f"Prediction: **{LABEL_MAP.get(pred_label, str(pred_label))}** (raw={pred_label})")
            st.write("Model input row:")
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Endpoint invocation failed: {e}")
        st.info("Verify AWS secrets/region/endpoint name and that your input columns match the model’s training inputs.")

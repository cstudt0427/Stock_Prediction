# =========================
# Portfolio/Streamlit_HW3.py   (or HW3.py)
# =========================
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
# Ensure repo root is on sys.path so `src` is importable
# (Adjust if your Streamlit file is in a different folder depth)
# ---------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.feature_utils import get_bitcoin_close_history

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
    # CSV works well with typical SageMaker input_fn using pd.read_csv(StringIO(...))
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
with st.form("prediction_form"):
    st.subheader("Inputs")

    col1, col2 = st.columns(2)
    with col1:
        close_price = st.number_input(
            "Close (USD)",
            min_value=0.0,
            value=50000.0,
            step=10.0,
            help="Latest BTC close price to append to the recent history window."
        )

    with col2:
        days = st.number_input(
            "History range (days from CoinGecko)",
            min_value=30,
            max_value=3650,
            value=365,
            step=30,
            help="How much history to pull from CoinGecko."
        )
        tail_n = st.number_input(
            "Rows sent to endpoint (tail_n)",
            min_value=50,
            max_value=2000,
            value=300,
            step=50,
            help="How many most-recent rows to send. Must be >= largest rolling/EMA window used in training."
        )

    show_debug = st.checkbox("Show debug tables", value=True)
    submitted = st.form_submit_button("Run Prediction")

# ---------------------------
# Prediction helpers
# ---------------------------
LABEL_MAP = {-1: "Sell", 0: "Hold", 1: "Buy"}

def extract_last_label(raw):
    """
    Supports JSON outputs like:
      - {"predictions": [..]}
      - [..] or [[..]]
      - scalar
    Returns last label (int) or None.
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
        last = raw[-1]
        if isinstance(last, list) and len(last) > 0:
            last = last[0]
        try:
            return int(round(float(last)))
        except Exception:
            return None

    try:
        return int(round(float(raw)))
    except Exception:
        return None

# ---------------------------
# Run prediction
# ---------------------------
if submitted:
    try:
        # 1) Pull recent close history from CoinGecko
        history = get_bitcoin_close_history(days=int(days), tail_n=int(tail_n))

        # 2) Append the user-provided close to the end
        input_df = pd.concat(
            [history, pd.DataFrame({"Close": [float(close_price)]})],
            ignore_index=True
        )

        if show_debug:
            st.subheader("Debug: Payload sent to endpoint (last 10 rows)")
            st.dataframe(input_df.tail(10))
            st.write("Payload shape:", input_df.shape)
            st.write("Columns:", list(input_df.columns))

        # 3) Invoke endpoint
        raw_pred = predictor.predict(input_df)

        pred_label = extract_last_label(raw_pred)
        if pred_label is None:
            st.error(f"Could not parse prediction output: {raw_pred}")
        else:
            st.success(f"Prediction: **{LABEL_MAP.get(pred_label, str(pred_label))}** (raw={pred_label})")

    except Exception as e:
        st.error(f"Endpoint invocation failed: {e}")
        st.info("If this persists, your endpoint likely expects different input columns than a single 'Close' series.")

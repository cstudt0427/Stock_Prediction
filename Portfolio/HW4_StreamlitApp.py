import os
import sys
import warnings
import tempfile
import tarfile
import posixpath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import joblib
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline
import shap

# ── Setup ─────────────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

# Make sure the project's 'src' folder is importable (mirrors notebook structure)
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_pair   # returns a DataFrame of recent features

# ── AWS Credentials from Streamlit Secrets ────────────────────────────────────
# Expected secrets.toml structure:
#
# [aws_credentials]
# AWS_ACCESS_KEY_ID     = "..."
# AWS_SECRET_ACCESS_KEY = "..."
# AWS_SESSION_TOKEN     = "..."
# AWS_BUCKET            = "your-s3-bucket-name"
# AWS_ENDPOINT          = "logistic-pipeline-endpoint-auto-6"
# AWS_TARGET_TICKER     = "GOOGL"    # target stock used during training
# AWS_PARTNER_TICKER    = "GOOG"     # cointegrated partner stock

aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]
target_ticker  = st.secrets["aws_credentials"].get("AWS_TARGET_TICKER",  "GOOGL")
partner_ticker = st.secrets["aws_credentials"].get("AWS_PARTNER_TICKER", "GOOG")

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint":   aws_endpoint,
    "explainer":  "explainer_pair.shap",
    "pipeline":   "finalized_pair_model.tar.gz",
    # Two price inputs: partner first, then target (matches training column order)
    "keys":       [partner_ticker, target_ticker],
    "inputs": [
        {
            "name":    partner_ticker,
            "label":   f"{partner_ticker} Price ($)",
            "type":    "number",
            "min":     0.01,
            "max":     10000.0,
            "default": 150.0,
            "step":    0.01,
        },
        {
            "name":    target_ticker,
            "label":   f"{target_ticker} Price ($)",
            "type":    "number",
            "min":     0.01,
            "max":     10000.0,
            "default": 150.0,
            "step":    0.01,
        },
    ],
}

SIGNAL_MAP   = {1: "🟢 BUY", 0: "⚪ HOLD", -1: "🔴 SELL"}
SIGNAL_COLOR = {1: "green",  0: "grey",    -1: "red"}

# ── AWS Session (cached to avoid re-creating on every rerun) ──────────────────
@st.cache_resource
def get_session(key_id, secret, token):
    return boto3.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        aws_session_token=token,
        region_name="us-east-1",
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Feature extraction (cached) ───────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_base_features():
    """Load the rolling feature history needed by PairFeatureEngineer."""
    return extract_features_pair()

df_features = load_base_features()

# ── S3 helpers ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket):
    """Download and deserialize the sklearn pipeline from S3."""
    s3_client = _session.client("s3")
    filename  = MODEL_INFO["pipeline"]
    key       = f"sklearn-pipeline-deployment/{filename}"
    tmp_path  = os.path.join(tempfile.gettempdir(), filename)

    s3_client.download_file(Bucket=bucket, Key=key, Filename=tmp_path)

    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        joblib_name = [f for f in tar.getnames() if f.endswith(".joblib")][0]

    return joblib.load(os.path.join(tempfile.gettempdir(), joblib_name))


@st.cache_resource
def load_shap_explainer(_session, bucket):
    """Download and deserialize the SHAP explainer from S3."""
    s3_client  = _session.client("s3")
    expl_name  = MODEL_INFO["explainer"]
    local_path = os.path.join(tempfile.gettempdir(), expl_name)
    key        = posixpath.join("explainer", expl_name)

    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# ── Prediction via SageMaker endpoint ────────────────────────────────────────
def call_endpoint(input_array: np.ndarray):
    """Send a 2-D numpy array to the SageMaker endpoint and return (prediction, status)."""
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )
    try:
        raw = predictor.predict(input_array)
        # The endpoint returns the predicted class label
        pred = int(np.array(raw).flatten()[-1])
        return pred, 200
    except Exception as exc:
        return str(exc), 500

# ── SHAP explanation panel ────────────────────────────────────────────────────
def show_explanation(input_df: pd.DataFrame):
    explainer   = load_shap_explainer(session, aws_bucket)
    shap_values = explainer(input_df)

    st.subheader("🔍 Decision Transparency (SHAP)")

    fig, _ = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)

    top_feature = shap_values[0].feature_names[
        int(np.argmax(np.abs(shap_values[0].values)))
    ]
    st.info(
        f"**Business Insight:** The most influential factor driving this signal "
        f"was **{top_feature}**."
    )

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pair Trading Signal", layout="wide")
st.title("📈 Pair Trading Signal Predictor")
st.markdown(
    f"""
    This app predicts **BUY / HOLD / SELL** signals for the **{target_ticker}** stock
    using a cointegrated pair with **{partner_ticker}**.

    The model is a tuned Logistic Regression pipeline (ElasticNet regularization + SMOTE)
    trained on S&P 500 historical price data and deployed on **AWS SageMaker**.
    """
)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("Enter Today's Closing Prices")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                label=inp["label"],
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
                format="%.2f",
            )

    submitted = st.form_submit_button("🔮 Generate Signal", use_container_width=True)

# ── Prediction & display ──────────────────────────────────────────────────────
if submitted:
    with st.spinner("Calling SageMaker endpoint…"):
        # Build a single-row DataFrame matching the training column order
        new_row = [[user_inputs[k] for k in MODEL_INFO["keys"]]]
        input_df = pd.concat(
            [df_features, pd.DataFrame(new_row, columns=df_features.columns)],
            ignore_index=True,
        )
        input_array = input_df.to_numpy(dtype=np.float64)

        pred, status = call_endpoint(input_array)

    if status == 200:
        signal_label = SIGNAL_MAP.get(pred, str(pred))
        color        = SIGNAL_COLOR.get(pred, "grey")

        st.markdown("---")
        col_metric, col_desc = st.columns([1, 3])

        with col_metric:
            st.metric(label="Predicted Signal", value=signal_label)

        with col_desc:
            descriptions = {
                 1: f"The model expects **{target_ticker}** to rise more than 1% tomorrow — consider opening or holding a long position.",
                 0: f"The model expects **{target_ticker}** to trade flat (< 1% move) — hold current position and monitor.",
                -1: f"The model expects **{target_ticker}** to fall more than 1% tomorrow — consider reducing exposure or shorting.",
            }
            st.info(descriptions.get(pred, "Signal generated."))

        # SHAP waterfall
        with st.spinner("Computing SHAP explanation…"):
            try:
                show_explanation(input_df.tail(1).reset_index(drop=True))
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

    else:
        st.error(f"Endpoint error: {pred}")

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        f"""
        **Strategy:** Statistical Arbitrage / Pairs Trading

        **Target stock:** `{target_ticker}`
        **Partner stock:** `{partner_ticker}`

        **Model:** Logistic Regression
        - ElasticNet regularization
        - SMOTE class-imbalance handling
        - Tuned via 10-fold GridSearchCV

        **Features:** Generated by `PairFeatureEngineer`
        (rolling spread, z-score, lag returns, etc.)

        **Deployment:** AWS SageMaker real-time endpoint
        """
    )
    st.markdown("---")
    st.caption("Supply Chain & FinTech — ML Deployment Project")

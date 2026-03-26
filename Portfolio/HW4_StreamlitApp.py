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

# On Streamlit Cloud, feature_utils.py lives in the same folder as this app
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from feature_utils import extract_features_pair

# ── AWS Credentials from Streamlit Secrets ────────────────────────────────────
# secrets.toml structure:
#
# [aws_credentials]
# AWS_ACCESS_KEY_ID     = "..."
# AWS_SECRET_ACCESS_KEY = "..."
# AWS_SESSION_TOKEN     = "..."
# AWS_BUCKET            = "carson-studt-s3-bucket"
# AWS_ENDPOINT          = "logistic-pipeline-endpoint-auto-6-v8"
# AWS_TARGET_TICKER     = "GOOGL"
# AWS_PARTNER_TICKER    = "GOOG"   # set to whichever partner was selected

aws_id         = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret     = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token      = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket     = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint   = st.secrets["aws_credentials"]["AWS_ENDPOINT"]
target_ticker  = st.secrets["aws_credentials"].get("AWS_TARGET_TICKER",  "GOOGL")
partner_ticker = st.secrets["aws_credentials"].get("AWS_PARTNER_TICKER", "GOOG")

# ── Constants ─────────────────────────────────────────────────────────────────
SIGNAL_MAP   = {1: "🟢 BUY", 0: "⚪ HOLD", -1: "🔴 SELL"}
SIGNAL_COLOR = {1: "green",  0: "grey",    -1: "red"}
SIGNAL_DESC  = {
     1: lambda t: f"The model expects **{t}** to rise more than 1% tomorrow — consider opening or holding a long position.",
     0: lambda t: f"The model expects **{t}** to trade flat (< 1% move) — hold current position and monitor.",
    -1: lambda t: f"The model expects **{t}** to fall more than 1% tomorrow — consider reducing exposure or shorting.",
}

# ── AWS Session (created once per app lifetime) ───────────────────────────────
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

# ── Predictor (created once, reused on every button click) ────────────────────
@st.cache_resource
def get_predictor(_sm_session, endpoint_name):
    return Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=_sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )

# ── Feature history (1-hour cache — avoids re-fetching live prices each rerun) ─
@st.cache_data(ttl=3600)
def load_base_features():
    """Returns rolling price history used to build context for PairFeatureEngineer."""
    return extract_features_pair()

df_features = load_base_features()

# ── SHAP explainer (downloaded from S3 once per app lifetime) ─────────────────
@st.cache_resource
def load_shap_explainer(_session, bucket):
    s3_client  = _session.client("s3")
    local_path = os.path.join(tempfile.gettempdir(), "explainer_pair.shap")
    s3_client.download_file(
        Bucket=bucket,
        Key="explainer/explainer_pair.shap",
        Filename=local_path,
    )
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# ── Prediction ────────────────────────────────────────────────────────────────
def call_endpoint(input_array: np.ndarray):
    """Send a single-row numpy array to the SageMaker endpoint."""
    predictor = get_predictor(sm_session, aws_endpoint)
    try:
        raw  = predictor.predict(input_array)
        pred = int(np.array(raw).flatten()[-1])
        return pred, 200
    except Exception as exc:
        return str(exc), 500

# ── SHAP waterfall panel ──────────────────────────────────────────────────────
def show_explanation(input_df: pd.DataFrame):
    explainer   = load_shap_explainer(session, aws_bucket)
    shap_values = explainer(input_df)

    st.subheader("🔍 Decision Transparency (SHAP)")

    # LinearExplainer multiclass → shape (n_samples, n_features, n_classes)
    # Plot sample 0, class 0 (SELL)
    fig, _ = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[:, :, 0][0], max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)

    top_feature = shap_values[:, :, 0][0].feature_names[
        int(np.argmax(np.abs(shap_values[:, :, 0][0].values)))
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
    This app predicts **BUY / HOLD / SELL** signals for **{target_ticker}**
    using a cointegrated pair with **{partner_ticker}**.

    The model is a tuned Logistic Regression pipeline (ElasticNet + SMOTE)
    trained on S&P 500 historical price data and deployed on **AWS SageMaker**.
    """
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("Enter Today's Closing Prices")
    col1, col2 = st.columns(2)

    with col1:
        partner_price = st.number_input(
            label=f"{partner_ticker} Price ($)",
            min_value=0.01, max_value=10000.0, value=150.0, step=0.01, format="%.2f"
        )
    with col2:
        target_price = st.number_input(
            label=f"{target_ticker} Price ($)",
            min_value=0.01, max_value=10000.0, value=150.0, step=0.01, format="%.2f"
        )

    submitted = st.form_submit_button("🔮 Generate Signal", use_container_width=True)

# ── Prediction & display ──────────────────────────────────────────────────────
if submitted:
    with st.spinner("Calling SageMaker endpoint…"):
        # Build full context DataFrame so PairFeatureEngineer rolling windows work
        new_row  = [[partner_price, target_price]]
        input_df = pd.concat(
            [df_features, pd.DataFrame(new_row, columns=df_features.columns)],
            ignore_index=True,
        )

        # Send only the last row to the endpoint
        input_array = input_df.to_numpy(dtype=np.float64)[-1:, :]
        pred, status = call_endpoint(input_array)

    if status == 200:
        signal_label = SIGNAL_MAP.get(pred, str(pred))

        st.markdown("---")
        col_metric, col_desc = st.columns([1, 3])

        with col_metric:
            st.metric(label="Predicted Signal", value=signal_label)

        with col_desc:
            st.info(SIGNAL_DESC.get(pred, lambda t: "Signal generated.")(target_ticker))

        with st.spinner("Computing SHAP explanation…"):
            try:
                show_explanation(input_df.tail(1).reset_index(drop=True))
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")
    else:
        st.error(f"Endpoint error: {pred}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

        **Features:** `PairFeatureEngineer`
        (rolling spread, z-score, beta, lag returns)

        **Deployment:** AWS SageMaker real-time endpoint
        """
    )
    st.markdown("---")
    st.caption("Supply Chain & FinTech — ML Deployment Project")

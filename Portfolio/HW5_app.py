import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import json
import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline
import shap

warnings.simplefilter("ignore")

# ── Path setup so 'src' is importable on Streamlit Cloud ─────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ── Secrets ───────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint":  aws_endpoint,
    "explainer": "explainer_pca.shap",
    "pipeline":  "finalized_pca_model.tar.gz",
    "inputs": [
        {"name": "IBM_CR_Cum",  "min": -100.0, "max": 100.0, "default": 0.0, "step": 0.5},
        {"name": "JPM_CR_Cum", "min": -100.0, "max": 100.0, "default": 0.0, "step": 0.5},
    ]
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    s3_client.download_file(
        Bucket=bucket,
        Key=f"sklearn-pipeline-deployment/{filename}",
        Filename=filename
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

@st.cache_resource
def load_shap_explainer(_session, bucket):
    s3_client  = _session.client('s3')
    local_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"])
    if not os.path.exists(local_path):
        s3_client.download_file(
            Bucket=bucket,
            Key=f"explainer/{MODEL_INFO['explainer']}",
            Filename=local_path
        )
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

def call_endpoint(user_inputs: dict):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw = predictor.predict(user_inputs)
        pred_val = float(pd.DataFrame(raw).values[-1][0])
        return round(pred_val, 6), 200
    except Exception as e:
        return str(e), 500

def display_explanation(user_inputs: dict):
    best_pipeline = load_pipeline(session, aws_bucket)
    explainer     = load_shap_explainer(session, aws_bucket)

    # Load the full dataset to find the closest matching row (same as inference_pca.py)
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    file_path    = os.path.join(project_root, 'Portfolio/SP500Data.csv')
    dataset      = pd.read_csv(file_path, index_col=0)

    target        = 'GOOGL'
    return_period = 5
    SP500_1 = 'IBM_CR_Cum'
    SP500_2 = 'JPM_CR_Cum'

    X = np.log(dataset.drop([target], axis=1)).diff(return_period)
    X = np.exp(X).cumsum()
    X.columns = [name + "_CR_Cum" for name in X.columns]

    distances = np.sqrt(
        (X[SP500_1] - user_inputs[SP500_1]) ** 2 +
        (X[SP500_2] - user_inputs[SP500_2]) ** 2
    )
    closest_row = X.loc[[distances.idxmin()]].copy()
    closest_row[SP500_1] = user_inputs[SP500_1]
    closest_row[SP500_2] = user_inputs[SP500_2]

    # Now transform the full row through preprocessing steps only
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])
    transformed = preprocessing_pipeline.transform(closest_row)
    feature_names = [f"kpca_{i}" for i in range(transformed.shape[1])]
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    shap_values = explainer(transformed_df)

    st.subheader("Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    top_feature = pd.Series(
        shap_values[0].values, index=shap_values[0].feature_names
    ).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor was **{top_feature}**.")

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GOOGL Return Predictor", page_icon="📈", layout="wide")
st.title("📈 GOOGL Cumulative Return Predictor")
st.markdown(
    "Enter the **IBM** and **JPM** cumulative returns. "
    "The model finds the closest matching historical date and predicts "
    "GOOGL's 5-day forward cumulative return."
)

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " "),
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"],
                format="%.4f"
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    with st.spinner("Invoking SageMaker endpoint..."):
        result, status = call_endpoint(user_inputs)

    if status == 200:
        st.success("Prediction received!")
        st.metric("Predicted GOOGL Cumulative Return (5-day forward)", result)
        with st.spinner("Generating SHAP explanation..."):
            display_explanation(user_inputs)
    else:
        st.error(f"Endpoint error: {result}")

st.markdown("---")
st.caption("HW5 - Dimensionality Reduction - Option 1 - GOOGL Return Prediction via KernelPCA + Lasso")

import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer        # rec #3 – JSON preserves column names
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap
from joblib import dump, load

# -----------------------------------------------------------------------
# Setup & Path Configuration
# -----------------------------------------------------------------------
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# -----------------------------------------------------------------------
# Load X_train (saved from notebook, uploaded to Portfolio/ in GitHub)
# rec #5 – used to fill in all pipeline columns the user doesn't touch
# -----------------------------------------------------------------------
file_path = os.path.join(project_root, 'Portfolio/X_train.csv')
dataset = pd.read_csv(file_path)
dataset = dataset.drop(columns=[c for c in dataset.columns if 'Unnamed' in c])

# -----------------------------------------------------------------------
# AWS Secrets  (set these in .streamlit/secrets.toml)
# -----------------------------------------------------------------------
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# -----------------------------------------------------------------------
# AWS Session
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# Model / UI Configuration
# rec #6 – only expose a handful of top features; fill the rest from
# the first row of X_train so the full pipeline still works.
#
# UPDATE THESE KEYS to whichever features your SHAP analysis ranked
# highest (e.g., from feature_importances_ or a SHAP summary plot).
# -----------------------------------------------------------------------
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_sentiment.shap",
    "pipeline"  : "finalized_fraud_model.tar.gz",

    # ---- Top features to expose in the UI ----
    # Replace / extend these with YOUR model's actual top features.
    "keys": ['TransactionAmt', 'card6_freq_enc', 'card3', 'C12'],

    # Slider / number-input spec for each key
    "inputs": [
        {"name": "TransactionAmt",  "type": "number", "min": 0.0,   "max": 20000.0, "default": 100.0,  "step": 1.0},
        {"name": "card6_freq_enc",  "type": "number", "min": 0.0,   "max": 1.0,     "default": 0.5,    "step": 0.01},
        {"name": "card3",           "type": "number", "min": 100.0, "max": 231.0,   "default": 150.0,  "step": 1.0},
        {"name": "C12",             "type": "number", "min": 0.0,   "max": 2720.0,  "default": 1.0,    "step": 1.0},
    ]
}

# -----------------------------------------------------------------------
# Helper: load the sklearn pipeline from S3
# -----------------------------------------------------------------------
@st.cache_resource
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)

# -----------------------------------------------------------------------
# Helper: load the SHAP explainer from S3
# -----------------------------------------------------------------------
@st.cache_resource
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return load(f)

# -----------------------------------------------------------------------
# Prediction: call the SageMaker endpoint
# rec #3 – JSONSerializer keeps column names so inference_project.py
#           can rebuild the DataFrame correctly on the endpoint side.
# -----------------------------------------------------------------------
def call_model_api(input_dict: dict):
    """
    Send a single-row dict to the SageMaker endpoint.
    Returns (label_string, http_status_code).
    """
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),       # rec #3
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_dict)
        # raw_pred is a numpy array; grab the last predicted class
        pred_val  = int(np.array(raw_pred).flat[-1])
        mapping   = {0: "✅ Legitimate", 1: "🚨 Fraud"}
        return mapping.get(pred_val, f"Unknown ({pred_val})"), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# -----------------------------------------------------------------------
# SHAP explanation display
# -----------------------------------------------------------------------
def display_explanation(input_dict: dict, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    local_path     = os.path.join(tempfile.gettempdir(), explainer_name)
    s3_key         = posixpath.join('explainer', explainer_name)

    explainer     = load_shap_explainer(session, aws_bucket, s3_key, local_path)
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Build a preprocessing-only pipeline (drop the classifier step)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])

    input_df             = pd.DataFrame([input_dict])
    input_df_transformed = preprocessing_pipeline.transform(input_df)

    # Recover feature names after selection
    dataset_features   = dataset.iloc[:, 0:]
    all_feature_names  = dataset_features.columns[1:]          # drop index col if present
    selector           = best_pipeline.named_steps['selector']
    selected_features  = all_feature_names[selector.get_support()]
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=selected_features)

    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1], show=False)   # class 1 = Fraud
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 1].values, index=shap_values[0, :, 1].feature_names)
        .abs()
        .idxmax()
    )
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# -----------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection – ML Deployment", layout="wide")
st.title("🛡️ Fraud Detection – ML Deployment")
st.markdown(
    "Adjust the key transaction features below. "
    "All other pipeline inputs are filled automatically from training data."
)

with st.form("pred_form"):
    st.subheader("Transaction Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').title(),
                min_value=float(inp['min']),
                max_value=float(inp['max']),
                value=float(inp['default']),
                step=float(inp['step']),
            )

    submitted = st.form_submit_button("Run Prediction")

# -----------------------------------------------------------------------
# Build the full row the pipeline expects:
#   start from the FIRST row of X_train (fills all 100+ columns),
#   then overwrite the handful the user just set.
# rec #5 – avoids asking the user to fill every column manually.
# -----------------------------------------------------------------------
base_row = dataset.iloc[0].to_dict()
base_row.update(user_inputs)          # user values take precedence

if submitted:
    res, status = call_model_api(base_row)

    if status == 200:
        color = "red" if "Fraud" in res else "green"
        st.markdown(f"### Prediction: <span style='color:{color}'>{res}</span>", unsafe_allow_html=True)
        st.metric("Result", res)

        with st.spinner("Generating SHAP explanation…"):
            display_explanation(base_row, session, aws_bucket)
    else:
        st.error(res)

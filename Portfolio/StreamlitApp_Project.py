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
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sklearn.pipeline import Pipeline
import shap
from joblib import load

warnings.simplefilter("ignore")

# -----------------------------------------------------------------------
# X_train.csv lives in the same Portfolio/ folder as this script
# -----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path   = os.path.join(current_dir, 'X_train.csv')
dataset     = pd.read_csv(file_path)
dataset     = dataset.drop(columns=[c for c in dataset.columns if 'Unnamed' in c])

# -----------------------------------------------------------------------
# AWS Secrets
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
def get_session(key_id, secret, token):
    return boto3.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        aws_session_token=token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# -----------------------------------------------------------------------
# Model config — top 4 SHAP features exposed in UI
# Update ranges/defaults to match your actual data after running Section 14
# -----------------------------------------------------------------------
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_sentiment.shap",
    "pipeline"  : "finalized_loan_model.tar.gz",   # matches S3 upload in notebook
    "inputs"    : [
        {"name": "int_rate",       "min": 5.0,  "max": 30.0, "default": 13.0, "step": 0.1},
        {"name": "loan_to_income", "min": 0.0,  "max": 1.0,  "default": 0.05, "step": 0.01},
        {"name": "dti",            "min": 0.0,  "max": 50.0, "default": 18.0, "step": 0.1},
        {"name": "grade_num",      "min": 1.0,  "max": 7.0,  "default": 3.0,  "step": 1.0},
    ]
}

# -----------------------------------------------------------------------
# Load pipeline from S3
# Key: sklearn-pipeline-deployment/finalized_loan_model.tar.gz
# -----------------------------------------------------------------------
@st.cache_resource
def load_pipeline(_session, bucket):
    s3      = _session.client('s3')
    fname   = MODEL_INFO["pipeline"]
    s3_key  = f"sklearn-pipeline-deployment/{fname}"
    local   = os.path.join(tempfile.gettempdir(), fname)
    s3.download_file(Bucket=bucket, Key=s3_key, Filename=local)
    with tarfile.open(local, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        jfile = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(os.path.join(tempfile.gettempdir(), jfile))

# -----------------------------------------------------------------------
# Load SHAP explainer from S3
# Key: explainer/explainer_sentiment.shap
# -----------------------------------------------------------------------
@st.cache_resource
def load_shap_explainer(_session, bucket):
    s3     = _session.client('s3')
    fname  = MODEL_INFO["explainer"]
    s3_key = f"explainer/{fname}"
    local  = os.path.join(tempfile.gettempdir(), fname)
    if not os.path.exists(local):
        s3.download_file(Bucket=bucket, Key=s3_key, Filename=local)
    with open(local, "rb") as f:
        return load(f)

# -----------------------------------------------------------------------
# Call endpoint
# -----------------------------------------------------------------------
def call_model_api(input_dict: dict):
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v)
             for k, v in input_dict.items()}
    predictor = Predictor(
        endpoint_name     = MODEL_INFO["endpoint"],
        sagemaker_session = sm_session,
        serializer        = JSONSerializer(),
        deserializer      = JSONDeserializer()
    )
    try:
        raw      = predictor.predict(clean)
        pred_val = int(np.array(raw).flat[-1])
        mapping  = {0: "✅ Fully Paid", 1: "🚨 Charged Off (Default)"}
        return mapping.get(pred_val, f"Unknown ({pred_val})"), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# -----------------------------------------------------------------------
# SHAP explanation
# -----------------------------------------------------------------------
def display_explanation(input_dict: dict):
    try:
        best_pipeline = load_pipeline(session, aws_bucket)
        explainer     = load_shap_explainer(session, aws_bucket)

        # Preprocessing only — drop SMOTE + classifier (last 2 steps)
        pre_pipeline  = Pipeline(steps=best_pipeline.steps[:-2])
        input_df      = pd.DataFrame([input_dict])
        X_transformed = pre_pipeline.transform(input_df)

        # Reconstruct feature names
        fitted_pre    = best_pipeline.named_steps['pre']
        num_features  = list(dataset.select_dtypes(include=np.number).columns)
        cat_features  = list(dataset.select_dtypes(exclude=np.number).columns)
        ohe_names     = (fitted_pre.named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names_out(cat_features).tolist())
        all_names     = num_features + ohe_names
        X_df          = pd.DataFrame(X_transformed,
                                     columns=all_names[:X_transformed.shape[1]])

        shap_vals = explainer(X_df)

        st.subheader("🔍 Decision Transparency (SHAP)")
        fig, _ = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_vals[0], show=False)
        st.pyplot(fig)

        top_feat = (pd.Series(shap_vals[0].values, index=shap_vals[0].feature_names)
                    .abs().idxmax())
        st.info(f"**Key Driver:** The most influential factor was **{top_feat}**.")

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

# -----------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("🏦 Loan Default Prediction — ML Deployment")
st.markdown("Adjust the key loan features below. All other inputs are filled automatically from training data.")

with st.form("pred_form"):
    st.subheader("Loan & Borrower Inputs")
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

base_row = dataset.iloc[0].to_dict()
base_row.update(user_inputs)

if submitted:
    with st.spinner("Calling model endpoint..."):
        res, status = call_model_api(base_row)
    if status == 200:
        color = "red" if "Default" in res or "Charged" in res else "green"
        st.markdown(f"### Prediction: <span style='color:{color}'>{res}</span>",
                    unsafe_allow_html=True)
        st.metric("Result", res)
        with st.spinner("Generating SHAP explanation..."):
            display_explanation(base_row)
    else:
        st.error(res)

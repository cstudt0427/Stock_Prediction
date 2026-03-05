import os, warnings
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
# Secrets
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
def get_sm_session(_aws_id,_aws_secret,_aws_token,_aws_region):
    session = boto3.Session(
        aws_access_key_id=_aws_id,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        region_name=_aws_region,
    )
    return sagemaker.Session(boto_session=session)

sm_session = get_sm_session(aws_id,aws_secret,aws_token,aws_region)

@st.cache_resource
def get_predictor(endpoint_name:str):
    return Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sm_session,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer()
    )

predictor = get_predictor(aws_endpoint)

# ---------------------------
# Load BTC history
# ---------------------------
DATA_PATH = "BitstampData.csv"

@st.cache_data
def load_history():
    df = pd.read_csv(DATA_PATH)
    df = df[["Close"]]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    return df

# ---------------------------
# UI
# ---------------------------
with st.form("prediction_form"):

    st.subheader("Inputs")

    close_price = st.number_input(
        "Close",
        min_value=0.0,
        value=50000.0,
        step=10.0
    )

    show_debug = st.checkbox("Show debug tables", value=True)

    submitted = st.form_submit_button("Run Prediction")

# ---------------------------
# Prediction helper
# ---------------------------
LABEL_MAP = {
    -1:"Sell",
    0:"Hold",
    1:"Buy"
}

def extract_prediction(raw):

    if isinstance(raw, dict):
        if "predictions" in raw:
            raw = raw["predictions"]

    if isinstance(raw, list):
        raw = raw[-1]

    try:
        return int(round(float(raw)))
    except:
        return None

# ---------------------------
# Run prediction
# ---------------------------
if submitted:

    try:

        history = load_history()

        # take recent window
        history = history.tail(300)

        # append user price
        new_row = pd.DataFrame({"Close":[float(close_price)]})

        input_df = pd.concat([history,new_row],ignore_index=True)

        if show_debug:
            st.write("Data sent to endpoint:")
            st.dataframe(input_df.tail(10))

        raw_pred = predictor.predict(input_df)

        pred = extract_prediction(raw_pred)

        if pred is None:
            st.error(f"Could not parse prediction: {raw_pred}")
        else:
            st.success(f"Prediction: **{LABEL_MAP.get(pred,str(pred))}**")

    except Exception as e:

        st.error(f"Endpoint invocation failed: {e}")
        st.info("Check that BitstampData.csv exists in the repo and the endpoint name is correct.")

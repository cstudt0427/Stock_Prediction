import streamlit as st
import boto3
import json

st.set_page_config(page_title="GOOGL Return Predictor", page_icon="📈", layout="centered")

st.title("📈 GOOGL Cumulative Return Predictor")
st.markdown("Enter an **NVDA stock price** and the model finds the closest matching date in SP500Data.csv, then predicts GOOGL's cumulative forward return.")

st.sidebar.header("AWS Configuration")
endpoint_name = st.sidebar.text_input("Endpoint Name", value="logistic-pipeline-endpoint-auto-71")
aws_access_key = st.sidebar.text_input("AWS Access Key ID", type="password")
aws_secret_key = st.sidebar.text_input("AWS Secret Access Key", type="password")
aws_session_token = st.sidebar.text_input("AWS Session Token (optional)", type="password")
aws_region = st.sidebar.text_input("AWS Region", value="us-east-1")

st.markdown("---")
st.subheader("1. Enter NVDA Price")

nvda_price = st.number_input(
    "NVDA Stock Price (USD)",
    min_value=1.0,
    max_value=10000.0,
    value=500.0,
    step=0.01,
    format="%.2f"
)

st.info("The endpoint receives the NVDA price, finds the closest matching date in SP500Data.csv, and runs the KernelPCA + Lasso pipeline.")

st.markdown("---")
st.subheader("2. Run Prediction")

if st.button("Invoke Endpoint"):
    if not all([aws_access_key, aws_secret_key, endpoint_name]):
        st.error("Please fill in AWS Access Key, Secret Key, and Endpoint Name in the sidebar.")
        st.stop()

    try:
        session_kwargs = {
            "aws_access_key_id": aws_access_key,
            "aws_secret_access_key": aws_secret_key,
            "region_name": aws_region
        }
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        session = boto3.Session(**session_kwargs)
        runtime = session.client("sagemaker-runtime")

        payload = json.dumps({"NVDA": nvda_price})

        with st.spinner("Calling SageMaker endpoint..."):
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=payload.encode("utf-8")
            )

        result_str = response["Body"].read().decode("utf-8").strip()

        try:
            prediction = float(result_str.split("\n")[0].split(",")[0])
        except Exception:
            prediction = result_str

        st.success("Prediction received!")
        st.metric(
            label="Predicted GOOGL Cumulative Return (5-day forward)",
            value=f"{prediction:.6f}"
        )

    except Exception as e:
        st.error(f"Error invoking endpoint: {e}")

st.markdown("---")
st.caption("HW5 - Dimensionality Reduction - Option 1 - GOOGL Return Prediction via KernelPCA + Lasso")

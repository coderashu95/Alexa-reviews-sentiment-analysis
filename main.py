import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# flask --app api.py run --port=5000
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction - Upload the file and click on Predict",
    type="csv", accept_multiple_files=False
)

# Text input for sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Prediction on single sentence
if st.button("Predict"):
    #This line checks if the user has uploaded a file.
    if uploaded_file is not None:
        file = {"file": uploaded_file}
        # This line sends the uploaded file to a server (the prediction endpoint) using an HTTP POST request. 
        # The requests.post function is used to send the file.
        response = requests.post(prediction_endpoint, files=file)

        # response_bytes reads the response content from the server into an in-memory binary stream (like a file stored in memory).
        response_bytes = BytesIO(response.content)

        # response_df reads the CSV content from this stream into a pandas DataFrame. 
        # This DataFrame now contains the predictions returned by the server.
        response_df = pd.read_csv(response_bytes)

        st.download_button(
            label="Download Predictions",
            data=response_bytes,
            file_name="Predictions_SentimentBulkAnalysisResults.csv",
            key="result_download_button",
        )

    # This else block is executed if the condition for uploading a file is not met (i.e., no file was uploaded). 
    # It handles the scenario where the user provides text input.
    else:
        response = requests.post(prediction_endpoint, data={"text": user_input})
        response = response.json()
        st.write(f"Predicted sentiment: {response['prediction']}")
import streamlit as st
import pandas as pd
import pickle

import os

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# -------------------------------------
# Load Embedding + Vector DB
# -------------------------------------
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Yes Bank stock dataset contains OHLC prices.",
    "The regression model predicts Close price using Open High Low.",
    "Higher High values usually increase predicted Close price."
]

vector_db = FAISS.from_texts(documents, embedding)
vector_db.save_local("stock_vector_db")

# -------------------------------------
# Load Model and Scaler
# -------------------------------------

with open("models/Yes_bank_stock_price_prediction_regression_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------------
# Session State for History
# -------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -------------------------------------
# Sidebar
# -------------------------------------

st.sidebar.title("Prediction History")

if st.sidebar.button("Clear History"):
    st.session_state.prediction_history = []

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_df.index = range(1, len(history_df) + 1)
    st.sidebar.dataframe(history_df, use_container_width=True)
else:
    st.sidebar.write("No predictions yet.")

# -------------------------------------
# Streamlit UI
# -------------------------------------

st.title("Yes Bank Stock Closing Price Predictor")

st.write("Enter stock details below:")

# -------------------------------------
# User Inputs
# -------------------------------------

open_price = st.number_input("Open Price", min_value=0.0)
high_price = st.number_input("High Price", min_value=0.0)
low_price = st.number_input("Low Price", min_value=0.0)

# -------------------------------------
# Prediction
# -------------------------------------

if st.button("Predict"):

    unseen_data = pd.DataFrame({
        "Open": [open_price],
        "High": [high_price],
        "Low": [low_price]
    })

    unseen_data_scaled = scaler.transform(unseen_data)

    prediction = loaded_model.predict(unseen_data_scaled)[0]

    # Save prediction to history
    st.session_state.prediction_history.append({
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Predicted Close": round(float(prediction), 2)
    })

    # Show Result
    st.subheader("Prediction Result")
    st.success(f"Predicted Closing Price: ₹ {prediction:.2f}")
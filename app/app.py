from email.mime import message

import joblib
import streamlit as st
import pandas as pd
import os
import plotly.express as px

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

if not os.path.exists("stock_vector_db"):
    vector_db = FAISS.from_texts(documents, embedding)
    vector_db.save_local("stock_vector_db")
else:
    vector_db = FAISS.load_local(
        "stock_vector_db",
        embedding,
        allow_dangerous_deserialization=True
    )

# -------------------------------------
# AI Response Function
# -------------------------------------

def get_ai_response(user_input, prediction=None, features=None):

    docs = vector_db.similarity_search(user_input, k=2)
    context = " ".join([doc.page_content for doc in docs])

    model_context = f"""
    Prediction: {prediction}
    Features: {features}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a financial AI assistant."},
                {"role": "user", "content": context + "\n" + model_context + "\n" + user_input}
            ]
        )

    except Exception as e:
        # fallback model
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a financial AI assistant."},
                {"role": "user", "content": context + "\n" + model_context + "\n" + user_input}
            ]
        )

    return completion.choices[0].message.content
# -------------------------------------
# Load Model and Scaler (Cached)
# -------------------------------------

@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(BASE_DIR, "models", "Yes_bank_stock_price_prediction_regression_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


loaded_model, scaler = load_artifacts()

# -------------------------------------
# Load Dataset for Visualization
# -------------------------------------

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "data_YesBank_StockPrices.csv")

    data = pd.read_csv(data_path)

    data["Date"] = pd.to_datetime(data["Date"], format="%b-%y", errors="coerce")

    data = data.dropna(subset=["Date"])
    data = data.sort_values("Date")

    return data

data = load_data()

# -------------------------------------
# Session State for History
# -------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -------------------------------------
# Sidebar
# -------------------------------------

st.sidebar.title("Prediction History")

if st.sidebar.button("Clear History", key="clear_history_btn"):
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
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Prediction", "🤖 AI Copilot"])

with tab1:
    # your existing chart code
    # your KPI code
    

# -------------------------------------
# Historical Price Chart
# -------------------------------------

    st.subheader("📈 Historical Closing Price Trend")

# Convert pandas timestamps to python date
    min_date = data["Date"].min().date()
    max_date = data["Date"].max().date()

# Slider for date range
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

# Filter data based on slider
    filtered_data = data[(data["Date"] >= pd.to_datetime(start_date)) & 
                        (data["Date"] <= pd.to_datetime(end_date))]
    # ✅ Store globally
    st.session_state["filtered_data"] = filtered_data
# -------------------------------------
# KPI Cards (Dynamic based on filter)
# -------------------------------------

    st.subheader("📊 Key Stock Insights")

    col1, col2, col3, col4 = st.columns(4)

    max_price = filtered_data["Close"].max()
    min_price = filtered_data["Close"].min()
    avg_price = filtered_data["Close"].mean()
    total_days = filtered_data.shape[0]

    col1.metric("📈 Highest Price", f"₹ {max_price:.2f}")
    col2.metric("📉 Lowest Price", f"₹ {min_price:.2f}")
    col3.metric("📊 Average Price", f"₹ {avg_price:.2f}")
    col4.metric("🗓️ Data Points", total_days)

# Plot filtered data
    fig = px.line(
        filtered_data,
        x="Date",
        y="Close",
        title="Yes Bank Historical Closing Prices",
        labels={
            "Date": "Date",
            "Close": "Closing Price (₹)"
        }
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Closing Price (₹)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


with tab2:
    # inputs
    # prediction logic
# -------------------------------------
# User Inputs
# -------------------------------------
    st.write("NOTE: Stock prices below are suggested based on the history of Stock Prices for Yes Bank. " \
            "Enter manual stock details below:")
    # Get filtered data
    filtered_data = st.session_state.get("filtered_data", data)

# Compute dynamic values
    min_open = float(filtered_data["Open"].min())
    max_open = float(filtered_data["Open"].max())

    min_high = float(filtered_data["High"].min())
    max_high = float(filtered_data["High"].max())

    min_low = float(filtered_data["Low"].min())
    max_low = float(filtered_data["Low"].max())

# Inputs with dynamic defaults
    open_price = st.number_input(
        "Open Price:",
        min_value=min_open,
        max_value=max_open,
        value=min_open
    )

    high_price = st.number_input(
        "High Price:",
        min_value=min_high,
        max_value=max_high,
        value=max_high
    )

    low_price = st.number_input(
        "Low Price:",
        min_value=min_low,
        max_value=max_low,
        value=min_low
    )

# -------------------------------------
# Prediction
# -------------------------------------

    if st.button("Predict", key="predict_button_1"):

        unseen_data = pd.DataFrame({
                "Open": [open_price],
                "High": [high_price],
                "Low": [low_price]
        })

        unseen_data_scaled = scaler.transform(unseen_data)

        prediction = loaded_model.predict(unseen_data_scaled)[0]
        if st.button("Save Prediction", key="predict_button_2"):
        # ✅ Store for AI usage
            st.session_state["latest_prediction"] = prediction
            st.session_state["latest_features"] = {
                "Open": open_price,
                "High": high_price,
                "Low": low_price
            }

        # ✅ Save prediction to history
        st.session_state.prediction_history.append({
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Predicted Close": round(float(prediction), 2)
        })

    # ✅ Show Result
        st.subheader("Prediction Result")
        st.success(f"Predicted Closing Price: ₹ {prediction:.2f}")

        if "latest_prediction" in st.session_state:

            insight = get_ai_response(
                "Explain this prediction",
                st.session_state["latest_prediction"],
                st.session_state["latest_features"]
            )

            st.info(f"💡 AI Insight:\n\n{insight}")


with tab3:

    st.subheader("🤖 AI Copilot")

    if st.session_state.prediction_history:

        history_df = pd.DataFrame(st.session_state.prediction_history)

        history_df["label"] = history_df.apply(
            lambda row: f"Open: {row['Open']} | High: {row['High']} | Low: {row['Low']} → Close: {row['Predicted Close']}",
            axis=1
        )

        selected_index = st.selectbox(
            "Select a prediction to analyze:",
            options=history_df.index,
            format_func=lambda x: history_df.loc[x, "label"]
        )

        selected_row = history_df.loc[selected_index]

        prediction = selected_row["Predicted Close"]
        features = {
            "Open": selected_row["Open"],
            "High": selected_row["High"],
            "Low": selected_row["Low"]
        }

        # ✅ SAFE ID (exists here)
        prediction_id = str(selected_index)

        st.success(f"Selected Prediction: ₹ {prediction:.2f}")

    else:
        st.info("No prediction available yet.")

        prediction = "N/A"
        features = "N/A"

        # ✅ DEFAULT ID (VERY IMPORTANT)
        prediction_id = "default"

    # -------------------------------------
    # Chat Memory Per Prediction
    # -------------------------------------

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    if prediction_id not in st.session_state.chat_history:
        st.session_state.chat_history[prediction_id] = []

    user_input = st.chat_input("Ask anything about stock or prediction...")

    if user_input:

        response = get_ai_response(user_input, prediction, features)

        st.session_state.chat_history[prediction_id].append(("user", user_input))
        st.session_state.chat_history[prediction_id].append(("ai", response))

    # ✅ Render only selected prediction chat
    for role, msg in st.session_state.chat_history[prediction_id]:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(msg)
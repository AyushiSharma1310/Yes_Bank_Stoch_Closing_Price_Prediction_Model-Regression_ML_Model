import joblib
import streamlit as st
import pandas as pd
import os
import plotly.express as px

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------------------
# 🎨 CSS (FIXED)
# -------------------------------------

st.markdown(
    """
<style>

/* ================================
   📦 MAIN CHAT WRAPPER (TAB 3 ONLY)
================================ */
.chat-wrapper {
    position: relative;
    height: 100%;
    padding-bottom: 80px;  /* space for input */
}

/* ================================
   💬 CHAT INPUT FIXED AT BOTTOM
================================ */
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    left: calc(var(--sidebar-width, 0px) + 1rem);
    right: 1rem;
    z-index: 999;
    background: #0e1117;
    margin-left: 400px !important;
    padding: 10px 0;
}

/* ================================
   📏 HANDLE SIDEBAR COLLAPSE
================================ */

/* When sidebar collapsed */
section[data-testid="stSidebar"][aria-expanded="false"] ~ div [data-testid="stChatInput"] {
    left: 1rem !important;
}

/* ================================
   📱 RIGHT COLUMN ADJUSTMENT (IMPORTANT)
================================ */

/* If you are using st.columns */
div[data-testid="column"] [data-testid="stChatInput"] {
    left: auto !important;
    right: 1rem !important;
}

/* ================================
   🧠 CHAT MESSAGE AREA SCROLLABLE
================================ */
.chat-wrapper > div:first-child {
    max-height: calc(100vh - 120px);
    overflow-y: auto;
    padding-bottom: 20px;
}

</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------
# 🔐 SESSION STATE INIT
# -------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# -------------------------------------
# 🤖 GROQ + VECTOR DB
# -------------------------------------

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Yes Bank stock dataset contains OHLC prices.",
    "The regression model predicts Close price using Open High Low.",
    "Higher High values usually increase predicted Close price.",
]


if not os.path.exists("stock_vector_db"):
    vector_db = FAISS.from_texts(documents, embedding)
    vector_db.save_local("stock_vector_db")
else:
    vector_db = FAISS.load_local(
        "stock_vector_db", embedding, allow_dangerous_deserialization=True
    )

# -------------------------------------
# 🤖 AI FUNCTION
# -------------------------------------


def get_ai_response(user_input, prediction=None, features=None, chat_history=None):

    # 🔍 RAG context
    docs = vector_db.similarity_search(user_input, k=2)
    rag_context = " ".join([doc.page_content for doc in docs])

    # 🧠 Chat history (last 4 messages only)
    history_context = ""
    if chat_history:
        for role, msg in chat_history[-4:]:
            history_context += f"{role.upper()}: {msg}\n"

    # 🎯 Strong structured prompt
    final_prompt = f"""
You are a Financial AI Assistant.

STRICT RULES:
- Always use the given prediction and input features
- Do NOT give generic stock advice
- Explain reasoning clearly using numbers
- Keep answers concise but insightful

----------------------------------------
📊 Prediction Context:
Predicted Close Price: {prediction}

Input Features:
{features}

----------------------------------------
📚 Knowledge Base:
{rag_context}

----------------------------------------
💬 Conversation History:
{history_context}

----------------------------------------
👤 User Question:
{user_input}

----------------------------------------
💡 Answer:
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3,
            max_tokens=500,
        )

    except Exception:
        # 🔁 Fallback model
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3,
            max_tokens=500,
        )

    return completion.choices[0].message.content


# -------------------------------------
# 📊 LOAD MODEL + DATA
# -------------------------------------


@st.cache_resource(show_spinner=False)
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(
        os.path.join(
            BASE_DIR, "models", "Yes_bank_stock_price_prediction_regression_model.pkl"
        )
    )
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    return model, scaler


loader = st.empty()

with loader.container():
    st.markdown("## 🤖 Initializing AI Model...")
    st.markdown("Please wait while we load everything 🚀")
    st.image(
        "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXV4cjkwMnJudHpja3BreXVzYzU1amQyMmh3aHU1ZzZxdndmaWpiYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YnkMcHgNIMW4Yfmjxr/giphy.gif"
    )

    loaded_model, scaler = load_artifacts()

loader.empty()


@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(os.path.join(BASE_DIR, "data", "data_YesBank_StockPrices.csv"))
    data["Date"] = pd.to_datetime(data["Date"], format="%b-%y", errors="coerce")
    return data.sort_values("Date").dropna(subset=["Date"])


data = load_data()

# -------------------------------------
# 📜 SIDEBAR (Prediction History ONLY)
# -------------------------------------

st.sidebar.title("📜 Prediction History")

# ✅ Ensure session state exists
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ✅ Clear ONLY prediction history
if st.sidebar.button("Clear History", key="clear_history_btn"):
    st.session_state.prediction_history = []
    st.rerun()

# -------------------------------------
# 📊 Show Prediction History
# -------------------------------------

if st.session_state.prediction_history:

    df = pd.DataFrame(st.session_state.prediction_history)

    # ✅ Add user-friendly labels
    df.insert(0, "ID", [f"Prediction-{i+1}" for i in range(len(df))])

    st.sidebar.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.sidebar.write("No predictions yet.")
# -------------------------------------
# 🖥️ MAIN UI
# -------------------------------------
if "show_right_sidebar" not in st.session_state:
    st.session_state.show_right_sidebar = False


st.title("Yes-Bank Closing Price Predictor")
tab1, tab2, tab3 = st.tabs(
    ["📊 Dashboard", "🔮 Prediction", "🤖 Prediction AI Support"]
)

# -------------------------------------
# 📊 TAB 1
# -------------------------------------

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
        value=(min_date, max_date),
    )

    # Filter data based on slider
    filtered_data = data[
        (data["Date"] >= pd.to_datetime(start_date))
        & (data["Date"] <= pd.to_datetime(end_date))
    ]
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
        labels={"Date": "Date", "Close": "Closing Price (₹)"},
    )

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Closing Price (₹)", hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------
# 🔮 TAB 2
# -------------------------------------

with tab2:
    # -------------------------------------
    # User Inputs
    # -------------------------------------
    st.write(
        "NOTE: Stock prices below are suggested based on the history of Stock Prices for Yes Bank. "
        "Enter manual stock details below:"
    )

    filtered_data = st.session_state.get("filtered_data", data)

    # Dynamic ranges
    min_open = float(filtered_data["Open"].min())
    max_open = float(filtered_data["Open"].max())

    min_high = float(filtered_data["High"].min())
    max_high = float(filtered_data["High"].max())

    min_low = float(filtered_data["Low"].min())
    max_low = float(filtered_data["Low"].max())

    # Inputs
    open_price = st.number_input(
        "Open Price:", min_value=min_open, max_value=max_open, value=min_open
    )

    high_price = st.number_input(
        "High Price:", min_value=min_high, max_value=max_high, value=max_high
    )

    low_price = st.number_input(
        "Low Price:", min_value=min_low, max_value=max_low, value=min_low
    )

    # -------------------------------------
    # 🔮 Prediction Button
    # -------------------------------------
    if st.button("Predict", key="predict_button_1"):

        unseen_data = pd.DataFrame(
            {"Open": [open_price], "High": [high_price], "Low": [low_price]}
        )

        unseen_data_scaled = scaler.transform(unseen_data)
        prediction = loaded_model.predict(unseen_data_scaled)[0]

        # ✅ Store for persistence
        st.session_state["latest_prediction"] = prediction
        st.session_state["latest_features"] = {
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
        }

        # ✅ Save history
        st.session_state.prediction_history.append(
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Predicted Close": round(float(prediction), 2),
            }
        )

    # -------------------------------------
    # ✅ SHOW RESULT (PERSISTENT)
    # -------------------------------------
    if "latest_prediction" in st.session_state:

        prediction = st.session_state["latest_prediction"]

        st.subheader("Prediction Result")
        st.success(f"Predicted Closing Price: ₹ {prediction:.2f}")

        # 🤖 AI Insight
        insight = get_ai_response(
            "Explain this prediction",
            prediction,
            st.session_state["latest_features"],
        )

        st.info(f"💡 AI Insight:\n\n{insight}")

        # -------------------------------------
        # 📊 GRAPH (SAFE NOW ✅)
        # -------------------------------------
        st.subheader("📊 Prediction vs Historical Data")
        st.warning(
            "⚠️ This prediction is based on historical data upto November 2020. "
            "It may not reflect real-time market conditions or external factors."
        )
        fig = px.line(
            filtered_data,
            x="Date",
            y="Close",
            title="Historical Close Price vs Prediction",
        )

        latest_date = pd.Timestamp.now().date() + pd.Timedelta(days=1)

        fig.add_scatter(
            x=[latest_date],
            y=[prediction],
            mode="markers",
            name="Prediction",
            marker=dict(size=12, symbol="diamond"),
        )

        fig.add_hline(
            y=prediction,
            line_dash="dash",
            annotation_text="Predicted Price",
            annotation_position="top right",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Make a prediction to see results and graph")
# -------------------------------------
# 🤖 TAB 3
# -------------------------------------

with tab3:

    # 🔘 Toggle Right Panel
    if st.button("📂 Chat History", key="toggle_sidebar_btn"):
        st.session_state.show_right_sidebar = not st.session_state.show_right_sidebar
        st.rerun()
    if not st.session_state.show_right_sidebar:
        st.info("Click above to open chat history")
    # Layout
    if st.session_state.show_right_sidebar:
        col1, col2 = st.columns([2.5, 1.5])
    else:
        col1, col2 = st.columns([1, 0.001])

    # =====================================
    # LEFT → MAIN CHAT
    # =====================================
    with col1:

        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

        # ---------------------------
        # 📊 Prediction Selector
        # ---------------------------
        if st.session_state.prediction_history:

            df = pd.DataFrame(st.session_state.prediction_history)
            labels = [f"Prediction-{i+1}" for i in range(len(df))]

            selected = st.selectbox("Select Prediction", labels)
            idx = labels.index(selected)

            row = df.iloc[idx]

            prediction = row["Predicted Close"]
            features = {
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
            }

            prediction_id = f"prediction_{idx}"
            st.session_state["active_prediction_id"] = prediction_id

            st.success(f"{selected}: ₹ {prediction:.2f}")

        else:
            st.info("No predictions yet")
            prediction, features, prediction_id = "N/A", {}, "default"

        # ---------------------------
        # 🧠 Thread Management
        # ---------------------------
        if prediction_id not in st.session_state.chat_history:
            st.session_state.chat_history[prediction_id] = {"threads": []}

        threads = st.session_state.chat_history[prediction_id]["threads"]

        if not threads:
            threads.append({"messages": []})

        current_thread = threads[-1]

        # ---------------------------
        # 💬 CHAT MESSAGES (SCROLLABLE)
        # ---------------------------
        st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

        for role, msg in current_thread["messages"]:
            with st.chat_message(role):
                st.write(msg)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================
    # RIGHT → CHAT HISTORY
    # =====================================
    with col2:

        if st.session_state.show_right_sidebar:

            st.markdown("### 💬 Chat History")

            threads = st.session_state.chat_history[prediction_id]["threads"]

            if st.button("➕ New Chat", key=f"new_chat_{prediction_id}"):
                threads.append({"messages": []})
                st.rerun()

            if threads:
                for i, thread in enumerate(reversed(threads)):

                    first_question = (
                        thread["messages"][0][1]
                        if thread["messages"]
                        else "No chat here yet"
                    )

                    if st.button(
                        f"🧵 {first_question[:30]}...",
                        key=f"thread_{prediction_id}_{i}",
                    ):

                        selected_thread = threads[-(i + 1)]
                        threads.remove(selected_thread)
                        threads.append(selected_thread)

                        st.rerun()
            else:
                st.write("No chat history yet.")

    # =====================================
    # 💬 CHAT INPUT (OUTSIDE COLUMNS)
    # =====================================
    user_input = st.chat_input("Ask about prediction...")

    if user_input:

        current_thread = st.session_state.chat_history[
            st.session_state["active_prediction_id"]
        ]["threads"][-1]

        response = get_ai_response(
            user_input,
            prediction,
            features,
            current_thread["messages"],
        )

        current_thread["messages"].append(("user", user_input))
        current_thread["messages"].append(("assistant", response))

        st.rerun()

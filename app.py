import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from fpdf import FPDF

# ✅ FIXED LangChain imports
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# =========================
#  STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(
    page_title="📊 AI-Powered BI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
#  SIDEBAR HEADER & TRACKER
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()

st.sidebar.title("📊 BI AI Assistant")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 How to Use")
st.sidebar.markdown("""
1. **Upload your CSV file** containing sales, profit, or KPIs.  
2. **Ask questions in plain English** — e.g., *"Show me sales by region"*  
3. **View instant answers** with tables or charts.  
4. **Export results to PDF** if needed.  
""")
st.sidebar.markdown("---")

# Tracker
elapsed_time = datetime.now() - st.session_state.start_time
st.sidebar.metric("⏱ Time Active", str(elapsed_time).split(".")[0])
st.sidebar.metric("💬 Total Queries", st.session_state.query_count)

st.sidebar.markdown("---")
st.sidebar.info("💡 This assistant specializes in **Business Intelligence** but can also answer general questions.")

# =========================
#  API KEY CHECK
# =========================
if "API_KEY" not in st.secrets:
    st.error("❌ API Key not found. Please add it in Streamlit secrets as `API_KEY`.")
    st.stop()

api_key = st.secrets["API_KEY"]

# =========================
#  MAIN HEADER
# =========================
st.markdown("<h1 style='text-align: center;'>📊 AI-Powered Business Intelligence Assistant</h1>", unsafe_allow_html=True)
st.write("Ask anything about your data — from **sales trends** to **profit forecasts** — or have a general chat.")

# =========================
#  FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload a CSV file for BI analysis", type=["csv"])

df = None
agent = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
    st.dataframe(df.head())

    # =========================
    #  LLM + AGENT SETUP
    # =========================
    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama3-8b-8192",
        temperature=0
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True  # ✅ Required to avoid ValueError
    )

# =========================
#  CONVERSATIONAL INPUT
# =========================
query = st.text_input("💬 Ask a question about your data or chat:")

if query:
    st.session_state.query_count += 1
    st.session_state.chat_history.append(("You", query))

    if agent and df is not None:
        try:
            # If BI-related question
            bi_keywords = ["sales", "profit", "growth", "revenue", "region", "trend", "forecast", "kpi"]
            if any(kw in query.lower() for kw in bi_keywords):
                response = agent.run(query)
                st.session_state.chat_history.append(("AI (BI Mode)", response))

                st.success(response)

                # Optional: Auto chart if keyword detected
                if "sales" in query.lower() and "region" in df.columns:
                    fig, ax = plt.subplots()
                    df.groupby("region")["sales"].sum().plot(kind="bar", ax=ax)
                    ax.set_title("Sales by Region")
                    st.pyplot(fig)

            else:
                # General chat mode
                st.session_state.chat_history.append(("AI (General)", f"I understand your question: {query}. However, I specialize in BI. Could you provide data or context?"))
                st.info("ℹ️ This assistant is optimized for BI queries. Please upload data for full insights.")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    else:
        st.info("📌 Please upload a CSV file for BI insights.")

# =========================
#  CHAT HISTORY
# =========================
st.markdown("## 🗨️ Conversation History")
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")

# =========================
#  FOOTER
# =========================
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>© 2025 BI AI Assistant | Designed for smarter business insights</div>", unsafe_allow_html=True)

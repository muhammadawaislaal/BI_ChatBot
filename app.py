import os
import streamlit as st
import pandas as pd
import requests
from langchain_experimental.agents import create_pandas_dataframe_agent

# ===== Settings =====
GROQ_DAILY_LIMIT = 100000  # Free tier daily token limit
WARNING_THRESHOLD = 0.9    # 90% usage triggers warning

# ===== Helpers =====
def get_groq_usage(api_key):
    """Fetch today's token usage from Groq API."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get("https://api.groq.com/v1/usage", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            used = data.get("total_tokens_today", 0)
            return used
    except Exception as e:
        st.warning(f"Could not fetch Groq usage: {e}")
    return None

def get_groq_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model_name="gemma2-9b-it",
        temperature=0,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

def get_local_llm():
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="llama3")

# ===== Streamlit UI =====
st.set_page_config(page_title="BI Chatbot (Groq + Tracker + Fallback)", layout="wide")
st.title("📊 AI-Powered BI Assistant")

if "GROQ_API_KEY" not in st.secrets:
    st.error("Please set GROQ_API_KEY in your Streamlit secrets.")
else:
    usage = get_groq_usage(st.secrets["GROQ_API_KEY"])
    if usage is not None:
        st.info(f"Groq tokens used today: **{usage:,} / {GROQ_DAILY_LIMIT:,}**")
        if usage > GROQ_DAILY_LIMIT:
            st.error("🚫 Groq daily quota exceeded! Switching to local model.")
        elif usage > WARNING_THRESHOLD * GROQ_DAILY_LIMIT:
            st.warning("⚠️ You're close to hitting your daily Groq limit.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    query = st.text_input("Ask me about your data:")

    if query:
        try:
            # Decide which LLM to use
            use_local = False
            if "GROQ_API_KEY" in st.secrets:
                usage = get_groq_usage(st.secrets["GROQ_API_KEY"])
                if usage is not None and usage >= GROQ_DAILY_LIMIT:
                    use_local = True
            else:
                use_local = True

            if not use_local:
                llm = get_groq_llm()
            else:
                st.warning("Using local Ollama model instead of Groq.")
                llm = get_local_llm()

            # Create and run agent
            agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, allow_dangerous_code=True
            )
            response = agent.run(query)

        except Exception as e:
            st.warning(f"⚠️ Error with Groq: {e} — switching to local model.")
            llm = get_local_llm()
            agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, allow_dangerous_code=True
            )
            response = agent.run(query)

        st.subheader("Answer")
        st.write(response)

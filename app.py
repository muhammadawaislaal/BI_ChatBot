import os
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent

# Try Groq first
def get_groq_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model_name="gemma2-9b-it",  # smaller, higher free quota
        temperature=0,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

# Fallback: Local Ollama model
def get_local_llm():
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="llama3")

st.set_page_config(page_title="BI Chatbot (Groq + Fallback)", layout="wide")
st.title("📊 AI-Powered BI Assistant")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    query = st.text_input("Ask me about your data:")

    if query:
        try:
            llm = get_groq_llm()
            agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, allow_dangerous_code=True
            )
            response = agent.run(query)

        except Exception as e:
            st.warning(f"⚠️ Groq failed: {e}\nSwitching to local model...")
            llm = get_local_llm()
            agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, allow_dangerous_code=True
            )
            response = agent.run(query)

        st.subheader("Answer")
        st.write(response)

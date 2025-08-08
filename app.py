import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load Groq API key from Streamlit secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="BI Chatbot (Groq)", layout="wide")
st.title("📊 AI-Powered BI Assistant (via Groq)")

uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="mixtral-8x7b-32768",
        temperature=0,
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
    )

    query = st.text_input("Ask me about your data:")
    if query:
        with st.spinner("Thinking..."):
            try:
                response = agent.run(query)
                st.success(response)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV file to get started.")

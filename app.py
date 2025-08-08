import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os

# --- Streamlit UI ---
st.set_page_config(page_title="📊 AI-Powered BI Assistant", layout="wide")
st.title("📊 AI-Powered BI Assistant")
st.write("Upload a CSV file and ask questions about your data!")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Initialize LLM (OpenAI GPT)
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",  # You can change to gpt-4-turbo or gpt-3.5-turbo
        api_key=os.environ.get("OPENAI_API_KEY")  # Pulls from Streamlit secrets
    )

    # Create LangChain Pandas Agent with dangerous code enabled
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

    # User query input
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

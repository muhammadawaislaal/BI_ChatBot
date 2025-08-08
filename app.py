import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# ===== Load API key from Streamlit secrets =====
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ===== Streamlit page config =====
st.set_page_config(page_title="BI Chatbot", layout="wide")
st.title("📊 AI-Powered Business Intelligence Chatbot")

# ===== Session state for chat history =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===== File uploader =====
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file:
    # Load CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")
    st.dataframe(df.head())

    # Create LangChain Pandas Agent
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # User query input
    query = st.text_input("Ask me about your data:")

    if query:
        with st.spinner("Analyzing..."):
            answer = agent.run(query)

        # Save to history
        st.session_state.chat_history.append({"user": query, "bot": answer})

        # Display chat history
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")

        # Auto-generate chart if possible
        try:
            fig, ax = plt.subplots()
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) >= 2:
                df.plot(x=numeric_cols[0], y=numeric_cols[1], kind='line', ax=ax)
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)
        except Exception as e:
            st.warning(f"No chart generated: {e}")
else:
    st.info("Please upload a CSV file to get started.")

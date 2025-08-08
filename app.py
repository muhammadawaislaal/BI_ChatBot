# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent

# --- Streamlit UI ---
st.set_page_config(page_title="BI Chatbot", layout="wide")
st.title("📊 AI-Powered BI Assistant")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")
    st.dataframe(df.head())

    # Initialize LangChain Pandas Agent
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # Ask a question
    query = st.text_input("Ask me about your data:")

    if query:
        with st.spinner("Thinking..."):
            answer = agent.run(query)
            st.write("**Answer:**", answer)

            # Attempt to auto-generate a chart
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

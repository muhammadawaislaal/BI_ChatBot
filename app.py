import subprocess
import sys

# Ensure "tabulate" is installed for pandas.to_markdown()
try:
    import tabulate
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
    import tabulate

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# =========================
#  CONFIGURATION & THEME
# =========================
st.set_page_config(
    page_title="📊 BI Conversational AI",
    page_icon="💬",
    layout="wide",
)

# =========================
#  API KEY SETUP
# =========================
API_KEY = (
    st.secrets.get("GROQ_API_KEY")
    or st.secrets.get("API_KEY")
    or st.secrets.get("api_key")
)

if not API_KEY:
    st.error("❌ API Key not found. Please add `GROQ_API_KEY` in your Streamlit secrets.")
    st.stop()

# =========================
#  HEADER
# =========================
st.markdown(
    """
    <h1 style="text-align:center; color:#4B9CD3;">
        💬 Business Intelligence AI Assistant
    </h1>
    <p style="text-align:center; font-size:16px;">
        Ask anything about your data — get instant answers, insights, and visuals.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# =========================
#  SIDEBAR - INFO & TRACKER
# =========================
st.sidebar.header("📌 Instructions")
st.sidebar.markdown(
    """
    1. **Upload** your CSV file.  
    2. **Ask questions** in plain English.  
    3. Get **instant BI insights** + charts.  
    4. **Export** results as PDF.  
    """
)

st.sidebar.header("📈 Usage Tracker")
if "queries_count" not in st.session_state:
    st.session_state.queries_count = 0
st.sidebar.metric("Questions Asked", st.session_state.queries_count)

st.sidebar.header("💡 BI Tips")
st.sidebar.info(
    """
    - Ask about **trends** over time.  
    - Compare **categories**.  
    - Request **charts** or **summaries**.  
    - Example:  
      *"Show me monthly sales trends for 2024"*  
    """
)

# =========================
#  FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())

    # =========================
    #  LLM SETUP
    # =========================
    llm = ChatGroq(
        groq_api_key=API_KEY,
        model="llama3-8b-8192",
        temperature=0
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True
    )

    # =========================
    #  USER INPUT
    # =========================
    user_query = st.text_input("💬 Ask a BI question or any data-related query:")

    if user_query:
        st.session_state.queries_count += 1

        with st.spinner("🤔 Thinking..."):
            try:
                response = agent.run(user_query)
                st.markdown(f"### 🧠 Answer:\n{response}")

                if any(word in user_query.lower() for word in ["trend", "chart", "plot", "graph"]):
                    st.markdown("#### 📊 Auto-generated Chart")
                    plt.figure(figsize=(8, 4))
                    try:
                        num_cols = df.select_dtypes(include='number').columns
                        if len(num_cols) >= 1:
                            df[num_cols[0]].plot()
                            plt.title(f"{num_cols[0]} Trend")
                            plt.xlabel("Index")
                            plt.ylabel(num_cols[0])
                            st.pyplot(plt)
                    except Exception as e:
                        st.warning(f"Could not auto-generate chart: {e}")

            except Exception as e:
                st.error(f"Error: {e}")

    # =========================
    #  EXPORT TO PDF
    # =========================
    def export_pdf(text):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.cell(200, 10, txt=line, ln=True)
        file_name = f"BI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(file_name)
        return file_name

    if st.button("📄 Export Last Answer to PDF"):
        if user_query:
            pdf_path = export_pdf(f"Question: {user_query}\n\nAnswer:\n{response}")
            st.success(f"✅ Exported to {pdf_path}")
        else:
            st.warning("Please ask a question first.")

else:
    st.info("⬆️ Please upload a CSV file to start.")

# =========================
#  FOOTER
# =========================
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray; font-size:13px;">
        © 2025 BI Conversational AI | Built for Business Insights & Data Storytelling
    </p>
    """,
    unsafe_allow_html=True,
)

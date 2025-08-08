import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from fpdf import FPDF

# LangChain imports
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# =========================
# PAGE CONFIG & HEADER
# =========================
st.set_page_config(
    page_title="BI Pro AI Assistant",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER & FOOTER
# =========================
st.markdown("<h1 style='text-align:center; color:#2c3e50;'>📊 BI Pro AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Your conversational BI partner — Data meets insights</p>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📌 Instructions")
st.sidebar.markdown("""
1. **Upload** your CSV file.  
2. **Ask anything** — casual or BI-focused.  
3. For BI: Use natural language (e.g., *Show monthly sales trends*).  
4. AI will respond conversationally and provide charts/tables.  
5. Export insights as PDF anytime.
""")

st.sidebar.header("📈 Tracker")
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
st.sidebar.metric("Queries Made", st.session_state.query_count)

# =========================
# API KEY
# =========================
API_KEY = st.secrets.get("API_KEY", None)
if not API_KEY:
    st.error("❌ API key missing! Add it to `secrets.toml` as `API_KEY = 'your_key_here'`")
    st.stop()

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV Data", type=["csv"])
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Data Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

# =========================
# AI MODEL
# =========================
llm = ChatGroq(
    groq_api_key=API_KEY,
    model="llama-3.1-70b-versatile",
    temperature=0
)

agent = None
if df is not None:
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True
    )

# =========================
# CHAT INPUT
# =========================
st.markdown("### 💬 Ask me anything (BI questions get special attention)")
user_query = st.text_input("Type your question...")

if user_query:
    st.session_state.query_count += 1
    try:
        if agent and df is not None:
            # Run as BI query
            with st.spinner("🔍 Analyzing..."):
                response = agent.run(user_query)
            st.write(response)
        else:
            # Run as general conversation
            with st.spinner("💡 Thinking..."):
                from langchain.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_template(
                    "You are a friendly and knowledgeable AI assistant. "
                    "Answer the following question in a conversational and clear way:\n\n{q}"
                )
                final_prompt = prompt.format(q=user_query)
                response = llm.invoke(final_prompt)
            st.write(response.content)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# =========================
# EXPORT TO PDF
# =========================
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf_path = f"BI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_path)
    return pdf_path

if st.button("📄 Export Last Response as PDF"):
    if "response" in locals():
        file_path = export_pdf(str(response))
        with open(file_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_path)
    else:
        st.warning("No response to export yet!")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 BI Pro AI — Turning data into decisions</p>",
    unsafe_allow_html=True
)

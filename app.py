import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

# LangChain + Groq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Business Intelligence Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
st.sidebar.title("📊 AI BI Assistant")
st.sidebar.markdown("Small, polished demo running on **Groq** (free tier)")

# API Key check
if "GROQ_API_KEY" not in st.secrets:
    st.sidebar.error("❌ No Groq API key found in `secrets.toml`")
    st.stop()
else:
    st.sidebar.success("✅ Groq API key loaded")

# Model selector
model = st.sidebar.selectbox(
    "Select Model",
    ["gemma2-9b-it", "llama3-70b-8192", "mixtral-8x7b-32768"]
)

st.sidebar.markdown("### 📌 Instructions")
st.sidebar.markdown("""
1. **Upload CSV** file.  
2. Ask questions in plain English.  
3. View answers + charts instantly.  
4. Export results if needed.
""")

# --- MAIN TITLE ---
st.title("📈 AI Business Intelligence Assistant")
st.caption("Upload a CSV → Ask in natural language → Get answers & charts instantly.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader(
    "📤 Upload CSV file",
    type=["csv"],
    help="Upload your data file (max 200MB)."
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # --- AGENT ---
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model=model,
        temperature=0
    )
    agent = create_pandas_dataframe_agent(llm, df, verbose=False)

    # --- QUERY INPUT ---
    st.markdown("### 💬 Ask a question about your data")
    query = st.text_input("Example: 'Show total sales by Region'")

    if query:
        with st.spinner("🤖 Thinking..."):
            try:
                response = agent.invoke(query)
                st.markdown(f"**Answer:** {response['output']}")
            except Exception as e:
                st.error(f"Error: {e}")

        # --- OPTIONAL: Auto plot if request contains keywords ---
        if any(word in query.lower() for word in ["plot", "chart", "graph", "show", "visualize"]):
            try:
                fig, ax = plt.subplots()
                df.plot(ax=ax)
                st.pyplot(fig)
            except Exception:
                st.warning("No valid columns to plot automatically.")

    # --- EXPORT OPTIONS ---
    def export_pdf(text):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        return pdf.output(dest="S").encode("latin1")

    st.markdown("### 📤 Export")
    if query and 'output' in locals():
        pdf_data = export_pdf(response['output'])
        st.download_button(
            "Download Answer as PDF",
            data=pdf_data,
            file_name="ai_bi_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("👆 Please upload a CSV file to begin.")

# --- FOOTER ---
st.markdown("---")
st.caption("🚀 Built for portfolio demos — Powered by Streamlit, LangChain, and Groq.")

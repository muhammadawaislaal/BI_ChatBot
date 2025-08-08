import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from datetime import datetime

# LangChain imports
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq  # You can swap with any LLM provider

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="📊 AI Data Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
st.sidebar.title("📊 AI Data Analyst")
st.sidebar.markdown("Upload your data → Ask questions → Get instant insights.")

# Tracker for interactions
if "history" not in st.session_state:
    st.session_state.history = []

# --- FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # --- AI Model Setup ---
    llm = ChatGroq(
        api_key=st.secrets["API_KEY"],  # Replace with your secret
        model="llama3-70b-8192",
        temperature=0
    )

    # FIX: Allow dangerous code explicitly
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True
    )

    # --- QUERY SECTION ---
    st.markdown("### 💬 Ask a question")
    query = st.text_input("Example: Show total sales by region")

    if query:
        with st.spinner("Analyzing your data..."):
            try:
                response = agent.invoke(query)
                answer = response.get("output", "No output returned.")
                st.markdown(f"**Answer:** {answer}")

                # Save to tracker
                st.session_state.history.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "answer": answer
                })

            except Exception as e:
                st.error(f"Error: {e}")

        # Auto chart if request contains keywords
        if any(word in query.lower() for word in ["plot", "chart", "graph", "show", "visualize"]):
            try:
                fig, ax = plt.subplots()
                df.plot(ax=ax)
                st.pyplot(fig)
            except Exception:
                st.warning("No valid columns to plot automatically.")

    # --- TRACKER SECTION ---
    if st.session_state.history:
        st.markdown("### 📜 Interaction History")
        for item in st.session_state.history:
            st.write(f"🕒 {item['time']}")
            st.write(f"**Q:** {item['query']}")
            st.write(f"**A:** {item['answer']}")
            st.markdown("---")

    # --- EXPORT AS PDF ---
    def export_pdf(history):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for item in history:
            pdf.multi_cell(0, 10, f"[{item['time']}]\nQ: {item['query']}\nA: {item['answer']}\n")
            pdf.ln()
        return pdf.output(dest="S").encode("latin1")

    if st.session_state.history:
        pdf_data = export_pdf(st.session_state.history)
        st.download_button(
            "📥 Download Report as PDF",
            data=pdf_data,
            file_name="data_analysis_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("👆 Please upload a CSV file to start.")

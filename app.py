import subprocess
import sys
import chardet

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

st.sidebar.header("🤖 Model Details")
st.sidebar.info(
    """
    **AI Model:** LLaMA 3 8B  
    **Provider:** Groq  
    **Capabilities:** 
    - Data analysis & insights
    - Trend identification
    - Chart generation
    - Statistical summaries
    - Predictive suggestions
    """
)

st.sidebar.header("🎯 How to Use")
st.sidebar.info(
    """
    **Best Practices:**
    - Ask specific questions for better results
    - Use clear, concise language
    - Request visualizations when needed
    - Verify critical insights with exports
    
    **Example Queries:**
    - "Show sales trends by month"
    - "Compare performance across regions"
    - "What are the top 5 products by revenue?"
    - "Generate a correlation matrix for numerical columns"
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
#  FILE UPLOAD WITH ENCODING DETECTION
# =========================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

def detect_encoding(file):
    """Detect the encoding of an uploaded file"""
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # Reset file pointer
    return result['encoding']

if uploaded_file:
    try:
        # Detect file encoding
        encoding = detect_encoding(uploaded_file)
        st.info(f"Detected file encoding: {encoding}")
        
        # Read CSV with detected encoding
        df = pd.read_csv(uploaded_file, encoding=encoding)
        st.success("✅ File uploaded successfully!")
        
        # Show basic file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Data Types", len(df.dtypes.unique()))
        
        # Show dataframe preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head())
            
        # Show data types
        with st.expander("🔍 Data Types"):
            st.write(df.dtypes)
            
        # Show basic statistics
        with st.expander("📊 Basic Statistics"):
            st.write(df.describe())

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

            with st.spinner("🤔 Analyzing your data..."):
                try:
                    response = agent.run(user_query)
                    st.markdown(f"### 🧠 Answer:\n{response}")

                    # Auto-generate charts for relevant queries
                    chart_keywords = ["trend", "chart", "plot", "graph", "visualize", "show me"]
                    if any(word in user_query.lower() for word in chart_keywords):
                        st.markdown("#### 📊 Auto-generated Chart")
                        plt.figure(figsize=(10, 6))
                        try:
                            # Try to create a meaningful chart based on common patterns
                            num_cols = df.select_dtypes(include='number').columns
                            if len(num_cols) >= 1:
                                # Simple line plot for the first numerical column
                                plt.plot(df[num_cols[0]].values)
                                plt.title(f"{num_cols[0]} Trend")
                                plt.xlabel("Index")
                                plt.ylabel(num_cols[0])
                                plt.grid(True, alpha=0.3)
                                st.pyplot(plt)
                        except Exception as e:
                            st.warning(f"Could not auto-generate chart: {e}")

                except Exception as e:
                    st.error(f"Error processing your query: {str(e)}")
                    st.info("💡 Try rephrasing your question or ask about a different aspect of your data.")

        # =========================
        #  EXPORT TO PDF
        # =========================
        def export_pdf(text):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="BI Analysis Report", ln=True, align='C')
            pdf.ln(10)
            
            # Add timestamp
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.ln(10)
            
            # Add content
            pdf.set_font("Arial", size=12)
            for line in text.split("\n"):
                pdf.multi_cell(0, 10, txt=line)
            
            file_name = f"BI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf.output(file_name)
            return file_name

        if st.button("📄 Export Last Answer to PDF"):
            if user_query and 'response' in locals():
                pdf_path = export_pdf(f"Question: {user_query}\n\nAnswer:\n{response}")
                with open(pdf_path, "rb") as file:
                    btn = st.download_button(
                        label="Download PDF Report",
                        data=file,
                        file_name=pdf_path,
                        mime="application/octet-stream"
                    )
            else:
                st.warning("Please ask a question first to generate a report.")

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("💡 Try saving your CSV file with UTF-8 encoding before uploading.")

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


import subprocess
import sys
import chardet
import io

# Ensure required packages are installed
try:
    import tabulate
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
    import tabulate

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import plotly.express as px

# =========================
#  CONFIGURATION & THEME
# =========================
st.set_page_config(
    page_title="📊 BI Conversational AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #636363;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
        color: white;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #3498db;
    }
    .data-preview {
        border-radius: 10px;
        padding: 1rem;
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
    }
    .response-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        color: #2c3e50;
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

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
st.markdown('<h1 class="main-header">📊 Business Intelligence AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask anything about your data — get instant answers, insights, and professional visualizations</p>', unsafe_allow_html=True)

# =========================
#  SIDEBAR - INFO & TRACKER
# =========================
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white;'>
        <h2 style='color: white; margin-bottom: 1rem;'>📌 Instructions</h2>
        <ol style='color: white;'>
            <li><strong>Upload</strong> your CSV file</li>
            <li><strong>Ask questions</strong> in plain English</li>
            <li>Get <strong>instant BI insights</strong> + charts</li>
            <li><strong>Export</strong> results as PDF</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); padding: 1rem; border-radius: 10px; color: white;'>
        <h2 style='color: white;'>🤖 Model Details</h2>
        <p><strong>AI Model:</strong> LLaMA 3.1 8B</p>
        <p><strong>Provider:</strong> Groq</p>
        <p><strong>Capabilities:</strong></p>
        <ul>
            <li>Data analysis & insights</li>
            <li>Trend identification</li>
            <li>Chart generation</li>
            <li>Statistical summaries</li>
            <li>Predictive suggestions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 1rem; border-radius: 10px; color: white;'>
        <h2 style='color: white;'>🎯 How to Use</h2>
        <p><strong>Best Practices:</strong></p>
        <ul>
            <li>Ask specific questions for better results</li>
            <li>Use clear, concise language</li>
            <li>Request visualizations when needed</li>
            <li>Verify critical insights with exports</li>
        </ul>
        <p><strong>Example Queries:</strong></p>
        <ul>
            <li>"Show sales trends by month"</li>
            <<li>"Compare performance across regions"</li>
            <li>"What are the top 5 products by revenue?"</li>
            <li>"Generate a correlation matrix"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Usage Tracker
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white;'>
        <h2 style='color: white;'>📈 Usage Tracker</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if "queries_count" not in st.session_state:
        st.session_state.queries_count = 0
    st.metric("Questions Asked", st.session_state.queries_count)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1rem; border-radius: 10px;'>
        <h2>💡 BI Tips</h2>
        <ul>
            <li>Ask about <strong>trends</strong> over time</li>
            <li>Compare <strong>categories</strong></li>
            <li>Request <strong>charts</strong> or <strong>summaries</strong></li>
            <li>Example: <em>"Show me monthly sales trends"</em></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================
#  FILE UPLOAD WITH ENCODING DETECTION
# =========================
st.markdown("### 📂 Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload your dataset for analysis")

if uploaded_file:
    try:
        # Read file content for encoding detection
        raw_data = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        # Detect encoding
        encoding_detected = chardet.detect(raw_data)
        encoding = encoding_detected['encoding']
        confidence = encoding_detected['confidence']
        
        # If confidence is low, try common encodings
        if confidence < 0.7:
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        else:
            encodings_to_try = [encoding, 'utf-8', 'latin-1']
        
        df = None
        successful_encoding = None
        
        for enc in encodings_to_try:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                successful_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if df is None:
            # Last resort: try reading with error handling
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            successful_encoding = 'utf-8 (with error handling)'
        
        st.markdown(f"<div class='success-box'>✅ File uploaded successfully! Detected encoding: {successful_encoding}</div>", unsafe_allow_html=True)
        
        # Show basic file info in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Rows", df.shape[0])
        with col2:
            st.metric("📈 Columns", df.shape[1])
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("🔢 Numeric Columns", numeric_cols)
        with col4:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("📝 Categorical Columns", categorical_cols)
        
        # Data preview with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "🔍 Data Types", "📊 Statistics", "📈 Visual Overview"])
        
        with tab1:
            st.dataframe(df.head(10), width='stretch')
        
        with tab2:
            # Fix for data type display issue - convert to string
            dtype_info = []
            for col in df.columns:
                dtype_info.append({"Column": col, "Data Type": str(df[col].dtype)})
            dtype_df = pd.DataFrame(dtype_info)
            st.dataframe(dtype_df, width='stretch')
        
        with tab3:
            if numeric_cols > 0:
                st.dataframe(df.describe(), width='stretch')
            else:
                st.info("No numeric columns to display statistics")
        
        with tab4:
            if numeric_cols > 0:
                # Auto-generate some basic visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution of first numeric column
                    num_col = df.select_dtypes(include=[np.number]).columns[0]
                    fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Correlation heatmap if multiple numeric columns
                    if numeric_cols > 1:
                        corr_matrix = df.select_dtypes(include=[np.number]).corr()
                        fig = px.imshow(corr_matrix, title="Correlation Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns to generate visualizations")

        # =========================
        #  LLM SETUP
        # =========================
        llm = ChatGroq(
            groq_api_key=API_KEY,
            model="llama-3.1-8b-instant",
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
        st.markdown("---")
        st.markdown("### 💬 Ask a Question About Your Data")
        
        # Example questions
        example_questions = [
            "What are the key trends in this data?",
            "Show me sales by category",
            "What is the correlation between different variables?",
            "Identify any outliers or anomalies",
            "What are the top 5 performing items?"
        ]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        
        for i, question in enumerate(example_questions):
            with cols[i]:
                if st.button(question[:20] + "..." if len(question) > 20 else question, key=f"btn_{i}"):
                    st.session_state.user_query = question
        
        user_query = st.text_input(
            "Enter your question:",
            value=st.session_state.get("user_query", ""),
            placeholder="e.g., Show me monthly sales trends...",
            key="query_input"
        )

        if user_query:
            st.session_state.queries_count += 1
            
            with st.spinner("🤔 Analyzing your data with AI..."):
                try:
                    # Use invoke instead of run to avoid deprecation warning
                    response = agent.invoke(user_query)
                    response_text = response['output'] if isinstance(response, dict) and 'output' in response else str(response)
                    
                    # Display response in a nice format
                    st.markdown("### 📊 AI Analysis Result")
                    st.markdown(f"<div class='response-box'>{response_text}</div>", unsafe_allow_html=True)
                    
                    # Auto-generate visualizations for relevant queries
                    chart_keywords = ["trend", "chart", "plot", "graph", "visualize", "show me", "compare", "distribution"]
                    if any(word in user_query.lower() for word in chart_keywords) and numeric_cols > 0:
                        st.markdown("### 📈 Auto-generated Visualizations")
                        
                        try:
                            # Try to create meaningful visualizations based on query
                            if "trend" in user_query.lower() or "time" in user_query.lower():
                                # Look for date columns
                                date_cols = df.select_dtypes(include=['datetime64']).columns
                                if len(date_cols) > 0:
                                    date_col = date_cols[0]
                                    num_col = df.select_dtypes(include=[np.number]).columns[0]
                                    fig = px.line(df, x=date_col, y=num_col, title=f"{num_col} Trend Over Time")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # Use index as x-axis if no date column
                                    num_col = df.select_dtypes(include=[np.number]).columns[0]
                                    fig = px.line(df, y=num_col, title=f"{num_col} Trend")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            elif "distribution" in user_query.lower() or "histogram" in user_query.lower():
                                num_col = df.select_dtypes(include=[np.number]).columns[0]
                                fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif "compare" in user_query.lower():
                                if numeric_cols >= 2:
                                    col1, col2 = df.select_dtypes(include=[np.number]).columns[:2]
                                    fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            else:
                                # Default visualization
                                num_col = df.select_dtypes(include=[np.number]).columns[0]
                                fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
                                st.plotly_chart(fig, use_container_width=True)
                                
                        except Exception as e:
                            st.warning(f"Could not auto-generate chart: {e}")

                except Exception as e:
                    st.markdown("<div class='error-box'>❌ Error processing your query. Please try rephrasing or ask a different question.</div>", 
                               unsafe_allow_html=True)
                    st.error(f"Technical details: {str(e)}")

        # =========================
        #  EXPORT TO PDF
        # =========================
        def export_pdf(text, df_info, query):
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="BI Analysis Report", ln=True, align='C')
            pdf.ln(10)
            
            # Metadata
            pdf.set_font("Arial", '', 10)
            pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(200, 10, txt=f"Dataset: {uploaded_file.name}", ln=True)
            pdf.cell(200, 10, txt=f"Rows: {df_info['rows']}, Columns: {df_info['cols']}", ln=True)
            pdf.ln(10)
            
            # Question
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Question:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 10, txt=query)
            pdf.ln(5)
            
            # Answer
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Analysis:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 10, txt=text)
            
            file_name = f"BI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf.output(file_name)
            return file_name

        if st.button("📄 Export to PDF Report", use_container_width=True):
            if user_query and 'response_text' in locals():
                df_info = {
                    'rows': df.shape[0],
                    'cols': df.shape[1]
                }
                pdf_path = export_pdf(response_text, df_info, user_query)
                
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download PDF Report",
                        data=file,
                        file_name=pdf_path,
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.warning("Please ask a question first to generate a report.")

    except Exception as e:
        st.markdown("<div class='error-box'>❌ Error reading file. Please check the file format and try again.</div>", 
                   unsafe_allow_html=True)
        st.error(f"Technical details: {str(e)}")
        st.info("💡 Tip: Try saving your CSV file with UTF-8 encoding before uploading.")

else:
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px;'>
        <h2 style='color: #6c757d;'>📊 Welcome to BI Conversational AI</h2>
        <p style='color: #6c757d;'>Upload a CSV file to begin your data analysis journey</p>
        <div style='font-size: 4rem;'>⬆️</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
#  FOOTER
# =========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 13px;'>
        <p>© 2025 BI Conversational AI | Built for Professional Business Insights & Data Storytelling</p>
        <p>Powered by Groq LLaMA 3.1 • Streamlit • LangChain</p>
    </div>
    """,
    unsafe_allow_html=True,
)



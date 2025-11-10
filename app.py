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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
#  CONFIGURATION & THEME
# =========================
st.set_page_config(
    page_title="📊 BI Conversational AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS with Modern Design
st.markdown("""
<style>
    /* Main Container */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        font-family: 'Inter', sans-serif;
    }
    
    /* Cards & Containers */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: none;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .data-preview {
        border-radius: 15px;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #3498db;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    /* Status Boxes */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
        color: #721c24;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #dc3545;
        box-shadow: 0 5px 15px rgba(220, 53, 69, 0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #b8e2eb 100%);
        color: #0c5460;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 5px 15px rgba(23, 162, 184, 0.1);
    }
    .response-box {
        background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #3498db;
        color: #2c3e50;
        font-size: 16px;
        line-height: 1.7;
        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.1);
    }
    
    /* Buttons & Inputs */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* Sidebar Enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0px 0px;
        gap: 8px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    
    /* File Uploader */
    .stFileUploader>div>div {
        border: 2px dashed #3498db;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(52, 152, 219, 0.05);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
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
st.markdown('<p class="sub-header">Transform Your Data into Actionable Insights with AI-Powered Analysis</p>', unsafe_allow_html=True)

# =========================
#  SIDEBAR - ENHANCED PROFESSIONAL DESIGN
# =========================
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
        <h2 style='color: white; margin-bottom: 1.5rem; text-align: center;'>🚀 Quick Start Guide</h2>
        <div style='display: flex; flex-direction: column; gap: 1rem;'>
            <div style='display: flex; align-items: center; gap: 0.5rem;'>
                <span style='background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;'>1</span>
                <span><strong>Upload</strong> your CSV dataset</span>
            </div>
            <div style='display: flex; align-items: center; gap: 0.5rem;'>
                <span style='background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;'>2</span>
                <span><strong>Ask questions</strong> in natural language</span>
            </div>
            <div style='display: flex; align-items: center; gap: 0.5rem;'>
                <span style='background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;'>3</span>
                <span>Get <strong>AI-powered insights</strong> instantly</span>
            </div>
            <div style='display: flex; align-items: center; gap: 0.5rem;'>
                <span style='background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;'>4</span>
                <span><strong>Export</strong> professional reports</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI Model Information
    st.markdown("""
    <div style='background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
        <h2 style='color: white; margin-bottom: 1rem; text-align: center;'>🤖 AI Engine</h2>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; text-align: center;'>
            <div style='background: rgba(255,255,255,0.2); padding: 0.75rem; border-radius: 10px;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>Model</div>
                <div style='font-weight: 700;'>LLaMA 3.1</div>
            </div>
            <div style='background: rgba(255,255,255,0.2); padding: 0.75rem; border-radius: 10px;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>Provider</div>
                <div style='font-weight: 700;'>Groq</div>
            </div>
        </div>
        <div style='margin-top: 1rem; font-size: 0.9rem;'>
            <strong>Capabilities:</strong> Data Analysis • Trend Identification • Statistical Insights • Visualization
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Usage Analytics
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
        <h2 style='color: white; margin-bottom: 1rem; text-align: center;'>📊 Analytics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if "queries_count" not in st.session_state:
        st.session_state.queries_count = 0
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions Asked", st.session_state.queries_count)
    with col2:
        st.metric("Files Processed", st.session_state.files_processed)
    
    st.markdown("---")
    
    # Best Practices
    st.markdown("""
    <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1.5rem; border-radius: 15px;'>
        <h2 style='margin-bottom: 1rem; color: #2c3e50;'>💡 Pro Tips</h2>
        <div style='display: flex; flex-direction: column; gap: 0.75rem;'>
            <div style='display: flex; align-items: start; gap: 0.5rem;'>
                <span style='color: #3498db; font-weight: bold;'>•</span>
                <span>Use specific, clear questions</span>
            </div>
            <div style='display: flex; align-items: start; gap: 0.5rem;'>
                <span style='color: #3498db; font-weight: bold;'>•</span>
                <span>Ask for trends and comparisons</span>
            </div>
            <div style='display: flex; align-items: start; gap: 0.5rem;'>
                <span style='color: #3498db; font-weight: bold;'>•</span>
                <span>Request visualizations explicitly</span>
            </div>
            <div style='display: flex; align-items: start; gap: 0.5rem;'>
                <span style='color: #3498db; font-weight: bold;'>•</span>
                <span>Export insights for sharing</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
#  FILE UPLOAD WITH ENCODING DETECTION
# =========================
st.markdown("### 📁 Upload Your Dataset")
uploaded_file = st.file_uploader(
    "Drag and drop your CSV file here", 
    type=["csv"], 
    help="Upload your business data for AI-powered analysis"
)

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
        
        st.session_state.files_processed += 1
        st.markdown(f"<div class='success-box'>✅ Dataset loaded successfully! Detected encoding: {successful_encoding}</div>", unsafe_allow_html=True)
        
        # Enhanced Data Overview
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("📈 Total Columns", df.shape[1])
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("🔢 Numeric Features", numeric_cols)
        with col4:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("📝 Text Features", categorical_cols)
        
        # Enhanced Data Exploration Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Data Preview", "📐 Data Structure", "📊 Statistics", "📈 Quick Insights"])
        
        with tab1:
            st.markdown("#### Sample Data (First 10 Rows)")
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            dtype_info = []
            for col in df.columns:
                non_null_count = df[col].count()
                null_count = df[col].isnull().sum()
                dtype_info.append({
                    "Column": col, 
                    "Data Type": str(df[col].dtype),
                    "Non-Null": non_null_count,
                    "Null Values": null_count,
                    "Unique Values": df[col].nunique()
                })
            dtype_df = pd.DataFrame(dtype_info)
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab3:
            if numeric_cols > 0:
                st.markdown("#### Statistical Summary")
                st.dataframe(df.describe(), use_container_width=True)
            else:
                st.info("📝 No numeric columns available for statistical analysis")
        
        with tab4:
            if numeric_cols > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Enhanced distribution plot
                    num_col = df.select_dtypes(include=[np.number]).columns[0]
                    fig = px.histogram(
                        df, x=num_col, 
                        title=f"📈 Distribution of {num_col}",
                        color_discrete_sequence=['#3498db'],
                        template='plotly_white'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Enhanced correlation heatmap
                    if numeric_cols > 1:
                        corr_matrix = df.select_dtypes(include=[np.number]).corr()
                        fig = px.imshow(
                            corr_matrix,
                            title="🔄 Correlation Matrix",
                            color_continuous_scale='RdBu_r',
                            aspect='auto'
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No numeric columns available for visual insights")

        # =========================
        #  LLM SETUP
        # =========================
        llm = ChatGroq(
            groq_api_key=API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=True
        )

        # =========================
        #  ENHANCED USER INPUT SECTION
        # =========================
        st.markdown("---")
        st.markdown("### 💬 Ask Anything About Your Data")
        
        # Enhanced Example Questions
        example_questions = [
            "What are the key trends and patterns in this data?",
            "Show me sales performance by category",
            "What is the correlation between key variables?",
            "Identify any outliers or data quality issues",
            "What are the top 5 performing segments?"
        ]
        
        st.markdown("#### 🚀 Try These Questions:")
        cols = st.columns(5)
        for i, (col, question) in enumerate(zip(cols, example_questions)):
            with col:
                if st.button(
                    question[:25] + "..." if len(question) > 25 else question, 
                    key=f"btn_{i}",
                    use_container_width=True
                ):
                    st.session_state.user_query = question
        
        user_query = st.text_input(
            "**Or type your own question:**",
            value=st.session_state.get("user_query", ""),
            placeholder="e.g., What are the monthly sales trends? Show top products by revenue...",
            key="query_input"
        )

        if user_query:
            st.session_state.queries_count += 1
            
            with st.spinner("🔍 Analyzing your data with advanced AI..."):
                try:
                    # Enhanced response handling
                    response = agent.invoke(user_query)
                    response_text = response['output'] if isinstance(response, dict) and 'output' in response else str(response)
                    
                    # Display enhanced response
                    st.markdown("### 📈 AI Analysis Results")
                    st.markdown(f"<div class='response-box'>{response_text}</div>", unsafe_allow_html=True)
                    
                    # Enhanced Auto-visualization
                    chart_keywords = ["trend", "chart", "plot", "graph", "visualize", "show me", "compare", "distribution", "correlation"]
                    if any(word in user_query.lower() for word in chart_keywords) and numeric_cols > 0:
                        st.markdown("### 📊 Automated Visualizations")
                        
                        try:
                            # Enhanced trend analysis
                            if "trend" in user_query.lower() or "time" in user_query.lower():
                                date_cols = df.select_dtypes(include=['datetime64']).columns
                                if len(date_cols) > 0:
                                    date_col = date_cols[0]
                                    num_col = df.select_dtypes(include=[np.number]).columns[0]
                                    fig = px.line(
                                        df, x=date_col, y=num_col, 
                                        title=f"📈 {num_col} Trend Over Time",
                                        color_discrete_sequence=['#e74c3c']
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Enhanced distribution analysis
                            elif "distribution" in user_query.lower() or "histogram" in user_query.lower():
                                num_col = df.select_dtypes(include=[np.number]).columns[0]
                                fig = px.histogram(
                                    df, x=num_col, 
                                    title=f"📊 Distribution of {num_col}",
                                    color_discrete_sequence=['#2ecc71']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Enhanced comparison analysis
                            elif "compare" in user_query.lower():
                                if numeric_cols >= 2:
                                    col1, col2 = df.select_dtypes(include=[np.number]).columns[:2]
                                    fig = px.scatter(
                                        df, x=col1, y=col2, 
                                        title=f"🔄 {col1} vs {col2}",
                                        color_discrete_sequence=['#9b59b6']
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            else:
                                # Default enhanced visualization
                                num_col = df.select_dtypes(include=[np.number]).columns[0]
                                fig = px.box(
                                    df, y=num_col, 
                                    title=f"📦 Distribution Analysis of {num_col}",
                                    color_discrete_sequence=['#f39c12']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate automated visualization: {str(e)}")

                except Exception as e:
                    st.markdown("<div class='error-box'>❌ Unable to process your query. Please try rephrasing or ask a different question.</div>", unsafe_allow_html=True)
                    st.error(f"Technical details: {str(e)}")

        # =========================
        #  ENHANCED PDF EXPORT
        # =========================
        def export_pdf(text, df_info, query):
            pdf = FPDF()
            pdf.add_page()
            
            # Professional Header
            pdf.set_font("Arial", 'B', 20)
            pdf.cell(200, 15, txt="Business Intelligence Analysis Report", ln=True, align='C')
            pdf.ln(10)
            
            # Metadata Section
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(200, 8, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(200, 8, txt=f"Dataset: {uploaded_file.name}", ln=True)
            pdf.cell(200, 8, txt=f"Dimensions: {df_info['rows']} rows × {df_info['cols']} columns", ln=True)
            pdf.ln(15)
            
            # Question Section
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Analysis Question:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, txt=query)
            pdf.ln(10)
            
            # Answer Section
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="AI Analysis:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, txt=text)
            
            file_name = f"BI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf.output(file_name)
            return file_name

        if st.button("📄 Generate Professional Report", use_container_width=True):
            if user_query and 'response_text' in locals():
                df_info = {
                    'rows': df.shape[0],
                    'cols': df.shape[1]
                }
                pdf_path = export_pdf(response_text, df_info, user_query)
                
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=file,
                        file_name=pdf_path,
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.warning("💡 Please ask a question first to generate a comprehensive report.")

    except Exception as e:
        st.markdown("<div class='error-box'>❌ Error processing your dataset. Please check the file format and try again.</div>", unsafe_allow_html=True)
        st.error(f"Technical details: {str(e)}")
        st.info("💡 Pro Tip: Ensure your CSV file uses UTF-8 encoding and has consistent column formatting.")

else:
    # Enhanced Welcome Section
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; border: 2px dashed #3498db;'>
        <h2 style='color: #2c3e50; margin-bottom: 1.5rem; font-size: 2.5rem;'>🚀 Welcome to BI Conversational AI</h2>
        <p style='color: #6c757d; font-size: 1.2rem; margin-bottom: 3rem;'>Transform your raw data into powerful business insights with AI-powered analysis</p>
        <div style='font-size: 5rem; margin-bottom: 2rem;'>📊</div>
        <div style='display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 2rem; border-radius: 50px; font-weight: 600; font-size: 1.1rem;'>
            Upload Your CSV to Begin
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
#  ENHANCED FOOTER
# =========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; font-size: 14px; padding: 2rem 0;'>
        <p style='margin-bottom: 0.5rem; font-weight: 600;'>© 2025 BI Conversational AI Platform</p>
        <p style='margin-bottom: 0; font-size: 0.9rem; opacity: 0.8;'>Powered by Groq LLaMA 3.1 • Streamlit • LangChain • Advanced Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

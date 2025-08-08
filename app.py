# app.py
import os
import io
import time
import json
import base64
import requests
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

# LangChain imports (Groq + Ollama)
from langchain_experimental.agents import create_pandas_dataframe_agent

# NOTE: these imports are lazy inside helpers to avoid import errors at startup.

# ---------------------------
# ---- Configuration ----
# ---------------------------
GROQ_DAILY_LIMIT = 100000     # adjust if your Groq plan differs
GROQ_WARNING_THRESHOLD = 0.9  # 90% threshold for warning
DEFAULT_GROQ_MODEL = "gemma2-9b-it"            # lightweight default
ALTERNATE_GROQ_MODELS = ["gemma2-9b-it", "mistral-saba-7b", "llama-3.3-70b-versatile"]

st.set_page_config(page_title="AI BI Chatbot — Groq + Ollama", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# ---- Styling (classy) ----
# ---------------------------
st.markdown(
    """
    <style>
    .reportview-container { background: linear-gradient(180deg, #ffffff, #f6fbff); }
    .stApp { font-family: 'Inter', sans-serif; color: #0b2545; }
    .title { font-size:30px; font-weight:700; color: #023e8a;}
    .subtitle { color:#0353a4; }
    .sidebar .stButton>button { background-color: #0077b6; color: white; border-radius:8px;}
    .card { background: white; padding: 12px; border-radius: 12px; box-shadow: 0 4px 20px rgba(2,46,89,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# ---- Helper functions ----
# ---------------------------

def safe_get_secret(key: str):
    """Return secret if available, otherwise None"""
    try:
        return st.secrets[key]
    except Exception:
        return None

def get_groq_usage(api_key: str):
    """Return an integer number of tokens used today or None if not available."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        # Groq usage endpoint may vary; this is a best-effort
        r = requests.get("https://api.groq.com/v1/usage", headers=headers, timeout=6)
        if r.status_code == 200:
            j = r.json()
            # Try a few common keys; adapt if Groq returns different shape
            used = j.get("total_tokens_today") or j.get("usage", {}).get("total_tokens_today") or j.get("total_tokens", 0)
            return int(used)
        else:
            return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def create_agent_for_llm(llm_provider: str, model_name: str, df_json: str):
    """
    Create an agent for given llm ('groq' or 'ollama') and dataframe JSON.
    We cache by LLM+model+df_json to avoid re-init frequently.
    """
    df = pd.read_json(io.StringIO(df_json), orient="split")
    if llm_provider == "groq":
        from langchain_groq import ChatGroq
        groq_key = safe_get_secret("GROQ_API_KEY")
        llm = ChatGroq(model_name=model_name, temperature=0, groq_api_key=groq_key)
    else:
        # local fallback via Ollama (requires local Ollama install & model)
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(model=model_name)  # model_name like "llama3"
    agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)
    return agent

def auto_plot_df(df: pd.DataFrame, pick_columns: tuple = None):
    """Create a matplotlib figure from df. If pick_columns provided uses them (x,y)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    numeric = df.select_dtypes(include="number").columns.tolist()
    if pick_columns:
        xcol, ycol = pick_columns
        df.plot(x=xcol, y=ycol, ax=ax, marker="o")
    elif "Date" in df.columns and len(numeric) >= 1:
        df_local = df.copy()
        try:
            df_local["Date"] = pd.to_datetime(df_local["Date"])
            for col in df_local.select_dtypes(include="number").columns:
                ax.plot(df_local["Date"], df_local[col], marker="o", label=col)
            ax.legend()
        except Exception:
            # fallback: plot first numeric column vs index
            ax.plot(df_local.index, df_local[numeric[0]], marker="o")
    elif len(numeric) >= 2:
        ax.plot(df[numeric[0]], df[numeric[1]], marker="o")
        ax.set_xlabel(numeric[0])
        ax.set_ylabel(numeric[1])
    elif len(numeric) == 1:
        ax.plot(df.index, df[numeric[0]], marker="o")
        ax.set_ylabel(numeric[0])
    else:
        ax.text(0.5, 0.5, "No numeric columns to plot", ha="center")
    ax.set_title("Auto-generated chart")
    plt.tight_layout()
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    return buf.read()

def create_pdf_from_chat_and_image(chat_history, image_bytes=None):
    pdf = FPDF(unit="pt", format="letter")
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.set_text_color(10, 30, 60)
    pdf.cell(0, 20, "AI BI Chatbot — Report", ln=1)
    pdf.ln(4)
    for turn in chat_history:
        pdf.set_font("Helvetica", style="B", size=11)
        pdf.cell(0, 14, f"You: {turn['user']}", ln=1)
        pdf.set_font("Helvetica", size=11)
        # wrap bot text
        for line in split_text(turn["bot"], 90):
            pdf.multi_cell(0, 12, line)
        pdf.ln(6)
    if image_bytes:
        # Save bytes to temp file and embed
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.flush()
        pdf.image(tmp.name, x=50, y=pdf.get_y(), w=500)
        tmp.close()
    out = io.BytesIO()
    out.write(pdf.output(dest="S").encode("latin1"))
    out.seek(0)
    return out

def split_text(text, width):
    """Split text into chunks for PDF multi_cell."""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

# ---------------------------
# ---- Sidebar (controls) ----
# ---------------------------
with st.sidebar:
    st.markdown("<div class='card'><h3 class='subtitle'>AI BI Chatbot</h3>", unsafe_allow_html=True)
    st.markdown("Small, polished demo that runs on Groq (free) with local Ollama fallback.")
    st.write("---")

    # Show secrets status
    groq_key = safe_get_secret("GROQ_API_KEY")
    if groq_key:
        st.success("✅ GROQ_API_KEY found in secrets.")
    else:
        st.error("❗ GROQ API key not found. Add `GROQ_API_KEY` to Streamlit secrets or .streamlit/secrets.toml")

    st.write("### Model / Mode")
    model_choice = st.selectbox("Groq model (default)", ALTERNATE_GROQ_MODELS, index=0)
    use_local_first = st.checkbox("Prefer local Ollama first (development)", value=False)
    st.write("---")

    st.write("### Demo Data")
    if st.button("Download demo CSV"):
        sample = """Date,Region,Product,Units Sold,Unit Price,Total Sales
2025-01-01,North,Widget A,120,10,1200
2025-01-01,South,Widget B,80,12,960
2025-01-02,North,Widget A,150,10,1500
2025-01-02,South,Widget B,60,12,720
2025-01-03,North,Widget A,90,10,900
2025-01-03,South,Widget B,100,12,1200
2025-01-04,North,Widget A,200,10,2000
2025-01-04,South,Widget B,50,12,600
2025-01-05,North,Widget A,130,10,1300
2025-01-05,South,Widget B,110,12,1320
"""
        b = base64.b64encode(sample.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b}" download="sales_demo.csv">Click to download demo CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    st.write("---")
    st.write("Status & Usage")

    groq_usage = None
    if groq_key:
        groq_usage = get_groq_usage(groq_key)
    if groq_usage is not None:
        st.write(f"Groq tokens used today: **{groq_usage:,} / {GROQ_DAILY_LIMIT:,}**")
        pct = min(groq_usage / GROQ_DAILY_LIMIT, 1.0)
        st.progress(pct)
        if pct >= 1.0:
            st.error("Groq quota exceeded — fallback will be used.")
        elif pct >= GROQ_WARNING_THRESHOLD:
            st.warning("Approaching Groq daily limit — consider switching model or using local fallback.")
    else:
        st.info("Groq usage not available (API may not expose usage to this key).")

    st.write("---")
    st.markdown("Built for portfolio demos — lightweight, single-file, and deployable.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# ---- Main UI ----
# ---------------------------
st.markdown("<h1 class='title'>📊 AI Business Intelligence Assistant</h1>", unsafe_allow_html=True)
st.markdown("Upload CSV → Ask questions in natural language → Get answers & charts. Export results for sharing.")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Upload a CSV with at least one numeric column.")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(100))
            # keep df JSON in session for caching agents
            st.session_state["df_json"] = df.to_json(date_format="iso", orient="split")
            st.session_state["df_preview"] = df.head(100).to_dict(orient="records")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
    else:
        st.info("Upload a CSV to start, or use the demo CSV from the sidebar.")
        df = None

    # Chat area
    if df is not None:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        prompt = st.text_input("Ask a question about your data (e.g., 'Show total sales by region' or 'Plot Units Sold over Date')", key="input_prompt")
        cols = df.columns.tolist()
        st.write("Detected columns:", cols)

        # manual plotting controls
        with st.expander("Plot controls (manual)"):
            x_col = st.selectbox("X column (optional)", options=[None] + cols, index=0)
            y_col = st.selectbox("Y column (optional)", options=[None] + cols, index=0)
            plot_button = st.button("Generate manual plot")

        # Buttons area
        run_btn, clear_btn, export_pdf_btn, export_txt_btn = st.columns([1,1,1,1])

        if run_btn.button("Run query"):
            if not prompt.strip():
                st.warning("Please enter a question.")
            else:
                # Decide LLM: prefer local if user set, otherwise Groq if quota ok
                use_local = False
                if use_local_first:
                    use_local = True
                else:
                    if groq_key:
                        usage = get_groq_usage(groq_key)
                        if usage is not None and usage >= GROQ_DAILY_LIMIT:
                            use_local = True

                # choose LLM provider and model
                provider = "groq" if not use_local else "ollama"
                model_for_provider = model_choice if provider == "groq" else "llama3"

                try:
                    with st.spinner(f"Initializing agent with {provider} ({model_for_provider})..."):
                        agent = create_agent_for_llm(provider, model_for_provider, st.session_state["df_json"])
                        # Some agent APIs deprecate .run; still supported. Keep try/except.
                        try:
                            result = agent.run(prompt)
                        except Exception:
                            # fallback to invoke if needed
                            try:
                                result = agent.invoke({"input": prompt})
                            except Exception as e:
                                raise e

                    st.session_state.chat_history.append({"user": prompt, "bot": str(result)})
                    st.success("Answer generated.")
                except Exception as e:
                    st.error(f"Agent error: {e}")
                    # attempt fallback to local if not already
                    if provider == "groq":
                        st.info("Attempting fallback to local Ollama...")
                        try:
                            agent = create_agent_for_llm("ollama", "llama3", st.session_state["df_json"])
                            result = agent.run(prompt)
                            st.session_state.chat_history.append({"user": prompt, "bot": str(result)})
                            st.success("Answer generated with local model.")
                        except Exception as e2:
                            st.error(f"Local fallback failed: {e2}")

        if clear_btn.button("Clear history"):
            st.session_state.chat_history = []
            st.success("Cleared chat history.")

        # Manual plot
        if plot_button:
            try:
                pick = None
                if x_col and y_col:
                    pick = (x_col, y_col)
                fig = auto_plot_df(df, pick_columns=pick)
                st.pyplot(fig)
                # save last plot to session
                st.session_state["last_plot_bytes"] = fig_to_bytes(fig)
            except Exception as e:
                st.error(f"Plot error: {e}")

        # Show chat history
        if st.session_state.get("chat_history"):
            st.markdown("### Conversation")
            for i, turn in enumerate(st.session_state.chat_history[::-1]):
                st.markdown(f"**You:** {turn['user']}")
                st.markdown(f"**Bot:** {turn['bot']}")
                st.write("---")

        # Auto plot if user asked for 'plot' or 'chart' in prompt and we have recent result
        if st.session_state.get("chat_history"):
            last_user = st.session_state["chat_history"][-1]["user"].lower()
            if any(k in last_user for k in ("plot", "chart", "graph", "visual", "draw", "show")):
                try:
                    fig = auto_plot_df(df)
                    st.pyplot(fig)
                    st.session_state["last_plot_bytes"] = fig_to_bytes(fig)
                except Exception as e:
                    st.warning(f"Auto-plot failed: {e}")

        # Exports
        if export_txt_btn.button("Download TXT"):
            if not st.session_state.get("chat_history"):
                st.warning("Nothing to export.")
            else:
                content = []
                for t in st.session_state.chat_history:
                    content.append(f"You: {t['user']}\nBot: {t['bot']}\n\n")
                b = "\n".join(content).encode()
                st.download_button("Click to download chat (.txt)", data=b, file_name="chat_history.txt")

        if export_pdf_btn.button("Download PDF"):
            if not st.session_state.get("chat_history"):
                st.warning("Nothing to export.")
            else:
                img_bytes = st.session_state.get("last_plot_bytes")
                pdf_io = create_pdf_from_chat_and_image(st.session_state.chat_history, image_bytes=img_bytes)
                st.download_button("Download PDF report", data=pdf_io, file_name="bi_report.pdf", mime="application/pdf")

with col2:
    st.markdown("## Quick actions & tips")
    st.markdown("- Use **'plot'** or **'show'** in a question to auto-generate charts.")
    st.markdown("- Try: *What is total sales by Region?*")
    st.markdown("- Try: *Plot Units Sold over Date.*")
    st.markdown("---")
    st.markdown("### Current session info")
    st.write(f"Rows loaded: **{len(df) if df is not None else 0}**")
    st.write(f"Columns: **{len(df.columns) if df is not None else 0}**")
    st.markdown("---")
    st.markdown("### Debug (for devs)")
    if st.checkbox("Show raw session state"):
        st.json({k: str(v)[:400] for k, v in st.session_state.items()})

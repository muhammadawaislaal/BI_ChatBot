# app.py — Single-file robust BI assistant (no external LLM required)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import textwrap
import re
from datetime import datetime
from typing import Optional, Tuple

# Page config
st.set_page_config(page_title="AI Data Analyst — Demo", page_icon="📊", layout="wide")

# ----- Utility helpers -----
def short_fmt(x):
    try:
        x = float(x)
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:.2f}M"
        if abs(x) >= 1_000:
            return f"{x/1_000:.1f}K"
        return f"{x:.2f}"
    except Exception:
        return str(x)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def fig_to_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def create_pdf_report(chat_history, chart_bytes: Optional[bytes]=None) -> bytes:
    pdf = FPDF(unit="pt", format="letter")
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.set_text_color(10, 30, 60)
    pdf.cell(0, 24, "AI Data Analyst — Report", ln=1)
    pdf.set_font("Helvetica", size=10)
    pdf.ln(4)
    for turn in chat_history:
        pdf.set_font("Helvetica", style="B", size=10)
        pdf.cell(0, 12, f"You: {turn['user']}", ln=1)
        pdf.set_font("Helvetica", size=10)
        wrapped = textwrap.wrap(turn["bot"], width=90)
        for line in wrapped:
            pdf.multi_cell(0, 12, line)
        pdf.ln(6)
    if chart_bytes:
        tmp_name = "/tmp/_chart.png"
        with open(tmp_name, "wb") as f:
            f.write(chart_bytes)
        # position chart
        pdf.image(tmp_name, x=50, y=pdf.get_y(), w=500)
    return pdf.output(dest="S").encode("latin1")

def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None

def try_parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    col = detect_date_column(df)
    if col:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass
    return df

# ----- Intent parser & executor (rule-based) -----
def parse_and_execute(query: str, df: pd.DataFrame) -> Tuple[str, Optional[pd.DataFrame], Optional[bytes]]:
    """
    Return: (answer_text, optional result dataframe, optional chart bytes)
    The function supports common BI intents:
      - total / sum by <column>
      - average / mean of <column>
      - top N by <column>
      - which day had highest/lowest <metric>
      - plot/chart: numeric trends (auto)
      - percentage change / growth between two periods (if date present)
      - count distinct / counts by category
    """
    q = query.lower().strip()
    df = try_parse_date_column(df.copy())
    date_col = detect_date_column(df)

    # shortcuts
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # 1) SUM / TOTAL by column
    m = re.search(r"(total|sum|aggregate|revenue|sales).+by ([\w\s]+)", q)
    if m:
        metric_words = m.group(1)
        group_col = m.group(2).strip()
        # find group column fuzzy
        group_col_found = None
        for c in df.columns:
            if group_col in c.lower():
                group_col_found = c
                break
        # choose numeric metric if exists
        if group_col_found:
            # if there's explicit metric in q like "total sales by region"
            metric_match = re.search(r"(total|sum|sales|revenue|amount|units|units sold)", q)
            if metric_match:
                # find candidate numeric col (Total Sales or Units Sold)
                cand = None
                for nc in numeric_cols:
                    if any(k in nc.lower() for k in ["total", "sales", "amount", "units"]):
                        cand = nc
                        break
                if cand is None and numeric_cols:
                    cand = numeric_cols[0]
            else:
                cand = numeric_cols[0] if numeric_cols else None

            if cand:
                res = df.groupby(group_col_found)[cand].sum().reset_index().sort_values(by=cand, ascending=False)
                text = f"Total **{cand}** by **{group_col_found}**:"
                return (text + "\n\n" + res.to_string(index=False), res, None)
        # fallback: sum all numeric columns
        totals = df[numeric_cols].sum().to_frame().T
        text = "Summed numeric columns:"
        return (text + "\n\n" + totals.to_string(index=False), totals, None)

    # 2) AVERAGE / MEAN
    m = re.search(r"(average|avg|mean).+of ([\w\s]+)", q)
    if m:
        col = m.group(2).strip()
        col_found = next((c for c in df.columns if col in c.lower()), None)
        if col_found and col_found in numeric_cols:
            val = df[col_found].mean()
            return (f"Average of **{col_found}** is **{short_fmt(val)}**.", None, None)
        # fallback: average of first numeric
        if numeric_cols:
            val = df[numeric_cols[0]].mean()
            return (f"Average of **{numeric_cols[0]}** is **{short_fmt(val)}**.", None, None)
        return ("No numeric column found to compute average.", None, None)

    # 3) TOP N
    m = re.search(r"top\s*(\d+)?\s*(\w+)\s*by\s*([\w\s]+)", q)
    if m:
        n = int(m.group(1)) if m.group(1) else 5
        metric = m.group(2)
        group = m.group(3).strip()
        # find metric column
        metric_col = next((c for c in df.columns if metric in c.lower()), None)
        group_col = next((c for c in df.columns if group in c.lower()), None)
        if not metric_col:
            metric_col = numeric_cols[0] if numeric_cols else None
        if group_col and metric_col:
            res = df.groupby(group_col)[metric_col].sum().reset_index().sort_values(by=metric_col, ascending=False).head(n)
            return (f"Top {n} {group_col} by {metric_col}:", res, None)

    # 4) DAY with HIGHEST / LOWEST metric
    if any(kw in q for kw in ["highest", "largest", "max", "peak", "top"]) and date_col and any(nc in q for nc in numeric_cols+["sales","units","total"]):
        # choose metric
        metric = None
        for nc in numeric_cols:
            if nc.lower() in q or any(k in nc.lower() for k in ["total", "sales", "units", "amount"]):
                metric = nc
                break
        if metric is None and numeric_cols:
            metric = numeric_cols[0]
        df_date = df.copy()
        try:
            df_date[date_col] = pd.to_datetime(df_date[date_col])
            grouped = df_date.groupby(date_col)[metric].sum().reset_index()
            best = grouped.loc[grouped[metric].idxmax()]
            text = f"Highest {metric} occurred on **{best[date_col].date()}** with value **{short_fmt(best[metric])}**."
            # also return grouped table & a chart
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(grouped[date_col], grouped[metric], marker='o')
            ax.set_title(f"{metric} by {date_col}")
            ax.set_xlabel(date_col)
            ax.set_ylabel(metric)
            plt.xticks(rotation=25)
            chart_bytes = fig_to_bytes(fig)
            return (text, grouped, chart_bytes)
        except Exception:
            pass

    # 5) PLOT / CHART requests - try to auto-plot sensible data
    if any(k in q for k in ["plot", "chart", "show", "visual", "draw"]):
        # if user mentions specific columns "plot Units Sold over Date" pattern
        m = re.search(r"plot\s+([\w\s]+)\s+(over|by)\s+([\w\s]+)", q)
        if m:
            y = m.group(1).strip()
            x = m.group(3).strip()
            y_col = next((c for c in df.columns if y in c.lower()), None)
            x_col = next((c for c in df.columns if x in c.lower()), None)
            if y_col and x_col:
                try:
                    fig, ax = plt.subplots(figsize=(8,4))
                    if pd.api.types.is_datetime64_any_dtype(df[x_col]) or "date" in x_col.lower():
                        df_local = df.copy()
                        df_local[x_col] = pd.to_datetime(df_local[x_col])
                        df_local.sort_values(x_col, inplace=True)
                        ax.plot(df_local[x_col], df_local[y_col], marker='o')
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"{y_col} over {x_col}")
                    else:
                        ax.plot(df[x_col], df[y_col], marker='o')
                        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                    chart_bytes = fig_to_bytes(fig)
                    return (f"Plotted **{y_col}** over **{x_col}**.", None, chart_bytes)
                except Exception as e:
                    return (f"Could not plot {y} over {x}: {e}", None, None)
        # generic: plot first numeric columns (trend)
        if numeric_cols:
            try:
                fig, ax = plt.subplots(figsize=(8,4))
                sample = df[numeric_cols].copy()
                sample.plot(ax=ax)
                ax.set_title("Numeric columns over index")
                chart_bytes = fig_to_bytes(fig)
                return ("Auto-generated chart of numeric columns.", None, chart_bytes)
            except Exception as e:
                return (f"Could not auto-plot: {e}", None, None)

    # 6) PERCENT CHANGE over time (simple first-to-last)
    if "growth" in q or "increase" in q or "percent" in q:
        if date_col and numeric_cols:
            metric = numeric_cols[0]
            df_local = df.copy()
            try:
                df_local[date_col] = pd.to_datetime(df_local[date_col])
                grouped = df_local.groupby(date_col)[metric].sum().reset_index().sort_values(date_col)
                if len(grouped) >= 2:
                    first = grouped.iloc[0][metric]
                    last = grouped.iloc[-1][metric]
                    pct = ((last - first) / first * 100) if first != 0 else None
                    text = f"{metric} changed from {short_fmt(first)} to {short_fmt(last)} ({pct:.2f}% )" if pct is not None else f"Change from {first} to {last}"
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.plot(grouped[date_col], grouped[metric], marker='o')
                    chart_bytes = fig_to_bytes(fig)
                    return (text, grouped, chart_bytes)
            except Exception:
                pass
        return ("Not enough date/metric info to compute growth.", None, None)

    # 7) COUNT / DISTINCT
    m = re.search(r"(count|how many).*(by|of)?\s*([\w\s]+)", q)
    if m:
        col = m.group(3).strip()
        col_found = next((c for c in df.columns if col in c.lower()), None)
        if col_found:
            counts = df[col_found].value_counts().reset_index()
            counts.columns = [col_found, "count"]
            return (f"Counts by **{col_found}**:", counts, None)

    # 8) fallback helpful summary
    # Provide basic summary stats and top 5 rows
    summary = df.describe(include='all').T
    top_rows = df.head(5)
    text = "Couldn't confidently parse a specific BI question. Here is a quick summary and the top 5 rows."
    text += "\n\nSummary:\n" + summary.to_string()
    return (text, top_rows, None)

# ----- Layout: professional header/footer/sidebar -----
# Header
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:16px">
      <div style="font-size:26px;font-weight:700;color:#0b3d91">📊 AI Data Analyst</div>
      <div style="color:#2b6cb0;font-size:14px">Instant insights from uploaded CSVs — portfolio demo</div>
    </div>
    <hr style="margin-top:6px;margin-bottom:18px">
    """,
    unsafe_allow_html=True
)

# Sidebar: instructions, usage tips, profit guide, tracker
with st.sidebar:
    st.markdown("### ▶️ How to use")
    st.markdown("""
    1. Upload a CSV with at least one numeric column (sales, units, price).  
    2. Ask plain-English questions (examples below).  
    3. The app auto-runs the best-fit operation and shows results + charts.  
    """)
    st.markdown("**Example questions:**")
    st.markdown(
        "- `Show total sales by Region`\n"
        "- `Which day had the highest Total Sales?`\n"
        "- `Plot Units Sold over Date`\n"
        "- `Top 3 Products by Units Sold`\n"
        "- `What is the average Unit Price?`"
    )
    st.markdown("---")
    st.markdown("### 💡 Profit & Action guide")
    st.markdown("""
    - Look for product/region with high sales but low margin — consider pricing.\n
    - Identify falling trends early — schedule promotions for those dates.\n
    - Use top-performing segments to expand marketing spend.\n
    """)
    st.markdown("---")
    st.markdown("### 📈 Session Tracker")
    if "tracker" not in st.session_state:
        st.session_state.tracker = []
    st.markdown(f"Queries this session: **{len(st.session_state.tracker)}**")
    if st.button("Clear Session Tracker"):
        st.session_state.tracker = []
        st.experimental_rerun()

# Main columns
col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="uploader")
    if uploaded is None:
        st.info("Upload a CSV to start — try the demo CSV from the sidebar examples.")
    else:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df):,} rows | {len(df.columns)} columns")
            st.dataframe(df.head(8))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # Query input
        user_q = st.text_input("Ask a question about your data", placeholder="e.g., Show total sales by Region")
        run_btn = st.button("Run")

        if run_btn and user_q.strip():
            with st.spinner("Processing your request..."):
                try:
                    answer_text, result_df, chart_bytes = parse_and_execute(user_q, df)
                    # display answer text
                    st.markdown("### ✅ Result")
                    # format answer_text nicely
                    st.markdown(answer_text.replace("\n", "  \n"))
                    # show dataframe result if present
                    if isinstance(result_df, pd.DataFrame):
                        st.markdown("#### Table")
                        st.dataframe(result_df)
                    # show chart if present
                    if chart_bytes:
                        st.image(chart_bytes)
                    # Update tracker
                    st.session_state.tracker.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": user_q,
                        "answer": answer_text,
                        "has_table": isinstance(result_df, pd.DataFrame),
                        "has_chart": chart_bytes is not None
                    })
                except Exception as e:
                    st.error(f"Error processing request: {e}")

        # Downloads & exports area
        st.markdown("---")
        st.markdown("### Export")
        if st.session_state.tracker:
            # download combined chat as txt
            if st.button("Download session TXT"):
                txt_lines = []
                for t in st.session_state.tracker:
                    txt_lines.append(f"[{t['time']}] Q: {t['query']}\nA: {t['answer']}\n")
                st.download_button("Download .txt", data="\n".join(txt_lines), file_name="session.txt")
            # download chart/report from last run if present
            last = st.session_state.tracker[-1]
            if last.get("has_chart") and 'chart_bytes' in locals() and chart_bytes:
                st.download_button("Download last chart (PNG)", data=chart_bytes, file_name="chart.png", mime="image/png")
            # PDF export
            if st.button("Download PDF report (entire session)"):
                # gather chat history in friendly form
                chat_history = [{"user": t["query"], "bot": t["answer"]} for t in st.session_state.tracker]
                pdf_bytes = create_pdf_report(chat_history, chart_bytes if 'chart_bytes' in locals() else None)
                st.download_button("Click to download PDF", data=pdf_bytes, file_name="bi_report.pdf", mime="application/pdf")

with col_right:
    st.markdown("### Quick Insights")
    if 'df' in locals():
        # Basic KPI cards
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            kpi1 = numeric_cols[0]
            kpi_sum = df[kpi1].sum()
            kpi_avg = df[kpi1].mean()
            st.metric(label=f"Σ {kpi1}", value=short_fmt(kpi_sum), delta=f"avg {short_fmt(kpi_avg)}")
        st.markdown("---")
        st.markdown("Detected columns")
        st.write(list(df.columns))
    else:
        st.info("Upload data to see quick insights here.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="display:flex;justify-content:space-between;color:#666;">
      <div>© AI Data Analyst — Demo (Portfolio)</div>
      <div>Built with Streamlit • Minimal, offline-capable BI assistant</div>
    </div>
    """,
    unsafe_allow_html=True
)

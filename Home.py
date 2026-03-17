import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import (
    DATA_DIR_DEFAULT,
    clean_css,
    list_scored_json_files,
    load_scored_json,
    build_segments_df,
    get_speakers,
    get_runtime_seconds,
    compute_summary,
    minutes_str,
)

st.set_page_config(
    page_title="Verity: A Digital Media Misinformation Detector",
    layout="wide",
)

st.markdown(clean_css(), unsafe_allow_html=True)


def load_current_selection():
    data_dir = st.session_state.get("DATA_DIR", DATA_DIR_DEFAULT)
    files = list_scored_json_files(data_dir)
    if not files:
        return None, None, None, [], None

    selected = st.session_state.get("selected_file")
    if selected not in files:
        selected = files[0]

    payload = load_scored_json(os.path.join(data_dir, selected))
    df = build_segments_df(payload)
    speakers = get_speakers(payload, df)
    runtime_sec = get_runtime_seconds(payload, df)
    summary = compute_summary(payload, df)
    return selected, payload, df, speakers, runtime_sec, summary


# Sidebar
with st.sidebar:
    st.markdown("### Verity")
    st.caption("Media Integrity Dashboard")

    st.divider()

    data_dir = DATA_DIR_DEFAULT

    files = list_scored_json_files(data_dir)
    if not files:
        st.warning("No scored files found.")
        st.stop()

    st.session_state["selected_file"] = st.selectbox(
        "Select a video",
        options=files,
        index=0 if st.session_state.get("selected_file") not in files else files.index(st.session_state["selected_file"]),
    )

# 🔥 ADD THIS LINE
selected = st.session_state["selected_file"]

# Now load data
selected, payload, df, speakers, runtime_sec, summary = load_current_selection()

# HOME PAGE

st.title("Verity: A Digital Media Misinformation Detector")
st.caption("Bias • Fakeness • Emotional Tone")

st.write("")

st.markdown(f"**Selected file:** `{selected}`")

st.write("")

def score_color(value):
    if value is None:
        return "#9CA3AF"  # gray
    if value < 0.25:
        return "#16A34A"  # green
    elif value < 0.75:
        return "#F59E0B"  # yellow
    else:
        return "#DC2626"  # red
    
# KPI Row
# -----------------------------
# KPI Row (Gauges)
# -----------------------------
k1, k2 = st.columns(2)

def score_color(value):
    if value is None:
        return "#9CA3AF"
    if value < 0.25:
        return "#16A34A"  # green
    elif value < 0.75:
        return "#F59E0B"  # yellow
    else:
        return "#DC2626"  # red


def make_gauge(title, value):

    v = float(value) if value is not None else 0.0
    v = max(0.0, min(1.0, v))

    num_color = score_color(value)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={
                "valueformat": ".2f",
                "font": {"size": 44, "color": num_color},
            },
            title={
                "text": title,
                "font": {"size": 20}
            },
            gauge={
                "axis": {
                    "range": [0, 1],
                    "tickmode": "array",
                    "tickvals": [0, 0.25, 0.5, 0.75, 1],
                    "ticktext": ["0", "0.25", "0.5", "0.75", "1"],
                },
                "bar": {
                    "color": "#111827",
                    "thickness": 0.18
                },
                "steps": [
                    {"range": [0.00, 0.25], "color": "#16A34A"},
                    {"range": [0.25, 0.75], "color": "#F59E0B"},
                    {"range": [0.75, 1.00], "color": "#DC2626"},
                ],
            },
        )
    )

    fig.update_layout(
        height=380,  # prevents cutoff
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


avg_bias = summary.get("avg_bias") if summary else None
avg_fake = summary.get("avg_fakeness") if summary else None


with k1:
    st.plotly_chart(make_gauge("Bias", avg_bias), use_container_width=True)

    with st.expander("What is Bias?"):
        st.markdown("""
**Bias (0–1)**  
Measures rhetorical framing and persuasive language.

**Scale Interpretation**
- 0.00–0.25 → Low framing  
- 0.25–0.75 → Moderate framing  
- 0.75–1.00 → Highly loaded language  

Bias does **not** indicate factual accuracy.
""")


with k2:
    st.plotly_chart(make_gauge("Fakeness", avg_fake), use_container_width=True)

    with st.expander("What is Fakeness?"):
        st.markdown("""
**Fakeness (0–1)**  
Represents likelihood that verified claims are false or misleading.

**Scale Interpretation**
- 0.00–0.25 → Likely accurate  
- 0.25–0.75 → Mixed / questionable  
- 0.75–1.00 → Likely false or contradicted  

Only applies when factual claims were verified.
""")
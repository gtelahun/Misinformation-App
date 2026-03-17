import os
import streamlit as st
import pandas as pd
import altair as alt

from utils import (
    DATA_DIR_DEFAULT,
    list_scored_json_files,
    load_scored_json,
    build_segments_df,
    get_speakers,
    per_speaker_summary,
    compute_summary,
    clean_css,
)

st.set_page_config(page_title="Overview — Verity", page_icon="📊", layout="wide")
st.markdown(clean_css(), unsafe_allow_html=True)

# -----------------------------
# Color Logic (Same as Home)
# -----------------------------
def score_color(value):
    if value is None:
        return "#9CA3AF"
    if value < 0.25:
        return "#16A34A"  # green
    elif value < 0.75:
        return "#F59E0B"  # yellow
    else:
        return "#DC2626"  # red

# --- Load Selection ---
data_dir = st.session_state.get("DATA_DIR", DATA_DIR_DEFAULT)
files = list_scored_json_files(data_dir)

if not files:
    st.warning("No *_scored.json files found in ./data/")
    st.stop()

selected = st.session_state.get("selected_file", files[0])
if selected not in files:
    selected = files[0]

payload = load_scored_json(os.path.join(data_dir, selected))
df = build_segments_df(payload)
speakers = get_speakers(payload, df)
summary = compute_summary(payload, df)

st.markdown("## Overview")
st.caption(f"Selected: {selected}")

# --- KPI Row ---
k1, k2 = st.columns(2)

bias_val = summary["avg_bias"]
fake_val = summary["avg_fakeness"]

bias_color = score_color(bias_val)
fake_color = score_color(fake_val)

k1.markdown(
    f"""
    <div class="signal-card">
        <div style="font-size:0.9rem;color:#6B7280;">Bias</div>
        <div style="font-size:2.5rem;font-weight:700;color:{bias_color};">
            {bias_val:.2f}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k2.markdown(
    f"""
    <div class="signal-card">
        <div style="font-size:0.9rem;color:#6B7280;">Fakeness</div>
        <div style="font-size:2.5rem;font-weight:700;color:{fake_color};">
            {fake_val:.2f}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.divider()

# --- Speaker Summary Table ---
st.markdown("### Speaker Summary")

speaker_df = per_speaker_summary(payload, df, speakers)

speaker_df = speaker_df.rename(columns={
    "speaker": "Speaker",
    "avg_bias": "Bias",
    "avg_fakeness": "Fakeness"
})

speaker_df = speaker_df[["Speaker", "Bias", "Fakeness"]]

speaker_df["Bias"] = speaker_df["Bias"].round(2)
speaker_df["Fakeness"] = speaker_df["Fakeness"].round(2)

# Apply color styling
def highlight_cell(val):
    if val is None:
        return ""

    if val < 0.25:
        return "background-color: #ECFDF5; color: #065F46; font-weight:600;"
    elif val < 0.75:
        return "background-color: #FFFBEB; color: #92400E; font-weight:600;"
    else:
        return "background-color: #FEF2F2; color: #7F1D1D; font-weight:600;"

styled_df = speaker_df.style.applymap(highlight_cell, subset=["Bias", "Fakeness"])

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True
)

st.write("")
st.divider()

# --- Bias Distribution Boxplot ---
st.markdown("### Bias Distribution by Speaker")

boxplot = (
    alt.Chart(df)
    .mark_boxplot(
        extent="min-max",
        size=90
    )
    .encode(
        x=alt.X(
            "speaker:N",
            title="Speaker",
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "bias_score:Q",
            title="Bias",
            scale=alt.Scale(domain=[0, 1])
        ),
        color=alt.value("#1f4e79"),
        tooltip=[
            alt.Tooltip("speaker:N", title="Speaker"),
            alt.Tooltip("bias_score:Q", format=".2f", title="Bias")
        ]
    )
    .properties(height=440)
    .configure_axis(gridColor="#E5E7EB")
)

st.altair_chart(boxplot, use_container_width=True)

st.write("")

with st.expander("What This Shows"):
    st.markdown("""
This boxplot visualizes the distribution of **bias scores** for each speaker across all segments.

Each box represents:

- **Median bias** (center line)  
- **Interquartile range (IQR)** — middle 50% of segments  
- **Minimum and maximum values**

Wider boxes or longer whiskers indicate greater variability in rhetorical framing.

This helps identify whether a speaker:

- Maintains consistent framing  
- Occasionally uses highly loaded language  
- Exhibits wide swings in rhetorical intensity  
""")
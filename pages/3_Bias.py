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
    clean_css,
)

st.set_page_config(page_title="Bias")
st.markdown(clean_css(), unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
data_dir = st.session_state.get("DATA_DIR", DATA_DIR_DEFAULT)
files = list_scored_json_files(data_dir)

if not files:
    st.warning("No *_scored.json files found.")
    st.stop()

selected = st.session_state.get("selected_file", files[0])
if selected not in files:
    selected = files[0]

payload = load_scored_json(os.path.join(data_dir, selected))
df = build_segments_df(payload)
speakers = get_speakers(payload, df)

st.markdown("### Bias")
st.caption("Visualizes bias intensity over time. Select a segment below to explore details.")

if df.empty:
    st.info("No segments available.")
    st.stop()

# -----------------------------
# Explanation
# -----------------------------
with st.expander("How bias is interpreted"):
    st.markdown("""
- **Low Bias** → neutral / factual  
- **Moderate Bias** → some framing or opinion  
- **High Bias** → strong framing or persuasive language  

Color intensity reflects bias score.
""")

# -----------------------------
# Speaker Filter
# -----------------------------
speaker_choice = st.selectbox(
    "Speaker",
    options=["All"] + speakers,
    index=0
)

# Reset slider on speaker change
if "prev_speaker_bias" not in st.session_state:
    st.session_state["prev_speaker_bias"] = speaker_choice

if st.session_state["prev_speaker_bias"] != speaker_choice:
    st.session_state["selected_segment_idx_bias"] = 0
    st.session_state["prev_speaker_bias"] = speaker_choice

# -----------------------------
# Filtering
# -----------------------------
work = df.copy()

if speaker_choice != "All":
    work = work[work["speaker"] == speaker_choice]

work = work.dropna(subset=["start_min", "end_min", "bias_score"])

if work.empty:
    st.warning("No data after filters.")
    st.stop()

# -----------------------------
# Prepare Data
# -----------------------------
bias_df = work.sort_values("start_min").reset_index(drop=True)
bias_df["segment_id"] = bias_df.index + 1

# -----------------------------
# Slider
# -----------------------------
st.write("")

if "selected_segment_idx_bias" not in st.session_state:
    st.session_state["selected_segment_idx_bias"] = 0

n_segments = len(bias_df)

st.session_state["selected_segment_idx_bias"] = min(
    st.session_state["selected_segment_idx_bias"], n_segments - 1
)

slider_val = st.slider(
    "Segment (timeline)",
    min_value=1,
    max_value=n_segments,
    value=st.session_state["selected_segment_idx_bias"] + 1,
    key="bias_slider"
)

selected_idx = slider_val - 1
st.session_state["selected_segment_idx_bias"] = selected_idx

selected_row = bias_df.iloc[selected_idx]
selected_df = bias_df.iloc[[selected_idx]]

st.caption(
    f"Segment {selected_idx + 1} of {n_segments} • "
    f"{selected_row['speaker']} • "
    f"{selected_row['start_min']:.2f}-{selected_row['end_min']:.2f} min"
)

# -----------------------------
# Heatmap
# -----------------------------
base = (
    alt.Chart(bias_df)
    .mark_rect(stroke="white", strokeWidth=2)
    .encode(
        x=alt.X("start_min:Q", title="Time (minutes)"),
        x2="end_min:Q",
        y=alt.Y(
            "speaker:N",
            title="Speaker",
            scale=alt.Scale(paddingInner=0.7)
        ),
        color=alt.Color(
            "bias_score:Q",
            scale=alt.Scale(
                domain=[0, 1],
                range=["#2563EB", "#7C3AED", "#DC2626"]
            ),
            title="Bias"
        ),
        tooltip=[
            alt.Tooltip("segment_id:Q", title="Segment"),
            alt.Tooltip("speaker:N", title="Speaker"),
            alt.Tooltip("start_min:Q", title="Start", format=".2f"),
            alt.Tooltip("end_min:Q", title="End", format=".2f"),
            alt.Tooltip("bias_score:Q", title="Bias", format=".2f"),
        ],
    )
)

highlight = (
    alt.Chart(selected_df)
    .mark_rect(
        fillOpacity=0,
        stroke="black",
        strokeWidth=4
    )
    .encode(
        x="start_min:Q",
        x2="end_min:Q",
        y="speaker:N"
    )
)

heatmap = (
    (base + highlight)
    .properties(height=140 + 80 * len(bias_df["speaker"].unique()))
    .configure_axis(gridColor="#E5E7EB", gridOpacity=0.4)
)

st.altair_chart(heatmap, use_container_width=True)

# -----------------------------
# Selected Segment Details
# -----------------------------
st.write("")
st.markdown("### Selected Segment Details")

start = selected_row.get("start_min")
end = selected_row.get("end_min")
time_str = f"{start:.2f}–{end:.2f} min"

st.markdown(
    f"""
    <div class="signal-card">
      <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
        <div><b>Speaker:</b> {selected_row.get("speaker")}</div>
        <div><b>Time:</b> {time_str}</div>
        <div><b>Bias Score:</b> {selected_row.get("bias_score"):.2f}</div>
      </div>

      <div style="height:0.6rem;"></div>

      <div><b>Segment:</b> {selected_row.get("text", "—")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
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

st.set_page_config(page_title="Fakeness")
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

st.markdown("### Fakeness")
st.caption("Visualizes verified claims and their truth risk over time.")

if df.empty:
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def fmt_sources(src_list):
    if not isinstance(src_list, list) or len(src_list) == 0:
        return "—"
    parts = []
    for s in src_list[:5]:
        title = s.get("title", "source")
        url = s.get("url", "")
        parts.append(f"- {title} ({url})" if url else f"- {title}")
    return "\n".join(parts)

def safe_text(row):
    claim = row.get("claim_text")
    if isinstance(claim, str) and claim.strip():
        return claim.strip()
    txt = row.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    return "—"

# -----------------------------
# Speaker Filter
# -----------------------------
speaker_choice = st.selectbox(
    "Speaker",
    options=["All"] + speakers,
    index=0
)

# reset slider when speaker changes
if "prev_speaker_fake" not in st.session_state:
    st.session_state["prev_speaker_fake"] = speaker_choice

if st.session_state["prev_speaker_fake"] != speaker_choice:
    st.session_state["selected_segment_idx_fake"] = 0
    st.session_state["prev_speaker_fake"] = speaker_choice

# -----------------------------
# Filter only segments WITH fakeness
# -----------------------------
work = df[df["fakeness_score"].notna()].copy()

if speaker_choice != "All":
    work = work[work["speaker"] == speaker_choice]

if work.empty:
    st.warning("No verified claims found.")
    st.stop()

# sort + segment id
fake_df = work.sort_values("start_min").reset_index(drop=True)
fake_df["segment_id"] = fake_df.index + 1

# -----------------------------
# Slider
# -----------------------------
st.write("")

if "selected_segment_idx_fake" not in st.session_state:
    st.session_state["selected_segment_idx_fake"] = 0

n_segments = len(fake_df)

st.session_state["selected_segment_idx_fake"] = min(
    st.session_state["selected_segment_idx_fake"], n_segments - 1
)

slider_val = st.slider(
    "Segment (timeline)",
    min_value=1,
    max_value=n_segments,
    value=st.session_state["selected_segment_idx_fake"] + 1,
    key="fake_slider"
)

selected_idx = slider_val - 1
st.session_state["selected_segment_idx_fake"] = selected_idx

selected_row = fake_df.iloc[selected_idx]
selected_df = fake_df.iloc[[selected_idx]]

st.caption(
    f"Segment {selected_idx + 1} of {n_segments} • "
    f"{selected_row['speaker']} • "
    f"{selected_row['start_min']:.2f}-{selected_row['end_min']:.2f} min"
)

# -----------------------------
# Heatmap
# -----------------------------
base = (
    alt.Chart(fake_df)
    .mark_rect(stroke="white", strokeWidth=2)
    .encode(
        x="start_min:Q",
        x2="end_min:Q",
        y=alt.Y("speaker:N", scale=alt.Scale(paddingInner=0.7)),
        color=alt.Color(
            "fakeness_score:Q",
            scale=alt.Scale(
                domain=[0, 1],
                range=["#10B981", "#F97316", "#EF4444"]
            ),
            title="Truth Risk"
        ),
        tooltip=[
            alt.Tooltip("segment_id:Q", title="Segment"),
            alt.Tooltip("speaker:N"),
            alt.Tooltip("fakeness_score:Q", format=".2f", title="Truth Risk"),
        ]
    )
)

highlight = (
    alt.Chart(selected_df)
    .mark_rect(fillOpacity=0, stroke="black", strokeWidth=4)
    .encode(
        x="start_min:Q",
        x2="end_min:Q",
        y="speaker:N"
    )
)

chart = (base + highlight).properties(
    height=140 + 80 * len(fake_df["speaker"].unique())
)

st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Selected Claim Details
# -----------------------------
st.write("")
st.markdown("### Selected Claim Details")

start = selected_row.get("start_min")
end = selected_row.get("end_min")
time_str = f"{start:.2f}–{end:.2f} min"

st.markdown(
    f"""
    <div class="signal-card">
      <div style="display:flex; justify-content:space-between;">
        <div><b>Speaker:</b> {selected_row.get("speaker")}</div>
        <div><b>Time:</b> {time_str}</div>
        <div><b>Truth Risk:</b> {selected_row.get("fakeness_score"):.2f}</div>
      </div>

      <div style="height:0.6rem;"></div>

      <div><b>Claim:</b> {safe_text(selected_row)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Transcript
# -----------------------------
transcript = selected_row.get("text")
excerpt = (
    transcript[:400] + "..."
    if isinstance(transcript, str) and len(transcript) > 400
    else transcript
)

with st.expander("View extracted transcript segment"):
    st.markdown(f'"{excerpt if excerpt else "—"}"')

# -----------------------------
# Sources (IMPORTANT)
# -----------------------------
with st.expander("Sources / evidence", expanded=True):
    st.markdown(fmt_sources(selected_row.get("sources")))
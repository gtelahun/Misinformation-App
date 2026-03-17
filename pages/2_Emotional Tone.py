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

st.set_page_config(page_title="Emotional Tone")
st.markdown(clean_css(), unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
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

st.markdown("### Emotional Tone")
st.caption("Visualizes emotional framing over time. Select a segment below to explore details.")

if df.empty:
    st.info("No segments available.")
    st.stop()

# -----------------------------
# Explanation
# -----------------------------
with st.expander("How emotion is grouped"):
    st.markdown("""
- **Positive** → supportive / optimistic  
- **Neutral** → informational  
- **Negative** → critical / emotional  

Each block represents a segment of speech.
""")

# -----------------------------
# Speaker Filter
# -----------------------------
speaker_choice = st.selectbox(
    "Speaker",
    options=["All"] + speakers,
    index=0
)

# make a working copy and filter FIRST
work = df.copy()

if speaker_choice != "All":
    work = work[work["speaker"] == speaker_choice]

# drop incomplete rows (must happen after filter)
work = work.dropna(subset=["start_min", "end_min", "emotion_tone"])

if work.empty:
    st.warning("No data after filters.")
    st.stop()

# -----------------------------
# Emotion grouping function
# -----------------------------
def group_emotion(e):
    # defensive: handle non-string
    e = str(e).lower()

    if e in ["enthusiastic", "optimistic", "supportive", "positive"]:
        return "Positive"
    elif e in ["angry", "frustration", "fear", "defensive", "critical", "sadness", "confrontational"]:
        return "Negative"
    else:
        return "Neutral"

# -----------------------------
# Build emotion_df from filtered work
# -----------------------------
work["emotion_group"] = work["emotion_tone"].apply(group_emotion)

# sort and assign segment ids AFTER filtering (this is the important fix)
emotion_df = work.sort_values("start_min").reset_index(drop=True)
emotion_df["segment_id"] = emotion_df.index + 1  # 1-based id for UI

# -----------------------------
# Slider Selection (human-friendly index)
# -----------------------------
st.write("")  # spacing

# ensure session state key exists
if "selected_segment_idx" not in st.session_state:
    st.session_state["selected_segment_idx"] = 0

n_segments = len(emotion_df)
if n_segments == 0:
    st.info("No emotion segments available.")
    st.stop()

# clamp stored index to valid range (important when switching speakers/files)
st.session_state["selected_segment_idx"] = min(max(0, st.session_state["selected_segment_idx"]), n_segments - 1)

# slider uses 1..n for human-friendly numbering
slider_val = st.slider(
    "Segment (timeline)",
    min_value=1,
    max_value=n_segments,
    value=st.session_state["selected_segment_idx"] + 1,
)

# convert back to 0-based index and store
selected_idx = int(slider_val) - 1
st.session_state["selected_segment_idx"] = selected_idx

# fetch selected row (safe because we've clamped above)
selected_row = emotion_df.iloc[selected_idx]
selected_df = emotion_df.iloc[[selected_idx]]  # small df for highlight

st.caption(
    f"Segment {selected_idx + 1} of {n_segments} • "
    f"{selected_row['speaker']} • "
    f"{selected_row['start_min']:.2f}-{selected_row['end_min']:.2f} min"
)

# -----------------------------
# Emotion Heatmap + Highlight
# -----------------------------
base = (
    alt.Chart(emotion_df)
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
            "emotion_group:N",
            scale=alt.Scale(
                domain=["Positive", "Neutral", "Negative"],
                range=["#16A34A", "#F59E0B", "#DC2626"]
            ),
            title="Emotion"
        ),
        tooltip=[
            alt.Tooltip("segment_id:Q", title="Segment"),
            alt.Tooltip("speaker:N", title="Speaker"),
            alt.Tooltip("start_min:Q", title="Start", format=".2f"),
            alt.Tooltip("end_min:Q", title="End", format=".2f"),
            alt.Tooltip("emotion_group:N", title="Emotion"),
        ],
    )
)

# highlight is an outline rect over the selected segment
highlight = (
    alt.Chart(selected_df)
    .mark_rect(fillOpacity=0, stroke="black", strokeWidth=4)
    .encode(
        x="start_min:Q",
        x2="end_min:Q",
        y=alt.Y("speaker:N", scale=alt.Scale(paddingInner=0.7))
    )
)

heatmap = (
    (base + highlight)
    .properties(height=140 + 80 * len(emotion_df["speaker"].unique()))
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

bias_val = selected_row.get("bias_score")
fake_val = selected_row.get("fakeness_score")

bias_str = f"{bias_val:.2f}" if pd.notna(bias_val) else "—"
fake_str = f"{fake_val:.2f}" if pd.notna(fake_val) else "—"

st.markdown(
    f"""
    <div class="signal-card">
      <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
        <div><b>Speaker:</b> {selected_row.get("speaker")}</div>
        <div><b>Time:</b> {time_str}</div>
        <div><b>Emotion:</b> {selected_row.get("emotion_group")}</div>
      </div>

      <div style="height:0.6rem;"></div>

      <div><b>Segment:</b> {selected_row.get("text", "—")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
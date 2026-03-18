import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

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
# Load data
# -----------------------------
data_dir = st.session_state.get("DATA_DIR", DATA_DIR_DEFAULT)
files = list_scored_json_files(data_dir)

if not files:
    st.warning("No *_scored.json files found in ./data/")
    st.stop()

selected_file = st.session_state.get("selected_file", files[0])
if selected_file not in files:
    selected_file = files[0]

payload = load_scored_json(os.path.join(data_dir, selected_file))
df = build_segments_df(payload)
speakers = get_speakers(payload, df)

st.markdown("### Emotional Tone")
st.caption("Click a segment to explore details.")

if df.empty:
    st.info("No segments available.")
    st.stop()

# -----------------------------
# Explanation
# -----------------------------
with st.expander("How emotion is grouped"):
    st.markdown(
        """
- **Positive** → supportive / optimistic  
- **Neutral** → informational  
- **Negative** → critical / emotional  

Each block represents a segment of speech.
"""
    )

# -----------------------------
# Speaker filter
# -----------------------------
speaker_choice = st.selectbox(
    "Speaker",
    options=["All"] + speakers,
    index=0,
    key="emotion_speaker_choice",
)

# -----------------------------
# Build filtered working df
# -----------------------------
work = df.copy()

if speaker_choice != "All":
    work = work[work["speaker"] == speaker_choice].copy()

work = work.dropna(subset=["start_min", "end_min", "emotion_tone"]).copy()

if work.empty:
    st.warning("No data after filters.")
    st.stop()

# -----------------------------
# Emotion grouping
# -----------------------------
def group_emotion(e):
    e = str(e).strip().lower()

    if e in ["enthusiastic", "optimistic", "supportive", "positive"]:
        return "Positive"
    elif e in [
        "angry",
        "frustration",
        "fear",
        "defensive",
        "critical",
        "sadness",
        "confrontational",
        "negative",
    ]:
        return "Negative"
    else:
        return "Neutral"

work["emotion_group"] = work["emotion_tone"].apply(group_emotion)

# -----------------------------
# Sort + stable ids
# -----------------------------
emotion_df = work.sort_values(["start_min", "end_min"]).reset_index(drop=True).copy()
emotion_df["segment_id"] = emotion_df.index + 1
emotion_df["duration"] = emotion_df["end_min"] - emotion_df["start_min"]

# remove any zero/negative durations just in case
emotion_df = emotion_df[emotion_df["duration"] > 0].reset_index(drop=True)
emotion_df["segment_id"] = emotion_df.index + 1

if emotion_df.empty:
    st.warning("No valid emotion segments available.")
    st.stop()

n_segments = len(emotion_df)

# -----------------------------
# Reset selection when file/filter changes
# -----------------------------
selection_context = (selected_file, speaker_choice, n_segments)
if st.session_state.get("emotion_selection_context") != selection_context:
    st.session_state["selected_segment_idx"] = 0
    st.session_state["emotion_selection_context"] = selection_context

if "selected_segment_idx" not in st.session_state:
    st.session_state["selected_segment_idx"] = 0

st.session_state["selected_segment_idx"] = min(
    max(0, st.session_state["selected_segment_idx"]),
    n_segments - 1,
)

selected_idx = st.session_state["selected_segment_idx"]

# -----------------------------
# Colors + trace order
# -----------------------------
color_map = {
    "Positive": "#16A34A",
    "Neutral": "#F59E0B",
    "Negative": "#DC2626",
}
trace_order = ["Positive", "Neutral", "Negative"]

# Keep speaker order consistent and match original feel
speaker_order = list(emotion_df["speaker"].drop_duplicates())
speaker_order = sorted(speaker_order)

# -----------------------------
# Build chart
# -----------------------------
fig = go.Figure()

# We store a reliable mapping from each trace point to the global segment index
trace_point_to_global_idx = {}

for emotion in trace_order:
    group = emotion_df[emotion_df["emotion_group"] == emotion].copy()
    if group.empty:
        continue

    group = group.sort_values(["start_min", "end_min"]).reset_index(drop=True)

    # map point number within this trace -> global dataframe index
    point_map = {}
    for point_num, (_, row) in enumerate(group.iterrows()):
        global_idx = int(row["segment_id"]) - 1
        point_map[point_num] = global_idx
    trace_point_to_global_idx[emotion] = point_map

    fig.add_trace(
        go.Bar(
            x=group["duration"],
            y=group["speaker"],
            base=group["start_min"],
            orientation="h",
            name=emotion,
            marker=dict(
                color=color_map[emotion],
                line=dict(color="white", width=2),
            ),
            customdata=group[["segment_id", "start_min", "end_min", "speaker"]].values,
            hovertemplate=(
                "Segment %{customdata[0]}<br>"
                "Speaker: %{customdata[3]}<br>"
                "Time: %{customdata[1]:.2f} - %{customdata[2]:.2f} min<br>"
                f"{emotion}<extra></extra>"
            ),
            showlegend=True,
        )
    )

# Selected segment
selected_row = emotion_df.iloc[selected_idx]

# Highlight selected segment with outline
fig.add_shape(
    type="rect",
    x0=float(selected_row["start_min"]),
    x1=float(selected_row["end_min"]),
    y0=selected_row["speaker"],
    y1=selected_row["speaker"],
    xref="x",
    yref="y",
    line=dict(color="black", width=4),
    fillcolor="rgba(0,0,0,0)",
    layer="above",
)

# Axis + layout styling
fig.update_layout(
    barmode="overlay",
    height=140 + 80 * len(speaker_order),
    xaxis_title="Time (minutes)",
    yaxis_title="Speaker",
    legend_title="Emotion",
    plot_bgcolor="white",
    paper_bgcolor="white",
    clickmode="event+select",
    margin=dict(l=40, r=20, t=20, b=70),
)

# -----------------------------
fig = go.Figure()

# We store a reliable mapping from each trace point to the global segment index
trace_point_to_global_idx = {}

for emotion in trace_order:
    group = emotion_df[emotion_df["emotion_group"] == emotion].copy()
    if group.empty:
        continue

    group = group.sort_values(["start_min", "end_min"]).reset_index(drop=True)

    # map point number within this trace -> global dataframe index
    point_map = {}
    for point_num, (_, row) in enumerate(group.iterrows()):
        global_idx = int(row["segment_id"]) - 1
        point_map[point_num] = global_idx
    trace_point_to_global_idx[emotion] = point_map

    fig.add_trace(
        go.Bar(
            x=group["duration"],
            y=group["speaker"],
            base=group["start_min"],
            orientation="h",
            name=emotion,
            marker=dict(
                color=color_map[emotion],
                line=dict(color="white", width=2),
            ),
            customdata=group[["segment_id", "start_min", "end_min", "speaker"]].values,
            hovertemplate=(
                "Segment %{customdata[0]}<br>"
                "Speaker: %{customdata[3]}<br>"
                "Time: %{customdata[1]:.2f} - %{customdata[2]:.2f} min<br>"
                f"{emotion}<extra></extra>"
            ),
            showlegend=True,
        )
    )
fig.update_xaxes(
    range=[0, emotion_df["end_min"].max()],
    showgrid=True,
    gridcolor="#E5E7EB",
    zeroline=False,
)

fig.update_yaxes(
    categoryorder="array",
    categoryarray=speaker_order,
)

# -----------------------------
# Click handling
# IMPORTANT: do not also call st.plotly_chart(fig)
# -----------------------------
selected_points = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_height=140 + 80 * len(speaker_order),
    key="emotion_plot",
)

# Robust click parsing
if selected_points:
    pt = selected_points[0]

    new_idx = None

    # First try customdata if available
    if "customdata" in pt and pt["customdata"]:
        try:
            clicked_segment_id = int(pt["customdata"][0])
            new_idx = clicked_segment_id - 1
        except Exception:
            new_idx = None

    # Fallback: use curveNumber + pointNumber
    if new_idx is None and "curveNumber" in pt and "pointNumber" in pt:
        try:
            curve_num = int(pt["curveNumber"])
            point_num = int(pt["pointNumber"])
            trace_name = fig.data[curve_num].name
            new_idx = trace_point_to_global_idx[trace_name][point_num]
        except Exception:
            new_idx = None

    if new_idx is not None:
        new_idx = min(max(0, new_idx), n_segments - 1)
        if new_idx != st.session_state["selected_segment_idx"]:
            st.session_state["selected_segment_idx"] = new_idx
            st.rerun()

# -----------------------------
# Re-read selected row after possible click/rerun
# -----------------------------
selected_idx = min(max(0, st.session_state["selected_segment_idx"]), n_segments - 1)
selected_row = emotion_df.iloc[selected_idx]

# -----------------------------
# Selected Segment Details
# -----------------------------
st.markdown("### Selected Segment Details")

# Safe values
start = selected_row.get("start_min") or 0
end = selected_row.get("end_min") or 0
time_str = f"{start:.2f}-{end:.2f} min"

emotion = selected_row.get("emotion_group", "Neutral")

emotion_color = {
    "Positive": "#16A34A",
    "Neutral": "#F59E0B",
    "Negative": "#DC2626"
}.get(emotion, "#6B7280")

html = f"""<div class="signal-card">

<div style="text-align:center; margin-bottom:10px;">
    <div style="
        display:inline-block;
        background:{emotion_color};
        color:white;
        padding:8px 18px;
        border-radius:999px;
        font-weight:700;
        font-size:1.1rem;
    ">
        {emotion}
    </div>
</div>

<div style="text-align:center; font-size:0.95rem; color:#6B7280; margin-bottom:12px;">
    {selected_row.get("speaker", "—")} • {time_str}
</div>

<div style="line-height:1.7;">
    <div style="font-weight:600; margin-bottom:4px;">Segment</div>
    {selected_row.get("text", "—")}
</div>

</div>"""

st.markdown(html, unsafe_allow_html=True)
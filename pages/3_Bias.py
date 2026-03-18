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

st.set_page_config(page_title="Bias")
st.markdown(clean_css(), unsafe_allow_html=True)

# -----------------------------
# Load data
# -----------------------------
data_dir = st.session_state.get("DATA_DIR", DATA_DIR_DEFAULT)
files = list_scored_json_files(data_dir)

if not files:
    st.warning("No *_scored.json files found.")
    st.stop()

if "selected_file" not in st.session_state or st.session_state["selected_file"] not in files:
    st.session_state["selected_file"] = files[0]

selected_file = st.session_state["selected_file"]

payload = load_scored_json(os.path.join(data_dir, selected_file))
df = build_segments_df(payload)
speakers = get_speakers(payload, df)

st.markdown("### Bias")
st.caption("Click a segment to explore details.")

if df.empty:
    st.info("No segments available.")
    st.stop()

# -----------------------------
# Explanation
# -----------------------------
with st.expander("How bias is interpreted"):
    st.markdown(
        """
- **Low Bias** → neutral / factual  
- **Moderate Bias** → some framing or opinion  
- **High Bias** → strong framing or persuasive language  

Color intensity reflects bias score.
"""
    )

# -----------------------------
# Speaker filter
# -----------------------------
speaker_choice = st.selectbox(
    "Speaker",
    options=["All"] + speakers,
    index=0,
    key="bias_speaker_choice",
)

# -----------------------------
# Build filtered working df
# -----------------------------
work = df.copy()

if speaker_choice != "All":
    work = work[work["speaker"] == speaker_choice].copy()

work = work.dropna(subset=["start_min", "end_min", "bias_score"]).copy()

if work.empty:
    st.warning("No data after filters.")
    st.stop()

# -----------------------------
# Sort + stable ids
# -----------------------------
bias_df = work.sort_values(["start_min", "end_min"]).reset_index(drop=True).copy()
bias_df["segment_id"] = bias_df.index + 1
bias_df["duration"] = bias_df["end_min"] - bias_df["start_min"]

bias_df = bias_df[bias_df["duration"] > 0].reset_index(drop=True)
bias_df["segment_id"] = bias_df.index + 1

if bias_df.empty:
    st.warning("No valid bias segments available.")
    st.stop()

n_segments = len(bias_df)

# -----------------------------
# Reset selection when file/filter changes
# -----------------------------
selection_context = (selected_file, speaker_choice, n_segments)
if st.session_state.get("bias_selection_context") != selection_context:
    st.session_state["selected_segment_idx_bias"] = 0
    st.session_state["bias_selection_context"] = selection_context

if "selected_segment_idx_bias" not in st.session_state:
    st.session_state["selected_segment_idx_bias"] = 0

st.session_state["selected_segment_idx_bias"] = min(
    max(0, st.session_state["selected_segment_idx_bias"]),
    n_segments - 1,
)

selected_idx = st.session_state["selected_segment_idx_bias"]

# -----------------------------
# Bias helpers
# -----------------------------
def bias_to_color(val):
    if pd.isna(val):
        return "#E5E7EB"
    if val < 0.3:
        return "#2563EB"   # low
    elif val < 0.6:
        return "#7C3AED"   # moderate
    else:
        return "#DC2626"   # high

def bias_label(val):
    if pd.isna(val):
        return "Unknown"
    if val < 0.3:
        return "Low"
    elif val < 0.6:
        return "Moderate"
    else:
        return "High"

def bias_bin(val):
    if pd.isna(val):
        return "unknown"
    if val < 0.3:
        return "low"
    elif val < 0.6:
        return "mid"
    else:
        return "high"

bias_df["bias_bin"] = bias_df["bias_score"].apply(bias_bin)

# Keep speaker order consistent and match emotion page feel
speaker_order = list(bias_df["speaker"].drop_duplicates())
speaker_order = sorted(speaker_order)

# -----------------------------
# Build chart (same structure as emotion)
# -----------------------------
fig = go.Figure()

trace_point_to_global_idx = {}

for b in ["low", "mid", "high"]:
    group = bias_df[bias_df["bias_bin"] == b].copy()
    if group.empty:
        continue

    group = group.sort_values(["start_min", "end_min"]).reset_index(drop=True)

    point_map = {}
    for point_num, (_, row) in enumerate(group.iterrows()):
        global_idx = int(row["segment_id"]) - 1
        point_map[point_num] = global_idx
    trace_point_to_global_idx[b] = point_map

    fig.add_trace(
        go.Bar(
            x=group["duration"],
            y=group["speaker"],
            base=group["start_min"],
            orientation="h",
            name=b,
            marker=dict(
                color=group["bias_score"].apply(bias_to_color),
                line=dict(color="white", width=2),
            ),
            customdata=group[
                ["segment_id", "start_min", "end_min", "speaker", "bias_score"]
            ].values,
            hovertemplate=(
                "Segment %{customdata[0]}<br>"
                "Speaker: %{customdata[3]}<br>"
                "Time: %{customdata[1]:.2f} - %{customdata[2]:.2f} min<br>"
                "Bias: %{customdata[4]:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

# Selected segment
selected_row = bias_df.iloc[selected_idx]

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
    plot_bgcolor="white",
    paper_bgcolor="white",
    clickmode="event+select",
    margin=dict(l=40, r=20, t=20, b=70),
)

fig.update_xaxes(
    range=[0, bias_df["end_min"].max()],
    constrain="domain"
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="#E5E7EB",
    zeroline=False,
)

fig.update_yaxes(
    categoryorder="array",
    categoryarray=speaker_order,
)

# -----------------------------
# Custom bias legend
# -----------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <div style="
            width:220px;
            height:12px;
            background: linear-gradient(to right, #2563EB, #7C3AED, #DC2626);
            border-radius:999px;
        "></div>
        <div style="font-size:0.85rem; color:#6B7280;">
            Low Bias → Moderate Bias → High Bias
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Click handling
# IMPORTANT: do not call st.plotly_chart(fig)
# -----------------------------
selected_points = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_height=140 + 80 * len(speaker_order),
    key="bias_plot",
)

if selected_points:
    pt = selected_points[0]
    new_idx = None

    if "customdata" in pt and pt["customdata"]:
        try:
            clicked_segment_id = int(pt["customdata"][0])
            new_idx = clicked_segment_id - 1
        except Exception:
            new_idx = None

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
        if new_idx != st.session_state["selected_segment_idx_bias"]:
            st.session_state["selected_segment_idx_bias"] = new_idx
            st.rerun()

# -----------------------------
# Re-read selected row after possible click/rerun
# -----------------------------
selected_idx = min(max(0, st.session_state["selected_segment_idx_bias"]), n_segments - 1)
selected_row = bias_df.iloc[selected_idx]

# -----------------------------
# Selected Segment Details
# -----------------------------
st.markdown("### Selected Segment Details")

start = selected_row.get("start_min") or 0
end = selected_row.get("end_min") or 0
time_str = f"{start:.2f}-{end:.2f} min"

bias_val = selected_row.get("bias_score") or 0
badge_label = bias_label(bias_val)
badge_color = bias_to_color(bias_val)

html = f"""<div class="signal-card">

<div style="text-align:center; margin-bottom:10px;">
    <div style="
        display:inline-block;
        background:{badge_color};
        color:white;
        padding:8px 18px;
        border-radius:999px;
        font-weight:700;
        font-size:1.1rem;
    ">
        {badge_label} Bias ({bias_val:.2f})
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
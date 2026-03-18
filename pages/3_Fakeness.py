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

st.set_page_config(page_title="Fakeness")
st.markdown(clean_css(), unsafe_allow_html=True)

# -----------------------------
# Load data
# -----------------------------
data_dir = st.session_state.get("DATA_DIR", DATA_DIR_DEFAULT)
files = list_scored_json_files(data_dir)

if not files:
    st.warning("No *_scored.json files found.")
    st.stop()

selected_file = st.session_state.get("selected_file", files[0])
if selected_file not in files:
    selected_file = files[0]

payload = load_scored_json(os.path.join(data_dir, selected_file))
df = build_segments_df(payload)
speakers = get_speakers(payload, df)

st.markdown("### Fakeness")
st.caption("Click a claim to explore details.")

if df.empty:
    st.info("No segments available.")
    st.stop()

# -----------------------------
# Explanation
# -----------------------------
with st.expander("How truth risk is interpreted"):
    st.markdown(
        """
- **Low Risk** → likely factual / well-supported  
- **Moderate Risk** → partially supported or uncertain  
- **High Risk** → likely inaccurate, misleading, or unsupported  

Color intensity reflects truth risk.
"""
    )

# -----------------------------
# Speaker filter
# -----------------------------
speaker_choice = st.selectbox(
    "Speaker",
    options=["All"] + speakers,
    index=0,
    key="fake_speaker_choice",
)

# -----------------------------
# Build filtered working df
# -----------------------------
work = df.copy()
work = work[work["fakeness_score"].notna()].copy()

if speaker_choice != "All":
    work = work[work["speaker"] == speaker_choice].copy()

if work.empty:
    st.warning("No verified claims found.")
    st.stop()

# -----------------------------
# Sort + stable ids
# -----------------------------
fake_df = work.sort_values(["start_min", "end_min"]).reset_index(drop=True).copy()
fake_df["segment_id"] = fake_df.index + 1
fake_df["duration"] = fake_df["end_min"] - fake_df["start_min"]

fake_df = fake_df[fake_df["duration"] > 0].reset_index(drop=True)
fake_df["segment_id"] = fake_df.index + 1

if fake_df.empty:
    st.warning("No valid claim segments available.")
    st.stop()

n_segments = len(fake_df)

# -----------------------------
# Reset selection when file/filter changes
# -----------------------------
selection_context = (selected_file, speaker_choice, n_segments)
if st.session_state.get("fake_selection_context") != selection_context:
    st.session_state["selected_segment_idx_fake"] = 0
    st.session_state["fake_selection_context"] = selection_context

if "selected_segment_idx_fake" not in st.session_state:
    st.session_state["selected_segment_idx_fake"] = 0

st.session_state["selected_segment_idx_fake"] = min(
    max(0, st.session_state["selected_segment_idx_fake"]),
    n_segments - 1,
)

selected_idx = st.session_state["selected_segment_idx_fake"]

# -----------------------------
# Helpers
# -----------------------------
def fake_to_color(val):
    if pd.isna(val):
        return "#E5E7EB"
    if val < 0.3:
        return "#10B981"   # low
    elif val < 0.6:
        return "#F97316"   # moderate
    else:
        return "#EF4444"   # high

def fake_label(val):
    if pd.isna(val):
        return "Unknown"
    if val < 0.3:
        return "Low Risk"
    elif val < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"

def fake_bin(val):
    if pd.isna(val):
        return "unknown"
    if val < 0.3:
        return "low"
    elif val < 0.6:
        return "mid"
    else:
        return "high"

def safe_claim_text(row):
    claim = row.get("claim_text")
    if isinstance(claim, str) and claim.strip():
        return claim.strip()

    txt = row.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    return "—"

def fmt_sources_html(src_list):
    if not isinstance(src_list, list) or len(src_list) == 0:
        return "<div style='color:#6B7280;'>No sources available.</div>"

    items = []
    for s in src_list[:5]:
        title = s.get("title", "source")
        url = s.get("url", "")
        if url:
            items.append(
                f"<li><a href='{url}' target='_blank' style='color:#2563EB; text-decoration:underline;'>{title}</a></li>"
            )
        else:
            items.append(f"<li>{title}</li>")

    return "<ul style='margin-top:0.25rem; padding-left:1.25rem;'>" + "".join(items) + "</ul>"

fake_df["fake_bin"] = fake_df["fakeness_score"].apply(fake_bin)

speaker_order = list(fake_df["speaker"].drop_duplicates())
speaker_order = sorted(speaker_order)

# -----------------------------
# Build chart (same structure as Bias)
# -----------------------------
fig = go.Figure()

trace_point_to_global_idx = {}

for b in ["low", "mid", "high"]:
    group = fake_df[fake_df["fake_bin"] == b].copy()
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
                color=group["fakeness_score"].apply(fake_to_color),
                line=dict(color="white", width=2),
            ),
            customdata=group[
                ["segment_id", "start_min", "end_min", "speaker", "fakeness_score"]
            ].values,
            hovertemplate=(
                "Segment %{customdata[0]}<br>"
                "Speaker: %{customdata[3]}<br>"
                "Time: %{customdata[1]:.2f} - %{customdata[2]:.2f} min<br>"
                "Risk: %{customdata[4]:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

# Selected segment
selected_row = fake_df.iloc[selected_idx]

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
    showgrid=True,
    gridcolor="#E5E7EB",
    zeroline=False,
)

fig.update_yaxes(
    categoryorder="array",
    categoryarray=speaker_order,
)

# -----------------------------
# Custom legend
# -----------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <div style="
            width:220px;
            height:12px;
            background: linear-gradient(to right, #10B981, #F97316, #EF4444);
            border-radius:999px;
        "></div>
        <div style="font-size:0.85rem; color:#6B7280;">
            Low Risk → Moderate Risk → High Risk
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
    key="fake_plot",
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
        if new_idx != st.session_state["selected_segment_idx_fake"]:
            st.session_state["selected_segment_idx_fake"] = new_idx
            st.rerun()

# -----------------------------
# Re-read selected row after possible click/rerun
# -----------------------------
selected_idx = min(max(0, st.session_state["selected_segment_idx_fake"]), n_segments - 1)
selected_row = fake_df.iloc[selected_idx]

# -----------------------------
# Selected Claim Details
# -----------------------------
st.markdown("### Selected Claim Details")

start = selected_row.get("start_min") or 0
end = selected_row.get("end_min") or 0
time_str = f"{start:.2f}-{end:.2f} min"

risk_val = selected_row.get("fakeness_score") or 0
badge_label = fake_label(risk_val)
badge_color = fake_to_color(risk_val)
claim_text = safe_claim_text(selected_row)
sources_html = fmt_sources_html(selected_row.get("sources"))

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
        {badge_label} ({risk_val:.2f})
    </div>
</div>

<div style="text-align:center; font-size:0.95rem; color:#6B7280; margin-bottom:12px;">
    {selected_row.get("speaker", "—")} • {time_str}
</div>

<div style="line-height:1.7; margin-bottom:1rem;">
    <div style="font-weight:600; margin-bottom:4px;">Claim</div>
    {claim_text}
</div>

<div style="margin-top:0.75rem;">
    <details>
        <summary style="cursor:pointer; font-weight:600;">View transcript segment</summary>
        <div style="margin-top:0.75rem; line-height:1.7;">
            {selected_row.get("text", "—")}
        </div>
    </details>
</div>

<div style="margin-top:1rem;">
    <details open>
        <summary style="cursor:pointer; font-weight:600;">Sources / evidence</summary>
        <div style="margin-top:0.75rem; line-height:1.7;">
            {sources_html}
        </div>
    </details>
</div>

</div>"""

st.markdown(html, unsafe_allow_html=True)
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Definitions — Verity", page_icon="📚", layout="wide")

st.markdown("## Definitions")
st.caption("How core metrics are defined and interpreted in this project.")

st.write("")

definitions = [
    {
        "Metric": "Bias",
        "Range": "0 – 1",
        "What it means": "Measures persuasive or emotionally loaded framing (not factual accuracy).",
        "How it’s computed": "Higher when language uses exaggeration, strong framing, or rhetorical emphasis."
    },
    {
        "Metric": "Fakeness",
        "Range": "0 – 1",
        "What it means": "Estimates the likelihood a factual claim is false or misleading.",
        "How it’s computed": "Higher when verification sources contradict or weaken the claim."
    },
    {
        "Metric": "Emotional Tone",
        "Range": "Category",
        "What it means": "Dominant emotional tone of the segment (e.g., neutral, angry, defensive).",
        "How it’s computed": "Inferred from linguistic cues in the transcript."
    },
    {
        "Metric": "Segment Kind",
        "Range": "Category",
        "What it means": "Type of segment (factual claim, policy claim, analysis, question, etc.).",
        "How it’s computed": "Classifies what the speaker is doing in that moment."
    }
]

df = pd.DataFrame(definitions)
st.dataframe(
    df.set_index("Metric"),
    use_container_width=True,
    hide_index=False
)

st.write("")
st.markdown("### Important Notes")

st.markdown(
"""
- **Bias ≠ False.** A segment can be persuasive and still factually correct.  
- **Truth Risk only appears when a factual claim was verified.** If no checkable claim exists, it may be absent.  
- **Misleading Index** provides a simplified “one-number” overview.  
- This system is **decision-support**, not a replacement for human judgment.
"""
)
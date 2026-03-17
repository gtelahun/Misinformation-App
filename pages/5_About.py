import streamlit as st

st.set_page_config(page_title="About — Verity", layout="wide")

st.markdown("## About Verity")

st.markdown(
"""
**Verity** is a media integrity dashboard for analyzing digital media.

It evaluates:

- **Bias** (rhetorical framing)
- **Fakeness** (verification-based factual risk)
- **Emotional Tone**
- **Source-backed claims**

This tool is designed to be understandable for general audiences."
"""
)

st.write("")

st.markdown("### What Verity Does NOT Do")

st.markdown(
"""
- It does not determine absolute truth.
- It does not infer speaker intent.
- It does not replace primary sources
- It does not make final judgments; it supports analysis.
"""
)
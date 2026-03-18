import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


DATA_DIR_DEFAULT = "data"


METRIC_DESCRIPTIONS = {
    "bias_score": {
        "label": "Bias",
        "range": "0–1",
        "what": "Measures persuasive / loaded framing (not truth).",
        "how": (
            "High when wording uses exaggeration, emotional manipulation, one-sided framing, "
            "or loaded language. Low when phrasing is neutral and descriptive."
        ),
    },
    "fakeness_score": {
        "label": "Fakeness",
        "range": "0–1",
        "what": "Probability-like truth-risk that a claim is false or misleading (verification-based).",
        "how": (
            "High when sources contradict the claim or evidence strongly suggests the claim is misleading. "
            "Low when reputable sources support it. Can be null when not verifiable or verification didn’t run."
        ),
    },
    "overall_misleading_index": {
        "label": "Overall Misleading Index",
        "range": "0–1",
        "what": "A single summary score of misleadingness.",
        "how": (
            "If fakeness exists, it averages Bias and Fakeness. Otherwise, it equals Bias. "
            "This helps summarize persuasion + truth-risk in one number."
        ),
    },
    "analysis_confidence": {
        "label": "Analysis Confidence",
        "range": "0–1",
        "what": "Confidence in non-verification outputs (bias, tone, kind, leaning, stance).",
        "how": (
            "High when the segment is clear and the model is confident about tone/framing. "
            "Lower when the segment is ambiguous or context is missing."
        ),
    },
    "fakeness_confidence": {
        "label": "Fakeness Confidence",
        "range": "0–1",
        "what": "Confidence in verification result only (when fakeness_score exists).",
        "how": (
            "Higher when sources directly address the claim and evidence is strong. "
            "Lower when evidence is partial or indirect."
        ),
    },
    "emotion_tone": {
        "label": "Emotion Tone",
        "range": "Category",
        "what": "Dominant emotional tone label for the segment.",
        "how": (
            "A categorical label inferred from language cues (e.g., anger, fear, neutral, defensive). "
            "Not a medical diagnosis — just a communication signal."
        ),
    },
    "segment_kind": {
        "label": "Segment Kind",
        "range": "Category",
        "what": "Type of segment: factual claim, policy claim, question, analysis, etc.",
        "how": (
            "Classifies what the speaker is doing (making a checkable claim vs expressing values vs asking). "
            "Used to focus verification on checkable factual claims."
        ),
    },
    "verification_verdict": {
        "label": "Verification Verdict",
        "range": "Category",
        "what": "Outcome of verification when it ran: supported/contradicted/mixed/insufficient.",
        "how": (
            "Based on whether cited sources align with the claim. "
            "Some pipelines may use slightly different labels; the app displays what your JSON provides."
        ),
    },
    "political_leaning": {
        "label": "Political Leaning",
        "range": "Category",
        "what": "Left/Right/Neutral/Not Applicable — linguistic and topical leaning classification.",
        "how": (
            "A rough classifier based on issue framing and language. "
            "Not a measure of truth, and can be 'Not Applicable' for non-political segments."
        ),
    },
}


def minutes_str(seconds: Optional[float]) -> str:
    """Format seconds as minutes with 2 decimals."""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "—"
    mins = float(seconds) / 60.0
    return f"{mins:.2f} min"


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


@st.cache_data(show_spinner=False)
def list_scored_json_files(data_dir: str = DATA_DIR_DEFAULT) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    files = []
    for f in os.listdir(data_dir):
        if f.lower().endswith(".json") and "scored" in f.lower():
            files.append(f)
    return sorted(files)


@st.cache_data(show_spinner=True)
def load_scored_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_segments_df(payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for seg in payload.get("segments", []):
        s = seg.get("scores", {}) or {}
        t = seg.get("time", {}) or {}
        rows.append(
            {
                "segment_id": seg.get("segment_id"),
                "speaker": seg.get("speaker"),
                "start_sec": t.get("start"),
                "end_sec": t.get("end"),
                "start_min": (t.get("start") or 0.0) / 60.0 if t.get("start") is not None else np.nan,
                "end_min": (t.get("end") or 0.0) / 60.0 if t.get("end") is not None else np.nan,
                "text": seg.get("text", ""),
                "claim_text": s.get("claim_text"),
                "bias_score": s.get("bias_score"),
                "fakeness_score": s.get("fakeness_score"),
                "overall_misleading_index": s.get("overall_misleading_index"),
                "analysis_confidence": s.get("analysis_confidence"),
                "fakeness_confidence": s.get("fakeness_confidence"),
                "verification_verdict": s.get("verification_verdict"),
                "emotion_tone": s.get("emotion_tone"),
                "segment_kind": s.get("segment_kind"),
                "political_leaning": s.get("political_leaning"),
                "sources": s.get("sources") or [],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        # Ensure numeric
        for col in ["start_min", "end_min", "bias_score", "fakeness_score", "overall_misleading_index",
                    "analysis_confidence", "fakeness_confidence"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["mid_min"] = (df["start_min"] + df["end_min"]) / 2.0
    return df


def get_speakers(payload: Dict[str, Any], df: pd.DataFrame) -> List[str]:
    # Prefer meta speakers_scored if present, else infer from segments
    meta_speakers = safe_get(payload, ["meta", "speakers_scored"], default=None)
    if isinstance(meta_speakers, list) and len(meta_speakers) > 0:
        return [str(x) for x in meta_speakers]
    if df.empty:
        return []
    speakers = sorted([s for s in df["speaker"].dropna().unique().tolist()])
    return speakers


def get_runtime_seconds(payload: Dict[str, Any], df: pd.DataFrame) -> Optional[float]:
    rt = safe_get(payload, ["meta", "runtime_sec"], default=None)
    if isinstance(rt, (int, float)):
        return float(rt)
    if df.empty:
        return None
    mx = df["end_sec"].max()
    return float(mx) if pd.notna(mx) else None


def compute_summary(payload: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    # Use payload summary if present, else compute fallback
    s = payload.get("summary", {}).get("global", {})
    if s:
        return s
    if df.empty:
        return {}
    out = {}
    out["segments_scored"] = int(df.shape[0])
    out["segments_verified"] = int(df["fakeness_score"].notna().sum())
    out["avg_bias"] = float(df["bias_score"].mean(skipna=True)) if "bias_score" in df else None
    out["avg_fakeness"] = float(df["fakeness_score"].mean(skipna=True)) if "fakeness_score" in df else None
    out["avg_overall_misleading_index"] = float(df["overall_misleading_index"].mean(skipna=True)) if "overall_misleading_index" in df else None
    out["model_agreement_score"] = None
    return out


def per_speaker_summary(payload: Dict[str, Any], df: pd.DataFrame, speakers: List[str]) -> pd.DataFrame:
    # Prefer payload summary.per_speaker when present
    ps = payload.get("summary", {}).get("per_speaker", {})
    if isinstance(ps, dict) and len(ps) > 0:
        rows = []
        for sp in speakers:
            item = ps.get(sp, {}) or {}
            rows.append(
                {
                    "speaker": sp,
                    "role": item.get("role", "—"),
                    "segments": item.get("segments"),
                    "avg_bias": item.get("avg_bias"),
                    "avg_fakeness": item.get("avg_fakeness"),
                    "avg_overall_misleading_index": item.get("avg_overall_misleading_index"),
                }
            )
        return pd.DataFrame(rows)

    # Fallback compute
    if df.empty:
        return pd.DataFrame(columns=["speaker", "segments", "avg_bias", "avg_fakeness", "avg_overall_misleading_index"])
    rows = []
    for sp in speakers:
        sdf = df[df["speaker"] == sp]
        rows.append(
            {
                "speaker": sp,
                "role": safe_get(payload, ["meta", "speaker_roles", sp, "role"], default="—"),
                "segments": int(sdf.shape[0]),
                "avg_bias": float(sdf["bias_score"].mean(skipna=True)),
                "avg_fakeness": float(sdf["fakeness_score"].mean(skipna=True)) if sdf["fakeness_score"].notna().any() else None,
                "avg_overall_misleading_index": float(sdf["overall_misleading_index"].mean(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def top_misleading_claims(df: pd.DataFrame, speaker: Optional[str] = None, n: int = 8) -> pd.DataFrame:
    if df.empty:
        return df
    sdf = df.copy()
    if speaker and speaker != "All":
        sdf = sdf[sdf["speaker"] == speaker]

    # Prioritize segments that have claim_text OR are factual_claim/policy_claim
    sdf["has_claim"] = sdf["claim_text"].notna() & (sdf["claim_text"].astype(str).str.len() > 0)
    sdf["is_claim_kind"] = sdf["segment_kind"].isin(["factual_claim", "policy_claim"])
    sdf = sdf[sdf["has_claim"] | sdf["is_claim_kind"]]

    # Rank by fakeness when available, else overall index
    sdf["rank_score"] = sdf["fakeness_score"].fillna(sdf["overall_misleading_index"])
    sdf = sdf.sort_values(["rank_score", "bias_score"], ascending=False)

    cols = [
        "speaker",
        "start_min",
        "end_min",
        "segment_kind",
        "emotion_tone",
        "bias_score",
        "fakeness_score",
        "overall_misleading_index",
        "verification_verdict",
        "claim_text",
        "text",
        "sources",
    ]
    cols = [c for c in cols if c in sdf.columns]
    return sdf.head(n)[cols]


def clean_css() -> str:
    return """
    <style>
      /* Layout tweaks */
      .block-container { padding-top: 3rem; padding-bottom: 2.5rem; }
      /* Card-like containers */
      .signal-card {
        border: 1px solid rgba(49, 51, 63, 0.10);
        border-radius: 18px;
        padding: 1.1rem 1.1rem;
        background: rgba(255,255,255,0.6);
        box-shadow: 0 8px 22px rgba(0,0,0,0.04);
      }
      /* Muted text */
      .signal-muted { color: rgba(49, 51, 63, 0.70); }
      /* Big title */
      .signal-title { font-size: 2.25rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
      .signal-subtitle { font-size: 1.05rem; font-weight: 500; color: rgba(49, 51, 63, 0.70); }
      /* Metric label alignment */
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.7); border: 1px solid rgba(49, 51, 63, 0.10);
        padding: 0.85rem 0.85rem; border-radius: 16px; }
      /* Sidebar spacing */
      section[data-testid="stSidebar"] { padding-top: 1rem; }
      /* Better radio spacing */
      div[role="radiogroup"] > label { padding: 0.15rem 0; }
      /* Tables */
      .stDataFrame { border-radius: 14px; overflow: hidden; }
    </style>
    """
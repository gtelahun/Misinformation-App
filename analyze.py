#!/usr/bin/env python3
# ============================================================
# UNIVERSAL TRANSCRIBE + SCORE (Merged Single Script)
# ============================================================
# This file merges:
# 1) UNIVERSAL TRANSCRIBE v5 (Stable + Speaker Consolidation)
# 2) UNIVERSAL SCORE v1.0 (Research-Grade)
#
# OUTPUTS (by default):
# - <audio>_transcript.json
# - <audio>_transcript_scored.json
#
# ------------------------------------------------------------
# NOTES ON MERGE
# ------------------------------------------------------------
# - Preserves ALL logic you pasted (transcribe + score) with only
#   minimal structural changes required to unify into one script.
# - Adds stage logging + timing.
# - Routing runs FIRST (before expand_segments_for_analysis), as you fixed.
# - Speaker selection for informational allows narrator to be included (so 1-speaker videos score).
# - Removed duplicate split_into_sentences definition (kept one identical version).
# ============================================================

from __future__ import annotations

# -------------------------
# Shared imports
# -------------------------
import argparse
import json
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from math import sqrt

from openai import OpenAI

client = OpenAI()

# ============================================================
# STAGE LOGGING + TIMING HELPERS
# ============================================================

def _ts() -> str:
    return time.strftime("%H:%M:%S", time.localtime())

def log_stage(msg: str) -> None:
    print(f"[{_ts()}] [STAGE] {msg}", flush=True)

def log_info(msg: str) -> None:
    print(f"[{_ts()}] [INFO]  {msg}", flush=True)

def log_done(msg: str) -> None:
    print(f"[{_ts()}] [DONE]  {msg}", flush=True)

@dataclass
class StageTimer:
    name: str
    t0: float

def stage_start(name: str) -> StageTimer:
    log_stage(name)
    return StageTimer(name=name, t0=time.time())

def stage_end(st: StageTimer) -> float:
    dt = time.time() - st.t0
    log_done(f"{st.name} ({dt:.3f}s)")
    return dt

def now_local_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# ============================================================
# ============================================================
# PART 1: TRANSCRIBE (Your pasted transcribe.py)
# ============================================================
# ============================================================

# ============================================================
# UNIVERSAL TRANSCRIBE v5 (Stable + Speaker Consolidation)
# - Dynamic chunk count
# - Parallel diarization
# - Improved stitching
# - Diarization split consolidation (NEW)
# - Safe for debates, panels, highlights
# ============================================================

MODEL_TRANSCRIBE = "gpt-4o-transcribe-diarize"
RESPONSE_FORMAT = "diarized_json"
CHUNKING_STRATEGY = "auto"
TRANSCRIBE_VERSION = "v5_universal_consolidated"

# --------------------------
# Stitch tuning
# --------------------------
MERGE_GAP_SECONDS = 1.6
MICRO_MAX_WORDS = 5
MICRO_MAX_DURATION = 0.8
MICRO_MAX_GAP = 0.6

# --------------------------
# Helpers
# --------------------------

def clean_text(text):
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([?.!,…])", r"\1", t)
    return t

def get_turn_times(turn):
    s = float(turn.get("start", 0))
    e = float(turn.get("end", s))
    return s, max(s, e)

def wc(text):
    return len(text.split())

def duration(seg):
    return seg["end"] - seg["start"]

def get_audio_duration_seconds(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())

# --------------------------
# Merge Adjacent Same Speaker
# --------------------------

def merge_adjacent(segs):
    if not segs:
        return []

    merged = [dict(segs[0])]

    for s in segs[1:]:
        last = merged[-1]
        gap = s["start"] - last["end"]

        if s["speaker_id"] == last["speaker_id"] and gap <= MERGE_GAP_SECONDS:
            last["text"] = clean_text(last["text"] + " " + s["text"])
            last["end"] = s["end"]
        else:
            merged.append(dict(s))

    return merged

# --------------------------
# Micro Cleanup
# --------------------------

def cleanup_micro(segs):
    if not segs:
        return []

    out = []
    i = 0

    while i < len(segs):
        cur = dict(segs[i])

        is_micro = (
            wc(cur["text"]) <= MICRO_MAX_WORDS and
            duration(cur) <= MICRO_MAX_DURATION
        )

        if not is_micro:
            out.append(cur)
            i += 1
            continue

        prev = out[-1] if out else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        merged = False

        if prev and prev["speaker_id"] == cur["speaker_id"]:
            gap = cur["start"] - prev["end"]
            if gap <= MICRO_MAX_GAP:
                prev["text"] = clean_text(prev["text"] + " " + cur["text"])
                prev["end"] = cur["end"]
                merged = True

        if not merged and nxt and nxt["speaker_id"] == cur["speaker_id"]:
            gap = nxt["start"] - cur["end"]
            if gap <= MICRO_MAX_GAP:
                nxt["text"] = clean_text(cur["text"] + " " + nxt["text"])
                nxt["start"] = cur["start"]
                merged = True

        i += 1

    return out

# --------------------------
# NEW: Speaker Consolidation
# --------------------------

def consolidate_split_speakers(segments):
    """
    Merge accidental diarization splits common in highlight edits.
    Only merges speakers that appear very few times and are structurally embedded.
    """

    speaker_counts = defaultdict(int)
    for seg in segments:
        speaker_counts[seg["speaker_id"]] += 1

    consolidated = []
    i = 0

    while i < len(segments):
        current = segments[i]
        sid = current["speaker_id"]

        # Candidate for merge if very rare speaker
        if speaker_counts[sid] <= 2:

            prev_seg = consolidated[-1] if consolidated else None
            next_seg = segments[i+1] if i+1 < len(segments) else None

            # Pattern: A -> D -> A
            if (
                prev_seg and next_seg and
                prev_seg["speaker_id"] == next_seg["speaker_id"] and
                prev_seg["speaker_id"] != sid
            ):
                prev_seg["text"] = clean_text(prev_seg["text"] + " " + current["text"])
                prev_seg["end"] = current["end"]
                i += 1
                continue

            # Merge rare speaker into previous if continuous
            if prev_seg and prev_seg["speaker_id"] != sid:
                prev_seg["text"] = clean_text(prev_seg["text"] + " " + current["text"])
                prev_seg["end"] = current["end"]
                i += 1
                continue

        consolidated.append(current)
        i += 1

    return consolidated


def run_transcription(AUDIO_FILE: str) -> str:
    """
    Runs your transcribe pipeline and writes <audio>_transcript.json.
    Returns transcript json path.
    """
    st_total = stage_start("TRANSCRIBE: start")

    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(AUDIO_FILE)

    st = stage_start("TRANSCRIBE: get duration")
    total_duration = get_audio_duration_seconds(AUDIO_FILE)
    stage_end(st)

    # Dynamic chunk sizing
    if total_duration < 300:
        NUM_CHUNKS = 2
    elif total_duration < 1200:
        NUM_CHUNKS = 6
    else:
        NUM_CHUNKS = 8

    chunk_len = total_duration / NUM_CHUNKS
    log_info(f"Duration: {total_duration:.2f}s | NUM_CHUNKS={NUM_CHUNKS} | chunk_len={chunk_len:.2f}s")

    st = stage_start(f"TRANSCRIBE: split audio into {NUM_CHUNKS} chunks")
    chunks = []
    for i in range(NUM_CHUNKS):
        start = i * chunk_len
        outfile = f"chunk_{i}.mp3"

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", AUDIO_FILE,
            "-ss", str(start),
            "-t", str(chunk_len),
            outfile
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        chunks.append((outfile, start))
    stage_end(st)

    st = stage_start("TRANSCRIBE: parallel diarized transcription")
    def transcribe_chunk(path, offset):
        with open(path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                file=f,
                model=MODEL_TRANSCRIBE,
                response_format=RESPONSE_FORMAT,
                chunking_strategy=CHUNKING_STRATEGY,
            )

        data = transcript.model_dump()
        raw = data.get("speaker_segments") or data.get("segments") or []

        for t in raw:
            s, e = get_turn_times(t)
            t["start"] = s + offset
            t["end"] = e + offset

        return raw

    all_turns = []
    with ThreadPoolExecutor(max_workers=NUM_CHUNKS) as executor:
        futures = [executor.submit(transcribe_chunk, p, o) for p, o in chunks]
        for f in as_completed(futures):
            all_turns.extend(f.result())

    all_turns.sort(key=lambda x: get_turn_times(x)[0])
    stage_end(st)

    st = stage_start("TRANSCRIBE: build segments + stitching")
    segments = []
    for t in all_turns:
        speaker = t.get("speaker") or "unknown"
        start, end = get_turn_times(t)
        text = clean_text(t.get("text", ""))

        segments.append({
            "speaker_id": speaker,
            "start": start,
            "end": end,
            "text": text
        })

    # Stitch passes
    segments = merge_adjacent(segments)
    segments = cleanup_micro(segments)
    segments = merge_adjacent(segments)

    # NEW consolidation
    segments = consolidate_split_speakers(segments)
    segments = merge_adjacent(segments)

    for i, s in enumerate(segments):
        s["segment_id"] = f"seg_{i}"
        s["start"] = round(s["start"], 3)
        s["end"] = round(s["end"], 3)

    stage_end(st)

    st = stage_start("TRANSCRIBE: write transcript json")
    out = {
        "meta": {
            "audio_file": AUDIO_FILE,
            "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": MODEL_TRANSCRIBE,
            "transcribe_version": TRANSCRIBE_VERSION,
            "segments_out": len(segments)
        },
        "segments": segments
    }

    out_path = f"{os.path.splitext(AUDIO_FILE)[0]}_transcript.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    stage_end(st)

    # Optional cleanup of chunk files
    st = stage_start("TRANSCRIBE: cleanup temp chunks")
    for p, _ in chunks:
        try:
            os.remove(p)
        except Exception:
            pass
    stage_end(st)

    stage_end(st_total)
    log_info(f"✅ Transcript saved: {out_path} | segments={len(segments)}")
    return out_path


# ============================================================
# ============================================================
# PART 2: SCORE (Your pasted score.py)
# ============================================================
# ============================================================

# =========================
# CONFIG
# =========================

SCORE_VERSION = "v1.0_universal_research_grade"

# Primary models
MODEL_ROUTER = os.getenv("MODEL_ROUTER", "gpt-4o-mini")
MODEL_ANALYZE_1 = os.getenv("MODEL_ANALYZE_1", "gpt-4o-mini")
MODEL_ANALYZE_2 = os.getenv("MODEL_ANALYZE_2", "")  # optional 2nd model for agreement
MODEL_VERIFY = os.getenv("MODEL_VERIFY", "gpt-4o-mini")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
DEFAULT_MIN_WORDS = int(os.getenv("DEFAULT_MIN_WORDS", "8"))
DEFAULT_MAX_SOURCES = int(os.getenv("DEFAULT_MAX_SOURCES", "2"))
DEFAULT_WORKERS = int(os.getenv("DEFAULT_WORKERS", "6"))

# If True, we try to compute model agreement using a second analysis pass.
DEFAULT_ENABLE_MODEL_AGREEMENT = bool(int(os.getenv("ENABLE_MODEL_AGREEMENT", "1" if MODEL_ANALYZE_2 else "0")))

# =========================
# SEGMENT SPLITTING CONFIG
# =========================

SPLIT_LONG_SEGMENTS = True
SPLIT_THRESHOLD_WORDS = 140
SPLIT_TARGET_SENTENCES = 3

# =========================
# DEFINITIONS (ALWAYS INCLUDED)
# =========================

DEFINITIONS: Dict[str, str] = {
    "fakeness_score": (
        "Probability-like truth-risk that a statement is false or misleading, based on verification of factual accuracy and "
        "claim reliability using web sources. 0–1. Null if verification did not run or no checkable claim exists."
    ),
    "bias_score": (
        "Degree of linguistic/ideological bias in phrasing, framing, exaggeration, loaded language, or manipulative rhetoric. "
        "0–1. This is NOT about factual truth."
    ),
    "political_leaning": (
        "Classification of overall political orientation in the segment (Left, Right, Neutral, Not Applicable)."
    ),
    "emotion_tone": (
        "Dominant emotional tone expressed (categorical label, e.g., calm, angry, fearful, sarcastic, critical)."
    ),
    "overall_misleading_index": (
        "Composite measure summarizing misleadingness. If fakeness_score exists: average(bias_score, fakeness_score). "
        "Else: equals bias_score."
    ),
    "model_agreement_score": (
        "Agreement between two independent model scoring passes (0–1). Reported when a second analysis model is enabled. "
        "Computed from similarity of bias scores and categorical matches (leaning/kind)."
    ),
    "analysis_confidence": (
        "Confidence (0–1) in NON-verification outputs: bias_score, emotion_tone, segment_kind, stance fields, political_leaning."
    ),
    "fakeness_confidence": (
        "Confidence (0–1) in verification result ONLY. Present only when fakeness_score exists."
    ),
    "verification_verdict": (
        "One of: supported, contradicted, mixed, insufficient. Only present if verification ran."
    ),
    "segment_kind": (
        "One of: factual_claim, policy_claim, values_statement, question, personal_attack, analysis, descriptive, other."
    ),
    "stance_target": "Entity/issue being discussed (string) or null if unclear.",
    "stance_direction": "One of: pro, anti, neutral/unclear (relative to stance_target).",
    "sources": "Verification sources used for fakeness_score (hard-capped by MAX_SOURCES_PER_VERIFIED_SEGMENT).",
    "speaker_roles": (
        "Per-speaker inferred role: candidate/debater, moderator/host, journalist/analyst, commentator, audience, narrator, other."
    ),
    "speaker_selection": (
        "Speakers to score are selected automatically. Moderators/hosts/journalists/audience/narrator are excluded from scoring. "
        "Debate content requires at least 2 substantive debaters; fallback selects top talk-time speakers if needed."
    ),
}

# =========================
# HELPERS (Score)
# =========================

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def clamp01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

def safe_speaker_id(seg: Dict[str, Any]) -> str:
    sid = seg.get("speaker_id") or seg.get("speaker") or "UNKNOWN"
    return str(sid)

def safe_time(seg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    try:
        return float(seg.get("start")), float(seg.get("end"))
    except Exception:
        return None, None

def safe_text(seg: Dict[str, Any]) -> str:
    return str(seg.get("text") or "").strip()

def avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def overall_misleading_index(bias_score: float, fakeness_score: Optional[float]) -> float:
    if fakeness_score is None:
        return float(bias_score)
    return (float(bias_score) + float(fakeness_score)) / 2.0

def pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs)
    deny = sum((y - my) ** 2 for y in ys)
    den = sqrt(denx * deny) if denx > 0 and deny > 0 else 0.0
    if den == 0.0:
        return None
    return num / den

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def expand_segments_for_analysis(
    segments: List[Dict[str, Any]],
    content_type: str
) -> List[Dict[str, Any]]:

    if not SPLIT_LONG_SEGMENTS:
        return segments

    # Do NOT split debates
    if content_type == "debate":
        return segments

    expanded = []

    for seg in segments:
        text = safe_text(seg)
        wc_ = word_count(text)

        # Only split long segments
        if wc_ < SPLIT_THRESHOLD_WORDS:
            expanded.append(seg)
            continue

        sentences = split_into_sentences(text)

        chunk = []
        chunk_count = 0
        sub_index = 0

        for sentence in sentences:
            chunk.append(sentence)
            chunk_count += 1

            if chunk_count >= SPLIT_TARGET_SENTENCES:
                new_seg = seg.copy()
                new_seg["text"] = " ".join(chunk)
                new_seg["segment_id"] = f'{seg["segment_id"]}_part{sub_index}'
                expanded.append(new_seg)

                chunk = []
                chunk_count = 0
                sub_index += 1

        # leftover sentences
        if chunk:
            new_seg = seg.copy()
            new_seg["text"] = " ".join(chunk)
            new_seg["segment_id"] = f'{seg["segment_id"]}_part{sub_index}'
            expanded.append(new_seg)

    return expanded

# =========================
# OPENAI JSON CALL (robust)
# =========================

def openai_json(model: str, system: str, user: str, tools=None) -> Dict[str, Any]:

    if tools:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=TEMPERATURE,
        )

        # Try direct text first
        raw = resp.output_text

        # If empty, try structured output
        if not raw and resp.output:
            for item in resp.output:
                if hasattr(item, "content"):
                    for c in item.content:
                        if c.get("type") == "output_text":
                            raw = c.get("text")
                            break

    else:
        resp = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content

    if not raw:
        return {}

    try:
        return json.loads(raw)
    except:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return {}

# =========================
# ROUTING (CONTENT TYPE)
# =========================

def detect_content_type(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_text = " ".join(safe_text(s) for s in segments[:30])

    system = f"""
You are a router for transcript analysis.
Return STRICT JSON: {{ "content_type": string, "confidence": number, "notes": string }}

Valid content_type:
- debate
- panel
- news
- sports
- informational
- unknown

Guidance:
- "debate": back-and-forth between >=2 substantive debaters/candidates with moderator questions.
- "panel": multiple speakers discussing, may include hosts/journalists + guests.
- "news": anchor/reporter segment.
- "sports": play-by-play/analysis of sports.
- "informational": one main speaker narrating/explaining.

Confidence 0–1.
"""
    out = openai_json(MODEL_ROUTER, system, sample_text)
    ct = str(out.get("content_type", "unknown")).strip().lower()
    if ct not in {"debate", "panel", "news", "sports", "informational", "unknown"}:
        ct = "unknown"
    return {
        "content_type": ct,
        "confidence": clamp01(out.get("confidence", 0.6)),
        "notes": str(out.get("notes", "")),
    }

# =========================
# SPEAKER STATS + ROLES
# =========================

def compute_speaker_stats(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for seg in segments:
        sid = safe_speaker_id(seg)
        st, en = safe_time(seg)
        dur = 0.0
        if st is not None and en is not None and en >= st:
            dur = en - st
        txt = safe_text(seg)

        if sid not in stats:
            stats[sid] = {"duration_sec": 0.0, "words": 0, "samples": []}
        stats[sid]["duration_sec"] += dur
        stats[sid]["words"] += word_count(txt)
        if txt and len(stats[sid]["samples"]) < 6:
            stats[sid]["samples"].append(txt)
    return stats

def detect_speaker_roles(segments: List[Dict[str, Any]], content_type: str) -> Dict[str, Dict[str, Any]]:
    stats = compute_speaker_stats(segments)
    roles: Dict[str, Dict[str, Any]] = {}

    system = f"""
You infer speaker roles from snippets in a transcript.

Return STRICT JSON: {{ "role": string, "confidence": number, "reason": string }}

Valid role:
- candidate/debater
- moderator/host
- journalist/analyst
- commentator
- audience
- narrator
- other

Rules:
- Moderators ask questions, manage turns, say "welcome" / "question for you".
- Candidates/debaters advocate positions, defend/attack opponents, propose policy.
- Journalists/analysts cite institutions, provide "FBI says...", "according to..."
- Commentators opine without moderating structure (less common in debates).
- Audience asks short questions or reacts.

Content type context: {content_type}
"""
    for sid, info in stats.items():
        joined = " ".join(info.get("samples", []))[:1800]
        out = openai_json(MODEL_ROUTER, system, joined)
        role = str(out.get("role", "other")).strip()
        if role not in {"candidate/debater", "moderator/host", "journalist/analyst", "commentator", "audience", "narrator", "other"}:
            role = "other"
        roles[sid] = {
            "role": role,
            "confidence": clamp01(out.get("confidence", 0.6)),
            "reason": str(out.get("reason", "")),
            "signals": {
                "duration_sec": round(float(info.get("duration_sec", 0.0)), 3),
                "words": int(info.get("words", 0)),
            }
        }
    return roles

# =========================
# SPEAKER SELECTION (SCORE ELIGIBILITY)
# =========================

EXCLUDE_FROM_SCORING = {"moderator/host", "audience", "narrator"}

def select_speakers_to_score(
    segments: List[Dict[str, Any]],
    roles: Dict[str, Dict[str, Any]],
    content_type: str
) -> Tuple[List[str], Dict[str, Any]]:

    stats = compute_speaker_stats(segments)

    ranked = sorted(
        stats.items(),
        key=lambda kv: (kv[1]["words"], kv[1]["duration_sec"]),
        reverse=True
    )

    debug: Dict[str, Any] = {
        "ranked_by_talk_time": [sid for sid, _ in ranked],
        "fallback_applied": False,
        "fallback_reason": "",
    }

    # ------------------------
    # PANEL
    # ------------------------
    if content_type == "panel":
        panel_speakers = [
            sid for sid, _ in ranked
            if roles.get(sid, {}).get("role") not in {"moderator/host", "audience"}
        ]
        debug["fallback_reason"] = "Panel: included all substantive speakers."
        return panel_speakers, debug

    # ------------------------
    # DEBATE
    # ------------------------
    if content_type == "debate":
        debate_speakers = [
            sid for sid, _ in ranked
            if roles.get(sid, {}).get("role") not in {"moderator/host", "audience"}
        ][:2]
        debug["fallback_reason"] = "Debate: selected top 2 substantive speakers."
        return debate_speakers, debug

    # ------------------------
    # INFORMATIONAL / NEWS / SPORTS
    # ------------------------
    # NOTE: For informational, we DO allow narrator to be included.
    default_speakers = [
        sid for sid, _ in ranked
        if roles.get(sid, {}).get("role") not in {"moderator/host", "audience"}
    ]

    debug["fallback_reason"] = "Informational: included primary speaker."
    return default_speakers[:1], debug

# =========================
# SEGMENT ANALYSIS (bias + claim extraction)
# =========================

ANALYSIS_SYSTEM_TEMPLATE = """
You are evaluating a transcript segment for:

1) rhetorical bias / persuasion (bias_score)
2) political leaning
3) emotional tone
4) classification of segment kind
5) extraction of ONE empirically verifiable claim if present

IMPORTANT:
Bias ≠ Truth.
Bias measures rhetorical framing intensity, exaggeration, loaded language, or persuasion.
Truth-risk is handled separately via web verification.

------------------------------------------------------------
PERFORMANCE METRICS (definitions you must follow)
------------------------------------------------------------

- Fakeness Score (0–1): truth-risk probability-like measure for verifiable claims ONLY (handled later by verifier).
- Bias Score (0–1): degree of linguistic/ideological framing bias.
- Political Leaning: Left / Right / Neutral / Not Applicable.
- Emotion/Tone Level: dominant emotional tone label.
- Overall Misleading Index: if fakeness exists, average(bias, fakeness); else equals bias.
- Model Agreement Score: computed separately (do not compute here).

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Return STRICT JSON with EXACT keys:

{
  "bias_score": number,
  "emotion_tone": string,
  "segment_kind": string,
  "stance_target": string or null,
  "stance_direction": string,
  "political_leaning": string,
  "analysis_confidence": number,
  "claim_text": string or null
}

Allowed values:
- segment_kind: factual_claim, policy_claim, values_statement, question, personal_attack, analysis, descriptive, other
- stance_direction: pro, anti, neutral/unclear
- political_leaning: Left, Right, Neutral, Not Applicable

------------------------------------------------------------
CRITICAL INSTRUCTION: CLAIM EXTRACTION
------------------------------------------------------------

Your primary task is to aggressively detect empirically testable content.

A verifiable claim is ANY statement that could be evaluated against:

• statistics
• laws or legal status
• institutional reports
• official documents
• court rulings
• election results
• public statements
• documented events
• measurable trends
• numerical quantities
• historical records

This includes:

• numeric assertions ("81 million", "over 20 states")
• trend claims ("crime is rising", "economy is collapsing")
• legal claims ("is illegal", "no state allows")
• institutional claims ("FBI says", "court ruled")
• attribution claims ("X said that...")
• comparative claims ("largest in history", "worst ever")
• existence claims ("there are bans", "this program exists")
• causal claims ("X causes Y")

------------------------------------------------------------
REWRITE RULE
------------------------------------------------------------

When extracting claim_text:

• Preserve the specific factual assertion that was actually made.
• Keep named individuals, states, institutions, and numerical values.
• Keep direct attributions ("X said Y").
• Remove emotional framing and rhetorical exaggeration only.
• Make the claim independently searchable without changing its substance.

CRITICAL:

Do NOT generalize, abstract, or reinterpret the claim.

Do NOT convert a specific attributed statement into a broad policy description.

Example:

"Governor X said the baby will be born and we will decide what to do."

Correct rewrite:
"Governor X stated that decisions about a baby born alive would be made after birth."

Incorrect rewrite:
"There are policies allowing decisions after birth."

Preserve superlative and comparative magnitude claims
(e.g., largest, biggest, most, worst, highest, lowest).
Do NOT reduce them to generic existence statements.

If a clear, specific, empirically testable factual assertion is present,
you should extract it.

If multiple factual claims exist, extract the SINGLE most specific,
concrete, and directly verifiable one.

If multiple factual claims exist, prioritize:
1) Numerical claims
2) Legal or institutional claims
3) Claims about an opponent’s actions or statements
4) Then self-referential pledges last

If the only possible extraction would require abstraction or reinterpretation,
set claim_text = null.

------------------------------------------------------------
DO NOT EXTRACT THESE AS FACTUAL CLAIMS
------------------------------------------------------------

Do NOT extract claims that depend primarily on:

• motive or intent attribution ("attempted to divide", "trying to undermine")
• moral judgment ("immoral", "evil", "dangerous ideology")
• ideological labeling ("Marxist", "fascist", unless self-declared and documented)
• purely speculative predictions without reference to stated plans or commitments
• purely subjective evaluation ("worst president ever")
• generalized character attacks

These are interpretive or opinion-based and not empirically verifiable.

In such cases, set claim_text = null.
"""

def analyze_segment_with_model(model: str, text: str, content_type: str) -> Dict[str, Any]:
    user = json.dumps({"content_type": content_type, "text": text[:2500]})
    out = openai_json(model, ANALYSIS_SYSTEM_TEMPLATE, user)

    bias = clamp01(out.get("bias_score", 0.3), default=0.3)
    conf = clamp01(out.get("analysis_confidence", 0.7), default=0.7)

    segment_kind = str(out.get("segment_kind", "other"))
    if segment_kind not in {"factual_claim", "policy_claim", "values_statement", "question", "personal_attack", "analysis", "descriptive", "other"}:
        segment_kind = "other"

    stance_direction = str(out.get("stance_direction", "neutral/unclear"))
    if stance_direction not in {"pro", "anti", "neutral/unclear"}:
        stance_direction = "neutral/unclear"

    political_leaning = str(out.get("political_leaning", "Not Applicable"))
    if political_leaning not in {"Left", "Right", "Neutral", "Not Applicable"}:
        political_leaning = "Not Applicable"

    claim_text = out.get("claim_text", None)
    if isinstance(claim_text, str):
        claim_text = claim_text.strip()
        if not claim_text:
            claim_text = None
        elif len(claim_text) > 220:
            claim_text = claim_text[:220].rstrip()
    else:
        claim_text = None

    return {
        "bias_score": bias,
        "emotion_tone": str(out.get("emotion_tone", "neutral")),
        "segment_kind": segment_kind,
        "stance_target": out.get("stance_target", None),
        "stance_direction": stance_direction,
        "political_leaning": political_leaning,
        "analysis_confidence": conf,
        "claim_text": claim_text,
    }

# =========================
# VERIFICATION (OpenAI web_search)
# =========================

VERIFY_SYSTEM_TEMPLATE = """
You are verifying a factual claim using web search.

Your task:
1. Clarify the claim into a precise, empirically testable statement.
2. Search for high-quality evidence.
3. Evaluate whether the claim is supported or contradicted.

IMPORTANT:

- Do NOT treat the fact that someone said the statement as evidence of truth.
- Reporting that a statement was made ≠ verifying it is accurate.
- You must evaluate factual accuracy.

SCORING:

fakeness_score represents the probability the claim is false.

Higher score = higher likelihood the claim is false.

Use the full continuous 0–1 range.
Avoid default anchor values.

Guidance:
- Clearly contradicted → >= 0.75
- Clearly supported → <= 0.25
- Mixed / partial → between 0.3–0.7

Return STRICT JSON:
{
  "clarified_claim": string,
  "fakeness_score": number,
  "fakeness_confidence": number,
  "reasoning": string,
  "sources": [{"title":string,"url":string}]
}
"""

@dataclass
class VerificationResult:
    verdict: str
    fakeness_score: float
    fakeness_confidence: float
    sources: List[Dict[str, str]]
    reasoning: str

_verify_cache: Dict[str, VerificationResult] = {}

def verify_claim(claim_text: str, max_sources: int) -> Optional[VerificationResult]:

    key = (claim_text or "").strip()
    if not key:
        return None

    if key in _verify_cache:
        return _verify_cache[key]

    user = json.dumps({
        "claim": claim_text,
        "context": "This claim was made in a public political transcript."
    })

    out = openai_json(
        MODEL_VERIFY,
        VERIFY_SYSTEM_TEMPLATE,
        user,
        tools=[{"type": "web_search"}],
    )

    if not out:
        return None

    score = out.get("fakeness_score", None)
    reasoning = str(out.get("reasoning", "")).strip()

    if score is None or not reasoning:
        return None

    score = clamp01(score)

    sources_raw = out.get("sources", []) or []
    sources: List[Dict[str, str]] = []

    for s in sources_raw:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title", "")).strip()
        url = str(s.get("url", "")).strip()
        if url:
            sources.append({"title": title, "url": url})
        if len(sources) >= max_sources:
            break

    # If no usable sources returned, attempt fallback
    if not sources:
        urls = re.findall(r'https?://[^\s)]+', reasoning)
        if urls:
            sources = [{"title": "Source", "url": u} for u in urls[:max_sources]]
        else:
            return None

    # Simple directional sanity check (lightweight, non-destructive)
    reasoning_l = reasoning.lower()

    false_terms = [
        "false", "contradict", "debunked",
        "misleading", "incorrect", "refuted"
    ]

    true_terms = [
        "clearly supported", "strongly supported",
        "confirmed by", "consistent with official data"
    ]

    if any(t in reasoning_l for t in false_terms) and score < 0.5:
        score = max(score, 0.7)

    if any(t in reasoning_l for t in true_terms) and score > 0.5:
        score = min(score, 0.3)

    confidence = clamp01(out.get("fakeness_confidence", 0.7))

    vr = VerificationResult(
        verdict="continuous",
        fakeness_score=score,
        fakeness_confidence=confidence,
        sources=sources,
        reasoning=reasoning,
    )

    _verify_cache[key] = vr
    return vr

# =========================
# MODEL AGREEMENT (optional)
# =========================

def compute_segment_agreement(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    0–1 agreement score from:
    - bias closeness (primary)
    - political leaning match (bonus)
    - segment_kind match (bonus)
    """
    bias_a = float(a["bias_score"])
    bias_b = float(b["bias_score"])
    bias_sim = 1.0 - min(1.0, abs(bias_a - bias_b))  # 1 when equal

    lean_match = 1.0 if a["political_leaning"] == b["political_leaning"] else 0.0
    kind_match = 1.0 if a["segment_kind"] == b["segment_kind"] else 0.0

    return clamp01(0.75 * bias_sim + 0.15 * lean_match + 0.10 * kind_match, default=0.5)


def run_scoring(input_transcript_json: str, out_path: str = "", workers: int = DEFAULT_WORKERS,
                min_words: int = DEFAULT_MIN_WORDS, max_sources: int = DEFAULT_MAX_SOURCES,
                enable_agreement_flag: bool = False) -> str:
    """
    Runs scoring on a transcript JSON and writes *_scored.json
    Returns scored json path.
    """
    st_total = stage_start("SCORE: start")

    if not os.path.exists(input_transcript_json):
        raise FileNotFoundError(input_transcript_json)

    st = stage_start("SCORE: load transcript json")
    with open(input_transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments: List[Dict[str, Any]] = data.get("segments") or []
    meta_in: Dict[str, Any] = data.get("meta") or {}
    stage_end(st)

    # Routing FIRST
    st = stage_start("SCORE: routing (detect content type)")
    content_info = detect_content_type(segments)
    content_type = content_info["content_type"]
    stage_end(st)
    log_info(f"Routing => content_type={content_type} | confidence={content_info['confidence']:.3f}")

    # Expand AFTER routing
    st = stage_start("SCORE: expand long segments (post-routing)")
    segments = expand_segments_for_analysis(segments, content_type)
    stage_end(st)
    log_info(f"Segments after expand: {len(segments)}")

    # Roles
    st = stage_start("SCORE: detect speaker roles")
    roles = detect_speaker_roles(segments, content_type)
    stage_end(st)

    # Speaker selection
    st = stage_start("SCORE: select speakers to score")
    speakers_to_score, selection_debug = select_speakers_to_score(segments, roles, content_type)
    speakers_to_score_set = set(speakers_to_score)
    stage_end(st)
    log_info(f"Speakers scored: {speakers_to_score}")

    enable_agreement = bool(enable_agreement_flag or DEFAULT_ENABLE_MODEL_AGREEMENT) and bool(MODEL_ANALYZE_2)
    log_info(f"Model agreement enabled: {enable_agreement} (MODEL_ANALYZE_2={'set' if MODEL_ANALYZE_2 else 'unset'})")

    scored_segments: List[Dict[str, Any]] = []
    verified_count = 0

    agreement_vals: List[float] = []

    st = stage_start("SCORE: parallel segment processing (analysis + optional verify)")
    def process(seg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sid = safe_speaker_id(seg)
        txt = safe_text(seg)
        if sid not in speakers_to_score_set:
            return None
        if word_count(txt) < min_words and not txt.endswith("?"):
            return None

        # Analysis pass 1
        a1 = analyze_segment_with_model(MODEL_ANALYZE_1, txt, content_type)

        # Analysis pass 2 (optional agreement)
        a2 = None
        seg_agree = None
        if enable_agreement and MODEL_ANALYZE_2:
            a2 = analyze_segment_with_model(MODEL_ANALYZE_2, txt, content_type)
            seg_agree = compute_segment_agreement(a1, a2)

        # Verification (ONLY if claim_text exists)
        fakeness_score = None
        fakeness_confidence = None
        verdict = None
        sources: List[Dict[str, str]] = []
        verification_reasoning = None

        claim_text = a1.get("claim_text")
        if claim_text:
            vr = verify_claim(claim_text, max_sources)
            if vr is not None:
                verdict = vr.verdict
                fakeness_score = float(vr.fakeness_score)
                fakeness_confidence = float(vr.fakeness_confidence)
                sources = vr.sources[: max_sources]
                verification_reasoning = vr.reasoning

        om = overall_misleading_index(float(a1["bias_score"]), fakeness_score)

        st_, en_ = safe_time(seg)

        return {
            "segment_id": seg.get("segment_id"),
            "speaker": sid,
            "time": {"start": st_, "end": en_},
            "text": txt,
            "scores": {
                "bias_score": round(float(a1["bias_score"]), 3),
                "analysis_confidence": round(float(a1["analysis_confidence"]), 3),

                "fakeness_score": None if fakeness_score is None else round(float(fakeness_score), 3),
                "fakeness_confidence": None if fakeness_confidence is None else round(float(fakeness_confidence), 3),
                "verification_verdict": verdict,
                "sources": sources,
                "verification_reasoning": verification_reasoning,

                "overall_misleading_index": round(float(om), 3),

                "emotion_tone": a1["emotion_tone"],
                "segment_kind": a1["segment_kind"],
                "stance_target": a1["stance_target"],
                "stance_direction": a1["stance_direction"],
                "political_leaning": a1["political_leaning"],

                "claim_text": claim_text,

                "model_agreement_score": None if seg_agree is None else round(float(seg_agree), 3),
            },
            "debug": {
                "analysis_model_1": MODEL_ANALYZE_1,
                "analysis_model_2": MODEL_ANALYZE_2 if enable_agreement else None,
            }
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(process, s) for s in segments]
        for fut in as_completed(futures):
            r = fut.result()
            if not r:
                continue
            scored_segments.append(r)
            if r["scores"]["fakeness_score"] is not None:
                verified_count += 1
            if r["scores"]["model_agreement_score"] is not None:
                agreement_vals.append(float(r["scores"]["model_agreement_score"]))

    stage_end(st)

    st = stage_start("SCORE: sort + summarize")
    # Sort chronologically
    scored_segments.sort(key=lambda x: (x["time"]["start"] if isinstance(x.get("time", {}).get("start"), (int, float)) else 1e18))

    # Build summaries
    global_bias: List[float] = []
    global_fake: List[float] = []
    global_overall: List[float] = []

    per_speaker_scores: Dict[str, List[Dict[str, Any]]] = {}

    for s in scored_segments:
        sc = s["scores"]
        global_bias.append(float(sc["bias_score"]))
        global_overall.append(float(sc["overall_misleading_index"]))
        if sc["fakeness_score"] is not None:
            global_fake.append(float(sc["fakeness_score"]))
        sp = s["speaker"]
        per_speaker_scores.setdefault(sp, []).append(sc)

    summary_per_speaker: Dict[str, Any] = {}
    for sp, scores in per_speaker_scores.items():
        bias_vals = [float(x["bias_score"]) for x in scores]
        overall_vals = [float(x["overall_misleading_index"]) for x in scores]
        fake_vals = [float(x["fakeness_score"]) for x in scores if x["fakeness_score"] is not None]
        agree_vals_sp = [float(x["model_agreement_score"]) for x in scores if x.get("model_agreement_score") is not None]

        summary_per_speaker[sp] = {
            "role": roles.get(sp, {}).get("role"),
            "role_confidence": round(float(roles.get(sp, {}).get("confidence", 0.0)), 3),
            "segments": len(scores),
            "avg_bias": round(avg(bias_vals), 3) if bias_vals else 0.0,
            "avg_fakeness": None if not fake_vals else round(avg(fake_vals), 3),
            "avg_overall_misleading_index": round(avg(overall_vals), 3) if overall_vals else 0.0,
            "avg_model_agreement_score": None if not agree_vals_sp else round(avg(agree_vals_sp), 3),
        }

    # Agreement global
    model_agreement_global = None
    if agreement_vals:
        model_agreement_global = round(avg(agreement_vals), 3)

    stage_end(st)

    st = stage_start("SCORE: write scored json")
    out: Dict[str, Any] = {
        "meta": {
            "input_file": os.path.basename(input_transcript_json),
            "created_local": now_local_str(),
            "score_version": SCORE_VERSION,
            "content_type": content_type,
            "routing_confidence": round(float(content_info["confidence"]), 3),
            "routing_notes": content_info["notes"],
            "runtime_sec": None,  # filled below
            "settings": {
                "MIN_WORDS_TO_SCORE": min_words,
                "MAX_SOURCES_PER_VERIFIED_SEGMENT": max_sources,
                "PARALLEL_WORKERS": workers,
                "WEB_VERIFICATION_ENABLED": True,
                "MODEL_ROUTER": MODEL_ROUTER,
                "MODEL_ANALYZE_1": MODEL_ANALYZE_1,
                "MODEL_ANALYZE_2": MODEL_ANALYZE_2 if enable_agreement else None,
                "MODEL_VERIFY": MODEL_VERIFY,
                "MODEL_AGREEMENT_ENABLED": bool(enable_agreement),
            },
            "speakers_scored": speakers_to_score,
            "speaker_roles": roles,
            "speaker_selection_debug": selection_debug,
            "transcribe_meta": meta_in,
        },
        "definitions": DEFINITIONS,
        "summary": {
            "global": {
                "avg_bias": round(avg(global_bias), 3) if global_bias else 0.0,
                "avg_fakeness": None if not global_fake else round(avg(global_fake), 3),
                "avg_overall_misleading_index": round(avg(global_overall), 3) if global_overall else 0.0,
                "segments_scored": len(scored_segments),
                "segments_verified": verified_count,
                "model_agreement_score": model_agreement_global,
            },
            "per_speaker": summary_per_speaker,
        },
        "segments": scored_segments,
    }

    # runtime fill
    runtime = round(time.time() - st_total.t0, 3)
    out["meta"]["runtime_sec"] = runtime

    base = os.path.splitext(input_transcript_json)[0]

# Remove trailing "_transcript" if present
    # Always write scored file to current working directory unless overridden
    if out_path.strip():
        scored_path = out_path.strip()
    else:
        base = os.path.splitext(os.path.basename(input_transcript_json))[0]

    # Remove trailing "_transcript" if present
    if base.endswith("_transcript"):
        base = base[:-11]

    scored_path = os.path.join(os.getcwd(), base + "_scored.json")

    # Write scored JSON file
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    stage_end(st)
    stage_end(st_total)

    log_info(f"✅ Scored JSON saved: {scored_path}")
    log_info(f"Content type: {content_type} | Segments scored: {len(scored_segments)} | Verified: {verified_count}")
    return scored_path


# ============================================================
# ============================================================
# MAIN (Unified CLI)
# ============================================================
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_file", help="Audio file to transcribe and score (e.g., debate.mp3)")

    # scoring args (preserved)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--min_words", type=int, default=DEFAULT_MIN_WORDS)
    ap.add_argument("--max_sources", type=int, default=DEFAULT_MAX_SOURCES)
    ap.add_argument("--enable_agreement", action="store_true", help="Enable model agreement if MODEL_ANALYZE_2 is set")

    # optional outputs
    ap.add_argument("--transcript_out", default="", help="Override transcript output path")
    ap.add_argument("--score_out", default="", help="Override scored output path")

    args = ap.parse_args()

    st_all = stage_start("PIPELINE: total")

    # 1) TRANSCRIBE
    transcript_path = run_transcription(args.audio_file)
    if args.transcript_out.strip():
        # rename/move transcript if requested
        st = stage_start("PIPELINE: apply transcript_out override")
        try:
            os.replace(transcript_path, args.transcript_out.strip())
            transcript_path = args.transcript_out.strip()
        finally:
            stage_end(st)

    # 2) SCORE (based on transcript json)
    scored_path = run_scoring(
        input_transcript_json=transcript_path,
        out_path=args.score_out,
        workers=args.workers,
        min_words=args.min_words,
        max_sources=args.max_sources,
        enable_agreement_flag=args.enable_agreement
    )

    stage_end(st_all)

    print("\n========================")
    print("✅ FINAL OUTPUTS")
    print("========================")
    print("Transcript:", transcript_path)
    print("Scored:    ", scored_path)
    print("========================\n")


if __name__ == "__main__":
    main()
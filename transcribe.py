# ============================================================
# UNIVERSAL TRANSCRIBE v5 (Stable + Speaker Consolidation)
# - Dynamic chunk count
# - Parallel diarization
# - Improved stitching
# - Diarization split consolidation (NEW)
# - Safe for debates, panels, highlights
# ============================================================

from openai import OpenAI
import json, os, re, time, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

client = OpenAI()

MODEL = "gpt-4o-transcribe-diarize"
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

# --------------------------
# Main
# --------------------------

def main():

    if len(sys.argv) < 2:
        raise ValueError("Usage: python transcribe.py <audiofile.mp3>")

    AUDIO_FILE = sys.argv[1]

    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(AUDIO_FILE)

    print("🔎 Getting duration...")
    total_duration = get_audio_duration_seconds(AUDIO_FILE)

    # Dynamic chunk sizing
    if total_duration < 300:
        NUM_CHUNKS = 2
    elif total_duration < 1200:
        NUM_CHUNKS = 6
    else:
        NUM_CHUNKS = 8

    chunk_len = total_duration / NUM_CHUNKS

    print(f"✂️ Splitting into {NUM_CHUNKS} chunks...")

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

    print("⚡ Parallel transcription...")

    def transcribe_chunk(path, offset):
        with open(path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                file=f,
                model=MODEL,
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

    out = {
        "meta": {
            "audio_file": AUDIO_FILE,
            "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": MODEL,
            "transcribe_version": TRANSCRIBE_VERSION,
            "segments_out": len(segments)
        },
        "segments": segments
    }

    out_path = f"{os.path.splitext(AUDIO_FILE)[0]}_transcript.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("✅ Saved:", out_path)
    print("Segments:", len(segments))

if __name__ == "__main__":
    main()
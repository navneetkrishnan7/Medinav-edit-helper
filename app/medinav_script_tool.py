"""
Medinav Script Tool - single-file desktop app.

Drag a video in -> extract audio -> transcribe -> separate the 2 speakers ->
pick the doctor's voice -> drop repeated takes (keep the LAST) and fix grammar.

Source of truth is the GitHub repo below. On launch the app checks the repo for a
newer version of itself and self-updates, so editing on your Mac and pushing is all
it takes to update every Windows machine.
"""
import os
import re
import sys
import time
import traceback
import tempfile
import shutil
import subprocess
import wave
import json
import csv
import html
import logging
import zipfile
import xml.sax.saxutils as xml_escape
from datetime import datetime

__version__ = "1.1.3"

# --------------------------------------------------------------------------- #
# Config (.env lives next to this file)
# --------------------------------------------------------------------------- #

def _load_dotenv():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(_HERE, "medinav-error.log")
LAST_PROJECT = os.path.join(_HERE, "last-session.medinav")
GLOSSARY_PATH = os.path.join(_HERE, "glossary.txt")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def log_error(label, tb):
    logging.error("%s\n%s", label, tb)

def install_exception_logger():
    def hook(exc_type, exc, tb):
        log_error("Unhandled exception", "".join(traceback.format_exception(exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = hook

def _model_path(env_name, default_name):
    p = os.environ.get(env_name, "")
    return p if p else os.path.join(_HERE, "models", default_name)

# ---- self-update source ----
GITHUB_REPO   = os.environ.get("GITHUB_REPO", "navneetkrishnan7/medinav-edit-helper")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
RAW_BASE      = "https://raw.githubusercontent.com/%s/%s/app" % (GITHUB_REPO, GITHUB_BRANCH)
AUTO_UPDATE   = os.environ.get("AUTO_UPDATE", "1") != "0"

WHISPER_MODEL    = os.environ.get("WHISPER_MODEL", "medium")
ASR_DEVICE       = os.environ.get("ASR_DEVICE", "auto")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "auto")
ASR_LANGUAGE     = os.environ.get("ASR_LANGUAGE", "en")
NUM_SPEAKERS     = int(os.environ.get("NUM_SPEAKERS", "2"))
SEG_MODEL        = _model_path("SEG_MODEL", "segmentation.onnx")
EMB_MODEL        = _model_path("EMB_MODEL", "embedding.onnx")
CLEANUP_BACKEND  = os.environ.get("CLEANUP_BACKEND", "claude")
ANTHROPIC_API_KEY= os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL     = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST      = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# --------------------------------------------------------------------------- #
# Self-update from GitHub
# --------------------------------------------------------------------------- #

def _vtuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or ""))

def _extract_version(text):
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None

def maybe_update():
    """Check the repo for a newer version; if found, replace this file and relaunch."""
    if not AUTO_UPDATE:
        return
    import urllib.request
    try:
        url = RAW_BASE + "/medinav_script_tool.py?nocache=" + str(int(time.time()))
        code = urllib.request.urlopen(url, timeout=8).read().decode("utf-8")
    except Exception:
        return  # offline or unreachable: just run what we have
    remote = _extract_version(code)
    if not remote or _vtuple(remote) <= _vtuple(__version__):
        return
    try:
        compile(code, "medinav_script_tool.py", "exec")  # never apply a broken update
    except SyntaxError:
        return
    try:
        with open(os.path.abspath(__file__), "w", encoding="utf-8") as f:
            f.write(code)
    except OSError:
        return
    subprocess.Popen([sys.executable, os.path.abspath(__file__)] + sys.argv[1:])
    sys.exit(0)

# --------------------------------------------------------------------------- #
# Stage 1: audio
# --------------------------------------------------------------------------- #

def extract_audio(video_path, sample_rate=16000):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Re-run the installer.")
    fd, out = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-f", "wav", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("Could not read audio from this file:\n" + p.stderr[-1200:])
    return out

# --------------------------------------------------------------------------- #
# Stage 2: transcription (faster-whisper, local)
# --------------------------------------------------------------------------- #

_whisper = None

def _device():
    return ASR_DEVICE if ASR_DEVICE != "auto" else "cpu"

def _compute(dev):
    if ASR_COMPUTE_TYPE != "auto":
        return ASR_COMPUTE_TYPE
    return "float16" if dev == "cuda" else "int8"

def transcribe(audio_path):
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        dev = _device()
        _whisper = WhisperModel(WHISPER_MODEL, device=dev, compute_type=_compute(dev))
    segments, _ = _whisper.transcribe(audio_path, language=ASR_LANGUAGE,
                                      vad_filter=True, beam_size=5)
    out = []
    for s in segments:
        t = (s.text or "").strip()
        if t:
            out.append({"start": float(s.start), "end": float(s.end), "text": t})
    return out

# --------------------------------------------------------------------------- #
# Stage 3: speaker separation (sherpa-onnx, fully local, no token)
# --------------------------------------------------------------------------- #

_diar = None

def _diar_engine():
    global _diar
    if _diar is None:
        import sherpa_onnx
        if not (os.path.exists(SEG_MODEL) and os.path.exists(EMB_MODEL)):
            raise RuntimeError("Speaker models were not found. Re-run the installer.")
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL)),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=NUM_SPEAKERS),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not cfg.validate():
            raise RuntimeError("Could not initialize the speaker model.")
        _diar = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return _diar

def read_wave_mono(path):
    import numpy as np
    with wave.open(path, "rb") as f:
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        sample_rate = f.getframerate()
        frames = f.readframes(f.getnframes())
    if channels != 1:
        raise RuntimeError("Expected mono audio but found %d channels." % channels)
    if sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError("Unsupported WAV sample width: %d bytes." % sample_width)
    return samples, sample_rate

def diarize_and_label(audio_path, segments):
    sd = _diar_engine()
    samples, sr = read_wave_mono(audio_path)
    if sr != sd.sample_rate:
        raise RuntimeError("Unexpected audio sample rate (%d)." % sr)
    result = sd.process(samples).sort_by_start_time()
    turns = [{"start": float(s.start), "end": float(s.end),
              "speaker": "Speaker " + str(s.speaker + 1)} for s in result]
    for seg in segments:
        best, best_ov = None, 0.0
        for t in turns:
            ov = max(0.0, min(seg["end"], t["end"]) - max(seg["start"], t["start"]))
            if ov > best_ov:
                best_ov, best = ov, t["speaker"]
        seg["speaker"] = best or "Speaker 1"
    return segments

def speaker_samples(segments, max_chars=280):
    samples, counts = {}, {}
    for seg in segments:
        spk = seg.get("speaker", "Speaker 1")
        counts[spk] = counts.get(spk, 0) + 1
        samples.setdefault(spk, "")
        if len(samples[spk]) < max_chars:
            samples[spk] += " " + seg["text"]
    return {spk: {"sample": samples[spk].strip(), "segments": counts[spk]} for spk in samples}

def selection_name(speaker):
    if isinstance(speaker, (list, tuple)):
        return "Merged: " + ", ".join(str(s) for s in speaker)
    return speaker or ""

def selected_utterances(segments, speaker):
    utterances = sorted(segments or [], key=lambda s: (s.get("start", 0), s.get("end", 0)))
    if speaker == ALL_SPEECH:
        return utterances
    if isinstance(speaker, (list, tuple, set)):
        labels = {str(s) for s in speaker}
        return [s for s in utterances if s.get("speaker") in labels]
    return [s for s in utterances if s.get("speaker") == speaker]

def utterance_sample(utterances, max_chars=280):
    text = ""
    for seg in utterances or []:
        if len(text) >= max_chars:
            break
        text += " " + seg.get("text", "")
    return text.strip()

def write_audio_preview(audio_path, utterances, label, preview_dir, max_seconds=12.0):
    if not utterances:
        return ""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower() or "speaker"
    out = os.path.join(preview_dir, safe + ".wav")
    try:
        with wave.open(audio_path, "rb") as src, wave.open(out, "wb") as dst:
            dst.setparams(src.getparams())
            rate = src.getframerate()
            width = src.getsampwidth()
            channels = src.getnchannels()
            total_frames = src.getnframes()
            silence = b"\x00" * int(rate * 0.15) * width * channels
            written = 0.0
            for seg in utterances:
                if written >= max_seconds:
                    break
                start = max(0.0, float(seg.get("start", 0)))
                end = max(start, float(seg.get("end", start)))
                dur = min(end - start, max_seconds - written)
                if dur <= 0:
                    continue
                start_frame = min(total_frames, int(start * rate))
                frames = max(1, int(dur * rate))
                src.setpos(start_frame)
                data = src.readframes(frames)
                if not data:
                    continue
                if written > 0:
                    dst.writeframes(silence)
                dst.writeframes(data)
                written += dur
        return out if os.path.getsize(out) > 44 else ""
    except Exception:
        logging.exception("Could not create audio preview for %s", label)
        return ""

def add_audio_previews(audio_path, segments, samples):
    preview_dir = tempfile.mkdtemp(prefix="medinav_previews_")
    samples[ALL_SPEECH] = {
        "sample": utterance_sample(segments),
        "segments": len(segments or []),
    }
    for label, info in list(samples.items()):
        info["preview_path"] = write_audio_preview(
            audio_path, selected_utterances(segments, label), label, preview_dir)
    return preview_dir

def fmt_time(seconds):
    seconds = max(0.0, float(seconds or 0))
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%02d.%03d" % (h, m, s, ms)

def fmt_srt_time(seconds):
    return fmt_time(seconds).replace(".", ",")

def split_script_lines(script):
    chunks = []
    for para in re.split(r"\n\s*\n", (script or "").strip()):
        para = para.strip()
        if not para:
            continue
        parts = re.split(r"(?<=[.!?])\s+", para)
        chunks.extend(p.strip() for p in parts if p.strip())
    return chunks or ([script.strip()] if script.strip() else [])

def review_flags_for_segments(segments):
    text = " ".join(s.get("text", "") for s in segments).strip()
    flags = []
    if len(text.split()) <= 3:
        flags.append("Very short source")
    if re.search(r"\b(um+|uh+|erm+|hmm+)\b", text, re.I):
        flags.append("Filler words")
    if re.search(r"\b(again|cut|from the top|one more|retake)\b", text, re.I):
        flags.append("Possible director cue")
    if len(text) > 260:
        flags.append("Long source span")
    if re.search(r"[^A-Za-z0-9\s.,;:'\"!?()/%+-]", text) and len(text) < 40:
        flags.append("Check transcription")
    return "; ".join(flags)

def build_edit_map(script, utterances):
    lines = split_script_lines(script)
    if not lines or not utterances:
        return []
    per = max(1, int(round(len(utterances) / max(1, len(lines)))))
    out, cursor = [], 0
    for i, line in enumerate(lines, 1):
        remaining_lines = len(lines) - i + 1
        remaining_utts = len(utterances) - cursor
        take = max(1, int(round(remaining_utts / remaining_lines))) if remaining_lines else per
        chunk = utterances[cursor: cursor + take] or utterances[-1:]
        cursor += take
        start = min(u["start"] for u in chunk)
        end = max(u["end"] for u in chunk)
        out.append({
            "line": i,
            "text": line,
            "start": start,
            "end": end,
            "timecode": fmt_time(start) + " - " + fmt_time(end),
            "source": " ".join(u.get("text", "") for u in chunk).strip(),
            "flags": review_flags_for_segments(chunk),
        })
    return out

def processing_summary(video_path, segments, speaker, script, edit_map, timings):
    selected = selected_utterances(segments, speaker)
    words = len((script or "").split())
    duration = max((s.get("end", 0) for s in segments), default=0)
    selected_duration = sum(max(0, s.get("end", 0) - s.get("start", 0)) for s in selected)
    return {
        "video": video_path or "",
        "duration_seconds": duration,
        "selected_speaker": selection_name(speaker),
        "selected_speaker_seconds": selected_duration,
        "segments": len(segments),
        "selected_segments": len(selected),
        "script_words": words,
        "edit_lines": len(edit_map or []),
        "estimated_tokens": max(1, int(words * 1.35)) + 900,
        "timings": timings or {},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

def summary_text(summary):
    if not summary:
        return ""
    timings = summary.get("timings", {})
    lines = [
        "Video: " + os.path.basename(summary.get("video", "")),
        "Duration: " + fmt_time(summary.get("duration_seconds", 0)),
        "Selected speaker: " + str(summary.get("selected_speaker", "")),
        "Selected speaker time: " + fmt_time(summary.get("selected_speaker_seconds", 0)),
        "Transcript segments: %s total / %s selected" % (
            summary.get("segments", 0), summary.get("selected_segments", 0)),
        "Script words: %s" % summary.get("script_words", 0),
        "Edit map lines: %s" % summary.get("edit_lines", 0),
        "Approx cleanup tokens: %s" % summary.get("estimated_tokens", 0),
    ]
    if timings:
        lines.append("")
        lines.append("Processing time")
        for k, v in timings.items():
            lines.append("- %s: %.1fs" % (k, float(v)))
    return "\n".join(lines)

def load_glossary():
    if not os.path.exists(GLOSSARY_PATH):
        return ""
    try:
        return open(GLOSSARY_PATH, encoding="utf-8").read().strip()
    except OSError:
        return ""

def save_glossary(text):
    with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
        f.write((text or "").strip() + "\n")

def project_data(video_path, segments, selected_speaker, script, tamil, edit_map, summary):
    return {
        "app": "Medinav Script Tool",
        "version": __version__,
        "video_path": video_path or "",
        "segments": segments or [],
        "selected_speaker": selected_speaker or "",
        "english_script": script or "",
        "tamil_reference": tamil or "",
        "edit_map": edit_map or [],
        "summary": summary or {},
        "glossary": load_glossary(),
    }

def save_project_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_project_file(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def write_edit_csv(path, edit_map):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line", "timecode", "start", "end", "text", "source", "flags"])
        w.writeheader()
        for row in edit_map or []:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})

def write_srt(path, edit_map):
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(edit_map or [], 1):
            f.write("%d\n%s --> %s\n%s\n\n" % (
                i, fmt_srt_time(row.get("start", 0)), fmt_srt_time(row.get("end", 0)), row.get("text", "")))

def _docx_para(text):
    text = xml_escape.escape(text or "")
    return "<w:p><w:r><w:t xml:space=\"preserve\">%s</w:t></w:r></w:p>" % text

def write_docx(path, title, sections):
    body = [_docx_para(title)]
    for heading, content in sections:
        body.append(_docx_para(""))
        body.append(_docx_para(heading))
        for para in re.split(r"\n\s*\n", content or ""):
            if para.strip():
                body.append(_docx_para(para.strip()))
    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>%s<w:sectPr/></w:body></w:document>""" % "".join(body)
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)

def write_bilingual_html(path, english, tamil):
    page = """<!doctype html><meta charset="utf-8"><title>Medinav Script</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;color:#132A3A}table{width:100%%;border-collapse:collapse}th,td{width:50%%;vertical-align:top;border:1px solid #d6e2ea;padding:14px;line-height:1.5}th{background:#f0f7f8;text-align:left}</style>
<h1>Medinav Script</h1><table><tr><th>English</th><th>Tamil reference</th></tr><tr><td>%s</td><td>%s</td></tr></table>""" % (
        html.escape(english or "").replace("\n", "<br>"),
        html.escape(tamil or "").replace("\n", "<br>"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)

# --------------------------------------------------------------------------- #
# Stage 4: cleanup (retake dedup + grammar) via Claude or local Ollama
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a transcript editor for a medical video production company.

The on-camera speaker is a doctor recording short marketing / educational videos. Everything is captured in ONE continuous take file, so the doctor usually repeats the same line several times before getting it right. THE LAST ATTEMPT AT ANY GIVEN LINE IS THE FINAL, BEST TAKE - earlier attempts at that same line must be discarded. After nailing a line the doctor moves on to the next sentence.

The doctor is often not a native English speaker, and the transcript comes from automatic speech recognition, so expect false starts, filler words, broken grammar, and mis-transcribed words.

Your job:
1. Read the time-ordered transcript segments (all from this one speaker).
2. Find runs of consecutive segments that are repeated attempts (retakes) of the SAME intended line. Keep ONLY the last attempt in each run; drop the earlier attempts.
3. Stitch the kept lines together, in order, into one continuous script.
4. Fix grammar, spelling, punctuation, and obvious ASR errors so it reads as fluent, natural, professional English - while preserving the speaker's meaning and ALL medical terminology (drug names, procedures, conditions, anatomy, dosages). Correct clearly mis-transcribed medical terms (e.g. "myocardial in fraction" -> "myocardial infarction").
5. If a line is too garbled to make perfectly grammatical, produce the CLOSEST fluent version that preserves the intended meaning. NEVER invent claims, statistics, names, or content that was not spoken.
6. Occasionally a stray instruction from the off-camera director may leak in ("again", "from the top", "cut", "look at the camera"). Drop these - they are not part of the script.

Output ONLY the final cleaned script as plain text, with natural paragraph breaks. Do not include timestamps, segment numbers, speaker labels, or any commentary."""

def _user_prompt(utterances, glossary=""):
    lines = [f"[{i}] ({u['start']:.1f}s-{u['end']:.1f}s) {u['text']}"
             for i, u in enumerate(utterances, 1)]
    glossary_note = ""
    if glossary.strip():
        glossary_note = "\n\nPreserve and prefer these approved terms exactly when relevant:\n" + glossary.strip()
    return ("Here are the time-ordered transcript segments from the selected speaker. "
            "Produce the final cleaned script." + glossary_note + "\n\n" + "\n".join(lines))

def cleanup(utterances, glossary=""):
    if not utterances:
        return ""
    if CLEANUP_BACKEND == "ollama":
        import json, urllib.request
        payload = {"model": OLLAMA_MODEL, "system": SYSTEM_PROMPT,
                   "prompt": _user_prompt(utterances, glossary), "stream": False,
                   "options": {"temperature": 0.2}}
        req = urllib.request.Request(OLLAMA_HOST + "/api/generate",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return (json.loads(r.read().decode()).get("response") or "").strip()
    from anthropic import Anthropic
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to the .env file.")
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=4096,
                                 system=SYSTEM_PROMPT,
                                 messages=[{"role": "user", "content": _user_prompt(utterances, glossary)}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()

TRANSLATE_PROMPT = """Translate the supplied English medical video script into natural Tamil for an editor's reference.

Preserve the meaning, names, medical terminology, dosages, and factual claims. Keep paragraph breaks aligned with the English where possible. Do not add explanations, labels, notes, timestamps, or commentary. Output only Tamil text."""

def translate_to_tamil(script):
    script = (script or "").strip()
    if not script:
        return ""
    if CLEANUP_BACKEND == "ollama":
        import json, urllib.request
        payload = {"model": OLLAMA_MODEL, "system": TRANSLATE_PROMPT,
                   "prompt": script, "stream": False,
                   "options": {"temperature": 0.1}}
        req = urllib.request.Request(OLLAMA_HOST + "/api/generate",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return (json.loads(r.read().decode()).get("response") or "").strip()
    from anthropic import Anthropic
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to the .env file.")
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=4096,
                                 system=TRANSLATE_PROMPT,
                                 messages=[{"role": "user", "content": script}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()

SELECTION_PROMPT = """You are helping a Tamil-speaking video editor understand a selected English word, phrase, or sentence from a medical video script.

Translate the selected text into natural Tamil. If it is a medical, dental, procedural, anatomy, dosage, or clinic workflow term, add one short editor-friendly meaning/explanation in Tamil or simple bilingual Tamil-English. Preserve proper nouns and medical terms where direct translation would be confusing.

Output exactly:
Tamil:
...

Meaning:
..."""

def translate_selection_to_tamil(text):
    text = (text or "").strip()
    if not text:
        return ""
    if CLEANUP_BACKEND == "ollama":
        import json, urllib.request
        payload = {"model": OLLAMA_MODEL, "system": SELECTION_PROMPT,
                   "prompt": text, "stream": False,
                   "options": {"temperature": 0.1}}
        req = urllib.request.Request(OLLAMA_HOST + "/api/generate",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return (json.loads(r.read().decode()).get("response") or "").strip()
    from anthropic import Anthropic
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to the .env file.")
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=1200,
                                 system=SELECTION_PROMPT,
                                 messages=[{"role": "user", "content": text}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()

# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QButtonGroup, QCheckBox, QFileDialog,
    QProgressBar, QFrame, QMessageBox, QScrollArea, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialog, QPlainTextEdit, QListWidget,
)
try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
except Exception:
    QAudioOutput = QMediaPlayer = None

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".wmv", ".flv"}
LOGO_NAME = "medinav-logo.jpg"
ICON_NAME = "medinav-icon.ico"
ALL_SPEECH = "All speech"
MERGED_SPEAKERS = "Merged speakers"


def logo_path():
    path = os.path.join(_HERE, LOGO_NAME)
    if os.path.exists(path):
        return path
    if AUTO_UPDATE:
        try:
            import urllib.request
            url = RAW_BASE + "/" + LOGO_NAME + "?nocache=" + str(int(time.time()))
            urllib.request.urlretrieve(url, path)
        except Exception:
            pass
    return path if os.path.exists(path) else ""

def icon_path():
    path = os.path.join(_HERE, ICON_NAME)
    if os.path.exists(path):
        return path
    if AUTO_UPDATE:
        try:
            import urllib.request
            url = RAW_BASE + "/" + ICON_NAME + "?nocache=" + str(int(time.time()))
            urllib.request.urlretrieve(url, path)
        except Exception:
            pass
    return path if os.path.exists(path) else logo_path()

def logo_pixmap(path, logical_size):
    pix = QPixmap(path) if path else QPixmap()
    if pix.isNull():
        return pix
    screen = QApplication.primaryScreen()
    ratio = max(1.0, screen.devicePixelRatio() if screen else 1.0)
    px = int(logical_size * ratio)
    scaled = pix.scaled(px, px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    scaled.setDevicePixelRatio(ratio)
    return scaled

def set_windows_app_id():
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Medinav.ScriptTool")
    except Exception:
        pass

def ensure_windows_shortcut_icon():
    if not sys.platform.startswith("win"):
        return
    ico = icon_path()
    if not ico or not ico.lower().endswith(".ico") or not os.path.exists(ico):
        return
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    shortcut = os.path.join(desktop, "Medinav Script Tool.lnk")
    if not os.path.exists(shortcut):
        return
    script = (
        "$ws=New-Object -ComObject WScript.Shell; "
        "$sc=$ws.CreateShortcut($args[0]); "
        "$sc.IconLocation=$args[1]; "
        "$sc.Save()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-Command", script, shortcut, ico],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except Exception:
        pass


class AnalyzeWorker(QThread):
    progress = Signal(str)
    done = Signal(list, dict, dict, str)
    failed = Signal(str)

    def __init__(self, path):
        super().__init__(); self.path = path

    def run(self):
        wav = None
        timings = {}
        try:
            started = time.time()
            self.progress.emit("Extracting audio...")
            t = time.time()
            wav = extract_audio(self.path)
            timings["audio extraction"] = time.time() - t
            self.progress.emit("Transcribing (this is the slow part)...")
            t = time.time()
            segments = transcribe(wav)
            timings["transcription"] = time.time() - t
            if not segments:
                raise RuntimeError("No speech was detected in this file.")
            self.progress.emit("Separating speakers...")
            t = time.time()
            diarize_and_label(wav, segments)
            timings["speaker separation"] = time.time() - t
            samples = speaker_samples(segments)
            preview_dir = add_audio_previews(wav, segments, samples)
            timings["analysis total"] = time.time() - started
            self.done.emit(segments, samples, timings, preview_dir)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if wav and os.path.exists(wav):
                try: os.remove(wav)
                except OSError: pass


class CleanupWorker(QThread):
    progress = Signal(str)
    done = Signal(str, dict)
    failed = Signal(str)

    def __init__(self, segments, speaker, glossary=""):
        super().__init__(); self.segments = segments; self.speaker = speaker; self.glossary = glossary

    def run(self):
        try:
            utt = selected_utterances(self.segments, self.speaker)
            label = "Claude" if CLEANUP_BACKEND == "claude" else "local model"
            self.progress.emit("Cleaning up the script with " + label + "...")
            t = time.time()
            self.done.emit(cleanup(utt, self.glossary), {"cleanup": time.time() - t})
        except Exception:
            self.failed.emit(traceback.format_exc())


class TranslateWorker(QThread):
    progress = Signal(str)
    done = Signal(str, dict)
    failed = Signal(str)

    def __init__(self, script):
        super().__init__(); self.script = script

    def run(self):
        try:
            label = "Claude" if CLEANUP_BACKEND == "claude" else "local model"
            self.progress.emit("Translating to Tamil with " + label + "...")
            t = time.time()
            self.done.emit(translate_to_tamil(self.script), {"Tamil translation": time.time() - t})
        except Exception:
            self.failed.emit(traceback.format_exc())


class SelectionTranslateWorker(QThread):
    progress = Signal(str)
    done = Signal(str, str)
    failed = Signal(str)

    def __init__(self, text):
        super().__init__(); self.text = text

    def run(self):
        try:
            label = "Claude" if CLEANUP_BACKEND == "claude" else "local model"
            self.progress.emit("Translating selected text with " + label + "...")
            self.done.emit(self.text, translate_selection_to_tamil(self.text))
        except Exception:
            self.failed.emit(traceback.format_exc())


class BatchWorker(QThread):
    progress = Signal(str)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, paths, out_dir, glossary=""):
        super().__init__(); self.paths = paths; self.out_dir = out_dir; self.glossary = glossary

    def run(self):
        results = []
        try:
            for idx, path in enumerate(self.paths, 1):
                base = os.path.splitext(os.path.basename(path))[0]
                self.progress.emit("Batch %d/%d: %s" % (idx, len(self.paths), base))
                wav = None
                try:
                    timings, started = {}, time.time()
                    t = time.time(); wav = extract_audio(path); timings["audio extraction"] = time.time() - t
                    t = time.time(); segments = transcribe(wav); timings["transcription"] = time.time() - t
                    if not segments:
                        raise RuntimeError("No speech was detected.")
                    t = time.time(); diarize_and_label(wav, segments); timings["speaker separation"] = time.time() - t
                    samples = speaker_samples(segments)
                    ordered = sorted(samples.items(), key=lambda kv: kv[1]["segments"], reverse=True)
                    speaker = ALL_SPEECH if len(ordered) <= 2 else ordered[0][0]
                    utt = selected_utterances(segments, speaker)
                    t = time.time(); script = cleanup(utt, self.glossary); timings["cleanup"] = time.time() - t
                    edit_map = build_edit_map(script, utt)
                    timings["analysis total"] = time.time() - started
                    summary = processing_summary(path, segments, speaker, script, edit_map, timings)
                    stem = os.path.join(self.out_dir, base)
                    with open(stem + ".txt", "w", encoding="utf-8") as f:
                        f.write(script)
                    write_edit_csv(stem + "-edit-map.csv", edit_map)
                    write_srt(stem + ".srt", edit_map)
                    write_docx(stem + ".docx", "Medinav Script", [("English", script)])
                    save_project_file(stem + ".medinav", project_data(path, segments, speaker, script, "", edit_map, summary))
                    results.append({"video": path, "ok": True, "speaker": speaker})
                except Exception as e:
                    logging.exception("Batch failed for %s", path)
                    results.append({"video": path, "ok": False, "error": str(e)})
                finally:
                    if wav and os.path.exists(wav):
                        try: os.remove(wav)
                        except OSError: pass
            self.done.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


class GlossaryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Glossary")
        self.resize(560, 420)
        L = QVBoxLayout(self)
        label = QLabel("Terms to preserve exactly when relevant")
        label.setObjectName("sectionLabel")
        L.addWidget(label)
        self.text = QPlainTextEdit()
        self.text.setPlainText(load_glossary())
        self.text.setPlaceholderText("Dr. Wahaab\nInvisalign\nZirconia crown\nMedinav")
        L.addWidget(self.text, 1)
        row = QHBoxLayout(); row.addStretch(1)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        save = QPushButton("Save glossary"); save.setObjectName("primaryButton"); save.clicked.connect(self.accept)
        row.addWidget(cancel); row.addWidget(save); L.addLayout(row)

    def glossary(self):
        return self.text.toPlainText()


class SelectionTranslationDialog(QDialog):
    def __init__(self, original, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Selection translation")
        self.resize(620, 460)
        L = QVBoxLayout(self)
        original_label = QLabel("Selected English")
        original_label.setObjectName("sectionLabel")
        L.addWidget(original_label)
        self.original = QPlainTextEdit()
        self.original.setPlainText(original)
        self.original.setReadOnly(True)
        self.original.setMaximumHeight(110)
        L.addWidget(self.original)
        result_label = QLabel("Tamil help")
        result_label.setObjectName("sectionLabel")
        L.addWidget(result_label)
        self.result = QPlainTextEdit()
        self.result.setPlainText(result)
        self.result.setReadOnly(True)
        L.addWidget(self.result, 1)
        row = QHBoxLayout(); row.addStretch(1)
        copy = QPushButton("Copy Tamil help"); copy.setObjectName("secondaryButton")
        copy.clicked.connect(self.copy_result)
        close = QPushButton("Close"); close.setObjectName("primaryButton")
        close.clicked.connect(self.accept)
        row.addWidget(copy); row.addWidget(close); L.addLayout(row)

    def copy_result(self):
        QApplication.clipboard().setText(self.result.toPlainText())


class DropFrame(QFrame):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropFrame")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(176)
        lay = QVBoxLayout(self); lay.setAlignment(Qt.AlignCenter); lay.setSpacing(8)
        self.label = QLabel("Drop a doctor video")
        self.label.setAlignment(Qt.AlignCenter); self.label.setObjectName("dropLabel")
        self.hint = QLabel("or click to browse")
        self.hint.setAlignment(Qt.AlignCenter); self.hint.setObjectName("dropHint")
        lay.addWidget(self.label); lay.addWidget(self.hint)

    def set_file_name(self, path):
        self.label.setText(os.path.basename(path))
        self.hint.setText("Ready to transcribe")

    def mousePressEvent(self, _):
        path, _f = QFileDialog.getOpenFileName(
            self, "Choose a video", "",
            "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.wmv *.flv);;All files (*.*)")
        if path:
            self.file_dropped.emit(path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.splitext(p)[1].lower() in VIDEO_EXT:
                self.file_dropped.emit(p); return
        QMessageBox.warning(self, "Unsupported file", "Please drop a video file.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medinav Script Tool  v" + __version__)
        app_icon = QIcon(icon_path())
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.resize(940, 680)
        self.segments = []
        self.video_path = ""
        self.selected_speaker = ""
        self.edit_map = []
        self.summary = {}
        self.analysis_timings = {}
        self.speaker_buttons = QButtonGroup(self)
        self.merge_checks = []
        self.merge_radio = None
        self.preview_dir = ""
        self.player = None
        self.audio_output = None
        self._a = self._c = self._t = self._b = self._sel = None

        scroll = QScrollArea()
        scroll.setObjectName("windowScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(scroll)
        root = QWidget()
        scroll.setWidget(root)
        L = QVBoxLayout(root); L.setContentsMargins(28, 24, 28, 24); L.setSpacing(16)

        header = QFrame(); header.setObjectName("header")
        h = QHBoxLayout(header); h.setContentsMargins(16, 14, 16, 14); h.setSpacing(14)
        logo = QLabel(); logo.setObjectName("logo"); logo.setFixedSize(104, 104); logo.setAlignment(Qt.AlignCenter)
        lp = logo_path()
        pix = logo_pixmap(lp, 96)
        if not pix.isNull():
            logo.setPixmap(pix)
        else:
            logo.setText("N")
        h.addWidget(logo)
        brand = QVBoxLayout(); brand.setSpacing(2)
        title = QLabel("Medinav"); title.setObjectName("title")
        subtitle = QLabel("Script Tool"); subtitle.setObjectName("subtitle")
        brand.addWidget(title); brand.addWidget(subtitle)
        h.addLayout(brand, 1)
        version = QLabel("v" + __version__); version.setObjectName("versionPill"); h.addWidget(version, 0, Qt.AlignTop)
        L.addWidget(header)

        tools = QHBoxLayout()
        open_project = QPushButton("Open project"); open_project.setObjectName("secondaryButton"); open_project.clicked.connect(self.open_project)
        load_last = QPushButton("Load last"); load_last.setObjectName("secondaryButton"); load_last.clicked.connect(self.load_last_project)
        save_project = QPushButton("Save project"); save_project.setObjectName("secondaryButton"); save_project.clicked.connect(self.save_project)
        glossary = QPushButton("Glossary"); glossary.setObjectName("secondaryButton"); glossary.clicked.connect(self.edit_glossary)
        batch = QPushButton("Batch process"); batch.setObjectName("secondaryButton"); batch.clicked.connect(self.start_batch)
        tools.addWidget(open_project); tools.addWidget(load_last); tools.addWidget(save_project); tools.addWidget(glossary); tools.addWidget(batch); tools.addStretch(1)
        L.addLayout(tools)

        self.drop = DropFrame(); self.drop.file_dropped.connect(self.start_analysis); L.addWidget(self.drop)

        self.status = QLabel(""); self.status.setObjectName("status"); L.addWidget(self.status)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.hide(); L.addWidget(self.progress)

        self.speaker_panel = QWidget(); self.speaker_layout = QVBoxLayout(self.speaker_panel)
        self.speaker_layout.setContentsMargins(0, 0, 0, 0); self.speaker_panel.hide(); L.addWidget(self.speaker_panel)

        self.generate_btn = QPushButton("Generate script"); self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.clicked.connect(self.start_cleanup)
        self.generate_btn.hide(); L.addWidget(self.generate_btn)

        self.tabs = QTabWidget(); self.tabs.setObjectName("scriptTabs")

        english_tab = QWidget(); english_wrap = QVBoxLayout(english_tab)
        english_wrap.setContentsMargins(12, 12, 12, 12); english_wrap.setSpacing(8)
        english_head = QHBoxLayout()
        english_label = QLabel("Cleaned script"); english_label.setObjectName("sectionLabel")
        english_head.addWidget(english_label); english_head.addStretch(1)
        cp = QPushButton("Copy"); cp.setObjectName("secondaryButton"); cp.clicked.connect(self.copy_output)
        sv = QPushButton("Save as .txt"); sv.setObjectName("secondaryButton"); sv.clicked.connect(self.save_output)
        self.selection_btn = QPushButton("Translate selection"); self.selection_btn.setObjectName("secondaryButton")
        self.selection_btn.setEnabled(False); self.selection_btn.clicked.connect(self.start_selection_translation)
        docx = QPushButton("Export DOCX"); docx.setObjectName("secondaryButton"); docx.clicked.connect(self.export_docx)
        side = QPushButton("Side-by-side HTML"); side.setObjectName("secondaryButton"); side.clicked.connect(self.export_side_by_side)
        english_head.addWidget(cp); english_head.addWidget(sv); english_head.addWidget(self.selection_btn)
        english_head.addWidget(docx); english_head.addWidget(side); english_wrap.addLayout(english_head)
        self.output = QPlainTextEdit(); self.output.setObjectName("output")
        self.output.setPlaceholderText("The cleaned script will appear here.")
        self.output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.output.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.output.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.output.setMinimumHeight(360)
        self.output.copyAvailable.connect(self.selection_btn.setEnabled)
        english_wrap.addWidget(self.output, 1)
        self.tabs.addTab(english_tab, "English")

        self.tamil_tab = QWidget(); tamil_wrap = QVBoxLayout(self.tamil_tab)
        tamil_wrap.setContentsMargins(12, 12, 12, 12); tamil_wrap.setSpacing(8)
        tamil_head = QHBoxLayout()
        tamil_label = QLabel("Tamil reference"); tamil_label.setObjectName("sectionLabel")
        tamil_head.addWidget(tamil_label); tamil_head.addStretch(1)
        self.translate_btn = QPushButton("Translate to Tamil"); self.translate_btn.setObjectName("primaryButton")
        self.translate_btn.clicked.connect(self.start_translation)
        copy_tamil = QPushButton("Copy Tamil"); copy_tamil.setObjectName("secondaryButton")
        copy_tamil.clicked.connect(self.copy_tamil)
        save_tamil = QPushButton("Save Tamil"); save_tamil.setObjectName("secondaryButton")
        save_tamil.clicked.connect(self.save_tamil)
        tamil_head.addWidget(self.translate_btn); tamil_head.addWidget(copy_tamil); tamil_head.addWidget(save_tamil)
        tamil_wrap.addLayout(tamil_head)
        self.tamil_output = QPlainTextEdit(); self.tamil_output.setObjectName("tamilOutput")
        self.tamil_output.setPlaceholderText("Tamil translation for reference will appear here.")
        self.tamil_output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.tamil_output.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tamil_output.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.tamil_output.setMinimumHeight(360)
        tamil_wrap.addWidget(self.tamil_output, 1)
        self.tabs.addTab(self.tamil_tab, "Tamil")
        self.tabs.setTabEnabled(1, False)

        map_tab = QWidget(); map_wrap = QVBoxLayout(map_tab)
        map_wrap.setContentsMargins(12, 12, 12, 12); map_wrap.setSpacing(8)
        map_head = QHBoxLayout()
        map_label = QLabel("Edit map"); map_label.setObjectName("sectionLabel")
        map_head.addWidget(map_label); map_head.addStretch(1)
        copy_map = QPushButton("Copy map"); copy_map.setObjectName("secondaryButton"); copy_map.clicked.connect(self.copy_edit_map)
        csv_btn = QPushButton("Export CSV"); csv_btn.setObjectName("secondaryButton"); csv_btn.clicked.connect(self.export_csv)
        srt_btn = QPushButton("Export SRT"); srt_btn.setObjectName("secondaryButton"); srt_btn.clicked.connect(self.export_srt)
        map_head.addWidget(copy_map); map_head.addWidget(csv_btn); map_head.addWidget(srt_btn)
        map_wrap.addLayout(map_head)
        self.map_table = QTableWidget(0, 5); self.map_table.setObjectName("mapTable")
        self.map_table.setHorizontalHeaderLabels(["Line", "Source time", "Final script", "Review flags", "Source transcript"])
        self.map_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.map_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.map_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.map_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.map_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        map_wrap.addWidget(self.map_table, 1)
        self.tabs.addTab(map_tab, "Edit map")

        summary_tab = QWidget(); summary_wrap = QVBoxLayout(summary_tab)
        summary_wrap.setContentsMargins(12, 12, 12, 12); summary_wrap.setSpacing(8)
        summary_label = QLabel("Processing summary"); summary_label.setObjectName("sectionLabel")
        summary_wrap.addWidget(summary_label)
        self.summary_output = QPlainTextEdit(); self.summary_output.setObjectName("summaryOutput")
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText("Processing summary will appear here.")
        self.summary_output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.summary_output.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.summary_output.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        summary_wrap.addWidget(self.summary_output, 1)
        self.tabs.addTab(summary_tab, "Summary")
        L.addWidget(self.tabs, 1)

        self.setStyleSheet(STYLE)

    def start_analysis(self, path):
        self.reset()
        self.video_path = path
        self.drop.set_file_name(path)
        self.busy(True, "Starting...")
        self._a = AnalyzeWorker(path)
        self._a.progress.connect(self.status.setText)
        self._a.done.connect(self.on_analyzed)
        self._a.failed.connect(self.on_error)
        self._a.start()

    def add_speaker_choice(self, label, speaker, info, checked=False, mergeable=False):
        box = QFrame(); box.setObjectName("speakerRow"); v = QVBoxLayout(box)
        v.setContentsMargins(14, 12, 14, 12); v.setSpacing(6)
        top = QHBoxLayout()
        rb = QRadioButton(label)
        rb.setProperty("speaker", speaker)
        if checked:
            rb.setChecked(True)
        self.speaker_buttons.addButton(rb)
        top.addWidget(rb); top.addStretch(1)
        if mergeable:
            cb = QCheckBox("Merge")
            cb.setProperty("speaker", speaker)
            cb.toggled.connect(self.on_merge_check_changed)
            self.merge_checks.append(cb)
            top.addWidget(cb)
        preview = info.get("preview_path", "")
        play = QPushButton("Play"); play.setObjectName("secondaryButton")
        play.setEnabled(bool(preview))
        play.clicked.connect(lambda _checked=False, p=preview: self.play_preview(p))
        top.addWidget(play)
        stop = QPushButton("Stop"); stop.setObjectName("secondaryButton")
        stop.clicked.connect(self.stop_preview)
        top.addWidget(stop)
        v.addLayout(top)
        samp = QLabel(info.get("sample") or "(no clear speech)")
        samp.setWordWrap(True); samp.setObjectName("sampleText")
        v.addWidget(samp)
        self.speaker_layout.addWidget(box)

    def add_merge_choice(self):
        box = QFrame(); box.setObjectName("speakerRow"); v = QVBoxLayout(box)
        v.setContentsMargins(14, 12, 14, 12); v.setSpacing(6)
        self.merge_radio = QRadioButton("Merge checked speakers")
        self.merge_radio.setProperty("speaker", MERGED_SPEAKERS)
        self.speaker_buttons.addButton(self.merge_radio)
        v.addWidget(self.merge_radio)
        note = QLabel("Use this when the same doctor was split into multiple speaker labels.")
        note.setWordWrap(True); note.setObjectName("sampleText")
        v.addWidget(note)
        self.speaker_layout.addWidget(box)

    def on_merge_check_changed(self, checked):
        if checked and self.merge_radio:
            self.merge_radio.setChecked(True)

    def play_preview(self, path):
        if not path or not os.path.exists(path):
            QMessageBox.information(self, "No audio sample", "No audio sample is available for this speaker.")
            return
        try:
            if QMediaPlayer and QAudioOutput:
                if not self.player:
                    self.player = QMediaPlayer(self)
                    self.audio_output = QAudioOutput(self)
                    self.audio_output.setVolume(1.0)
                    self.player.setAudioOutput(self.audio_output)
                self.player.stop()
                self.player.setSource(QUrl.fromLocalFile(path))
                self.player.play()
            elif sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            self.status.setText("Playing speaker sample.")
        except Exception:
            tb = traceback.format_exc()
            log_error("Audio preview failed", tb)
            QMessageBox.critical(self, "Could not play audio", tb.strip().splitlines()[-1])

    def stop_preview(self):
        try:
            if self.player:
                self.player.stop()
                self.status.setText("Audio stopped.")
        except Exception:
            pass

    def on_analyzed(self, segments, samples, timings, preview_dir):
        speaker_items = sorted(
            [(spk, info) for spk, info in samples.items() if spk != ALL_SPEECH],
            key=lambda kv: kv[1]["segments"],
            reverse=True,
        )
        if len(speaker_items) > 1:
            self.busy(False, "More than one voice was identified. Play samples, choose one, merge labels, or use all speech.")
        else:
            self.busy(False, "Pick whose voice to turn into the script:")
        self.segments = segments
        self.analysis_timings = timings or {}
        self.preview_dir = preview_dir or ""
        while self.speaker_layout.count():
            w = self.speaker_layout.takeAt(0).widget()
            if w: w.deleteLater()
        for b in list(self.speaker_buttons.buttons()):
            self.speaker_buttons.removeButton(b)
        self.merge_checks = []
        self.merge_radio = None
        use_all_default = len(speaker_items) <= 2
        all_info = samples.get(ALL_SPEECH, {"sample": utterance_sample(segments), "segments": len(segments)})
        self.add_speaker_choice(
            "Use all speech  -  " + str(all_info.get("segments", len(segments))) + " segments",
            ALL_SPEECH,
            all_info,
            checked=use_all_default,
        )
        if len(speaker_items) > 1:
            self.add_merge_choice()
        for i, (spk, info) in enumerate(speaker_items):
            self.add_speaker_choice(
                spk + "  -  " + str(info["segments"]) + " segments",
                spk,
                info,
                checked=(i == 0 and not use_all_default),
                mergeable=(len(speaker_items) > 1),
            )
        self.speaker_panel.show(); self.generate_btn.show()

    def start_cleanup(self):
        b = self.speaker_buttons.checkedButton()
        if not b:
            QMessageBox.information(self, "Pick a speaker", "Choose which voice to use."); return
        self.busy(True, "Working..."); self.generate_btn.setEnabled(False)
        self.tamil_output.clear(); self.tabs.setTabEnabled(1, False); self.tabs.setCurrentIndex(0)
        choice = b.property("speaker")
        if choice == MERGED_SPEAKERS:
            selected = [cb.property("speaker") for cb in self.merge_checks if cb.isChecked()]
            if not selected:
                self.busy(False, "Pick speakers to merge.")
                self.generate_btn.setEnabled(True)
                QMessageBox.information(self, "Pick speakers to merge", "Check the speaker labels that belong together.")
                return
            self.selected_speaker = selected
        else:
            self.selected_speaker = choice
        self._c = CleanupWorker(self.segments, self.selected_speaker, load_glossary())
        self._c.progress.connect(self.status.setText)
        self._c.done.connect(self.on_script)
        self._c.failed.connect(self.on_error)
        self._c.start()

    def on_script(self, script, cleanup_timings):
        self.busy(False, "Done. Review the script below.")
        self.generate_btn.setEnabled(True)
        self.output.setPlainText(script)
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().minimum())
        self.tabs.setCurrentIndex(0)
        self.translate_btn.setEnabled(bool(script.strip()))
        selected = selected_utterances(self.segments, self.selected_speaker)
        self.edit_map = build_edit_map(script, selected)
        self.render_edit_map()
        timings = dict(self.analysis_timings)
        timings.update(cleanup_timings or {})
        self.summary = processing_summary(self.video_path, self.segments, self.selected_speaker, script, self.edit_map, timings)
        self.summary_output.setPlainText(summary_text(self.summary))
        self.autosave()

    def start_translation(self):
        script = self.output.toPlainText().strip()
        if not script:
            QMessageBox.information(self, "No script", "Generate the English script first."); return
        self.busy(True, "Translating..."); self.translate_btn.setEnabled(False)
        self._t = TranslateWorker(script)
        self._t.progress.connect(self.status.setText)
        self._t.done.connect(self.on_tamil)
        self._t.failed.connect(self.on_error)
        self._t.start()

    def on_tamil(self, text, translation_timings):
        self.busy(False, "Tamil reference ready.")
        self.translate_btn.setEnabled(True)
        self.tamil_output.setPlainText(text)
        self.tamil_output.verticalScrollBar().setValue(self.tamil_output.verticalScrollBar().minimum())
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)
        if translation_timings:
            self.summary.setdefault("timings", {}).update(translation_timings)
            self.summary_output.setPlainText(summary_text(self.summary))
        self.autosave()

    def selected_english_text(self):
        return self.output.textCursor().selectedText().replace("\u2029", "\n").strip()

    def start_selection_translation(self):
        text = self.selected_english_text()
        if not text:
            QMessageBox.information(self, "No selection", "Highlight a word, phrase, or sentence in the English tab first.")
            return
        self.busy(True, "Translating selected text...")
        self.selection_btn.setEnabled(False)
        self._sel = SelectionTranslateWorker(text)
        self._sel.progress.connect(self.status.setText)
        self._sel.done.connect(self.on_selection_translation)
        self._sel.failed.connect(self.on_error)
        self._sel.start()

    def on_selection_translation(self, original, result):
        self.busy(False, "Selection translated.")
        self.selection_btn.setEnabled(bool(self.selected_english_text()))
        dlg = SelectionTranslationDialog(original, result, self)
        dlg.setStyleSheet(STYLE)
        dlg.exec()

    def copy_output(self):
        QApplication.clipboard().setText(self.output.toPlainText()); self.status.setText("Copied.")

    def copy_tamil(self):
        QApplication.clipboard().setText(self.tamil_output.toPlainText()); self.status.setText("Tamil copied.")

    def save_output(self):
        text = self.output.toPlainText().strip()
        if not text: return
        path, _f = QFileDialog.getSaveFileName(self, "Save script", "script.txt", "Text (*.txt)")
        if path:
            open(path, "w", encoding="utf-8").write(text); self.status.setText("Saved to " + path)

    def save_tamil(self):
        text = self.tamil_output.toPlainText().strip()
        if not text: return
        path, _f = QFileDialog.getSaveFileName(self, "Save Tamil reference", "script-ta.txt", "Text (*.txt)")
        if path:
            open(path, "w", encoding="utf-8").write(text); self.status.setText("Saved to " + path)

    def current_project(self):
        return project_data(
            self.video_path,
            self.segments,
            self.selected_speaker,
            self.output.toPlainText(),
            self.tamil_output.toPlainText(),
            self.edit_map,
            self.summary,
        )

    def autosave(self):
        try:
            save_project_file(LAST_PROJECT, self.current_project())
        except Exception:
            logging.exception("Autosave failed")

    def save_project(self):
        path, _f = QFileDialog.getSaveFileName(self, "Save project", "medinav-project.medinav", "Medinav project (*.medinav)")
        if path:
            save_project_file(path, self.current_project())
            self.status.setText("Project saved.")

    def open_project(self):
        path, _f = QFileDialog.getOpenFileName(self, "Open project", "", "Medinav project (*.medinav)")
        if not path:
            return
        self.load_project(path)

    def load_last_project(self):
        if not os.path.exists(LAST_PROJECT):
            QMessageBox.information(self, "No saved session", "No previous session was found."); return
        self.load_project(LAST_PROJECT)

    def load_project(self, path):
        try:
            data = load_project_file(path)
            self.video_path = data.get("video_path", "")
            self.segments = data.get("segments", [])
            self.selected_speaker = data.get("selected_speaker", "")
            self.output.setPlainText(data.get("english_script", ""))
            self.tamil_output.setPlainText(data.get("tamil_reference", ""))
            self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().minimum())
            self.tamil_output.verticalScrollBar().setValue(self.tamil_output.verticalScrollBar().minimum())
            self.edit_map = data.get("edit_map", [])
            self.summary = data.get("summary", {})
            if data.get("glossary"):
                save_glossary(data.get("glossary", ""))
            self.render_edit_map()
            self.summary_output.setPlainText(summary_text(self.summary))
            self.tabs.setTabEnabled(1, bool(self.tamil_output.toPlainText().strip()))
            self.tabs.setCurrentIndex(0)
            self.status.setText("Project loaded.")
        except Exception:
            tb = traceback.format_exc()
            log_error("Open project failed", tb)
            QMessageBox.critical(self, "Could not open project", tb.strip().splitlines()[-1])

    def edit_glossary(self):
        dlg = GlossaryDialog(self)
        dlg.setStyleSheet(STYLE)
        if dlg.exec() == QDialog.Accepted:
            save_glossary(dlg.glossary())
            self.status.setText("Glossary saved.")

    def render_edit_map(self):
        if not hasattr(self, "map_table"):
            return
        self.map_table.setRowCount(len(self.edit_map or []))
        for row_idx, row in enumerate(self.edit_map or []):
            vals = [
                str(row.get("line", row_idx + 1)),
                row.get("timecode", ""),
                row.get("text", ""),
                row.get("flags", ""),
                row.get("source", ""),
            ]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.map_table.setItem(row_idx, col, item)
        self.map_table.resizeRowsToContents()

    def edit_map_text(self):
        lines = []
        for row in self.edit_map or []:
            flag = (" [" + row.get("flags", "") + "]") if row.get("flags") else ""
            lines.append("%s  %s%s\n%s" % (row.get("line"), row.get("timecode"), flag, row.get("text", "")))
        return "\n\n".join(lines)

    def copy_edit_map(self):
        QApplication.clipboard().setText(self.edit_map_text())
        self.status.setText("Edit map copied.")

    def export_csv(self):
        if not self.edit_map: return
        path, _f = QFileDialog.getSaveFileName(self, "Export edit map", "edit-map.csv", "CSV (*.csv)")
        if path:
            write_edit_csv(path, self.edit_map); self.status.setText("CSV exported.")

    def export_srt(self):
        if not self.edit_map: return
        path, _f = QFileDialog.getSaveFileName(self, "Export SRT", "script.srt", "SubRip (*.srt)")
        if path:
            write_srt(path, self.edit_map); self.status.setText("SRT exported.")

    def export_docx(self):
        script = self.output.toPlainText().strip()
        if not script: return
        path, _f = QFileDialog.getSaveFileName(self, "Export DOCX", "script.docx", "Word document (*.docx)")
        if path:
            sections = [("English", script)]
            tamil = self.tamil_output.toPlainText().strip()
            if tamil:
                sections.append(("Tamil reference", tamil))
            if self.edit_map:
                sections.append(("Edit map", self.edit_map_text()))
            write_docx(path, "Medinav Script", sections)
            self.status.setText("DOCX exported.")

    def export_side_by_side(self):
        english = self.output.toPlainText().strip()
        tamil = self.tamil_output.toPlainText().strip()
        if not english and not tamil: return
        path, _f = QFileDialog.getSaveFileName(self, "Export side-by-side", "script-side-by-side.html", "HTML (*.html)")
        if path:
            write_bilingual_html(path, english, tamil)
            self.status.setText("Side-by-side HTML exported.")

    def start_batch(self):
        paths, _f = QFileDialog.getOpenFileNames(
            self, "Choose videos for batch", "",
            "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.wmv *.flv);;All files (*.*)")
        if not paths:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return
        self.busy(True, "Starting batch...")
        self._b = BatchWorker(paths, out_dir, load_glossary())
        self._b.progress.connect(self.status.setText)
        self._b.done.connect(self.on_batch_done)
        self._b.failed.connect(self.on_error)
        self._b.start()

    def on_batch_done(self, results):
        self.busy(False, "Batch complete.")
        ok = sum(1 for r in results if r.get("ok"))
        failed = len(results) - ok
        QMessageBox.information(self, "Batch complete", "%d completed, %d failed." % (ok, failed))

    def busy(self, on, msg=""):
        self.status.setText(msg); self.progress.setVisible(on); self.drop.setEnabled(not on)

    def cleanup_preview_dir(self):
        if self.player:
            self.player.stop()
        path = getattr(self, "preview_dir", "")
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        self.preview_dir = ""

    def reset(self):
        self.cleanup_preview_dir()
        self.segments = []; self.video_path = ""; self.selected_speaker = ""
        self.edit_map = []; self.summary = {}; self.analysis_timings = {}
        self.merge_checks = []; self.merge_radio = None
        self.speaker_panel.hide()
        self.generate_btn.hide(); self.generate_btn.setEnabled(True); self.output.clear()
        self.tamil_output.clear(); self.translate_btn.setEnabled(True)
        self.selection_btn.setEnabled(False)
        self.tabs.setTabEnabled(1, False); self.tabs.setCurrentIndex(0)
        self.render_edit_map(); self.summary_output.clear()

    def on_error(self, tb):
        self.busy(False, "Something went wrong."); self.generate_btn.setEnabled(True)
        if hasattr(self, "translate_btn"):
            self.translate_btn.setEnabled(bool(self.output.toPlainText().strip()))
        if hasattr(self, "selection_btn"):
            self.selection_btn.setEnabled(bool(self.selected_english_text()))
        last = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        log_error("App error", tb)
        QMessageBox.critical(self, "Error", last)
        print(tb, file=sys.stderr)

    def closeEvent(self, event):
        self.cleanup_preview_dir()
        super().closeEvent(event)


STYLE = """
QMainWindow, QWidget { background: #F5F8FA; color: #132A3A; font-size: 14px; }
QLabel { background: transparent; }
#header { background: #FFFFFF; border: 1px solid #DCE7EF; border-radius: 8px; }
#logo { background: #FFFFFF; border: 1px solid #E6EDF3; border-radius: 8px; color: #E61F2B; font-size: 30px; font-weight: 800; }
#title { background: transparent; font-size: 28px; font-weight: 800; color: #E61F2B; }
#subtitle { background: transparent; font-size: 15px; font-weight: 650; color: #243B53; }
#versionPill { background: #EEF6F7; color: #0C5A68; border: 1px solid #CFE3E7; border-radius: 8px; padding: 5px 10px; font-weight: 650; }
#dropFrame { background: #FFFFFF; border: 2px dashed #8FB6C8; border-radius: 8px; }
#dropFrame:hover { background: #F0F7F8; border-color: #0F7892; }
#dropLabel { color: #0E4C62; font-size: 20px; font-weight: 760; }
#dropHint { color: #698295; font-size: 13px; }
#status { color: #526B7A; font-weight: 600; min-height: 18px; }
#sectionLabel { color: #1B3445; font-size: 15px; font-weight: 760; }
#scriptTabs { background: #FFFFFF; border: 1px solid #D6E2EA; border-radius: 8px; }
QTabWidget::pane { background: #FFFFFF; border: 1px solid #D6E2EA; border-radius: 8px; top: -1px; }
QTabBar::tab { background: #EAF2F5; color: #446273; border: 1px solid #CFE0E8; padding: 8px 18px; min-width: 96px; font-weight: 750; }
QTabBar::tab:selected { background: #FFFFFF; color: #0E4C62; border-bottom-color: #FFFFFF; }
QTabBar::tab:disabled { color: #94A8B3; background: #EEF3F5; }
#speakerRow { background: #FFFFFF; border: 1px solid #DCE7EF; border-radius: 8px; }
#speakerRow:hover { border-color: #89B7C7; background: #FBFDFE; }
#sampleText { color: #5F7380; font-style: italic; line-height: 1.35; }
QRadioButton { font-weight: 700; color: #183446; spacing: 8px; }
#output, #tamilOutput, #summaryOutput, QPlainTextEdit { background: #FFFFFF; color: #12283A; border: 1px solid #D6E2EA; border-radius: 8px; padding: 10px; selection-background-color: #0F7892; }
#tamilOutput { font-size: 15px; }
#mapTable { background: #FFFFFF; color: #12283A; border: 1px solid #D6E2EA; border-radius: 8px; gridline-color: #E4EEF3; selection-background-color: #DCEEF3; selection-color: #12283A; }
QHeaderView::section { background: #EEF6F7; color: #0E4C62; border: none; border-right: 1px solid #D6E2EA; padding: 7px; font-weight: 750; }
QPushButton { border: none; border-radius: 8px; padding: 9px 16px; font-weight: 700; }
#primaryButton { background: #E61F2B; color: #FFFFFF; }
#primaryButton:hover { background: #C91824; }
#secondaryButton { background: #FFFFFF; color: #0F5F75; border: 1px solid #BFD3DC; }
#secondaryButton:hover { background: #EEF6F7; border-color: #85AFC0; }
QPushButton:disabled { background: #D5E0E8; color: #8FA2AF; }
QProgressBar { border: none; background: #DDE8EE; border-radius: 5px; height: 8px; }
QProgressBar::chunk { background: #0F7892; border-radius: 5px; }
"""


def main():
    install_exception_logger()
    maybe_update()  # self-update from GitHub before showing the window
    set_windows_app_id()
    ensure_windows_shortcut_icon()
    app = QApplication(sys.argv)
    app_icon = QIcon(icon_path())
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

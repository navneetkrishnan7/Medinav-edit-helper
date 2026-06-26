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

__version__ = "1.0.7"

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

def _user_prompt(utterances):
    lines = [f"[{i}] ({u['start']:.1f}s-{u['end']:.1f}s) {u['text']}"
             for i, u in enumerate(utterances, 1)]
    return ("Here are the time-ordered transcript segments from the selected speaker. "
            "Produce the final cleaned script.\n\n" + "\n".join(lines))

def cleanup(utterances):
    if not utterances:
        return ""
    if CLEANUP_BACKEND == "ollama":
        import json, urllib.request
        payload = {"model": OLLAMA_MODEL, "system": SYSTEM_PROMPT,
                   "prompt": _user_prompt(utterances), "stream": False,
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
                                 messages=[{"role": "user", "content": _user_prompt(utterances)}])
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

# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QRadioButton, QButtonGroup, QFileDialog,
    QProgressBar, QFrame, QMessageBox, QTabWidget,
)

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".wmv", ".flv"}
LOGO_NAME = "medinav-logo.jpg"


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


class AnalyzeWorker(QThread):
    progress = Signal(str)
    done = Signal(list, dict)
    failed = Signal(str)

    def __init__(self, path):
        super().__init__(); self.path = path

    def run(self):
        wav = None
        try:
            self.progress.emit("Extracting audio...")
            wav = extract_audio(self.path)
            self.progress.emit("Transcribing (this is the slow part)...")
            segments = transcribe(wav)
            if not segments:
                raise RuntimeError("No speech was detected in this file.")
            self.progress.emit("Separating speakers...")
            diarize_and_label(wav, segments)
            self.done.emit(segments, speaker_samples(segments))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if wav and os.path.exists(wav):
                try: os.remove(wav)
                except OSError: pass


class CleanupWorker(QThread):
    progress = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, segments, speaker):
        super().__init__(); self.segments = segments; self.speaker = speaker

    def run(self):
        try:
            utt = [s for s in self.segments if s.get("speaker") == self.speaker]
            label = "Claude" if CLEANUP_BACKEND == "claude" else "local model"
            self.progress.emit("Cleaning up the script with " + label + "...")
            self.done.emit(cleanup(utt))
        except Exception:
            self.failed.emit(traceback.format_exc())


class TranslateWorker(QThread):
    progress = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, script):
        super().__init__(); self.script = script

    def run(self):
        try:
            label = "Claude" if CLEANUP_BACKEND == "claude" else "local model"
            self.progress.emit("Translating to Tamil with " + label + "...")
            self.done.emit(translate_to_tamil(self.script))
        except Exception:
            self.failed.emit(traceback.format_exc())


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
        self.resize(960, 760)
        self.segments = []
        self.speaker_buttons = QButtonGroup(self)
        self._a = self._c = self._t = None

        root = QWidget(); self.setCentralWidget(root)
        L = QVBoxLayout(root); L.setContentsMargins(28, 24, 28, 24); L.setSpacing(16)

        header = QFrame(); header.setObjectName("header")
        h = QHBoxLayout(header); h.setContentsMargins(16, 14, 16, 14); h.setSpacing(14)
        logo = QLabel(); logo.setObjectName("logo"); logo.setFixedSize(82, 82); logo.setAlignment(Qt.AlignCenter)
        lp = logo_path()
        pix = QPixmap(lp) if lp else QPixmap()
        if not pix.isNull():
            logo.setPixmap(pix.scaled(74, 74, Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
        english_head.addWidget(cp); english_head.addWidget(sv); english_wrap.addLayout(english_head)
        self.output = QTextEdit(); self.output.setObjectName("output")
        self.output.setPlaceholderText("The cleaned script will appear here.")
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
        self.tamil_output = QTextEdit(); self.tamil_output.setObjectName("tamilOutput")
        self.tamil_output.setPlaceholderText("Tamil translation for reference will appear here.")
        tamil_wrap.addWidget(self.tamil_output, 1)
        self.tabs.addTab(self.tamil_tab, "Tamil")
        self.tabs.setTabEnabled(1, False)
        L.addWidget(self.tabs, 1)

        self.setStyleSheet(STYLE)

    def start_analysis(self, path):
        self.reset()
        self.drop.set_file_name(path)
        self.busy(True, "Starting...")
        self._a = AnalyzeWorker(path)
        self._a.progress.connect(self.status.setText)
        self._a.done.connect(self.on_analyzed)
        self._a.failed.connect(self.on_error)
        self._a.start()

    def on_analyzed(self, segments, samples):
        self.busy(False, "Pick whose voice to turn into the script:")
        self.segments = segments
        while self.speaker_layout.count():
            w = self.speaker_layout.takeAt(0).widget()
            if w: w.deleteLater()
        for b in list(self.speaker_buttons.buttons()):
            self.speaker_buttons.removeButton(b)
        ordered = sorted(samples.items(), key=lambda kv: kv[1]["segments"], reverse=True)
        for i, (spk, info) in enumerate(ordered):
            box = QFrame(); box.setObjectName("speakerRow"); v = QVBoxLayout(box)
            v.setContentsMargins(14, 12, 14, 12); v.setSpacing(6)
            rb = QRadioButton(spk + "  -  " + str(info["segments"]) + " segments")
            rb.setProperty("speaker", spk)
            if i == 0: rb.setChecked(True)
            self.speaker_buttons.addButton(rb)
            samp = QLabel(info["sample"] or "(no clear speech)")
            samp.setWordWrap(True); samp.setObjectName("sampleText")
            v.addWidget(rb); v.addWidget(samp); self.speaker_layout.addWidget(box)
        self.speaker_panel.show(); self.generate_btn.show()

    def start_cleanup(self):
        b = self.speaker_buttons.checkedButton()
        if not b:
            QMessageBox.information(self, "Pick a speaker", "Choose which voice to use."); return
        self.busy(True, "Working..."); self.generate_btn.setEnabled(False)
        self.tamil_output.clear(); self.tabs.setTabEnabled(1, False); self.tabs.setCurrentIndex(0)
        self._c = CleanupWorker(self.segments, b.property("speaker"))
        self._c.progress.connect(self.status.setText)
        self._c.done.connect(self.on_script)
        self._c.failed.connect(self.on_error)
        self._c.start()

    def on_script(self, script):
        self.busy(False, "Done. Review the script below.")
        self.generate_btn.setEnabled(True)
        self.output.setPlainText(script)
        self.tabs.setCurrentIndex(0)
        self.translate_btn.setEnabled(bool(script.strip()))

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

    def on_tamil(self, text):
        self.busy(False, "Tamil reference ready.")
        self.translate_btn.setEnabled(True)
        self.tamil_output.setPlainText(text)
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)

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

    def busy(self, on, msg=""):
        self.status.setText(msg); self.progress.setVisible(on); self.drop.setEnabled(not on)

    def reset(self):
        self.segments = []; self.speaker_panel.hide()
        self.generate_btn.hide(); self.generate_btn.setEnabled(True); self.output.clear()
        self.tamil_output.clear(); self.translate_btn.setEnabled(True)
        self.tabs.setTabEnabled(1, False); self.tabs.setCurrentIndex(0)

    def on_error(self, tb):
        self.busy(False, "Something went wrong."); self.generate_btn.setEnabled(True)
        if hasattr(self, "translate_btn"):
            self.translate_btn.setEnabled(bool(self.output.toPlainText().strip()))
        last = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        QMessageBox.critical(self, "Error", last)
        print(tb, file=sys.stderr)


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
#output, #tamilOutput { background: #FFFFFF; color: #12283A; border: 1px solid #D6E2EA; border-radius: 8px; padding: 10px; selection-background-color: #0F7892; }
#tamilOutput { font-size: 15px; }
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
    maybe_update()  # self-update from GitHub before showing the window
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

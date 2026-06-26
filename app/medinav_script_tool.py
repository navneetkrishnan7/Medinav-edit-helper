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

__version__ = "1.0.1"

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

def diarize_and_label(audio_path, segments):
    import sherpa_onnx
    sd = _diar_engine()
    samples, sr = sherpa_onnx.read_wave(audio_path)
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

# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QRadioButton, QButtonGroup, QFileDialog,
    QProgressBar, QFrame, QMessageBox,
)

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".wmv", ".flv"}


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


class DropFrame(QFrame):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropFrame")
        self.setMinimumHeight(170)
        lay = QVBoxLayout(self); lay.setAlignment(Qt.AlignCenter)
        self.label = QLabel("Drag a video here\nor click to browse")
        self.label.setAlignment(Qt.AlignCenter); self.label.setObjectName("dropLabel")
        lay.addWidget(self.label)

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
        self.resize(820, 720)
        self.segments = []
        self.speaker_buttons = QButtonGroup(self)
        self._a = self._c = None

        root = QWidget(); self.setCentralWidget(root)
        L = QVBoxLayout(root); L.setContentsMargins(20, 20, 20, 20); L.setSpacing(14)

        title = QLabel("Medinav Script Tool"); title.setObjectName("title"); L.addWidget(title)

        self.drop = DropFrame(); self.drop.file_dropped.connect(self.start_analysis); L.addWidget(self.drop)

        self.status = QLabel(""); self.status.setObjectName("status"); L.addWidget(self.status)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.hide(); L.addWidget(self.progress)

        self.speaker_panel = QWidget(); self.speaker_layout = QVBoxLayout(self.speaker_panel)
        self.speaker_layout.setContentsMargins(0, 0, 0, 0); self.speaker_panel.hide(); L.addWidget(self.speaker_panel)

        self.generate_btn = QPushButton("Generate script"); self.generate_btn.clicked.connect(self.start_cleanup)
        self.generate_btn.hide(); L.addWidget(self.generate_btn)

        self.output = QTextEdit(); self.output.setObjectName("output")
        self.output.setPlaceholderText("The cleaned script will appear here."); L.addWidget(self.output, 1)

        row = QHBoxLayout(); row.addStretch(1)
        cp = QPushButton("Copy"); cp.clicked.connect(self.copy_output)
        sv = QPushButton("Save as .txt"); sv.clicked.connect(self.save_output)
        row.addWidget(cp); row.addWidget(sv); L.addLayout(row)

        self.setStyleSheet(STYLE)

    def start_analysis(self, path):
        self.reset()
        self.drop.label.setText(os.path.basename(path))
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
        self._c = CleanupWorker(self.segments, b.property("speaker"))
        self._c.progress.connect(self.status.setText)
        self._c.done.connect(self.on_script)
        self._c.failed.connect(self.on_error)
        self._c.start()

    def on_script(self, script):
        self.busy(False, "Done. Review the script below.")
        self.generate_btn.setEnabled(True)
        self.output.setPlainText(script)

    def copy_output(self):
        QApplication.clipboard().setText(self.output.toPlainText()); self.status.setText("Copied.")

    def save_output(self):
        text = self.output.toPlainText().strip()
        if not text: return
        path, _f = QFileDialog.getSaveFileName(self, "Save script", "script.txt", "Text (*.txt)")
        if path:
            open(path, "w", encoding="utf-8").write(text); self.status.setText("Saved to " + path)

    def busy(self, on, msg=""):
        self.status.setText(msg); self.progress.setVisible(on); self.drop.setEnabled(not on)

    def reset(self):
        self.segments = []; self.speaker_panel.hide()
        self.generate_btn.hide(); self.generate_btn.setEnabled(True); self.output.clear()

    def on_error(self, tb):
        self.busy(False, "Something went wrong."); self.generate_btn.setEnabled(True)
        last = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        QMessageBox.critical(self, "Error", last)
        print(tb, file=sys.stderr)


STYLE = """
QMainWindow, QWidget { background: #0C447C; color: #E6F1FB; font-size: 14px; }
#title { font-size: 22px; font-weight: 700; color: #FFFFFF; }
#dropFrame { background: #185FA5; border: 2px dashed #E6F1FB; border-radius: 12px; }
#dropFrame:hover { background: #1d6fbf; }
#dropLabel { color: #E6F1FB; font-size: 16px; }
#status { color: #C7DEF5; }
#speakerRow { background: #185FA5; border-radius: 10px; padding: 4px; }
#sampleText { color: #C7DEF5; font-style: italic; }
QRadioButton { font-weight: 600; }
#output { background: #FFFFFF; color: #0C447C; border-radius: 8px; padding: 8px; }
QPushButton { background: #185FA5; color: #FFFFFF; border: none; border-radius: 8px; padding: 8px 16px; }
QPushButton:hover { background: #1d6fbf; }
QPushButton:disabled { background: #3a5f85; color: #9bb6d0; }
QProgressBar { border: none; background: #185FA5; border-radius: 6px; height: 8px; }
QProgressBar::chunk { background: #E6F1FB; border-radius: 6px; }
"""


def main():
    maybe_update()  # self-update from GitHub before showing the window
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    MainWindow().show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

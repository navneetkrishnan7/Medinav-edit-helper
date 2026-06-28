# Medinav Script Tool

Desktop app for turning a raw doctor-to-camera video into a clean, grammatically
correct script. Editors drag a video in; it transcribes, separates the two speakers,
lets you pick the doctor's voice, drops the repeated takes (keeping the last one),
and fixes the grammar with Claude.

## The development workflow (edit on Mac, push to Windows)

```
  Mac (you edit)  ──git push──►  GitHub (source of truth)  ──auto──►  Windows editors
```

- The app's source of truth is `app/medinav_script_tool.py` in this repo.
- Every Windows machine runs a copy that, on launch, checks GitHub for a newer
  version of itself and self-updates before opening. So you don't touch the
  Windows machines at all — you push, they update on next open.

### To ship a change
1. Edit `app/medinav_script_tool.py` on your Mac.
2. Bump the version line near the top: `__version__ = "1.0.1"` (must increase, or
   the Windows copies won't see it as newer).
3. Commit and push:
   ```
   git add app/medinav_script_tool.py
   git commit -m "tweak cleanup prompt"
   git push
   ```
4. Each editor's app picks it up the next time they open it. (raw.githubusercontent
   caches for a few minutes, so allow a short delay.)

No version bump = no update. That's the safety switch: you can push docs or installer
changes without forcing an app update.

### Testing on your Mac before you push
The app is cross-platform. With `ffmpeg` installed (`brew install ffmpeg`) and the
deps (`pip install PySide6 faster-whisper anthropic numpy "sherpa-onnx>=1.13,<2"`),
plus a `.env` and the two ONNX models in `app/models/`, you can run it directly:
```
AUTO_UPDATE=0 python app/medinav_script_tool.py
```
`AUTO_UPDATE=0` stops it from overwriting your local edits with the GitHub copy
while you're developing.

## First-time install on a Windows machine
1. Make sure this repo is **public** (so the Windows machines can fetch the app
   without auth), and that `$repo`/`$branch` at the top of `install.bat` match it.
2. Download `install.bat` onto the Windows machine and double-click it.
3. It downloads the app, sets up Python + libraries + ffmpeg + the speaker models,
   asks once for your Anthropic key, and makes a desktop icon. Done.

After that, updates flow automatically via GitHub — `install.bat` only needs to be
re-run if the *installer itself* changes (e.g. a new library dependency).

## Speaker selection

After transcription, the app may identify one or more speaker labels. If there is
only one real voice but the model splits it into two labels, choose **Use all
speech** or tick both labels and choose **Merge checked speakers** before
generating the script.

- **Play** lets the editor hear a short sample before choosing.
- **Use all speech** keeps every transcript segment in order.
- **Merge checked speakers** combines selected labels, useful when one doctor was
  incorrectly split into multiple speakers.
- Choosing one speaker keeps only that detected voice, useful when there is a
  real off-camera voice that should be excluded.

## First-time install on a Mac
1. Make sure this repo is public and that `repo`/`branch` at the top of
   `install_mac.command` match it.
2. Download `install_mac.command` onto the Mac.
3. If macOS blocks it because it came from the internet, right-click it, choose
   **Open**, then confirm.
4. The installer downloads the app, creates a Python environment, installs the
   libraries, checks/installs ffmpeg through Homebrew when available, downloads
   the speaker models, asks once for your Anthropic key, and creates:
   `~/Applications/Medinav Script Tool.app`

After that, app updates flow through the same GitHub self-update path. Re-run
`install_mac.command` only when the installer, launcher, or dependency setup changes.

## Files
- `app/medinav_script_tool.py` — the whole app (edit this)
- `app/launcher.py` — crash-reporting wrapper used by desktop shortcuts/apps
- `app/medinav-logo.jpg` / `app/medinav-icon.ico` / `app/medinav-icon.icns` — branding assets
- `install.bat` — one-click Windows installer (pulls the app from GitHub)
- `install_mac.command` — one-click macOS installer (pulls the app from GitHub)
- `.env` — created per-machine at install; holds keys; **never committed**

## Notes
- Keys live only in each machine's `.env`, never in the repo.
- A manual updater is also placed in the install folder as a fallback if you ever
  turn auto-update off.

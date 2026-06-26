#!/bin/bash
set -euo pipefail

# =====================================================================
#  Medinav Script Tool - macOS installer.
#  Double-click this file. It sets up everything and creates a Mac app.
#  The app itself is pulled from GitHub and self-updates after install.
# =====================================================================

repo="navneetkrishnan7/Medinav-edit-helper"
branch="main"

echo ""
echo "=============================================="
echo "  Medinav Script Tool - macOS installer"
echo "=============================================="
echo ""

app_dir="$HOME/Library/Application Support/MedinavScriptTool"
bundle_dir="$HOME/Applications/Medinav Script Tool.app"
raw_base="https://raw.githubusercontent.com/$repo/$branch/app"

mkdir -p "$app_dir"

download() {
  local url="$1"
  local out="$2"
  curl -fL --retry 3 --connect-timeout 20 "$url" -o "$out"
}

version_ok() {
  "$1" - "$2" <<'PY'
import sys
need = tuple(map(int, sys.argv[1].split(".")))
raise SystemExit(0 if sys.version_info[:2] >= need else 1)
PY
}

find_python() {
  local candidate
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && version_ok "$candidate" "3.10"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

echo "[1/8] Downloading the app from GitHub..."
bust="$(date +%s)"
download "$raw_base/medinav_script_tool.py?nocache=$bust" "$app_dir/medinav_script_tool.py"
download "$raw_base/launcher.py?nocache=$bust" "$app_dir/launcher.py"

echo "[2/8] Checking for Python..."
python_cmd="$(find_python || true)"
if [ -z "$python_cmd" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "      Python 3.10+ not found. Installing Python 3.12 with Homebrew..."
    brew install python@3.12
    python_cmd="$(find_python || true)"
  fi
fi
if [ -z "$python_cmd" ]; then
  echo "Python 3.10+ could not be found."
  echo "Install Python 3.12 from https://www.python.org/downloads/macos/ or install Homebrew, then run this again."
  exit 1
fi

echo "[3/8] Creating environment and installing libraries..."
venv="$app_dir/venv"
if [ ! -x "$venv/bin/python" ]; then
  "$python_cmd" -m venv "$venv"
fi
vpy="$venv/bin/python"
"$vpy" -m pip install --upgrade pip
"$vpy" -m pip install PySide6 faster-whisper anthropic numpy "sherpa-onnx>=1.13,<2"

echo "[4/8] Checking ffmpeg..."
ffmpeg_path="$(command -v ffmpeg || true)"
if [ -z "$ffmpeg_path" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "      ffmpeg not found. Installing ffmpeg with Homebrew..."
    brew install ffmpeg
    ffmpeg_path="$(command -v ffmpeg || true)"
  fi
fi
if [ -z "$ffmpeg_path" ]; then
  echo "ffmpeg could not be found."
  echo "Install Homebrew from https://brew.sh, then run: brew install ffmpeg"
  echo "After that, run this installer again."
  exit 1
fi
ffmpeg_dir="$(dirname "$ffmpeg_path")"

echo "[5/8] Downloading the speaker-separation models..."
models="$app_dir/models"
mkdir -p "$models"
seg="$models/segmentation.onnx"
emb="$models/embedding.onnx"
if [ ! -f "$seg" ]; then
  tmp="$(mktemp -d)"
  seg_tar="$tmp/segmentation.tar.bz2"
  download "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" "$seg_tar"
  tar -xf "$seg_tar" -C "$tmp"
  found="$(find "$tmp" -name model.onnx -print -quit)"
  if [ -z "$found" ]; then
    echo "Could not unpack the segmentation model."
    exit 1
  fi
  cp "$found" "$seg"
  rm -rf "$tmp"
fi
if [ ! -f "$emb" ]; then
  download "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx" "$emb"
fi

echo "[6/8] Settings..."
env_file="$app_dir/.env"
if [ ! -f "$env_file" ]; then
  echo ""
  echo "  Paste your Anthropic API key. Press Enter to add it later."
  read -r -p "  Anthropic API key: " anthropic_key
  {
    echo "CLEANUP_BACKEND=claude"
    echo "ANTHROPIC_API_KEY=$anthropic_key"
    echo "CLAUDE_MODEL=claude-sonnet-4-6"
    echo "WHISPER_MODEL=medium"
    printf 'SEG_MODEL="%s"\n' "$seg"
    printf 'EMB_MODEL="%s"\n' "$emb"
    echo "GITHUB_REPO=$repo"
    echo "GITHUB_BRANCH=$branch"
    echo "AUTO_UPDATE=1"
    echo "# For local cleanup instead of Claude: CLEANUP_BACKEND=ollama"
  } > "$env_file"
fi

echo "[7/8] Creating updater and Mac app..."
update_script="$app_dir/update.command"
cat > "$update_script" <<EOF
#!/bin/bash
set -euo pipefail
raw_base="https://raw.githubusercontent.com/$repo/$branch/app"
app_dir="\$HOME/Library/Application Support/MedinavScriptTool"
bust="\$(date +%s)"
curl -fL --retry 3 "\$raw_base/medinav_script_tool.py?nocache=\$bust" -o "\$app_dir/medinav_script_tool.py"
curl -fL --retry 3 "\$raw_base/launcher.py?nocache=\$bust" -o "\$app_dir/launcher.py"
echo "Done."
read -r -p "Press Enter to close."
EOF
chmod +x "$update_script"

mkdir -p "$bundle_dir/Contents/MacOS" "$bundle_dir/Contents/Resources"
cat > "$bundle_dir/Contents/MacOS/run.sh" <<EOF
#!/bin/bash
app_dir="\$HOME/Library/Application Support/MedinavScriptTool"
export PATH="$ffmpeg_dir:/opt/homebrew/bin:/usr/local/bin:\$PATH"
cd "\$app_dir"
exec "\$app_dir/venv/bin/python" "\$app_dir/launcher.py"
EOF
chmod +x "$bundle_dir/Contents/MacOS/run.sh"

cat > "$bundle_dir/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>Medinav Script Tool</string>
  <key>CFBundleDisplayName</key>
  <string>Medinav Script Tool</string>
  <key>CFBundleIdentifier</key>
  <string>in.medinav.script-tool</string>
  <key>CFBundleVersion</key>
  <string>1.0</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleExecutable</key>
  <string>run.sh</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

echo "[8/8] Finishing..."
echo ""
echo "=============================================="
echo "  Done. The Mac app is here:"
echo "  $bundle_dir"
echo "=============================================="
echo ""
read -r -p "Launch it now? (y/n): " launch_now
if [ "$launch_now" = "y" ] || [ "$launch_now" = "Y" ]; then
  open "$bundle_dir"
fi

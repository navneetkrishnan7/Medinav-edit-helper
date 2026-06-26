@echo off
rem =====================================================================
rem  Medinav Script Tool - one-click installer.
rem  Double-click this file. It sets up everything and makes a desktop icon.
rem  The app itself is pulled from GitHub and self-updates after install.
rem =====================================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $c=Get-Content -LiteralPath '%~f0' -Raw; iex ($c.Substring($c.IndexOf(('#::'+'PS::'))))"
pause
exit /b
#::PS::
$repo   = "navneetkrishnan7/medinav-edit-helper"
$branch = "main"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Medinav Script Tool - installer"            -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$enc = New-Object System.Text.UTF8Encoding($false)
$App = Join-Path $env:LOCALAPPDATA "MedinavScriptTool"
New-Item -ItemType Directory -Force -Path $App | Out-Null
$rawApp = "https://raw.githubusercontent.com/$repo/$branch/app/medinav_script_tool.py"

Write-Host "[1/7] Downloading the app from GitHub..." -ForegroundColor Yellow
$bust = [DateTimeOffset]::Now.ToUnixTimeSeconds()
Invoke-WebRequest -Uri ($rawApp + "?nocache=$bust") -OutFile (Join-Path $App "medinav_script_tool.py")
# Fixed by Codex: download launcher.py from its exact raw GitHub URL.
$rawLauncher = "https://raw.githubusercontent.com/$repo/$branch/app/launcher.py"
Invoke-WebRequest -Uri ($rawLauncher + "?nocache=$bust") -OutFile (Join-Path $App "launcher.py")
$rawLogo = "https://raw.githubusercontent.com/$repo/$branch/app/medinav-logo.jpg"
Invoke-WebRequest -Uri ($rawLogo + "?nocache=$bust") -OutFile (Join-Path $App "medinav-logo.jpg")

Write-Host "[2/7] Checking for Python..." -ForegroundColor Yellow
function Get-Py {
  if (Get-Command py -ErrorAction SilentlyContinue)     { return ,@("py","-3") }
  if (Get-Command python -ErrorAction SilentlyContinue) { return ,@("python") }
  return $null
}
$py = Get-Py
if (-not $py) {
  Write-Host "      Python not found. Trying winget..." -ForegroundColor Yellow
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    $py = Get-Py
  }
}
if (-not $py) {
  Write-Host "Python could not be installed automatically." -ForegroundColor Red
  Write-Host "Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add to PATH'), then run this again." -ForegroundColor Red
  return
}
if ($py.Count -gt 1) { $pyExe=$py[0]; $pyRest=$py[1..($py.Count-1)] } else { $pyExe=$py[0]; $pyRest=@() }

Write-Host "[3/7] Creating environment and installing libraries..." -ForegroundColor Yellow
$venv = Join-Path $App "venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
  & $pyExe @pyRest -m venv $venv
}
$vpy = Join-Path $venv "Scripts\python.exe"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install PySide6 faster-whisper anthropic numpy "sherpa-onnx>=1.13,<2"
if ($LASTEXITCODE -ne 0) {
  Write-Host "Library install failed. Check your internet connection and run the installer again." -ForegroundColor Red
  return
}

Write-Host "[4/7] Downloading ffmpeg..." -ForegroundColor Yellow
$ffBin = Join-Path $App "ffmpeg\bin"
if (-not (Test-Path (Join-Path $ffBin "ffmpeg.exe"))) {
  $zip = Join-Path $env:TEMP "mst_ffmpeg.zip"
  $tmp = Join-Path $env:TEMP "mst_ffmpeg"
  Invoke-WebRequest -Uri "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" -OutFile $zip
  if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
  Expand-Archive -Path $zip -DestinationPath $tmp -Force
  $inner = Get-ChildItem -Path $tmp -Directory | Select-Object -First 1
  New-Item -ItemType Directory -Force -Path $ffBin | Out-Null
  Copy-Item -Path (Join-Path $inner.FullName "bin\*") -Destination $ffBin -Force
  Remove-Item -Force $zip
}

Write-Host "[5/7] Downloading the speaker-separation models (no token needed)..." -ForegroundColor Yellow
$models = Join-Path $App "models"
New-Item -ItemType Directory -Force -Path $models | Out-Null
$seg = Join-Path $models "segmentation.onnx"
$emb = Join-Path $models "embedding.onnx"
if (-not (Test-Path $seg)) {
  $segTar = Join-Path $env:TEMP "mst_seg.tar.bz2"
  $segTmp = Join-Path $env:TEMP "mst_seg"
  Invoke-WebRequest -Uri "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" -OutFile $segTar
  if (Test-Path $segTmp) { Remove-Item -Recurse -Force $segTmp }
  New-Item -ItemType Directory -Force -Path $segTmp | Out-Null
  tar -xf $segTar -C $segTmp
  $found = Get-ChildItem -Path $segTmp -Recurse -Filter "model.onnx" | Select-Object -First 1
  if (-not $found) {
    Write-Host "Could not unpack the segmentation model (needs 'tar', Windows 10 1803+)." -ForegroundColor Red
    return
  }
  Copy-Item $found.FullName $seg -Force
  Remove-Item -Force $segTar
}
if (-not (Test-Path $emb)) {
  Invoke-WebRequest -Uri "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx" -OutFile $emb
}

Write-Host "[6/7] Settings..." -ForegroundColor Yellow
$envFile = Join-Path $App ".env"
if (-not (Test-Path $envFile)) {
  Write-Host ""
  Write-Host "  Paste your Anthropic API key (used to clean up the script)." -ForegroundColor Green
  $ak = Read-Host "  Anthropic API key (Enter to add later)"
  $lines = @(
    "CLEANUP_BACKEND=claude",
    ("ANTHROPIC_API_KEY=" + $ak),
    "CLAUDE_MODEL=claude-sonnet-4-6",
    "WHISPER_MODEL=medium",
    ("SEG_MODEL=" + $seg),
    ("EMB_MODEL=" + $emb),
    ("GITHUB_REPO=" + $repo),
    ("GITHUB_BRANCH=" + $branch),
    "AUTO_UPDATE=1",
    "# For best quality on an NVIDIA GPU: WHISPER_MODEL=large-v3 and ASR_DEVICE=cuda"
  )
  [System.IO.File]::WriteAllLines($envFile, $lines, $enc)
}

Write-Host "[7/7] Creating launcher, updater and desktop icon..." -ForegroundColor Yellow
$run = Join-Path $App "run.bat"
$runLines = @(
  "@echo off",
  "set `"PATH=%~dp0ffmpeg\bin;%PATH%`"",
  "start `"`" `"%~dp0venv\Scripts\pythonw.exe`" `"%~dp0medinav_script_tool.py`""
)
[System.IO.File]::WriteAllLines($run, $runLines, $enc)

$upd = Join-Path $App "update.bat"
$updLines = @(
  "@echo off",
  "echo Pulling the latest app from GitHub...",
  "powershell -NoProfile -ExecutionPolicy Bypass -Command ""Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/$repo/$branch/app/medinav_script_tool.py?t=%RANDOM%' -OutFile '%~dp0medinav_script_tool.py'""",
  "echo Done.",
  "pause"
)
[System.IO.File]::WriteAllLines($upd, $updLines, $enc)

$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop "Medinav Script Tool.lnk"
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = $run
$sc.WorkingDirectory = $App
$sc.IconLocation = (Join-Path $venv "Scripts\pythonw.exe")
$sc.Save()

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "  Done. A 'Medinav Script Tool' icon is on"     -ForegroundColor Green
Write-Host "  your desktop. It self-updates from GitHub."    -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
$go = Read-Host "Launch it now? (y/n)"
if ($go -eq "y") { Start-Process -FilePath $run }
return

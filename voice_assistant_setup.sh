Í#!/usr/bin/env bash
#
# Setup for a local voice assistant on macOS (Apple Silicon):
#   whisper.cpp (STT) + Qwen via Ollama (LLM) + Kokoro (TTS)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Override defaults with env vars, e.g.:
#   OLLAMA_MODEL=qwen3:32b ./setup.sh
#
set -euo pipefail

# ---- Config ------------------------------------------------------------
INSTALL_DIR="${INSTALL_DIR:-$HOME/voice-assistant}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:14b}"       # or qwen3:32b, qwen3:30b-a3b
WHISPER_MODEL="${WHISPER_MODEL:-large-v3-turbo}" # or large-v3 for max accuracy
PYTHON="python3.12"                              # Kokoro needs 3.10-3.12

# ---- Helpers -----------------------------------------------------------
info() { printf "\n\033[1;34m==>\033[0m %s\n" "$1"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---- Preflight ---------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS only."; exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  warn "Not Apple Silicon — Metal acceleration won't apply; expect slow performance."
fi

# ---- Homebrew ----------------------------------------------------------
# (The Homebrew installer also pulls in the Xcode Command Line Tools, which
#  provide git and the compilers whisper.cpp needs.)
if ! have brew; then
  info "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"   # put brew on PATH for this session
else
  info "Homebrew already installed."
fi

# ---- System dependencies ----------------------------------------------
info "Installing system dependencies via Homebrew..."
brew install cmake espeak-ng ffmpeg portaudio ollama python@3.12

# ---- Ollama: start server, then pull the model -------------------------
info "Ensuring the Ollama server is running..."
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  ollama serve >/tmp/ollama.log 2>&1 &
  for _ in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  warn "Ollama server didn't respond; check /tmp/ollama.log. Skipping model pull."
else
  info "Pulling $OLLAMA_MODEL (several GB — this may take a while)..."
  ollama pull "$OLLAMA_MODEL"
fi

# ---- whisper.cpp: clone, build, download model -------------------------
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
if [[ ! -d whisper.cpp ]]; then
  info "Cloning whisper.cpp..."
  git clone https://github.com/ggml-org/whisper.cpp
fi
cd whisper.cpp
info "Building whisper.cpp (Metal is auto-enabled on Apple Silicon)..."
cmake -B build
cmake --build build -j --config Release
info "Downloading whisper model: $WHISPER_MODEL..."
sh ./models/download-ggml-model.sh "$WHISPER_MODEL"
cd "$INSTALL_DIR"

# ---- Python environment + Kokoro --------------------------------------
info "Creating Python virtual environment with $PYTHON..."
"$PYTHON" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install kokoro soundfile sounddevice requests numpy

# ---- Done --------------------------------------------------------------
info "Setup complete!"
cat <<EOF

Everything is installed under: $INSTALL_DIR
  - whisper binary:  whisper.cpp/build/bin/whisper-cli
  - whisper model:   whisper.cpp/models/ggml-$WHISPER_MODEL.bin
  - Ollama model:    $OLLAMA_MODEL
  - Python venv:     venv/  (kokoro, sounddevice, soundfile, requests, numpy)

Next steps:
  1. Copy voice_assistant.py into $INSTALL_DIR
  2. cd $INSTALL_DIR && source venv/bin/activate
  3. python voice_assistant.py

Notes:
  - Kokoro downloads its ~330MB voice model from Hugging Face on first run.
  - The paths in voice_assistant.py assume it lives in $INSTALL_DIR
    (with ./whisper.cpp/... directly beneath it).
  - If you picked a different model, edit MODEL at the top of voice_assistant.py.
EOF

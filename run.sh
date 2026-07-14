#!/usr/bin/env bash
#
# Launcher for the local voice assistant.
# Ensures Ollama is running, then starts the assistant using the venv's
# Python directly (so it ignores any python/pip shell aliases).
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
#
set -euo pipefail

# Always run from this script's own directory, so ./venv and
# ./voice_assistant.py resolve no matter where you launch it from.
cd "$(dirname "$0")"

VENV_PY="./venv/bin/python"
APP="voice_assistant_with_search.py"

info() { printf "\n\033[1;34m==>\033[0m %s\n" "$1"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$1"; }

# ---- Sanity checks -----------------------------------------------------
if [[ ! -x "$VENV_PY" ]]; then
  warn "No venv Python found at $VENV_PY"
  echo "Run ./setup.sh first, or recreate the venv with python3.12 -m venv venv"
  exit 1
fi
if [[ ! -f "$APP" ]]; then
  warn "$APP not found in $(pwd). Copy the orchestrator here first."
  exit 1
fi

# ---- Ensure Ollama is running ------------------------------------------
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  info "Ollama is already running."
else
  if ! command -v ollama >/dev/null 2>&1; then
    warn "ollama not found on PATH. Install it (brew install ollama) or start it manually."
    exit 1
  fi
  info "Starting Ollama server..."
  ollama serve >/tmp/ollama.log 2>&1 &
  for _ in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    warn "Ollama didn't come up within 30s. Check /tmp/ollama.log"
    exit 1
  fi
  info "Ollama is up."
fi

# ---- Launch the assistant ----------------------------------------------
info "Starting the voice assistant (Ctrl+C to quit)..."
exec "$VENV_PY" "$APP"

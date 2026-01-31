#!/bin/bash
# Start IndexTTS Server (WebUI)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TTS_DIR="$PROJECT_ROOT/external/indextts/IndexTTS"

cd "$TTS_DIR"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at $TTS_DIR/.venv"
    echo "Please run setup_indextts.sh first"
    exit 1
fi

# Activate virtual environment (uv uses .venv)
source .venv/bin/activate

# Default configuration
HOST="${INDEXTTS_HOST:-0.0.0.0}"
PORT="${INDEXTTS_PORT:-7860}"

echo "Starting IndexTTS WebUI..."
echo "Host: $HOST"
echo "Port: $PORT"
echo ""

# Start WebUI server
python webui.py --host "$HOST" --port "$PORT" "$@"

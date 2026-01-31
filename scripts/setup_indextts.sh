#!/bin/bash
# IndexTTS 2.5 Local Setup Script
# Requires Python 3.10+ and CUDA/MPS for acceleration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TTS_DIR="$PROJECT_ROOT/external/indextts"

echo "=========================================="
echo "IndexTTS 2.5 Local Deployment Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$python_version < 3.10" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.10+ required, found $python_version"
    exit 1
fi

# Create directory
mkdir -p "$TTS_DIR"
cd "$TTS_DIR"

# Clone or update IndexTTS
if [ -d "IndexTTS" ]; then
    echo "Updating IndexTTS..."
    cd IndexTTS
    git pull
else
    echo "Cloning IndexTTS..."
    git clone https://github.com/index-tts/index-tts.git IndexTTS
    cd IndexTTS
fi

# Remove old venv if exists (uv uses .venv)
if [ -d "venv" ]; then
    rm -rf venv
fi

# Check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies using uv (recommended by IndexTTS)
echo "Installing dependencies with uv..."
if [[ "$(uname)" == "Darwin" ]]; then
    echo "macOS detected, MPS will be used for acceleration..."
    uv sync --extra webui
else
    uv sync --extra webui
fi

# Activate venv
source .venv/bin/activate

# Download model weights
echo "Downloading model weights (this may take a while)..."
python -c "
from indextts import IndexTTS
# This will auto-download models on first use
tts = IndexTTS(model_dir='checkpoints', cfg_path='checkpoints/config.yaml')
print('Model initialization complete!')
" 2>/dev/null || echo "Models will be downloaded on first run."

echo ""
echo "=========================================="
echo "IndexTTS Setup Complete!"
echo "=========================================="
echo ""
echo "To start the WebUI:"
echo "  cd $TTS_DIR/IndexTTS"
echo "  source .venv/bin/activate"
echo "  python webui.py"
echo ""
echo "Or use the start script:"
echo "  $SCRIPT_DIR/start_indextts.sh"
echo ""

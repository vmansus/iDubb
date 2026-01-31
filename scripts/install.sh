#!/bin/bash

# NextGenMedia Installation Script

set -e

echo "=== NextGenMedia Installation ==="

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Check system requirements
echo ""
echo "Checking system requirements..."

# Python 3.9+
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "  Python: $PYTHON_VERSION"
else
    echo "  Error: Python 3 is required"
    exit 1
fi

# Node.js 18+
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "  Node.js: $NODE_VERSION"
else
    echo "  Error: Node.js is required"
    exit 1
fi

# FFmpeg
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | awk '{print $3}')
    echo "  FFmpeg: $FFMPEG_VERSION"
else
    echo "  Warning: FFmpeg not found"
    echo "  Please install FFmpeg for video processing"
    echo "    macOS:  brew install ffmpeg"
    echo "    Ubuntu: sudo apt install ffmpeg"
fi

# Setup Python environment
echo ""
echo "Setting up Python environment..."

cd "$PROJECT_ROOT/backend"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created virtual environment"
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Installed Python dependencies"

# Download Whisper model
echo ""
echo "Pre-downloading Whisper model (this may take a while)..."
python3 -c "import whisper; whisper.load_model('base')" 2>/dev/null || true

# Setup Node.js environment
echo ""
echo "Setting up frontend..."

cd "$PROJECT_ROOT/frontend"
npm install --silent
echo "  Installed Node.js dependencies"

# Setup configuration
echo ""
echo "Setting up configuration..."

cd "$PROJECT_ROOT"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env file from template"
else
    echo "  .env file already exists"
fi

# Create data directories
mkdir -p data/downloads data/processed data/uploads
echo "  Created data directories"

# Make scripts executable
chmod +x scripts/*.sh

# Done
echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your platform credentials"
echo "  2. Run: ./scripts/start.sh"
echo ""
echo "For platform credentials:"
echo "  - Bilibili: Get SESSDATA, bili_jct, buvid3 from browser cookies"
echo "  - Douyin/Xiaohongshu: Copy full cookie string from browser"
echo ""

#!/bin/bash
# CosyVoice 2.0 Local Setup Script
# Requires Python 3.10+ and CUDA/MPS for acceleration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TTS_DIR="$PROJECT_ROOT/external/cosyvoice"

echo "=========================================="
echo "CosyVoice 2.0 Local Deployment Setup"
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

# Clone or update CosyVoice
if [ -d "CosyVoice" ]; then
    echo "Updating CosyVoice..."
    cd CosyVoice
    git pull
else
    echo "Cloning CosyVoice..."
    git clone https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install pynini separately (required for text normalization)
echo "Installing pynini..."
conda install -c conda-forge pynini==2.1.6 -y 2>/dev/null || pip install pynini

# Install WeTextProcessing for Chinese text normalization
pip install WeTextProcessing

# Install main dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "macOS detected, MPS will be used for acceleration..."
    pip install torch torchvision torchaudio
else
    echo "No GPU detected, using CPU mode..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Download pretrained models
echo "Downloading pretrained models..."
mkdir -p pretrained_models

# CosyVoice-300M model (for SFT inference)
if [ ! -d "pretrained_models/CosyVoice-300M" ]; then
    echo "Downloading CosyVoice-300M model..."
    # Use modelscope to download
    pip install modelscope
    python -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
print('CosyVoice-300M downloaded!')
"
fi

# CosyVoice-300M-Instruct model (for instruction-based synthesis)
if [ ! -d "pretrained_models/CosyVoice-300M-Instruct" ]; then
    echo "Downloading CosyVoice-300M-Instruct model..."
    python -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
print('CosyVoice-300M-Instruct downloaded!')
"
fi

# CosyVoice-300M-SFT model (for speaker fine-tuned synthesis)
if [ ! -d "pretrained_models/CosyVoice-300M-SFT" ]; then
    echo "Downloading CosyVoice-300M-SFT model..."
    python -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
print('CosyVoice-300M-SFT downloaded!')
"
fi

echo ""
echo "=========================================="
echo "CosyVoice Setup Complete!"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  cd $TTS_DIR/CosyVoice"
echo "  source venv/bin/activate"
echo "  python webui.py --port 50000"
echo ""
echo "Or use the start script:"
echo "  $SCRIPT_DIR/start_cosyvoice.sh"
echo ""
echo "API Endpoints:"
echo "  POST /inference_sft - SFT synthesis"
echo "  POST /inference_zero_shot - Zero-shot voice cloning"
echo "  POST /inference_cross_lingual - Cross-lingual synthesis"
echo "  POST /inference_instruct - Instruction-based synthesis"
echo ""

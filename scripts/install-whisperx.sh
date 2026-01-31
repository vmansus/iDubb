#!/bin/bash
# Install WhisperX for word-level alignment in transcription
# Run this from the project root or scripts directory

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Find project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == */scripts ]]; then
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
else
    PROJECT_DIR="$SCRIPT_DIR"
fi
BACKEND_DIR="$PROJECT_DIR/backend"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installing WhisperX${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if backend venv exists
if [ ! -d "$BACKEND_DIR/venv" ]; then
    echo -e "${RED}Error: Backend venv not found at $BACKEND_DIR/venv${NC}"
    echo -e "${YELLOW}Please run the main startup script first to create the venv${NC}"
    exit 1
fi

# Activate venv
echo -e "${YELLOW}Activating backend venv...${NC}"
source "$BACKEND_DIR/venv/bin/activate"

# Check Python version
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install whisperx from GitHub
echo -e "${YELLOW}Installing WhisperX from GitHub...${NC}"
pip install git+https://github.com/m-bain/whisperx.git

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
if python3 -c "import whisperx; print(f'WhisperX version: {whisperx.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✓ WhisperX installed successfully!${NC}"
else
    # whisperx might not have __version__, try just importing
    if python3 -c "import whisperx; print('WhisperX imported successfully')" 2>/dev/null; then
        echo -e "${GREEN}✓ WhisperX installed successfully!${NC}"
    else
        echo -e "${RED}✗ WhisperX installation may have failed${NC}"
        exit 1
    fi
fi

echo -e ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "WhisperX provides word-level alignment for better subtitle timing."
echo -e ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Restart the backend service"
echo -e "2. Select a WhisperX model in the UI (🎯 WhisperX)"
echo -e ""

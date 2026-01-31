#!/bin/bash
# NextGenMedia - Local Development Startup Script
# Uses Apple MPS GPU acceleration for Whisper (48GB Mac optimized)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   NextGenMedia - Local Development    ${NC}"
echo -e "${BLUE}========================================${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"

# Check if running on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo -e "${GREEN}✓ Apple Silicon detected - MPS acceleration available${NC}"
else
    echo -e "${YELLOW}⚠ Not Apple Silicon - will use CPU${NC}"
fi

# Function to check and clear port
clear_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}⚠ Port $port is in use, clearing...${NC}"
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 1
        echo -e "${GREEN}✓ Port $port cleared${NC}"
    fi
}

# Function to start backend
start_backend() {
    echo -e "\n${BLUE}Starting Backend...${NC}"
    cd "$BACKEND_DIR"

    # Check/create virtual environment
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install dependencies if needed or requirements.txt changed
    if [ ! -f "venv/.deps_installed" ]; then
        echo -e "${YELLOW}Installing dependencies (first time setup)...${NC}"
        pip install --upgrade pip
        pip install -r requirements.txt

        # Ensure PyTorch with MPS support
        echo -e "${YELLOW}Ensuring PyTorch with MPS support...${NC}"
        pip install torch torchvision torchaudio

        cp requirements.txt venv/.requirements.txt.installed
        touch venv/.deps_installed
        echo -e "${GREEN}✓ Dependencies installed${NC}"
    elif [ ! -f "venv/.requirements.txt.installed" ] || ! diff -q requirements.txt venv/.requirements.txt.installed > /dev/null 2>&1; then
        echo -e "${YELLOW}requirements.txt changed, updating dependencies...${NC}"
        pip install -r requirements.txt
        cp requirements.txt venv/.requirements.txt.installed
        echo -e "${GREEN}✓ Dependencies updated${NC}"
    fi

    # Create data directories
    mkdir -p "$PROJECT_DIR/data/downloads"
    mkdir -p "$PROJECT_DIR/data/processed"
    mkdir -p "$PROJECT_DIR/data/uploads"

    # Check Whisper model
    echo -e "\n${BLUE}Whisper Configuration:${NC}"
    echo -e "  Model: large-v3 (best quality)"
    echo -e "  Device: auto (will use MPS on Apple Silicon)"
    echo -e "${YELLOW}Note: First run will download ~3GB model${NC}"

    # Clear port if in use
    clear_port 8888

    # Set HuggingFace token for WhisperX/pyannote (word-level alignment)
    # Get your token at: https://huggingface.co/settings/tokens
    # Also accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
    #                   and: https://huggingface.co/pyannote/segmentation-3.0
    export HF_TOKEN="${HF_TOKEN:-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}"

    # Start uvicorn
    echo -e "\n${GREEN}Starting API server on http://localhost:8888${NC}"
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8888 --reload
}

# Function to start frontend
start_frontend() {
    echo -e "\n${BLUE}Starting Frontend...${NC}"
    cd "$FRONTEND_DIR"

    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Error: Node.js not found${NC}"
        exit 1
    fi

    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓ Node.js: $NODE_VERSION${NC}"

    # Install dependencies if needed or package.json changed
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        npm install
        cp package.json node_modules/.package.json.installed
    elif [ ! -f "node_modules/.package.json.installed" ] || ! diff -q package.json node_modules/.package.json.installed > /dev/null 2>&1; then
        echo -e "${YELLOW}package.json changed, updating dependencies...${NC}"
        npm install
        cp package.json node_modules/.package.json.installed
    fi

    # Clear port if in use
    clear_port 3005

    # Start dev server
    echo -e "\n${GREEN}Starting frontend on http://localhost:3005${NC}"
    npm run dev -- --port 3005
}

# Function to start both
start_all() {
    # Create logs directory
    LOGS_DIR="$PROJECT_DIR/logs"
    mkdir -p "$LOGS_DIR"
    BACKEND_LOG="$LOGS_DIR/backend.log"

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}   Starting Services...${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "  Backend logs: $BACKEND_LOG"
    echo -e "  Use 'tail -f $BACKEND_LOG' to view backend logs"
    echo -e ""

    # Clear ports if in use
    clear_port 8888
    clear_port 3005

    # Start backend in background with logs
    cd "$BACKEND_DIR"
    source venv/bin/activate
    
    # Set HuggingFace token for WhisperX/pyannote
    export HF_TOKEN="${HF_TOKEN:-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}"
    
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8888 --reload 2>&1 | tee "$BACKEND_LOG" &
    BACKEND_PID=$!

    # Wait for backend to be ready
    echo -e "\n${YELLOW}Waiting for backend to start...${NC}"
    sleep 5

    # Check if backend is running
    if curl -s http://localhost:8888/api/tasks > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend started successfully${NC}"
    else
        echo -e "${RED}✗ Backend may not have started properly, check logs${NC}"
    fi

    # Start frontend in foreground (so we can see its output)
    cd "$FRONTEND_DIR"
    npm run dev -- --port 3005 &
    FRONTEND_PID=$!

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}   Services Started!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "  Backend:  http://localhost:8888"
    echo -e "  Frontend: http://localhost:3005"
    echo -e "  API Docs: http://localhost:8888/docs"
    echo -e "  Backend Logs: tail -f $BACKEND_LOG"
    echo -e "\nPress Ctrl+C to stop all services"

    # Trap Ctrl+C to cleanup
    trap "echo -e '\n${YELLOW}Stopping services...${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

    # Wait for both processes
    wait $BACKEND_PID $FRONTEND_PID
}

# Parse command line arguments
case "${1:-all}" in
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    all)
        start_all
        ;;
    *)
        echo "Usage: $0 {backend|frontend|all}"
        echo ""
        echo "Commands:"
        echo "  backend   - Start only the backend API server"
        echo "  frontend  - Start only the frontend dev server"
        echo "  all       - Start both (default)"
        exit 1
        ;;
esac

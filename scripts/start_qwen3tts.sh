#!/bin/bash
# Start Qwen3-TTS Server
# High-quality TTS with voice cloning from Alibaba Qwen team

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Qwen3-TTS Server                    ${NC}"
echo -e "${BLUE}========================================${NC}"

# Default configuration
PORT="${QWEN3_TTS_PORT:-50001}"
MODEL="${QWEN3_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
LANGUAGE="${QWEN3_TTS_LANGUAGE:-Chinese}"
VOICE="${QWEN3_TTS_VOICE:-vivian}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --voice)
            VOICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --port PORT       Server port (default: 50001)"
            echo "  --model MODEL     Model name (default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)"
            echo "  --language LANG   Default language (default: Chinese)"
            echo "  --voice VOICE     Default voice (default: vivian)"
            echo ""
            echo "Available voices:"
            echo "  vivian, serena, dylan, eric, ryan, aiden, uncle_fu, ono_anna, sohee"
            echo ""
            echo "Available languages:"
            echo "  Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Port:     $PORT"
echo -e "  Model:    $MODEL"
echo -e "  Language: $LANGUAGE"
echo -e "  Voice:    $VOICE"
echo ""

# Check for virtual environment
cd "$BACKEND_DIR"
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
fi

# Check/install dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
pip install -q qwen-tts soundfile fastapi uvicorn python-multipart 2>/dev/null || true

# Check for GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"

echo ""
echo -e "${GREEN}Starting Qwen3-TTS server on http://0.0.0.0:$PORT${NC}"
echo -e "${YELLOW}Note: First run will download the model (~3-6GB)${NC}"
echo ""

# Start the server
python3 << EOF
import asyncio
import io
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import soundfile as sf

# Configuration
PORT = int("$PORT")
MODEL_NAME = "$MODEL"
DEFAULT_LANGUAGE = "$LANGUAGE"
DEFAULT_VOICE = "$VOICE"

app = FastAPI(
    title="Qwen3-TTS API",
    description="High-quality TTS with voice cloning powered by Qwen3-TTS",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (lazy loaded)
_model = None
_model_name = None

VOICES = {
    "vivian": {"name": "Vivian", "gender": "female", "languages": ["zh", "en"]},
    "serena": {"name": "Serena", "gender": "female", "languages": ["zh", "en"]},
    "dylan": {"name": "Dylan", "gender": "male", "languages": ["zh", "en"]},
    "eric": {"name": "Eric", "gender": "male", "languages": ["zh", "en"]},
    "ryan": {"name": "Ryan", "gender": "male", "languages": ["zh", "en"]},
    "aiden": {"name": "Aiden", "gender": "male", "languages": ["zh", "en"]},
    "uncle_fu": {"name": "Uncle_Fu", "gender": "male", "languages": ["zh"]},
    "ono_anna": {"name": "Ono_Anna", "gender": "female", "languages": ["ja", "en"]},
    "sohee": {"name": "Sohee", "gender": "female", "languages": ["ko", "en"]},
}

def get_model(model_name: str = MODEL_NAME):
    global _model, _model_name

    if _model is None or _model_name != model_name:
        import torch
        from qwen_tts import Qwen3TTSModel

        # Determine device
        if torch.cuda.is_available():
            device_map = "cuda:0"
            dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_map = "mps"
            dtype = torch.float16
            attn_impl = "eager"
        else:
            device_map = "cpu"
            dtype = torch.float32
            attn_impl = "eager"

        print(f"Loading model {model_name} on {device_map}...")

        try:
            _model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
            _model_name = model_name
            print(f"Model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load with flash_attention_2, falling back to eager: {e}")
            _model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device_map,
                dtype=dtype,
                attn_implementation="eager",
            )
            _model_name = model_name

    return _model


@app.get("/")
async def root():
    return {"message": "Qwen3-TTS API", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "default_voice": DEFAULT_VOICE,
        "default_language": DEFAULT_LANGUAGE
    }


@app.get("/voices")
async def list_voices():
    return {"voices": VOICES}


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE
    language: Optional[str] = DEFAULT_LANGUAGE
    instruct: Optional[str] = None


@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """Synthesize speech from text"""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text")

        voice_info = VOICES.get(request.voice.lower(), VOICES[DEFAULT_VOICE])
        speaker = voice_info["name"]

        model = get_model()
        wavs, sr = model.generate_custom_voice(
            text=request.text,
            language=request.language,
            speaker=speaker,
            instruct=request.instruct or ""
        )

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format='WAV')
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_form")
async def synthesize_form(
    text: str = Form(...),
    voice: str = Form(DEFAULT_VOICE),
    language: str = Form(DEFAULT_LANGUAGE),
    instruct: str = Form(None)
):
    """Synthesize speech from form data (for compatibility)"""
    request = TTSRequest(text=text, voice=voice, language=language, instruct=instruct)
    return await synthesize(request)


@app.post("/clone")
async def voice_clone(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...),
    language: str = Form(DEFAULT_LANGUAGE)
):
    """Clone voice from reference audio"""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Empty text")

        # Read reference audio
        audio_bytes = await ref_audio.read()

        # Save to temp file (qwen-tts needs file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            ref_path = f.name

        try:
            # Use base model for cloning
            clone_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            model = get_model(clone_model_name)

            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_path,
                ref_text=ref_text
            )

            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format='WAV')
            buffer.seek(0)

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=cloned.wav"}
            )
        finally:
            os.unlink(ref_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Preload model on startup
    print("Preloading model...")
    try:
        get_model()
    except Exception as e:
        print(f"Warning: Model preload failed: {e}")
        print("Model will be loaded on first request.")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
EOF

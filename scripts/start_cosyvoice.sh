#!/bin/bash
# Start CosyVoice Server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TTS_DIR="$PROJECT_ROOT/external/cosyvoice/CosyVoice"

cd "$TTS_DIR"

# Activate virtual environment
source venv/bin/activate

# Default configuration
PORT="${COSYVOICE_PORT:-50000}"
MODEL="${COSYVOICE_MODEL:-CosyVoice-300M-SFT}"

echo "Starting CosyVoice server..."
echo "Port: $PORT"
echo "Model: $MODEL"
echo ""

# Check if running with FastAPI or Gradio
if [ "$1" == "--api" ]; then
    # FastAPI mode (headless)
    echo "Starting in API-only mode..."
    python -c "
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import io
import sys
sys.path.append('.')
from cosyvoice.cli.cosyvoice import CosyVoice

app = FastAPI(title='CosyVoice API')
cosyvoice = CosyVoice('pretrained_models/$MODEL')

@app.get('/health')
async def health():
    return {'status': 'healthy', 'model': '$MODEL'}

@app.post('/inference_sft')
async def inference_sft(tts_text: str = Form(...), spk_id: str = Form(...)):
    output = cosyvoice.inference_sft(tts_text, spk_id)
    audio = next(output)
    buffer = io.BytesIO()
    import torchaudio
    torchaudio.save(buffer, audio['tts_speech'], 22050, format='wav')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type='audio/wav')

@app.post('/inference_zero_shot')
async def inference_zero_shot(
    tts_text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...)
):
    import torchaudio
    wav_data = await prompt_wav.read()
    prompt_speech, sr = torchaudio.load(io.BytesIO(wav_data))
    if sr != 16000:
        prompt_speech = torchaudio.transforms.Resample(sr, 16000)(prompt_speech)
    output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech)
    audio = next(output)
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio['tts_speech'], 22050, format='wav')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type='audio/wav')

@app.post('/inference_cross_lingual')
async def inference_cross_lingual(
    tts_text: str = Form(...),
    prompt_wav: UploadFile = File(...)
):
    import torchaudio
    wav_data = await prompt_wav.read()
    prompt_speech, sr = torchaudio.load(io.BytesIO(wav_data))
    if sr != 16000:
        prompt_speech = torchaudio.transforms.Resample(sr, 16000)(prompt_speech)
    output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech)
    audio = next(output)
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio['tts_speech'], 22050, format='wav')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type='audio/wav')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=$PORT)
"
else
    # Gradio WebUI mode
    echo "Starting Gradio WebUI..."
    python webui.py --port "$PORT" "$@"
fi

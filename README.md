# iDubb 🎬

[中文](./README_CN.md) | English

> ⚠️ **Development Status**: This project is under active development. Features may change, and bugs are expected. Use at your own risk.

> 🤖 **AI-Generated Code**: This project was entirely written by AI coding assistants including [Claude Code](https://claude.ai), [Clawdbot](https://github.com/clawdbot/clawdbot), and other AI tools. Human involvement was limited to product direction and testing.

All-in-one video translation and dubbing pipeline. Download, transcribe, translate, dub, and upload - fully automated.

## ✨ Features

### 🎯 Core Pipeline
- **Video Download** - YouTube, TikTok, Bilibili and other platforms via yt-dlp
- **Speech Transcription** - Multiple backends:
  - Whisper (OpenAI)
  - Faster Whisper (4-8x faster, GPU optimized)
  - WhisperX (word-level alignment, speaker diarization)
- **Smart Translation** - Multiple engines:
  - Google Translate (free)
  - OpenAI GPT-4/GPT-4o
  - Anthropic Claude
  - DeepSeek
- **AI Dubbing** - Text-to-speech with multiple engines:
  - Edge TTS (Microsoft, free, 400+ voices)
  - CosyVoice (voice cloning)
  - Index TTS (voice cloning)
  - Qwen3 TTS (voice cloning)
- **Subtitle Processing** - Dual subtitles, custom styles, ASS/SRT export, hardcoded burn-in
- **One-click Upload** - Bilibili, Douyin (抖音), Xiaohongshu (小红书) with multi-account support

### 🔄 Processing Modes
| Mode | Description | Use Case |
|------|-------------|----------|
| **Full** | Transcribe → Translate → Dub → Compose | YouTube → Bilibili translation |
| **Subtitle** | Transcribe → Translate → Embed subtitles | Videos with dialogue, no dubbing needed |
| **Direct** | Download → Upload directly | Viral clips, no dialogue |
| **Auto** | AI analyzes content and decides | Uncertain content type |

### 📡 Subscription System
- Subscribe to YouTube/TikTok channels
- Auto-detect new videos with configurable intervals
- Auto-processing pipeline (download → process → upload)
- Batch import historical videos

### 🤖 AI Enhancement
- **AI Proofreading**: Grammar check, terminology consistency, timing optimization
- **AI Metadata**: Auto-generate platform-specific titles, descriptions, tags
  - Different styles for Douyin (short, trendy), Bilibili (detailed), Xiaohongshu (lifestyle)
- **Custom Glossary**: Maintain terminology consistency across translations

### 🎨 Additional Features
- Modern React UI with dark theme
- Multi-language interface (Chinese/English)
- Processing presets for quick setup
- Trending video discovery (YouTube)
- Task management with progress tracking
- Multi-account support for all upload platforms

## 🛠️ Tech Stack

**Backend**
- Python 3.10+ / FastAPI / SQLAlchemy + SQLite
- yt-dlp (video download)
- FFmpeg (video processing)
- Whisper/Faster-Whisper/WhisperX (transcription)
- Playwright (browser automation for TikTok, Douyin)

**Frontend**
- React 18 + TypeScript
- Tailwind CSS
- Vite
- React Query
- i18next

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- FFmpeg
- GPU (optional, recommended for Whisper)

### Quick Start

```bash
# Clone
git clone https://github.com/vmansus/iDubb.git
cd iDubb

# Backend
cd backend
pip install -r requirements.txt
playwright install chromium  # For TikTok/Douyin support

# Start backend
uvicorn api.main:app --host 0.0.0.0 --port 8888

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173

## ⚙️ Configuration

### API Keys

Configure in Settings page:

| Service | Required For | Notes |
|---------|--------------|-------|
| OpenAI API Key | GPT translation, AI proofreading | Optional if using Google Translate |
| Anthropic API Key | Claude translation | Optional |
| DeepSeek API Key | DeepSeek translation | Optional, cost-effective |
| YouTube Data API | Trending videos | Optional |

### Platform Credentials

For auto-upload features, configure platform credentials in Settings:
- **Bilibili**: Login via QR code scan
- **Douyin**: Login via QR code or cookies
- **Xiaohongshu**: Login via QR code or cookies

### Environment Variables

```bash
# .env (optional)
WHISPER_MODEL=small          # tiny, base, small, medium, large-v3
WHISPER_DEVICE=auto          # auto, cpu, cuda, mps
```

## 📖 Usage

### Basic Workflow

1. **Create Task** - Paste video URL or upload local file
2. **Choose Mode** - Full translation, subtitle only, or direct upload
3. **Configure** - Select languages, TTS voice, subtitle style
4. **Process** - One click to start, monitor progress in real-time
5. **Review** - Preview results, edit subtitles if needed
6. **Upload** - One-click upload to multiple platforms

### Channel Subscription

1. Go to "Subscriptions" page
2. Click "Add Subscription"
3. Paste YouTube/TikTok channel URL
4. Configure check interval and processing options
5. Enable auto-process for hands-free operation

## 📁 Project Structure

```
iDubb/
├── backend/
│   ├── api/              # FastAPI routes
│   ├── database/         # SQLAlchemy models
│   ├── downloaders/      # Video download (yt-dlp)
│   ├── transcription/    # Whisper backends
│   ├── translation/      # Translation engines
│   ├── tts/              # TTS engines
│   ├── dubbing/          # Audio/video composition
│   ├── subtitles/        # Subtitle processing
│   ├── uploaders/        # Platform uploaders
│   ├── subscriptions/    # Subscription scheduler
│   ├── metadata/         # AI metadata generation
│   ├── proofreading/     # AI proofreading
│   └── pipeline.py       # Main processing pipeline
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── services/     # API client
│   │   └── locales/      # i18n translations
│   └── public/
└── docs/                 # Documentation
```

## 🔧 Voice Cloning Setup

### CosyVoice
```bash
# Install CosyVoice to external/cosyvoice
git clone https://github.com/FunAudioLLM/CosyVoice external/cosyvoice
cd external/cosyvoice && pip install -r requirements.txt
```

### Index TTS
```bash
# Install IndexTTS to external/indextts
git clone https://github.com/indexteam/IndexTTS external/indextts
cd external/indextts && pip install -r requirements.txt
```

## 🤝 Contributing

Contributions welcome! Please read the existing code style and submit PRs.

## 📄 License

MIT

---

**Made with ❤️ by vmansus & Chad 🐕**

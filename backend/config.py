"""
iDubb - Configuration Module
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional, List

# Compute paths at module level for consistency
_BASE_DIR = Path(__file__).parent.parent

# Data directory: use IDUBB_DATA_DIR env var, or default to ~/.idubb (outside worktree)
# This allows multiple worktrees to share the same database
_DATA_DIR = Path(os.environ.get("IDUBB_DATA_DIR", Path.home() / ".idubb"))
_DATABASE_PATH = _DATA_DIR / "idubb.db"


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "iDubb"
    DEBUG: bool = False  # Default to False for security; enable via env var in development

    # Paths
    BASE_DIR: Path = _BASE_DIR
    DATA_DIR: Path = _DATA_DIR
    DOWNLOADS_DIR: Path = _DATA_DIR / "downloads"
    PROCESSED_DIR: Path = _DATA_DIR / "processed"
    UPLOADS_DIR: Path = _DATA_DIR / "uploads"

    # Database - use absolute path for consistent resolution
    DATABASE_URL: str = f"sqlite+aiosqlite:///{_DATABASE_PATH}"

    # Redis (for Celery)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Whisper Settings
    # Backend: "faster" (faster-whisper, 4-8x faster) or "openai" (standard whisper)
    WHISPER_BACKEND: str = "faster"
    # Models: tiny, base, small, medium, large, large-v2, large-v3
    # large-v3 is best quality, needs ~10GB RAM (fine for 48GB Mac)
    WHISPER_MODEL: str = "large-v3"
    # Devices: cpu, cuda (NVIDIA), mps (Apple Silicon GPU)
    # Note: Docker cannot use MPS, set to 'mps' only for native runs
    WHISPER_DEVICE: str = "auto"  # auto-detect best device

    # Translation Settings
    TRANSLATION_SERVICE: str = "google"  # google, deepl_free
    SOURCE_LANG: str = "en"
    TARGET_LANG: str = "zh-CN"

    # TTS Settings
    TTS_SERVICE: str = "edge"  # edge, qwen3, index, cosyvoice
    TTS_VOICE: str = "zh-CN-XiaoxiaoNeural"  # Edge TTS voice
    TTS_RATE: str = "+0%"

    # IndexTTS Settings (Local voice cloning via Gradio)
    INDEX_TTS_HOST: str = "127.0.0.1"
    INDEX_TTS_PORT: int = 9880
    INDEX_TTS_REF_AUDIO: str = ""  # Path to reference audio for voice cloning
    INDEX_TTS_EMO_MODE: str = "same_as_voice"  # same_as_voice, from_ref_audio, from_vector, from_text
    INDEX_TTS_EMO_WEIGHT: float = 0.65  # Emotion weight 0.0-1.0

    # CosyVoice Settings (Alibaba voice cloning)
    COSYVOICE_HOST: str = "127.0.0.1"
    COSYVOICE_PORT: int = 50000
    COSYVOICE_MODE: str = "preset"  # preset, zero_shot, cross_lingual, instruct
    COSYVOICE_SPEAKER: str = "中文女"

    # Qwen3-TTS Settings (Alibaba open-source TTS with voice cloning)
    QWEN3_TTS_MODEL: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"  # or 0.6B variant
    QWEN3_TTS_LANGUAGE: str = "Chinese"  # Chinese, English, Japanese, Korean, etc.
    QWEN3_TTS_VOICE: str = "vivian"  # vivian, serena, dylan, eric, ryan, aiden, uncle_fu, ono_anna, sohee

    # Video Download Settings
    MAX_VIDEO_DURATION: int = 3600  # 1 hour max
    VIDEO_QUALITY: str = "1080p"

    # Platform Credentials (loaded from environment)
    # Bilibili
    BILIBILI_SESSDATA: Optional[str] = None
    BILIBILI_BILI_JCT: Optional[str] = None
    BILIBILI_BUVID3: Optional[str] = None

    # Douyin (抖音)
    DOUYIN_COOKIES: Optional[str] = None

    # Xiaohongshu (小红书)
    XHS_COOKIES: Optional[str] = None

    # Proxy Settings (for YouTube access)
    PROXY_URL: Optional[str] = None

    # API Settings
    API_HOST: str = "127.0.0.1"  # Default to localhost; use 0.0.0.0 only in production with proper security
    API_PORT: int = 8888

    # Task Queue Settings
    MAX_CONCURRENT_TASKS: int = 2  # Maximum number of tasks running concurrently

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
for dir_path in [settings.DOWNLOADS_DIR, settings.PROCESSED_DIR, settings.UPLOADS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

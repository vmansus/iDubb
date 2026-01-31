"""
Transcription API endpoints
Provides model information and estimated transcription times
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger


class WhisperModel(BaseModel):
    """Whisper model info"""
    id: str
    name: str
    backend: str  # "faster" or "openai"
    size_mb: int
    vram_mb: int
    quality: str  # "basic", "good", "better", "great", "best"
    speed_factor_cpu: float  # relative to realtime on CPU (e.g., 15 = 15x realtime)
    speed_factor_gpu: float  # relative to realtime on GPU
    recommended_for: List[str]
    description: str


class TranscriptionEstimate(BaseModel):
    """Estimated transcription time"""
    model_id: str
    duration_seconds: float
    estimated_time_cpu_seconds: float
    estimated_time_gpu_seconds: float
    estimated_time_cpu_formatted: str
    estimated_time_gpu_formatted: str


# Model definitions with performance data
WHISPER_MODELS: List[Dict[str, Any]] = [
    # Faster-Whisper models (CTranslate2 backend) - 4-8x faster than OpenAI
    {
        "id": "faster:tiny",
        "name": "⚡ Faster-Whisper Tiny (极速)",
        "backend": "faster",
        "size_mb": 75,
        "vram_mb": 500,
        "quality": "basic",
        "speed_factor_cpu": 20.0,  # 20x realtime on CPU
        "speed_factor_gpu": 50.0,
        "recommended_for": ["快速测试", "简单对话"],
        "description": "Faster-Whisper: 最快速度，适合测试。准确率较低。",
    },
    {
        "id": "faster:base",
        "name": "⚡ Faster-Whisper Base (快速)",
        "backend": "faster",
        "size_mb": 150,
        "vram_mb": 700,
        "quality": "good",
        "speed_factor_cpu": 15.0,  # 15x realtime on CPU
        "speed_factor_gpu": 40.0,
        "recommended_for": ["日常视频", "清晰语音"],
        "description": "Faster-Whisper: 速度快，准确率适中。推荐日常使用。",
    },
    {
        "id": "faster:small",
        "name": "⚡ Faster-Whisper Small (均衡) ⭐推荐",
        "backend": "faster",
        "size_mb": 500,
        "vram_mb": 1500,
        "quality": "better",
        "speed_factor_cpu": 8.0,  # 8x realtime on CPU
        "speed_factor_gpu": 25.0,
        "recommended_for": ["大部分视频", "轻度口音"],
        "description": "Faster-Whisper: 速度与质量均衡。推荐作为默认选择。",
    },
    {
        "id": "faster:medium",
        "name": "⚡ Faster-Whisper Medium (高质量)",
        "backend": "faster",
        "size_mb": 1500,
        "vram_mb": 3000,
        "quality": "great",
        "speed_factor_cpu": 3.0,  # 3x realtime on CPU
        "speed_factor_gpu": 15.0,
        "recommended_for": ["专业内容", "多语言混合"],
        "description": "Faster-Whisper: 高准确率，适合专业内容。速度中等。",
    },
    {
        "id": "faster:large-v3",
        "name": "⚡ Faster-Whisper Large-V3 (最佳)",
        "backend": "faster",
        "size_mb": 3000,
        "vram_mb": 6000,
        "quality": "best",
        "speed_factor_cpu": 0.8,  # 0.8x realtime on CPU (slower than realtime)
        "speed_factor_gpu": 8.0,
        "recommended_for": ["困难音频", "强口音", "背景噪音"],
        "description": "Faster-Whisper: 最高准确率，CPU较慢。适合困难音频。",
    },
    # WhisperX models (word-level alignment, best for subtitles)
    {
        "id": "whisperx:tiny",
        "name": "🎯 WhisperX Tiny (极速)",
        "backend": "whisperx",
        "size_mb": 75,
        "vram_mb": 600,
        "quality": "basic",
        "speed_factor_cpu": 15.0,
        "speed_factor_gpu": 40.0,
        "recommended_for": ["快速测试", "词级对齐"],
        "description": "WhisperX: 极速+词级对齐，适合测试。",
    },
    {
        "id": "whisperx:base",
        "name": "🎯 WhisperX Base (快速)",
        "backend": "whisperx",
        "size_mb": 150,
        "vram_mb": 800,
        "quality": "good",
        "speed_factor_cpu": 12.0,
        "speed_factor_gpu": 35.0,
        "recommended_for": ["日常视频", "词级对齐"],
        "description": "WhisperX: 快速+词级对齐，日常使用。",
    },
    {
        "id": "whisperx:small",
        "name": "🎯 WhisperX Small (均衡)",
        "backend": "whisperx",
        "size_mb": 500,
        "vram_mb": 2000,
        "quality": "better",
        "speed_factor_cpu": 6.0,
        "speed_factor_gpu": 20.0,
        "recommended_for": ["精确字幕", "分段优化"],
        "description": "WhisperX: 均衡+词级对齐，推荐日常使用。",
    },
    {
        "id": "whisperx:medium",
        "name": "🎯 WhisperX Medium (高质量)",
        "backend": "whisperx",
        "size_mb": 1500,
        "vram_mb": 4000,
        "quality": "great",
        "speed_factor_cpu": 2.5,
        "speed_factor_gpu": 12.0,
        "recommended_for": ["精确字幕", "多语言"],
        "description": "WhisperX: 高质量+词级对齐，专业内容。",
    },
    {
        "id": "whisperx:large-v3",
        "name": "🎯 WhisperX Large-V3 (最佳) ⭐推荐",
        "backend": "whisperx",
        "size_mb": 3000,
        "vram_mb": 7000,
        "quality": "best",
        "speed_factor_cpu": 0.6,
        "speed_factor_gpu": 6.0,
        "recommended_for": ["最佳字幕", "困难音频"],
        "description": "WhisperX: 最高质量+词级对齐，字幕效果最佳。",
    },
    # OpenAI Whisper models (standard, slower but more compatible)
    {
        "id": "openai:tiny",
        "name": "🐢 OpenAI Whisper Tiny",
        "backend": "openai",
        "size_mb": 75,
        "vram_mb": 500,
        "quality": "basic",
        "speed_factor_cpu": 8.0,
        "speed_factor_gpu": 30.0,
        "recommended_for": ["测试", "MPS/GPU加速"],
        "description": "原版Whisper: 支持MPS(Mac GPU)，速度较慢。",
    },
    {
        "id": "openai:base",
        "name": "🐢 OpenAI Whisper Base",
        "backend": "openai",
        "size_mb": 150,
        "vram_mb": 700,
        "quality": "good",
        "speed_factor_cpu": 5.0,
        "speed_factor_gpu": 20.0,
        "recommended_for": ["通用", "MPS/GPU加速"],
        "description": "原版Whisper: 支持MPS(Mac GPU)，速度较慢。",
    },
    {
        "id": "openai:small",
        "name": "🐢 OpenAI Whisper Small",
        "backend": "openai",
        "size_mb": 500,
        "vram_mb": 1500,
        "quality": "better",
        "speed_factor_cpu": 2.5,
        "speed_factor_gpu": 12.0,
        "recommended_for": ["通用", "MPS/GPU加速"],
        "description": "原版Whisper: 支持MPS(Mac GPU)，CPU较慢。",
    },
    {
        "id": "openai:medium",
        "name": "🐢 OpenAI Whisper Medium",
        "backend": "openai",
        "size_mb": 1500,
        "vram_mb": 3000,
        "quality": "great",
        "speed_factor_cpu": 1.0,
        "speed_factor_gpu": 6.0,
        "recommended_for": ["高质量需求", "MPS/GPU加速"],
        "description": "原版Whisper: 支持MPS(Mac GPU)，CPU很慢。",
    },
    {
        "id": "openai:large-v3",
        "name": "🐢 OpenAI Whisper Large-V3",
        "backend": "openai",
        "size_mb": 3000,
        "vram_mb": 6000,
        "quality": "best",
        "speed_factor_cpu": 0.3,  # Very slow on CPU
        "speed_factor_gpu": 4.0,
        "recommended_for": ["最高质量", "MPS/GPU加速"],
        "description": "原版Whisper: 支持MPS(Mac GPU)，CPU非常慢。",
    },
]


def get_whisper_models(backend: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get available Whisper models.

    Args:
        backend: Filter by backend ("faster" or "openai"), None for all

    Returns:
        List of model info dicts
    """
    if backend:
        return [m for m in WHISPER_MODELS if m["backend"] == backend]
    return WHISPER_MODELS


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model info by ID"""
    for model in WHISPER_MODELS:
        if model["id"] == model_id:
            return model
    return None


def format_time(seconds: float) -> str:
    """Format seconds as human readable time"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分"


def estimate_transcription_time(
    duration_seconds: float,
    model_id: str = "faster:base",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Estimate transcription time for a video.

    Args:
        duration_seconds: Video duration in seconds
        model_id: Model ID (e.g., "faster:base")
        device: "cpu" or "gpu"

    Returns:
        Estimation dict with times
    """
    model = get_model_by_id(model_id)
    if not model:
        # Default to faster:base
        model = get_model_by_id("faster:base")

    speed_cpu = model["speed_factor_cpu"]
    speed_gpu = model["speed_factor_gpu"]

    est_cpu = duration_seconds / speed_cpu
    est_gpu = duration_seconds / speed_gpu

    return {
        "model_id": model_id,
        "model_name": model["name"],
        "duration_seconds": duration_seconds,
        "duration_formatted": format_time(duration_seconds),
        "estimated_time_cpu_seconds": est_cpu,
        "estimated_time_gpu_seconds": est_gpu,
        "estimated_time_cpu_formatted": format_time(est_cpu),
        "estimated_time_gpu_formatted": format_time(est_gpu),
        "device": device,
        "estimated_time_seconds": est_cpu if device == "cpu" else est_gpu,
        "estimated_time_formatted": format_time(est_cpu) if device == "cpu" else format_time(est_gpu),
    }


def get_recommended_model(duration_seconds: float, device: str = "cpu") -> str:
    """
    Get recommended model based on video duration and device.

    Args:
        duration_seconds: Video duration
        device: "cpu" or "gpu"

    Returns:
        Recommended model ID
    """
    if device == "gpu":
        # GPU can handle larger models efficiently
        if duration_seconds > 3600:  # > 1 hour
            return "faster:medium"
        else:
            return "faster:large-v3"
    else:
        # CPU - balance speed and quality
        if duration_seconds > 3600:  # > 1 hour
            return "faster:base"
        elif duration_seconds > 1800:  # > 30 min
            return "faster:small"
        elif duration_seconds > 600:  # > 10 min
            return "faster:small"
        else:
            return "faster:medium"


def get_all_estimates(duration_seconds: float, backend: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get estimates for all models"""
    estimates = []
    for model in WHISPER_MODELS:
        # Filter by backend if specified
        if backend and model["backend"] != backend:
            continue
        est = estimate_transcription_time(duration_seconds, model["id"], "cpu")
        est["quality"] = model["quality"]
        est["description"] = model["description"]
        est["recommended_for"] = model["recommended_for"]
        est["backend"] = model["backend"]
        estimates.append(est)
    return estimates

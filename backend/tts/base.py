"""
Base TTS Engine Interface
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class TTSResult:
    """TTS generation result"""
    success: bool
    audio_path: Optional[Path]
    duration: float  # seconds
    error: Optional[str] = None


@dataclass
class TTSSegment:
    """TTS segment with timing"""
    text: str
    start: float
    end: float
    audio_path: Optional[Path] = None


class BaseTTSEngine(ABC):
    """Abstract base class for TTS engines"""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str = None,
        rate: str = "+0%"
    ) -> TTSResult:
        """Synthesize speech from text"""
        pass

    @abstractmethod
    async def get_available_voices(self, language: str = None) -> List[dict]:
        """Get available voices"""
        pass

    @abstractmethod
    async def synthesize_segments(
        self,
        segments: List[TTSSegment],
        output_dir: Path,
        voice: str = None
    ) -> List[TTSSegment]:
        """Synthesize multiple segments"""
        pass

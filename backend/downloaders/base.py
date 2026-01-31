"""
Base Downloader Interface
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class VideoInfo:
    """Video metadata"""
    video_id: str
    title: str
    description: str
    duration: int  # seconds
    thumbnail_url: str
    uploader: str
    upload_date: Optional[datetime]
    view_count: int
    like_count: int
    tags: List[str]
    platform: str
    original_url: str
    extra: Optional[Dict[str, Any]] = field(default_factory=dict)  # Additional metadata


@dataclass
class DownloadResult:
    """Download result"""
    success: bool
    video_path: Optional[Path]
    audio_path: Optional[Path]
    video_info: Optional[VideoInfo]
    error: Optional[str] = None
    subtitle_path: Optional[Path] = None  # Downloaded subtitle file (if available)
    subtitle_language: Optional[str] = None  # Language of downloaded subtitle
    extra: Optional[Dict[str, Any]] = field(default_factory=dict)  # Additional result data


class BaseDownloader(ABC):
    """Abstract base class for video downloaders"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def get_video_info(self, url: str) -> Optional[VideoInfo]:
        """Get video metadata without downloading"""
        pass

    @abstractmethod
    async def download(self, url: str, quality: str = "1080p", **kwargs) -> DownloadResult:
        """Download video and return result

        Args:
            url: Video URL to download
            quality: Quality preset (e.g., "1080p", "720p")
            **kwargs: Additional options like subtitle_language, format_id, etc.
        """
        pass

    @abstractmethod
    async def get_trending(self, count: int = 10) -> List[VideoInfo]:
        """Get trending/popular videos"""
        pass

    @abstractmethod
    def supports_url(self, url: str) -> bool:
        """Check if this downloader supports the given URL"""
        pass

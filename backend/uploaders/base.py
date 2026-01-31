"""
Base Uploader Interface
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class UploadResult:
    """Upload result"""
    success: bool
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    platform: str = ""
    error: Optional[str] = None
    extra_info: Optional[Dict[str, Any]] = None


@dataclass
class VideoMetadata:
    """Video metadata for upload"""
    title: str
    description: str
    tags: List[str]
    cover_path: Optional[Path] = None
    category: Optional[str] = None
    is_original: bool = True
    schedule_time: Optional[str] = None  # ISO format for scheduled publish
    source_url: Optional[str] = None  # Source URL for 转载 content


class BaseUploader(ABC):
    """Abstract base class for platform uploaders"""

    def __init__(self):
        self._authenticated = False

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    @abstractmethod
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Authenticate with platform"""
        pass

    @abstractmethod
    async def upload(
        self,
        video_path: Path,
        metadata: VideoMetadata
    ) -> UploadResult:
        """Upload video to platform"""
        pass

    @abstractmethod
    async def check_upload_status(self, video_id: str) -> Dict[str, Any]:
        """Check upload/processing status"""
        pass

    @abstractmethod
    def get_required_credentials(self) -> List[str]:
        """Get list of required credential keys"""
        pass

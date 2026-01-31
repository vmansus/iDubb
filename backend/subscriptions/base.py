"""
Base classes for channel fetchers.
Defines the interface for platform-specific implementations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from loguru import logger


@dataclass
class VideoInfo:
    """Information about a video from a channel"""
    video_id: str
    title: str
    url: str
    published_at: datetime
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None  # Duration in seconds
    description: Optional[str] = None


@dataclass
class ChannelInfo:
    """Information about a channel"""
    platform: str
    channel_id: str
    channel_name: str
    channel_url: str
    avatar_url: Optional[str] = None
    subscriber_count: Optional[int] = None
    video_count: Optional[int] = None
    description: Optional[str] = None


class BaseFetcher(ABC):
    """Abstract base class for platform-specific channel fetchers"""

    @property
    @abstractmethod
    def platform(self) -> str:
        """Return the platform name (e.g., 'youtube', 'bilibili')"""
        pass

    @abstractmethod
    async def lookup_channel(self, url_or_id: str) -> Optional[ChannelInfo]:
        """
        Look up channel information from a URL or ID.

        Args:
            url_or_id: Channel URL, handle (@name), or channel ID

        Returns:
            ChannelInfo if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_latest_videos(
        self,
        channel_id: str,
        limit: int = 10
    ) -> List[VideoInfo]:
        """
        Get the latest videos from a channel.

        Args:
            channel_id: The platform-specific channel identifier
            limit: Maximum number of videos to return

        Returns:
            List of VideoInfo, sorted by publish date (newest first)
        """
        pass

    async def get_new_videos(
        self,
        channel_id: str,
        after_video_id: Optional[str] = None,
        after_date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[VideoInfo]:
        """
        Get videos newer than a reference point.

        Args:
            channel_id: The platform-specific channel identifier
            after_video_id: Only return videos published after this video
            after_date: Only return videos published after this date
            limit: Maximum number of videos to return

        Returns:
            List of new VideoInfo, sorted by publish date (newest first)
        """
        videos = await self.get_latest_videos(channel_id, limit)

        if not after_video_id and not after_date:
            # First check (no baseline set) - return empty list to establish baseline
            # without creating tasks. The scheduler will set the baseline using videos[0].
            # This prevents "video flood" when subscribing to a channel with many videos.
            logger.debug(f"First check for {channel_id}: no baseline set, returning empty to establish baseline")
            return []

        # Normalize after_date to naive datetime for comparison
        after_date_naive = None
        if after_date:
            after_date_naive = after_date.replace(tzinfo=None) if after_date.tzinfo else after_date

        new_videos = []
        for video in videos:
            # Stop if we've reached the reference video
            if after_video_id and video.video_id == after_video_id:
                break

            # Filter by date if specified (normalize to naive for comparison)
            if after_date_naive:
                video_date = video.published_at
                if video_date.tzinfo:
                    video_date = video_date.replace(tzinfo=None)
                if video_date <= after_date_naive:
                    break

            new_videos.append(video)

        return new_videos

    def parse_channel_url(self, url: str) -> Optional[str]:
        """
        Parse a channel URL to extract the channel identifier.
        Override in subclasses for platform-specific parsing.

        Args:
            url: The channel URL

        Returns:
            Channel identifier or None if URL cannot be parsed
        """
        return None

    async def get_videos_in_date_range(
        self,
        channel_id: str,
        start_date: datetime,
        end_date: datetime,
        max_videos: int = 50
    ) -> List[VideoInfo]:
        """
        Get videos from a channel within a specific date range.
        Override in subclasses for platform-specific implementations.

        Args:
            channel_id: The platform-specific channel identifier
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            max_videos: Maximum number of videos to return

        Returns:
            List of VideoInfo for videos published within the date range
        """
        # Default implementation: fetch latest videos and filter by date
        videos = await self.get_latest_videos(channel_id, limit=max_videos)
        
        logger.debug(f"Fetched {len(videos)} videos for {channel_id}, filtering by date range {start_date} to {end_date}")

        # Normalize dates to naive datetime for comparison (handle timezone mismatch)
        start_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
        end_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date

        result = []
        for v in videos:
            video_date = v.published_at.replace(tzinfo=None) if v.published_at.tzinfo else v.published_at
            logger.debug(f"Video {v.video_id}: published_at={video_date}, in_range={start_naive <= video_date <= end_naive}")
            if start_naive <= video_date <= end_naive:
                result.append(v)
        
        logger.info(f"Date filter: {len(videos)} videos -> {len(result)} in range")
        return result

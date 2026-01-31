"""
Factory function for creating platform-specific fetchers.
"""
from typing import Optional

from .base import BaseFetcher
from .youtube_fetcher import YouTubeFetcher
from .tiktok_fetcher import TikTokFetcher
from .instagram_fetcher import InstagramFetcher


# Supported platforms
SUPPORTED_PLATFORMS = {
    "youtube": {
        "name": "YouTube",
        "description": "YouTube video platform",
        "url_patterns": [
            "youtube.com",
            "youtu.be"
        ]
    },
    "tiktok": {
        "name": "TikTok",
        "description": "TikTok short video platform",
        "url_patterns": [
            "tiktok.com"
        ]
    },
    "instagram": {
        "name": "Instagram Reels",
        "description": "Instagram Reels",
        "url_patterns": [
            "instagram.com"
        ]
    }
}


def get_fetcher(platform: str) -> Optional[BaseFetcher]:
    """
    Get a fetcher instance for the specified platform.

    Args:
        platform: Platform name (youtube, tiktok, instagram)

    Returns:
        BaseFetcher instance or None if platform is not supported
    """
    platform = platform.lower()

    if platform == "youtube":
        return YouTubeFetcher()
    elif platform == "tiktok":
        return TikTokFetcher()
    elif platform == "instagram":
        return InstagramFetcher()
    else:
        return None


def detect_platform(url: str) -> Optional[str]:
    """
    Detect platform from a URL.

    Args:
        url: Video or channel URL

    Returns:
        Platform name or None if not detected
    """
    url_lower = url.lower()

    for platform, info in SUPPORTED_PLATFORMS.items():
        for pattern in info["url_patterns"]:
            if pattern in url_lower:
                return platform

    return None


async def lookup_channel_auto(url: str):
    """
    Automatically detect platform and lookup channel info.

    Args:
        url: Channel URL

    Returns:
        ChannelInfo or None
    """
    platform = detect_platform(url)
    if not platform:
        return None

    fetcher = get_fetcher(platform)
    if not fetcher:
        return None

    try:
        return await fetcher.lookup_channel(url)
    finally:
        await fetcher.close()

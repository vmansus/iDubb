"""
Subscription tracking module for YouTube, TikTok, and Instagram channels.
Monitors channels for new videos and automatically creates processing tasks.
"""
from .base import BaseFetcher, ChannelInfo, VideoInfo
from .youtube_fetcher import YouTubeFetcher
from .tiktok_fetcher import TikTokFetcher
from .instagram_fetcher import InstagramFetcher
from .fetcher_factory import get_fetcher, SUPPORTED_PLATFORMS
from .scheduler import SubscriptionScheduler

__all__ = [
    "BaseFetcher",
    "ChannelInfo",
    "VideoInfo",
    "YouTubeFetcher",
    "TikTokFetcher",
    "InstagramFetcher",
    "get_fetcher",
    "SUPPORTED_PLATFORMS",
    "SubscriptionScheduler",
]

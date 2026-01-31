"""
YouTube channel fetcher using RSS feeds.
No API key required - uses public RSS feeds.
For historical video fetching, uses yt-dlp.
"""
import asyncio
import logging
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

import httpx
import yt_dlp
from loguru import logger

from .base import BaseFetcher, ChannelInfo, VideoInfo


class YouTubeFetcher(BaseFetcher):
    """Fetcher for YouTube channels using RSS feeds"""

    RSS_BASE_URL = "https://www.youtube.com/feeds/videos.xml"
    CHANNEL_URL_BASE = "https://www.youtube.com/channel/"
    HANDLE_URL_BASE = "https://www.youtube.com/@"

    # Regex patterns for YouTube URLs
    CHANNEL_ID_PATTERN = re.compile(r"^UC[\w-]{22}$")
    HANDLE_PATTERN = re.compile(r"@([\w.-]+)")
    CHANNEL_URL_PATTERN = re.compile(r"youtube\.com/channel/(UC[\w-]{22})")
    HANDLE_URL_PATTERN = re.compile(r"youtube\.com/@([\w.-]+)")
    USER_URL_PATTERN = re.compile(r"youtube\.com/user/([\w.-]+)")
    CUSTOM_URL_PATTERN = re.compile(r"youtube\.com/c/([\w.-]+)")

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def platform(self) -> str:
        return "youtube"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def parse_channel_url(self, url: str) -> Optional[str]:
        """Parse YouTube URL to extract channel identifier"""
        # Direct channel ID
        if self.CHANNEL_ID_PATTERN.match(url):
            return url

        # Handle (@name)
        handle_match = self.HANDLE_PATTERN.match(url)
        if handle_match:
            return f"@{handle_match.group(1)}"

        # Full URLs
        channel_match = self.CHANNEL_URL_PATTERN.search(url)
        if channel_match:
            return channel_match.group(1)

        handle_url_match = self.HANDLE_URL_PATTERN.search(url)
        if handle_url_match:
            return f"@{handle_url_match.group(1)}"

        user_match = self.USER_URL_PATTERN.search(url)
        if user_match:
            return f"user/{user_match.group(1)}"

        custom_match = self.CUSTOM_URL_PATTERN.search(url)
        if custom_match:
            return f"c/{custom_match.group(1)}"

        return None

    async def _resolve_handle_to_channel_id(self, handle: str) -> Optional[str]:
        """Resolve a YouTube handle (@name) to a channel ID"""
        client = await self._get_client()

        # Remove @ if present
        handle_name = handle.lstrip("@")
        url = f"{self.HANDLE_URL_BASE}{handle_name}"

        try:
            response = await client.get(url)
            if response.status_code != 200:
                logger.warning(f"Failed to resolve handle {handle}: HTTP {response.status_code}")
                return None

            # Extract channel ID from page source
            html = response.text

            # Priority 1: browseId is the most reliable for the main channel
            channel_id_match = re.search(r'"browseId":"(UC[\w-]{22})"', html)
            if channel_id_match:
                return channel_id_match.group(1)

            # Priority 2: externalId
            channel_id_match = re.search(r'"externalId":"(UC[\w-]{22})"', html)
            if channel_id_match:
                return channel_id_match.group(1)

            # Priority 3: channelId (less reliable, may pick up related channels)
            channel_id_match = re.search(r'"channelId":"(UC[\w-]{22})"', html)
            if channel_id_match:
                return channel_id_match.group(1)

            # Priority 4: URL pattern
            channel_id_match = re.search(r'/channel/(UC[\w-]{22})', html)
            if channel_id_match:
                return channel_id_match.group(1)

            logger.warning(f"Could not find channel ID for handle {handle}")
            return None

        except Exception as e:
            logger.error(f"Error resolving handle {handle}: {e}")
            return None

    async def _get_channel_info_from_page(self, channel_id: str) -> Optional[ChannelInfo]:
        """Get channel info by scraping the channel page"""
        client = await self._get_client()
        url = f"{self.CHANNEL_URL_BASE}{channel_id}"

        try:
            response = await client.get(url)
            if response.status_code != 200:
                return None

            html = response.text

            # Extract channel name
            name_match = re.search(r'"title":"([^"]+)".*?"channelId":"' + channel_id, html)
            if not name_match:
                name_match = re.search(r'<title>([^<]+)</title>', html)

            channel_name = name_match.group(1) if name_match else "Unknown Channel"
            # Clean up name (remove " - YouTube" suffix)
            channel_name = re.sub(r'\s*-\s*YouTube$', '', channel_name)

            # Extract avatar URL
            avatar_match = re.search(r'"avatar":\{"thumbnails":\[\{"url":"([^"]+)"', html)
            avatar_url = avatar_match.group(1) if avatar_match else None

            return ChannelInfo(
                platform="youtube",
                channel_id=channel_id,
                channel_name=channel_name,
                channel_url=url,
                avatar_url=avatar_url
            )

        except Exception as e:
            logger.error(f"Error getting channel info for {channel_id}: {e}")
            return None

    async def lookup_channel(self, url_or_id: str) -> Optional[ChannelInfo]:
        """Look up YouTube channel information"""
        identifier = self.parse_channel_url(url_or_id)
        if not identifier:
            identifier = url_or_id

        # If it's a handle, resolve to channel ID first
        if identifier.startswith("@"):
            channel_id = await self._resolve_handle_to_channel_id(identifier)
            if not channel_id:
                return None
        elif identifier.startswith("user/") or identifier.startswith("c/"):
            # Legacy user/custom URLs - need to resolve
            client = await self._get_client()
            try:
                url = f"https://www.youtube.com/{identifier}"
                response = await client.get(url)
                if response.status_code == 200:
                    channel_id_match = re.search(r'"channelId":"(UC[\w-]{22})"', response.text)
                    if channel_id_match:
                        channel_id = channel_id_match.group(1)
                    else:
                        return None
                else:
                    return None
            except Exception as e:
                logger.error(f"Error resolving {identifier}: {e}")
                return None
        else:
            channel_id = identifier

        # Get channel info
        return await self._get_channel_info_from_page(channel_id)

    async def get_latest_videos(
        self,
        channel_id: str,
        limit: int = 10
    ) -> List[VideoInfo]:
        """Get latest videos from YouTube RSS feed"""
        client = await self._get_client()

        # Build RSS URL
        rss_url = f"{self.RSS_BASE_URL}?channel_id={channel_id}"

        try:
            response = await client.get(rss_url)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch RSS for {channel_id}: HTTP {response.status_code}")
                return []

            # Parse XML
            root = ET.fromstring(response.text)

            # Define namespace
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "yt": "http://www.youtube.com/xml/schemas/2015",
                "media": "http://search.yahoo.com/mrss/"
            }

            videos = []
            for entry in root.findall("atom:entry", ns)[:limit]:
                video_id_elem = entry.find("yt:videoId", ns)
                title_elem = entry.find("atom:title", ns)
                link_elem = entry.find("atom:link", ns)
                published_elem = entry.find("atom:published", ns)
                media_group = entry.find("media:group", ns)

                if video_id_elem is None or title_elem is None:
                    continue

                video_id = video_id_elem.text
                title = title_elem.text
                url = link_elem.get("href") if link_elem is not None else f"https://www.youtube.com/watch?v={video_id}"

                # Parse published date (always use naive datetime for consistency)
                published_at = datetime.now()
                if published_elem is not None and published_elem.text:
                    try:
                        # ISO format: 2024-01-15T12:00:00+00:00
                        date_str = published_elem.text
                        parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        # Remove timezone info for consistent comparison
                        published_at = parsed_date.replace(tzinfo=None)
                    except ValueError:
                        pass

                # Get thumbnail and description from media group
                thumbnail_url = None
                description = None
                if media_group is not None:
                    thumbnail_elem = media_group.find("media:thumbnail", ns)
                    if thumbnail_elem is not None:
                        thumbnail_url = thumbnail_elem.get("url")

                    desc_elem = media_group.find("media:description", ns)
                    if desc_elem is not None:
                        description = desc_elem.text

                videos.append(VideoInfo(
                    video_id=video_id,
                    title=title,
                    url=url,
                    published_at=published_at,
                    thumbnail_url=thumbnail_url,
                    description=description
                ))

            return videos

        except ET.ParseError as e:
            logger.error(f"Failed to parse RSS for {channel_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching videos for {channel_id}: {e}")
            return []

    async def get_videos_in_date_range(
        self,
        channel_id: str,
        start_date: datetime,
        end_date: datetime,
        max_videos: int = 50
    ) -> List[VideoInfo]:
        """
        Get videos from a YouTube channel within a specific date range.

        Uses a hybrid approach:
        1. First tries RSS feed (no auth required, returns 15 most recent videos with dates)
        2. Falls back to yt-dlp if RSS doesn't cover the date range

        Args:
            channel_id: YouTube channel ID (UC...)
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            max_videos: Maximum number of videos to return (default 50, max 100)

        Returns:
            List of VideoInfo for videos published within the date range
        """
        max_videos = min(max_videos, 100)
        logger.info(f"Fetching videos for channel {channel_id} from {start_date.date()} to {end_date.date()}")

        # Normalize dates for comparison (remove time component)
        start_date_norm = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date_norm = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Step 1: Try RSS feed first (fast, no auth required)
        videos = await self._get_videos_from_rss_in_range(
            channel_id, start_date_norm, end_date_norm, max_videos
        )

        if videos:
            logger.info(f"RSS returned {len(videos)} videos in date range")
            return videos

        # Step 2: RSS didn't return videos in range, try yt-dlp
        # Note: RSS only returns the 15 most recent videos. For older videos, yt-dlp is needed.
        logger.info("RSS didn't return videos in range (RSS only has 15 most recent). Trying yt-dlp...")
        videos = await self._get_videos_from_ytdlp_in_range(
            channel_id, start_date_norm, end_date_norm, max_videos
        )

        if not videos:
            logger.info("No videos found in the specified date range. This may be because: "
                       "1) No videos were published in that range, or "
                       "2) YouTube cookies need to be refreshed for historical video access.")

        return videos

    async def _get_videos_from_rss_in_range(
        self,
        channel_id: str,
        start_date: datetime,
        end_date: datetime,
        max_videos: int
    ) -> List[VideoInfo]:
        """Get videos from RSS feed filtered by date range."""
        client = await self._get_client()
        rss_url = f"{self.RSS_BASE_URL}?channel_id={channel_id}"

        try:
            response = await client.get(rss_url)
            if response.status_code != 200:
                logger.warning(f"RSS fetch failed: HTTP {response.status_code}")
                return []

            root = ET.fromstring(response.text)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "yt": "http://www.youtube.com/xml/schemas/2015",
                "media": "http://search.yahoo.com/mrss/"
            }

            videos = []
            for entry in root.findall("atom:entry", ns):
                video_id_elem = entry.find("yt:videoId", ns)
                title_elem = entry.find("atom:title", ns)
                published_elem = entry.find("atom:published", ns)

                if video_id_elem is None or title_elem is None:
                    continue

                video_id = video_id_elem.text
                title = title_elem.text

                # Parse published date
                published_at = None
                if published_elem is not None and published_elem.text:
                    try:
                        date_str = published_elem.text
                        published_at = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        # Remove timezone for comparison
                        if published_at.tzinfo:
                            published_at = published_at.replace(tzinfo=None)
                    except ValueError as e:
                        logger.debug(f"Date parse error for {video_id}: {e}")
                        continue

                if not published_at:
                    continue

                # Filter by date range
                if published_at < start_date or published_at > end_date:
                    logger.debug(f"Video {video_id} ({published_at.date()}) outside range, skipping")
                    continue

                # Get thumbnail and description
                link_elem = entry.find("atom:link", ns)
                media_group = entry.find("media:group", ns)

                url = link_elem.get("href") if link_elem is not None else f"https://www.youtube.com/watch?v={video_id}"

                thumbnail_url = None
                description = None
                if media_group is not None:
                    thumbnail_elem = media_group.find("media:thumbnail", ns)
                    if thumbnail_elem is not None:
                        thumbnail_url = thumbnail_elem.get("url")
                    desc_elem = media_group.find("media:description", ns)
                    if desc_elem is not None:
                        description = desc_elem.text

                if not thumbnail_url:
                    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

                videos.append(VideoInfo(
                    video_id=video_id,
                    title=title,
                    url=url,
                    published_at=published_at,
                    thumbnail_url=thumbnail_url,
                    description=description
                ))

                if len(videos) >= max_videos:
                    break

            videos.sort(key=lambda v: v.published_at, reverse=True)
            return videos

        except ET.ParseError as e:
            logger.error(f"RSS parse error: {e}")
            return []
        except Exception as e:
            logger.error(f"RSS fetch error: {e}")
            return []

    async def _get_videos_from_ytdlp_in_range(
        self,
        channel_id: str,
        start_date: datetime,
        end_date: datetime,
        max_videos: int
    ) -> List[VideoInfo]:
        """Get videos using yt-dlp with date range filtering."""
        channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
        date_after = start_date.strftime("%Y%m%d")
        date_before = (end_date + timedelta(days=1)).strftime("%Y%m%d")

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'playlistend': max_videos * 2,
            'daterange': yt_dlp.DateRange(date_after, date_before),
            'ignoreerrors': True,
            'no_check_certificates': True,
            'logger': logging.getLogger('yt-dlp-silent'),  # Use a silent logger
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['mweb', 'web'],
                    'skip': ['hls', 'dash'],
                }
            }
        }

        # Set up a null logger to suppress yt-dlp errors
        silent_logger = logging.getLogger('yt-dlp-silent')
        silent_logger.setLevel(logging.CRITICAL + 1)  # Higher than CRITICAL = silent

        # Try to find cookies file
        possible_paths = [
            Path('/app/data/youtube_cookies.txt'),
            Path(__file__).parent.parent / 'data' / 'youtube_cookies.txt',
            Path(__file__).parent.parent.parent / 'data' / 'youtube_cookies.txt',
            Path.home() / '.idubb' / 'youtube_cookies.txt',
        ]

        for cookies_path in possible_paths:
            if cookies_path.exists():
                ydl_opts['cookiefile'] = str(cookies_path)
                logger.info(f"Using cookies: {cookies_path}")
                break
        else:
            logger.warning("No cookies file found, yt-dlp may fail")

        try:
            def extract():
                # Suppress stderr to hide yt-dlp's noisy error messages
                import io
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        return ydl.extract_info(channel_url, download=False)
                finally:
                    sys.stderr = old_stderr

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, extract)

            if not result or 'entries' not in result:
                logger.warning(f"yt-dlp returned no results for {channel_id}")
                return []

            videos = []
            entries = result.get('entries', [])
            logger.info(f"yt-dlp returned {len(entries) if entries else 0} entries")

            for entry in entries:
                if not entry:
                    continue

                video_id = entry.get('id')
                title = entry.get('title')

                if not video_id or not title:
                    continue

                # Parse date from multiple possible fields
                published_at = None
                upload_date_str = entry.get('upload_date') or entry.get('release_date')

                if upload_date_str:
                    try:
                        published_at = datetime.strptime(upload_date_str, "%Y%m%d")
                    except ValueError:
                        pass

                if not published_at and entry.get('timestamp'):
                    try:
                        published_at = datetime.fromtimestamp(entry['timestamp'])
                    except (ValueError, OSError):
                        pass

                if not published_at:
                    published_at = datetime.now()

                # Filter by date range
                if published_at < start_date or published_at > end_date:
                    continue

                thumbnail_url = entry.get('thumbnail') or f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

                videos.append(VideoInfo(
                    video_id=video_id,
                    title=title,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    published_at=published_at,
                    thumbnail_url=thumbnail_url,
                    duration=entry.get('duration'),
                    description=entry.get('description'),
                ))

            videos.sort(key=lambda v: v.published_at, reverse=True)
            return videos[:max_videos]

        except Exception as e:
            logger.error(f"yt-dlp error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

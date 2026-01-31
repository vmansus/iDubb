"""
Instagram Reels fetcher implementation.
Fetches user info and reels from Instagram.
"""
import re
import json
from typing import Optional, List
from datetime import datetime

import httpx
from loguru import logger

from .base import BaseFetcher, ChannelInfo, VideoInfo


class InstagramFetcher(BaseFetcher):
    """Fetcher for Instagram Reels."""

    @property
    def platform(self) -> str:
        return "instagram"

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "X-IG-App-ID": "936619743392459",
            }
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _extract_username(self, url: str) -> Optional[str]:
        """Extract username from Instagram URL."""
        # Patterns:
        # https://www.instagram.com/username/
        # https://instagram.com/username/reels/
        # @username
        # username
        patterns = [
            r'instagram\.com/([^/?]+)',
            r'^@([^/?]+)$',
        ]

        for pattern in patterns:
            match = re.search(pattern, url.strip())
            if match:
                username = match.group(1)
                # Filter out instagram paths
                if username not in ['p', 'reel', 'reels', 'stories', 'explore', 'accounts']:
                    return username

        # If it's just a username without @
        cleaned = url.strip().lstrip('@')
        if not cleaned.startswith('http') and '/' not in cleaned:
            return cleaned

        return None

    async def lookup_channel(self, url: str) -> Optional[ChannelInfo]:
        """
        Look up Instagram user information.

        Args:
            url: Instagram user URL or @username

        Returns:
            ChannelInfo or None if not found
        """
        username = self._extract_username(url)
        if not username:
            return None

        try:
            # Try the web profile info endpoint
            api_url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
            response = await self.client.get(api_url)

            if response.status_code == 200:
                try:
                    data = response.json()
                    user_data = data.get("data", {}).get("user", {})

                    if user_data:
                        return ChannelInfo(
                            channel_id=user_data.get("id", username),
                            channel_name=user_data.get("full_name") or username,
                            channel_url=f"https://www.instagram.com/{username}/",
                            channel_avatar=user_data.get("profile_pic_url", ""),
                            platform="instagram"
                        )
                except json.JSONDecodeError:
                    pass

            # Fallback: scrape the profile page
            profile_url = f"https://www.instagram.com/{username}/"
            response = await self.client.get(profile_url)

            if response.status_code != 200:
                return None

            html = response.text

            # Try to extract from meta tags
            name_match = re.search(r'<meta property="og:title" content="([^"]+)"', html)
            avatar_match = re.search(r'"profile_pic_url":"([^"]+)"', html)
            id_match = re.search(r'"profilePage_([0-9]+)"', html)

            channel_name = username
            if name_match:
                # Format is usually "Name (@username)"
                title = name_match.group(1)
                if '(@' in title:
                    channel_name = title.split('(@')[0].strip()
                else:
                    channel_name = title

            avatar = ""
            if avatar_match:
                avatar = avatar_match.group(1).replace('\\u0026', '&')

            channel_id = id_match.group(1) if id_match else username

            return ChannelInfo(
                channel_id=channel_id,
                channel_name=channel_name,
                channel_url=profile_url,
                channel_avatar=avatar,
                platform="instagram"
            )

        except Exception as e:
            logger.error(f"Instagram lookup error: {e}")
            return None

    async def get_latest_videos(self, channel_id: str, limit: int = 10) -> List[VideoInfo]:
        """
        Fetch recent reels from an Instagram user.

        Note: This requires either:
        1. User's numeric ID (not username)
        2. Scraping the profile page

        Args:
            channel_id: Instagram user ID or username
            limit: Maximum number of videos to fetch

        Returns:
            List of VideoInfo objects (reels only)
        """
        videos = []

        try:
            # Try to get reels from the profile page
            # First, we need the numeric user ID if we have username
            if not channel_id.isdigit():
                # Try to get user info first
                api_url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={channel_id}"
                response = await self.client.get(api_url)

                if response.status_code == 200:
                    try:
                        data = response.json()
                        user_data = data.get("data", {}).get("user", {})

                        # Get reels from edge_felix_video_timeline
                        reels_data = user_data.get("edge_felix_video_timeline", {})
                        edges = reels_data.get("edges", [])

                        for edge in edges[:limit]:
                            node = edge.get("node", {})
                            try:
                                # Parse timestamp
                                timestamp = node.get("taken_at_timestamp", 0)
                                if timestamp:
                                    published_at = datetime.fromtimestamp(timestamp)
                                else:
                                    published_at = datetime.now()

                                shortcode = node.get("shortcode", "")

                                videos.append(VideoInfo(
                                    video_id=node.get("id", shortcode),
                                    title=self._get_caption(node) or f"Reel {shortcode}",
                                    url=f"https://www.instagram.com/reel/{shortcode}/",
                                    published_at=published_at,
                                    thumbnail_url=node.get("thumbnail_src", ""),
                                    duration=node.get("video_duration", 0)
                                ))
                            except Exception:
                                continue

                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            logger.error(f"Instagram get_latest_videos error: {e}")

        return videos

    def _get_caption(self, node: dict) -> str:
        """Extract caption from node data."""
        edges = node.get("edge_media_to_caption", {}).get("edges", [])
        if edges:
            return edges[0].get("node", {}).get("text", "")[:100]
        return ""

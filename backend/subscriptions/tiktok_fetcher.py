"""
TikTok channel fetcher implementation.
Fetches user info and videos from TikTok using Playwright for dynamic content.
"""
import re
import json
import asyncio
from typing import Optional, List
from datetime import datetime

import httpx
from loguru import logger

from .base import BaseFetcher, ChannelInfo, VideoInfo

# Try to import playwright
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed - TikTok video fetching may be limited")


class TikTokFetcher(BaseFetcher):
    """Fetcher for TikTok user videos."""

    @property
    def platform(self) -> str:
        return "tiktok"

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        self._browser = None
        self._playwright = None

    async def close(self):
        """Close the HTTP client and browser."""
        await self.client.aclose()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    def _extract_username(self, url: str) -> Optional[str]:
        """Extract username from TikTok URL."""
        patterns = [
            r'tiktok\.com/@([^/?]+)',
            r'^@([^/?]+)$',
        ]

        for pattern in patterns:
            match = re.search(pattern, url.strip())
            if match:
                return match.group(1)

        if not url.startswith('http') and not url.startswith('@'):
            return url.strip()

        return None

    def _extract_rehydration_data(self, html: str) -> Optional[dict]:
        """Extract data from __UNIVERSAL_DATA_FOR_REHYDRATION__ script."""
        match = re.search(
            r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">(.+?)</script>',
            html
        )
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    def _extract_sigi_state(self, html: str) -> Optional[dict]:
        """Extract data from SIGI_STATE script (legacy)."""
        match = re.search(
            r'<script id="SIGI_STATE" type="application/json">(.+?)</script>',
            html
        )
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    async def lookup_channel(self, url: str) -> Optional[ChannelInfo]:
        """
        Look up TikTok user information.

        Args:
            url: TikTok user URL or @username

        Returns:
            ChannelInfo or None if not found
        """
        username = self._extract_username(url)
        if not username:
            return None

        try:
            user_url = f"https://www.tiktok.com/@{username}"
            response = await self.client.get(user_url)

            if response.status_code != 200:
                return None

            html = response.text

            # Try new format: __UNIVERSAL_DATA_FOR_REHYDRATION__
            rehydration_data = self._extract_rehydration_data(html)
            if rehydration_data:
                default_scope = rehydration_data.get("__DEFAULT_SCOPE__", {})
                user_detail = default_scope.get("webapp.user-detail", {})
                user_info = user_detail.get("userInfo", {})
                user = user_info.get("user", {})

                if user:
                    return ChannelInfo(
                        channel_id=user.get("uniqueId", username),
                        channel_name=user.get("nickname", username),
                        channel_url=user_url,
                        avatar_url=user.get("avatarMedium", ""),
                        platform="tiktok"
                    )

            # Try legacy format: SIGI_STATE
            sigi_data = self._extract_sigi_state(html)
            if sigi_data:
                user_module = sigi_data.get("UserModule", {})
                users = user_module.get("users", {})

                if username.lower() in users:
                    user_data = users[username.lower()]
                    return ChannelInfo(
                        channel_id=user_data.get("id", username),
                        channel_name=user_data.get("nickname", username),
                        channel_url=user_url,
                        avatar_url=user_data.get("avatarMedium", ""),
                        platform="tiktok"
                    )

            # Fallback: extract from meta tags
            name_match = re.search(r'<title>(.+?) \(@' + re.escape(username) + r'\)', html)
            avatar_match = re.search(r'"avatarMedium":"([^"]+)"', html)

            channel_name = name_match.group(1) if name_match else username
            avatar = avatar_match.group(1).replace('\\u002F', '/') if avatar_match else ""

            return ChannelInfo(
                channel_id=username,
                channel_name=channel_name,
                channel_url=user_url,
                avatar_url=avatar,
                platform="tiktok"
            )

        except Exception as e:
            logger.error(f"TikTok lookup error: {e}")
            return None

    async def _get_browser(self):
        """Get or create a browser instance."""
        if not PLAYWRIGHT_AVAILABLE:
            return None
            
        if not self._browser:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
        return self._browser

    async def _fetch_videos_with_playwright(self, channel_id: str, limit: int = 10) -> List[VideoInfo]:
        """Fetch videos using Playwright browser automation."""
        videos = []
        browser = await self._get_browser()
        
        if not browser:
            return videos

        context = None
        page = None
        
        try:
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080}
            )
            page = await context.new_page()
            
            # Intercept API responses to capture video data
            captured_videos = []
            
            async def handle_response(response):
                if "/api/post/item_list" in response.url or "item_list" in response.url:
                    try:
                        data = await response.json()
                        items = data.get("itemList", []) or data.get("items", [])
                        captured_videos.extend(items)
                    except Exception:
                        pass

            page.on("response", handle_response)
            
            user_url = f"https://www.tiktok.com/@{channel_id}"
            
            try:
                await page.goto(user_url, wait_until="networkidle", timeout=30000)
            except PlaywrightTimeout:
                logger.warning(f"TikTok page load timeout for {channel_id}, continuing anyway")
            
            # Wait for video grid to load
            try:
                await page.wait_for_selector('[data-e2e="user-post-item"]', timeout=10000)
            except PlaywrightTimeout:
                logger.debug(f"Video grid selector not found for {channel_id}")
            
            # Scroll multiple times to load more videos
            logger.debug(f"Starting scroll loop for {channel_id}, need {limit} videos")
            for scroll_count in range(8):  # More scrolls for historical videos
                await page.evaluate("window.scrollBy(0, 2000)")
                await asyncio.sleep(1)
                logger.debug(f"Scroll {scroll_count + 1}: captured {len(captured_videos)} videos so far")
                # Check if we have enough videos
                if len(captured_videos) >= limit:
                    break
            
            # Wait a bit more for any pending API responses
            await asyncio.sleep(2)
            logger.debug(f"Final captured videos count: {len(captured_videos)}")
            
            # Try to extract from captured API responses first
            if captured_videos:
                for item in captured_videos[:limit]:
                    try:
                        video_id = item.get("id", "")
                        create_time = item.get("createTime", 0)
                        published_at = datetime.fromtimestamp(int(create_time)) if create_time else datetime.now()
                        
                        videos.append(VideoInfo(
                            video_id=video_id,
                            title=item.get("desc", "")[:100] or f"Video {video_id}",
                            url=f"https://www.tiktok.com/@{channel_id}/video/{video_id}",
                            published_at=published_at,
                            thumbnail_url=item.get("video", {}).get("cover", ""),
                            duration=item.get("video", {}).get("duration", 0)
                        ))
                    except Exception:
                        continue
                
                if videos:
                    return videos
            
            # Fallback: extract from page HTML
            html = await page.content()
            
            # Try rehydration data
            rehydration_data = self._extract_rehydration_data(html)
            if rehydration_data:
                default_scope = rehydration_data.get("__DEFAULT_SCOPE__", {})
                user_detail = default_scope.get("webapp.user-detail", {})
                item_list = user_detail.get("itemList", [])
                
                for item in item_list[:limit]:
                    try:
                        video_id = item.get("id", "")
                        create_time = item.get("createTime", 0)
                        published_at = datetime.fromtimestamp(int(create_time)) if create_time else datetime.now()
                        
                        videos.append(VideoInfo(
                            video_id=video_id,
                            title=item.get("desc", "")[:100] or f"Video {video_id}",
                            url=f"https://www.tiktok.com/@{channel_id}/video/{video_id}",
                            published_at=published_at,
                            thumbnail_url=item.get("video", {}).get("cover", ""),
                            duration=item.get("video", {}).get("duration", 0)
                        ))
                    except Exception:
                        continue
            
            # Last resort: extract video links from DOM
            if not videos:
                video_links = await page.query_selector_all('a[href*="/video/"]')
                seen_ids = set()
                
                for link in video_links[:limit * 2]:  # Get more to account for duplicates
                    try:
                        href = await link.get_attribute("href")
                        match = re.search(r'/video/(\d+)', href)
                        if match:
                            video_id = match.group(1)
                            if video_id not in seen_ids:
                                seen_ids.add(video_id)
                                videos.append(VideoInfo(
                                    video_id=video_id,
                                    title=f"Video {video_id}",
                                    url=f"https://www.tiktok.com/@{channel_id}/video/{video_id}",
                                    published_at=datetime.now(),
                                    thumbnail_url="",
                                    duration=0
                                ))
                                if len(videos) >= limit:
                                    break
                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Playwright error fetching TikTok videos: {e}")
        finally:
            if page:
                await page.close()
            if context:
                await context.close()

        return videos

    async def get_latest_videos(self, channel_id: str, limit: int = 10) -> List[VideoInfo]:
        """
        Fetch recent videos from a TikTok user.

        Uses Playwright for browser-based scraping when available,
        falls back to HTTP requests otherwise.

        Args:
            channel_id: TikTok username
            limit: Maximum number of videos to fetch

        Returns:
            List of VideoInfo objects
        """
        videos = []

        # Try Playwright first (most reliable)
        if PLAYWRIGHT_AVAILABLE:
            logger.debug(f"Fetching TikTok videos with Playwright for {channel_id}")
            videos = await self._fetch_videos_with_playwright(channel_id, limit)
            if videos:
                logger.info(f"Found {len(videos)} TikTok videos for {channel_id} via Playwright")
                return videos

        # Fallback to HTTP request
        try:
            user_url = f"https://www.tiktok.com/@{channel_id}"
            response = await self.client.get(user_url)

            if response.status_code != 200:
                logger.warning(f"TikTok returned status {response.status_code} for {channel_id}")
                return videos

            html = response.text

            # Try new format: __UNIVERSAL_DATA_FOR_REHYDRATION__
            rehydration_data = self._extract_rehydration_data(html)
            if rehydration_data:
                default_scope = rehydration_data.get("__DEFAULT_SCOPE__", {})
                user_detail = default_scope.get("webapp.user-detail", {})
                item_list = user_detail.get("itemList", [])
                
                if item_list:
                    for item in item_list[:limit]:
                        try:
                            video_id = item.get("id", "")
                            create_time = item.get("createTime", 0)
                            published_at = datetime.fromtimestamp(int(create_time)) if create_time else datetime.now()

                            videos.append(VideoInfo(
                                video_id=video_id,
                                title=item.get("desc", "")[:100] or f"Video {video_id}",
                                url=f"https://www.tiktok.com/@{channel_id}/video/{video_id}",
                                published_at=published_at,
                                thumbnail_url=item.get("video", {}).get("cover", ""),
                                duration=item.get("video", {}).get("duration", 0)
                            ))
                        except Exception:
                            continue

                    if videos:
                        return videos

            # Try legacy format: SIGI_STATE
            sigi_data = self._extract_sigi_state(html)
            if sigi_data:
                item_module = sigi_data.get("ItemModule", {})

                for video_id, video_data in list(item_module.items())[:limit]:
                    try:
                        create_time = video_data.get("createTime", 0)
                        published_at = datetime.fromtimestamp(int(create_time)) if create_time else datetime.now()

                        videos.append(VideoInfo(
                            video_id=video_id,
                            title=video_data.get("desc", "")[:100] or f"Video {video_id}",
                            url=f"https://www.tiktok.com/@{channel_id}/video/{video_id}",
                            published_at=published_at,
                            thumbnail_url=video_data.get("video", {}).get("cover", ""),
                            duration=video_data.get("video", {}).get("duration", 0)
                        ))
                    except Exception:
                        continue

            # Try to extract video IDs from page content as last resort
            if not videos:
                video_pattern = r'/@' + re.escape(channel_id) + r'/video/(\d+)'
                video_ids = list(set(re.findall(video_pattern, html)))[:limit]
                
                for video_id in video_ids:
                    videos.append(VideoInfo(
                        video_id=video_id,
                        title=f"Video {video_id}",
                        url=f"https://www.tiktok.com/@{channel_id}/video/{video_id}",
                        published_at=datetime.now(),
                        thumbnail_url="",
                        duration=0
                    ))

            if not videos:
                logger.warning(f"Could not extract videos for TikTok user {channel_id}")

        except Exception as e:
            logger.error(f"TikTok get_latest_videos error: {e}")

        return videos

    async def get_videos_in_date_range(
        self,
        channel_id: str,
        start_date: datetime,
        end_date: datetime,
        max_videos: int = 50
    ) -> List[VideoInfo]:
        """
        Get videos from a TikTok user.
        
        Note: TikTok video dates are often unreliable (extracted from page or set to now()),
        so we skip date filtering and return all fetched videos up to max_videos.
        The videos are sorted by recency (newest first) based on page order.
        """
        logger.info(f"Fetching TikTok videos for {channel_id} (date filter disabled, max={max_videos})")
        videos = await self.get_latest_videos(channel_id, limit=max_videos)
        logger.info(f"Returning {len(videos)} TikTok videos (no date filtering)")
        return videos

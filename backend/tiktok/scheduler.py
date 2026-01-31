"""
TikTokScheduler - Fetches and caches TikTok trending videos by tag
Uses yt-dlp to fetch videos from TikTok
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from loguru import logger


# Default popular TikTok tags
DEFAULT_TIKTOK_TAGS = [
    "trending",
    "fyp",
    "viral",
    "foryou",
    "comedy",
    "dance",
    "music",
    "food",
    "travel",
    "tech",
    "gaming",
    "fitness",
]

# TikTok region codes
TIKTOK_REGIONS = {
    "US": "United States",
    "UK": "United Kingdom", 
    "CN": "China",
    "JP": "Japan",
    "KR": "South Korea",
    "DE": "Germany",
    "FR": "France",
    "BR": "Brazil",
    "IN": "India",
    "ID": "Indonesia",
    "TH": "Thailand",
    "VN": "Vietnam",
    "PH": "Philippines",
    "MY": "Malaysia",
    "SG": "Singapore",
    "TW": "Taiwan",
    "HK": "Hong Kong",
}


class TikTokScheduler:
    """Scheduler for fetching TikTok trending videos"""

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._update_interval = 60  # minutes
        self._last_updated: Optional[datetime] = None
        self._enabled_tags = DEFAULT_TIKTOK_TAGS[:3]  # Default to first 3 tags
        self._max_videos_per_tag = 20
        self._min_view_count = 10000
        self._min_like_count = 1000
        self._max_duration = 180  # 3 minutes
        self._region_code = "US"

        # Callbacks for database operations
        self._get_settings: Optional[Callable] = None
        self._update_settings: Optional[Callable] = None
        self._save_videos: Optional[Callable] = None

    def set_callbacks(
        self,
        get_settings: Callable,
        update_settings: Callable,
        save_videos: Callable,
    ):
        """Set callback functions for database operations"""
        self._get_settings = get_settings
        self._update_settings = update_settings
        self._save_videos = save_videos

    async def start(self):
        """Start the TikTok scheduler"""
        if self._running:
            logger.warning("TikTok scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("TikTok scheduler started")

    async def stop(self):
        """Stop the TikTok scheduler"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TikTok scheduler stopped")

    async def _run_loop(self):
        """Main scheduler loop"""
        # Initial delay before first fetch
        await asyncio.sleep(15)

        while self._running:
            try:
                # Load settings
                if self._get_settings:
                    settings = await self._get_settings()
                    if settings:
                        self._update_interval = settings.get("update_interval", 60)
                        self._enabled_tags = settings.get("enabled_tags", DEFAULT_TIKTOK_TAGS[:3])
                        self._max_videos_per_tag = settings.get("max_videos_per_tag", 20)
                        self._min_view_count = settings.get("min_view_count", 10000)
                        self._min_like_count = settings.get("min_like_count", 1000)
                        self._max_duration = settings.get("max_duration", 180)
                        self._region_code = settings.get("region_code", "US")

                        # Check if enabled
                        if not settings.get("enabled", False):
                            logger.debug("TikTok scheduler disabled, sleeping...")
                            await asyncio.sleep(60)
                            continue

                        # Check if update is needed
                        last_updated_str = settings.get("last_updated")
                        if last_updated_str:
                            self._last_updated = datetime.fromisoformat(last_updated_str)

                if self._should_update():
                    await self.fetch_tiktok_videos()

                # Sleep until next check
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in TikTok scheduler loop: {e}")
                await asyncio.sleep(60)

    def _should_update(self) -> bool:
        """Check if TikTok videos should be updated"""
        if not self._last_updated:
            return True

        time_since_update = datetime.now() - self._last_updated
        return time_since_update >= timedelta(minutes=self._update_interval)

    async def fetch_tiktok_videos(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch TikTok videos for all enabled tags"""
        logger.info("Fetching TikTok trending videos...")
        results = {}

        for tag in self._enabled_tags:
            try:
                videos = await self._fetch_tag(tag)
                results[tag] = videos
                logger.info(f"Fetched {len(videos)} TikTok videos for #{tag}")
            except Exception as e:
                logger.error(f"Failed to fetch TikTok for #{tag}: {e}")
                results[tag] = []

        # Update last_updated timestamp
        self._last_updated = datetime.now()
        if self._update_settings:
            await self._update_settings({
                "last_updated": self._last_updated.isoformat()
            })

        # Save videos to database
        if self._save_videos:
            for tag, videos in results.items():
                if videos:
                    await self._save_videos(f"tiktok_{tag}", videos)

        return results

    async def fetch_single_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Fetch TikTok videos for a single tag (for manual refresh)"""
        # Load latest settings first
        if self._get_settings:
            settings = await self._get_settings()
            if settings:
                self._max_videos_per_tag = settings.get("max_videos_per_tag", 20)
                self._min_view_count = settings.get("min_view_count", 10000)
                self._min_like_count = settings.get("min_like_count", 1000)
                self._max_duration = settings.get("max_duration", 180)
                self._region_code = settings.get("region_code", "US")

        try:
            videos = await self._fetch_tag(tag)
            logger.info(f"Fetched {len(videos)} TikTok videos for #{tag}")

            # Save to database
            if self._save_videos and videos:
                await self._save_videos(f"tiktok_{tag}", videos)

            return videos
        except Exception as e:
            logger.error(f"Failed to fetch TikTok for #{tag}: {e}")
            return []

    async def _fetch_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Fetch TikTok videos for a specific tag using Playwright scraper"""
        videos = []

        try:
            # Try Playwright scraper first
            from .scraper import scrape_tiktok_tag, PLAYWRIGHT_AVAILABLE
            
            if PLAYWRIGHT_AVAILABLE:
                logger.info(f"Using Playwright to scrape TikTok #{tag}")
                scraped = await scrape_tiktok_tag(
                    tag, 
                    max_videos=self._max_videos_per_tag * 2,
                    timeout=30000
                )
                
                logger.debug(f"Playwright scraped {len(scraped)} raw videos for #{tag}")
                
                # Apply filters (skip filter if value wasn't extracted)
                for video in scraped:
                    duration = video.get('duration')
                    view_count = video.get('view_count')
                    like_count = video.get('like_count')
                    
                    # Only filter on fields that were actually extracted
                    if duration is not None and duration > 0:
                        if self._max_duration > 0 and duration > self._max_duration:
                            continue
                    if view_count is not None and view_count > 0:
                        if view_count < self._min_view_count:
                            continue
                    if like_count is not None and like_count > 0:
                        if like_count < self._min_like_count:
                            continue
                    
                    videos.append(video)
                    
                    if len(videos) >= self._max_videos_per_tag:
                        break
                
                logger.debug(f"After filtering: {len(videos)} videos passed for #{tag}")
                
                if videos:
                    return videos
                
                logger.warning(f"Playwright returned no videos for #{tag} after filtering, trying yt-dlp fallback")
            else:
                logger.warning("Playwright not available, using yt-dlp fallback")
            
            # Fallback to yt-dlp (may not work due to TikTok restrictions)
            return await self._fetch_tag_ytdlp(tag)

        except Exception as e:
            logger.error(f"Failed to fetch TikTok tag #{tag}: {e}")
            return videos

    async def _fetch_tag_ytdlp(self, tag: str) -> List[Dict[str, Any]]:
        """Fallback: Fetch TikTok videos using yt-dlp (currently broken for tags)"""
        import yt_dlp

        videos = []
        loop = asyncio.get_event_loop()

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'ignoreerrors': True,
            'skip_download': True,
            'playlistend': self._max_videos_per_tag * 2,
        }

        try:
            tag_url = f"https://www.tiktok.com/tag/{tag}"
            logger.debug(f"Fetching TikTok tag via yt-dlp: {tag_url}")

            def extract_info():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(tag_url, download=False)

            result = await loop.run_in_executor(None, extract_info)

            if not result:
                return videos

            entries = result.get('entries', []) if 'entries' in result else [result]

            for entry in entries:
                if not entry or not entry.get('id'):
                    continue

                video_id = entry.get('id', '')
                duration = entry.get('duration', 0) or 0
                view_count = entry.get('view_count', 0) or 0
                like_count = entry.get('like_count', 0) or 0

                if self._max_duration > 0 and duration > self._max_duration:
                    continue
                if view_count < self._min_view_count:
                    continue
                if like_count < self._min_like_count:
                    continue

                if any(v['video_id'] == video_id for v in videos):
                    continue

                video_data = {
                    "video_id": video_id,
                    "title": entry.get('title', '') or entry.get('description', '')[:100] or f"TikTok #{tag}",
                    "channel_name": entry.get('uploader', '') or entry.get('creator', '') or 'Unknown',
                    "channel_url": entry.get('uploader_url', '') or entry.get('channel_url', ''),
                    "thumbnail_url": entry.get('thumbnail', ''),
                    "duration": duration,
                    "view_count": view_count,
                    "like_count": like_count,
                    "video_url": entry.get('webpage_url', '') or f"https://www.tiktok.com/@{entry.get('uploader', 'user')}/video/{video_id}",
                    "platform": "tiktok",
                    "published_at": entry.get('upload_date'),
                    "width": entry.get('width', 0) or 0,
                    "height": entry.get('height', 0) or 0,
                }
                videos.append(video_data)

                if len(videos) >= self._max_videos_per_tag:
                    break

            return videos

        except Exception as e:
            logger.error(f"yt-dlp fetch failed for #{tag}: {e}")
            return videos


# Global instance
tiktok_scheduler = TikTokScheduler()

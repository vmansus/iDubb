"""
TrendingScheduler - Fetches and caches YouTube trending videos by category
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
from loguru import logger


# YouTube category IDs for trending videos
YOUTUBE_CATEGORIES = {
    "tech": {
        "id": 28,  # Science & Technology
        "name": "Science & Technology",
        "display_name": "Tech & Programming",
        "search_queries": [
            "new tech review this week",
            "latest programming tutorial",
            "new AI news today",
            "tech news this week",
            "coding tutorial new",
            "latest software release",
        ],
    },
    "gaming": {
        "id": 20,  # Gaming
        "name": "Gaming",
        "display_name": "Gaming",
        "search_queries": [
            "new game release this week",
            "gaming highlights today",
            "latest gameplay",
            "new esports tournament",
            "game trailer new",
            "trending games this week",
        ],
    },
    "lifestyle": {
        "id": 22,  # People & Blogs
        "name": "People & Blogs",
        "display_name": "Lifestyle & Vlog",
        "search_queries": [
            "new vlog this week",
            "latest travel vlog",
            "new cooking video",
            "daily vlog today",
            "new fitness routine",
            "trending lifestyle",
        ],
    },
    "music": {
        "id": 10,  # Music
        "name": "Music",
        "display_name": "Music",
        "search_queries": [
            "new music video",
            "latest song release",
            "trending music",
            "new album review",
        ],
    },
    "entertainment": {
        "id": 24,  # Entertainment
        "name": "Entertainment",
        "display_name": "Entertainment",
        "search_queries": [
            "entertainment news",
            "celebrity interview",
            "trending entertainment",
            "new show review",
        ],
    },
    "education": {
        "id": 27,  # Education
        "name": "Education",
        "display_name": "Education",
        "search_queries": [
            "educational video",
            "learning tutorial",
            "online course",
            "science explained",
        ],
    },
    "news": {
        "id": 25,  # News & Politics
        "name": "News & Politics",
        "display_name": "News & Politics",
        "search_queries": [
            "breaking news today",
            "latest news",
            "world news",
            "political news",
        ],
    },
    "howto": {
        "id": 26,  # Howto & Style
        "name": "Howto & Style",
        "display_name": "Howto & Style",
        "search_queries": [
            "how to tutorial",
            "DIY project",
            "fashion tips",
            "beauty tutorial",
        ],
    },
    "comedy": {
        "id": 23,  # Comedy
        "name": "Comedy",
        "display_name": "Comedy",
        "search_queries": [
            "funny video",
            "comedy sketch",
            "stand up comedy",
            "funny moments",
        ],
    },
    "film": {
        "id": 1,  # Film & Animation
        "name": "Film & Animation",
        "display_name": "Film & Animation",
        "search_queries": [
            "movie trailer",
            "film review",
            "animation short",
            "new movie clips",
        ],
    },
    "sports": {
        "id": 17,  # Sports
        "name": "Sports",
        "display_name": "Sports",
        "search_queries": [
            "sports highlights",
            "game recap",
            "best plays",
            "sports news",
        ],
    },
}

# YouTube search filter parameters (sp parameter)
# These are base64 encoded protobuf filters
YOUTUBE_TIME_FILTERS = {
    "hour": "EgIIAQ%3D%3D",      # Last hour
    "today": "EgIIAg%3D%3D",     # Today
    "week": "EgIIAw%3D%3D",      # This week
    "month": "EgIIBA%3D%3D",     # This month
    "year": "EgIIBQ%3D%3D",      # This year
}

YOUTUBE_SORT_FILTERS = {
    "relevance": "",
    "upload_date": "CAISAhAB",   # Sort by upload date
    "view_count": "CAMSAhAB",    # Sort by view count
    "rating": "CAESAhAB",        # Sort by rating
}


class TrendingScheduler:
    """Scheduler for fetching YouTube trending videos"""

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._update_interval = 60  # minutes
        self._last_updated: Optional[datetime] = None
        self._enabled_categories = list(YOUTUBE_CATEGORIES.keys())
        self._max_videos_per_category = 20
        self._time_filter = "week"  # hour, today, week, month, year
        self._sort_by = "upload_date"  # relevance, upload_date, view_count, rating
        self._min_view_count = 10000  # Minimum views to include
        self._max_duration = 1800  # Maximum duration in seconds (30 min default)
        self._exclude_shorts = True  # Exclude YouTube Shorts (videos <= 60 seconds)
        
        # YouTube API settings
        self._youtube_api_key: Optional[str] = None
        self._use_official_api = True
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
        """Start the trending scheduler"""
        if self._running:
            logger.warning("Trending scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Trending scheduler started")

    async def stop(self):
        """Stop the trending scheduler"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Trending scheduler stopped")

    async def _run_loop(self):
        """Main scheduler loop"""
        # Initial delay before first fetch
        await asyncio.sleep(10)

        while self._running:
            try:
                # Load settings
                if self._get_settings:
                    settings = await self._get_settings()
                    if settings:
                        self._update_interval = settings.get("update_interval", 60)
                        self._enabled_categories = settings.get(
                            "enabled_categories", list(YOUTUBE_CATEGORIES.keys())
                        )
                        self._max_videos_per_category = settings.get(
                            "max_videos_per_category", 20
                        )
                        self._time_filter = settings.get("time_filter", "week")
                        self._sort_by = settings.get("sort_by", "upload_date")
                        self._min_view_count = settings.get("min_view_count", 10000)
                        self._max_duration = settings.get("max_duration", 1800)
                        self._exclude_shorts = settings.get("exclude_shorts", True)
                        
                        # YouTube API settings
                        self._youtube_api_key = settings.get("youtube_api_key")
                        self._use_official_api = settings.get("use_official_api", True)
                        self._region_code = settings.get("region_code", "US")

                        # Check if enabled
                        if not settings.get("enabled", True):
                            logger.debug("Trending scheduler disabled, sleeping...")
                            await asyncio.sleep(60)
                            continue

                        # Check if update is needed
                        last_updated_str = settings.get("last_updated")
                        if last_updated_str:
                            self._last_updated = datetime.fromisoformat(last_updated_str)

                if self._should_update():
                    await self.fetch_trending_videos()

                # Sleep until next check
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trending scheduler loop: {e}")
                await asyncio.sleep(60)

    def _should_update(self) -> bool:
        """Check if trending videos should be updated"""
        if not self._last_updated:
            return True

        time_since_update = datetime.now() - self._last_updated
        return time_since_update >= timedelta(minutes=self._update_interval)

    async def fetch_trending_videos(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch trending videos for all enabled categories"""
        logger.info("Fetching YouTube trending videos...")
        results = {}

        for category_key in self._enabled_categories:
            if category_key not in YOUTUBE_CATEGORIES:
                continue

            try:
                videos = await self._fetch_category(category_key)
                results[category_key] = videos
                logger.info(f"Fetched {len(videos)} trending videos for {category_key}")
            except Exception as e:
                logger.error(f"Failed to fetch trending for {category_key}: {e}")
                results[category_key] = []

        # Update last_updated timestamp
        self._last_updated = datetime.now()
        if self._update_settings:
            await self._update_settings({
                "last_updated": self._last_updated.isoformat()
            })

        # Save videos to database
        if self._save_videos:
            for category_key, videos in results.items():
                if videos:
                    await self._save_videos(category_key, videos)

        return results

    async def _fetch_category(self, category_key: str) -> List[Dict[str, Any]]:
        """Fetch trending/popular videos for a specific category"""
        # Use YouTube Data API if available and enabled
        if self._youtube_api_key and self._use_official_api:
            videos = await self._fetch_category_via_api(category_key)
        else:
            videos = await self._fetch_category_via_ytdlp(category_key)
        
        # Filter out Shorts if configured
        if self._exclude_shorts and videos:
            original_count = len(videos)
            videos = [v for v in videos if not self._is_short(v)]
            filtered_count = original_count - len(videos)
            if filtered_count > 0:
                logger.debug(f"Filtered out {filtered_count} Shorts from {category_key}")
        
        return videos

    async def fetch_single_category(self, category_key: str) -> List[Dict[str, Any]]:
        """Fetch trending videos for a single category (for manual refresh)"""
        if category_key not in YOUTUBE_CATEGORIES:
            logger.warning(f"Unknown category: {category_key}")
            return []
        
        # Load latest settings first
        if self._get_settings:
            settings = await self._get_settings()
            if settings:
                self._exclude_shorts = settings.get("exclude_shorts", True)
                self._youtube_api_key = settings.get("youtube_api_key")
                self._use_official_api = settings.get("use_official_api", True)
                self._region_code = settings.get("region_code", "US")
                self._max_videos_per_category = settings.get("max_videos_per_category", 20)
                self._min_view_count = settings.get("min_view_count", 10000)
                self._max_duration = settings.get("max_duration", 1800)
        
        try:
            videos = await self._fetch_category(category_key)
            logger.info(f"Fetched {len(videos)} trending videos for {category_key}")
            
            # Save to database
            if self._save_videos and videos:
                await self._save_videos(category_key, videos)
            
            return videos
        except Exception as e:
            logger.error(f"Failed to fetch trending for {category_key}: {e}")
            return []
    
    async def _fetch_category_via_api(self, category_key: str) -> List[Dict[str, Any]]:
        """Fetch trending videos using YouTube Data API v3"""
        import aiohttp
        from datetime import datetime, timedelta
        
        category = YOUTUBE_CATEGORIES[category_key]
        category_id = category.get("id")
        
        videos = []
        
        try:
            # Calculate published_after based on time_filter
            now = datetime.utcnow()
            time_deltas = {
                "hour": timedelta(hours=1),
                "today": timedelta(days=1),
                "week": timedelta(weeks=1),
                "month": timedelta(days=30),
                "year": timedelta(days=365),
            }
            delta = time_deltas.get(self._time_filter, timedelta(weeks=1))
            published_after = (now - delta).isoformat() + "Z"
            
            # Build API URL for mostPopular videos
            base_url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "snippet,contentDetails,statistics",
                "chart": "mostPopular",
                "regionCode": self._region_code,
                "maxResults": min(self._max_videos_per_category, 50),  # API max is 50
                "key": self._youtube_api_key,
            }
            
            # Add category filter if available
            if category_id:
                params["videoCategoryId"] = str(category_id)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"YouTube API error: {response.status} - {error_text}")
                        # Fallback to yt-dlp
                        return await self._fetch_category_via_ytdlp(category_key)
                    
                    data = await response.json()
                    
                    for item in data.get("items", []):
                        snippet = item.get("snippet", {})
                        content = item.get("contentDetails", {})
                        stats = item.get("statistics", {})
                        
                        # Parse duration (ISO 8601 format: PT1H2M3S)
                        duration_str = content.get("duration", "PT0S")
                        duration = self._parse_iso_duration(duration_str)
                        
                        # Apply duration filter
                        if self._max_duration > 0 and duration > self._max_duration:
                            continue
                        
                        # Apply view count filter
                        view_count = int(stats.get("viewCount", 0))
                        if view_count < self._min_view_count:
                            continue
                        
                        # Apply time filter (published_after)
                        published_at_str = snippet.get("publishedAt")
                        if published_at_str:
                            try:
                                published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                                published_at = published_at.replace(tzinfo=None)  # Make naive for comparison
                                cutoff_time = now - delta
                                if published_at < cutoff_time:
                                    continue
                            except (ValueError, TypeError):
                                pass  # Keep video if date parsing fails
                        
                        video_id = item.get("id")
                        video_data = {
                            "video_id": video_id,
                            "title": snippet.get("title", ""),
                            "channel_name": snippet.get("channelTitle", "Unknown"),
                            "channel_url": f"https://www.youtube.com/channel/{snippet.get('channelId', '')}",
                            "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"),
                            "duration": duration,
                            "view_count": view_count,
                            "video_url": f"https://www.youtube.com/watch?v={video_id}",
                            "platform": "youtube",
                            "published_at": snippet.get("publishedAt"),
                        }
                        videos.append(video_data)
            
            logger.info(f"Fetched {len(videos)} trending videos via YouTube API for {category_key}")
            return videos
            
        except Exception as e:
            logger.error(f"YouTube API fetch failed for {category_key}: {e}")
            # Fallback to yt-dlp method
            return await self._fetch_category_via_ytdlp(category_key)
    
    def _parse_iso_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration (PT1H2M3S) to seconds"""
        import re
        if not duration_str or not duration_str.startswith("PT"):
            return 0
        
        hours = minutes = seconds = 0
        
        # Match hours
        h_match = re.search(r'(\d+)H', duration_str)
        if h_match:
            hours = int(h_match.group(1))
        
        # Match minutes
        m_match = re.search(r'(\d+)M', duration_str)
        if m_match:
            minutes = int(m_match.group(1))
        
        # Match seconds
        s_match = re.search(r'(\d+)S', duration_str)
        if s_match:
            seconds = int(s_match.group(1))
        
        return hours * 3600 + minutes * 60 + seconds

    def _is_short(self, video: Dict[str, Any]) -> bool:
        """
        Check if a video is a YouTube Short.
        Detection methods (in priority order):
        1. URL contains /shorts/ path
        2. Vertical aspect ratio (height > width)
        3. Duration <= 60 seconds (fallback when no dimensions available)
        """
        # Method 1: Check URL for /shorts/ path (most reliable)
        video_url = video.get("video_url", "") or ""
        if "/shorts/" in video_url:
            return True
        
        # Method 2: Check aspect ratio if dimensions are available
        width = video.get("width", 0) or 0
        height = video.get("height", 0) or 0
        if width > 0 and height > 0:
            # Vertical video (height > width) is a Short
            if height > width:
                return True
            # Horizontal video is not a Short
            return False
        
        # Method 3: Duration-based fallback (when no dimensions available)
        # YouTube Shorts are max 60 seconds
        duration = video.get("duration", 0) or 0
        if duration > 0 and duration <= 60:
            return True
        
        return False

    async def _fetch_category_via_ytdlp(self, category_key: str) -> List[Dict[str, Any]]:
        """Fetch trending/popular videos for a specific category using yt-dlp search (fallback)"""
        import yt_dlp
        from urllib.parse import quote_plus

        category = YOUTUBE_CATEGORIES[category_key]

        # Use category-specific search queries
        queries = category.get("search_queries", ["trending"])

        # Get the time filter and sort filter sp parameters
        time_sp = YOUTUBE_TIME_FILTERS.get(self._time_filter, "")
        sort_sp = YOUTUBE_SORT_FILTERS.get(self._sort_by, "")

        # Step 1: Use flat extraction for fast initial fetch
        flat_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',
            'ignoreerrors': True,
            'skip_download': True,
            'playlistend': 20,
        }

        # Add cookies if available
        from config import settings as app_settings
        cookies_file = app_settings.DATA_DIR / "youtube_cookies.txt"
        if cookies_file.exists():
            flat_opts['cookiefile'] = str(cookies_file)

        candidate_videos = []
        loop = asyncio.get_event_loop()

        for query in queries:
            try:
                # Build YouTube search URL with time and sort filters
                encoded_query = quote_plus(query)
                # YouTube sp parameter: time_filter takes priority for filtering by date range,
                # sort_sp is used when no time filter to sort by upload date
                sp_param = time_sp if time_sp else sort_sp
                if sp_param:
                    search_url = f"https://www.youtube.com/results?search_query={encoded_query}&sp={sp_param}"
                    logger.debug(f"Search URL with sp={sp_param}: time_filter={self._time_filter}, sort_by={self._sort_by}")
                else:
                    search_url = f"ytsearch15:{query}"

                def extract_flat(url):
                    with yt_dlp.YoutubeDL(flat_opts) as ydl:
                        return ydl.extract_info(url, download=False)

                result = await loop.run_in_executor(None, lambda: extract_flat(search_url))

                if result and 'entries' in result:
                    for entry in result['entries']:
                        if not entry or not entry.get('id'):
                            continue

                        video_id = entry.get('id', '')
                        if any(v['video_id'] == video_id for v in candidate_videos):
                            continue

                        # Get duration and filter by max_duration
                        duration = entry.get('duration', 0) or 0
                        if duration > 0 and self._max_duration > 0 and duration > self._max_duration:
                            continue

                        # Get view count
                        view_count = entry.get('view_count', 0) or 0
                        if view_count > 0 and view_count < self._min_view_count:
                            continue

                        # Get original URL (may contain /shorts/ for Shorts)
                        original_url = entry.get('url', '') or entry.get('webpage_url', '') or ''
                        if not original_url or 'youtube.com' not in original_url:
                            original_url = f"https://www.youtube.com/watch?v={video_id}"

                        video_data = {
                            "video_id": video_id,
                            "title": entry.get('title', ''),
                            "channel_name": entry.get('uploader', '') or entry.get('channel', '') or 'Unknown',
                            "channel_url": entry.get('uploader_url', '') or entry.get('channel_url', ''),
                            "thumbnail_url": entry.get('thumbnail', '') or f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                            "duration": duration,
                            "view_count": view_count,
                            "video_url": original_url,
                            "platform": "youtube",
                            "published_at": None,  # Will be fetched in step 2
                            "width": entry.get('width', 0) or 0,
                            "height": entry.get('height', 0) or 0,
                        }
                        candidate_videos.append(video_data)

                        if len(candidate_videos) >= self._max_videos_per_category * 2:
                            break

            except Exception as e:
                logger.warning(f"Failed to search for '{query}': {e}")
                continue

        # Step 2: Fetch upload_date for candidate videos (limit to max needed)
        videos_to_fetch = candidate_videos[:self._max_videos_per_category]

        # Step 2: Try to fetch upload dates (optional - may fail due to rate limiting)
        # Note: This step is often rate-limited by YouTube. Videos are already sorted by
        # upload_date from the search API, so skipping this step is acceptable.
        fetch_upload_dates = False  # Disabled due to YouTube rate limiting; videos already sorted by upload_date

        if videos_to_fetch and fetch_upload_dates:
            single_opts = {
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
                'skip_download': True,
            }

            # Use cookie file for authentication
            if cookies_file.exists():
                single_opts['cookiefile'] = str(cookies_file)

            # Only try to fetch upload dates for a few videos to test if it works
            test_count = min(3, len(videos_to_fetch))
            success_count = 0

            logger.info(f"Testing upload date fetch for {test_count} videos...")

            for idx, video_data in enumerate(videos_to_fetch[:test_count]):
                try:
                    video_url = video_data['video_url']

                    def extract_single(url, opts):
                        with yt_dlp.YoutubeDL(opts) as ydl:
                            return ydl.extract_info(url, download=False)

                    info = await loop.run_in_executor(
                        None,
                        lambda u=video_url, o=single_opts: extract_single(u, o)
                    )

                    if info:
                        upload_date = info.get('upload_date')
                        if upload_date and len(upload_date) == 8:
                            try:
                                video_data['published_at'] = datetime.strptime(upload_date, '%Y%m%d')
                                success_count += 1
                            except ValueError:
                                pass

                        if not video_data.get('published_at') and info.get('timestamp'):
                            try:
                                video_data['published_at'] = datetime.fromtimestamp(info['timestamp'])
                                success_count += 1
                            except (ValueError, OSError):
                                pass

                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.debug(f"Failed to fetch upload date for {video_data['video_id']}: {e}")
                    await asyncio.sleep(0.5)

            # If test was successful, fetch remaining videos
            if success_count > 0:
                logger.info(f"Upload date fetch working, continuing with remaining videos...")
                for idx, video_data in enumerate(videos_to_fetch[test_count:], start=test_count):
                    try:
                        video_url = video_data['video_url']
                        info = await loop.run_in_executor(
                            None,
                            lambda u=video_url, o=single_opts: extract_single(u, o)
                        )
                        if info:
                            upload_date = info.get('upload_date')
                            if upload_date and len(upload_date) == 8:
                                try:
                                    video_data['published_at'] = datetime.strptime(upload_date, '%Y%m%d')
                                except ValueError:
                                    pass
                            if not video_data.get('published_at') and info.get('timestamp'):
                                try:
                                    video_data['published_at'] = datetime.fromtimestamp(info['timestamp'])
                                except (ValueError, OSError):
                                    pass
                        if (idx + 1) % 5 == 0:
                            logger.info(f"Fetched upload dates for {idx + 1}/{len(videos_to_fetch)} videos")
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.debug(f"Failed to fetch upload date: {e}")
                        await asyncio.sleep(0.5)
            else:
                logger.warning(f"Upload date fetch not working (rate limited or auth required), using search order")

        # Count how many videos have published_at
        videos_with_date = sum(1 for v in videos_to_fetch if v.get('published_at'))
        logger.info(f"Got upload dates for {videos_with_date}/{len(videos_to_fetch)} videos")

        # If we have enough published_at dates, sort by them
        # Otherwise, keep the search result order (already sorted by upload_date by YouTube)
        if videos_with_date >= len(videos_to_fetch) // 2:
            videos_to_fetch.sort(
                key=lambda x: x.get('published_at') or datetime.min,
                reverse=True
            )
        else:
            logger.info("Keeping videos in search result order (sorted by upload_date)")

        return videos_to_fetch

    def _is_tech_related(self, title: str) -> bool:
        """Check if video title is tech-related"""
        keywords = [
            'programming', 'coding', 'developer', 'software', 'tech', 'technology',
            'ai', 'machine learning', 'python', 'javascript', 'react', 'node',
            'tutorial', 'code', 'web', 'app', 'startup', 'computer', 'linux',
            'mac', 'windows', 'iphone', 'android', 'gadget', 'review', 'unboxing',
            'api', 'database', 'cloud', 'aws', 'google', 'microsoft', 'apple',
        ]
        return any(kw in title for kw in keywords)

    def _is_gaming_related(self, title: str) -> bool:
        """Check if video title is gaming-related"""
        keywords = [
            'game', 'gaming', 'gameplay', 'playthrough', 'walkthrough', 'stream',
            'esports', 'fps', 'rpg', 'minecraft', 'fortnite', 'valorant', 'league',
            'gta', 'call of duty', 'cod', 'pokemon', 'nintendo', 'xbox', 'playstation',
            'ps5', 'steam', 'twitch', 'speedrun', 'let\'s play', 'boss fight',
        ]
        return any(kw in title for kw in keywords)

    def _is_lifestyle_related(self, title: str) -> bool:
        """Check if video title is lifestyle-related"""
        keywords = [
            'vlog', 'day in', 'routine', 'haul', 'travel', 'food', 'cooking',
            'recipe', 'fitness', 'workout', 'fashion', 'beauty', 'makeup',
            'skincare', 'home', 'decor', 'diy', 'challenge', 'storytime',
            'q&a', 'get ready', 'grwm', 'what i eat', 'apartment', 'tour',
        ]
        return any(kw in title for kw in keywords)

    async def refresh_now(self) -> Dict[str, List[Dict[str, Any]]]:
        """Manually trigger a refresh of trending videos"""
        return await self.fetch_trending_videos()


# Global instance
trending_scheduler = TrendingScheduler()

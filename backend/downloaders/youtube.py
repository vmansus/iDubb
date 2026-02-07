"""
YouTube Video Downloader using yt-dlp
Enhanced with proper quality selection, thumbnail download, and progress tracking
"""
import asyncio
import os
import re
import aiohttp
from pathlib import Path
from typing import Optional, List, Callable
from datetime import datetime
import yt_dlp
from loguru import logger

from .base import BaseDownloader, VideoInfo, DownloadResult

# Ensure deno and rustypipe-botguard are in PATH for yt-dlp
deno_path = os.path.expanduser("~/.deno/bin")
local_bin_path = os.path.expanduser("~/.local/bin")
current_path = os.environ.get("PATH", "")

# Add paths if not already present
paths_to_add = []
if deno_path not in current_path:
    paths_to_add.append(deno_path)
if local_bin_path not in current_path:
    paths_to_add.append(local_bin_path)

if paths_to_add:
    os.environ["PATH"] = f"{':'.join(paths_to_add)}:{current_path}"


class YouTubeDownloader(BaseDownloader):
    """YouTube video downloader with proper quality selection"""

    YOUTUBE_PATTERNS = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/shorts/[\w-]+',
        r'(https?://)?youtu\.be/[\w-]+',
    ]

    # Quality presets with format selectors
    # Uses flexible format selection that falls back gracefully
    # bestvideo/bestaudio ensures highest available quality within constraints
    QUALITY_FORMATS = {
        "2160p": {
            "height": 2160,
            # Try 4K first, then fall back to best available
            "format": "bestvideo[height<=2160]+bestaudio/bestvideo+bestaudio/best",
        },
        "1440p": {
            "height": 1440,
            "format": "bestvideo[height<=1440]+bestaudio/bestvideo+bestaudio/best",
        },
        "1080p": {
            "height": 1080,
            "format": "bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio/best",
        },
        "720p": {
            "height": 720,
            "format": "bestvideo[height<=720]+bestaudio/bestvideo+bestaudio/best",
        },
        "480p": {
            "height": 480,
            "format": "bestvideo[height<=480]+bestaudio/bestvideo+bestaudio/best",
        },
        "360p": {
            "height": 360,
            "format": "bestvideo[height<=360]+bestaudio/bestvideo+bestaudio/best",
        },
    }

    def __init__(self, output_dir: Path, cookies_file: Optional[Path] = None, proxy_url: Optional[str] = None):
        super().__init__(output_dir)
        self.cookies_file = cookies_file
        self.proxy_url = proxy_url

        self.ydl_opts_base = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            # Disable cache to avoid stale player JS variants (e.g. main vs tv)
            'cachedir': False,
            # Retry settings for network resilience (SSL errors, connection drops)
            'retries': 10,
            'fragment_retries': 10,
            'file_access_retries': 3,
            'socket_timeout': 30,
        }

        # Check for cookies file
        # macOS Keychain blocks direct browser cookie access, so use exported cookie file
        # Install "Get cookies.txt LOCALLY" Chrome extension and export YouTube cookies
        from config import settings
        local_cookies = settings.DATA_DIR / "youtube_cookies.txt"
        if cookies_file and cookies_file.exists():
            self.ydl_opts_base['cookiefile'] = str(cookies_file)
            logger.info(f"Using YouTube cookies from: {cookies_file}")
        elif local_cookies.exists():
            self.ydl_opts_base['cookiefile'] = str(local_cookies)
            logger.info(f"Using YouTube cookies from: {local_cookies}")
        else:
            logger.warning("No YouTube cookies file found. Some videos may not be accessible.")
            logger.warning(f"Please export cookies to: {local_cookies}")

        # Add proxy if configured
        if proxy_url:
            self.ydl_opts_base['proxy'] = proxy_url
            logger.info(f"Using proxy for YouTube: {proxy_url}")
        else:
            # Check environment variable
            env_proxy = os.environ.get('PROXY_URL') or os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
            if env_proxy:
                self.ydl_opts_base['proxy'] = env_proxy
                logger.info(f"Using proxy from environment: {env_proxy}")

        # Add common headers to avoid bot detection
        self.ydl_opts_base['http_headers'] = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
        }

        # Additional options to bypass restrictions
        # Note: android/ios clients don't support cookies
        # mweb and web clients support cookies but may have SABR issues
        # tv client avoids n-challenge issues with player 4e51e895 (yt-dlp#15814)
        self.ydl_opts_base['extractor_args'] = {
            'youtube': {
                'player_client': ['tv'],
            }
        }

        # Enable remote components for n-challenge solver (REQUIRED for YouTube downloads in 2025+)
        # This downloads the JS solver from GitHub to handle YouTube's signature encryption
        self.ydl_opts_base['remote_components'] = ['ejs:github']

        # Note: For 4K downloads, rustypipe-botguard must be installed in ~/.local/bin
        # Install with: curl -L "https://codeberg.org/ThetaDev/rustypipe-botguard/releases/download/v0.1.2/rustypipe-botguard-v0.1.2-aarch64-apple-darwin.tar.xz" | xz -d | tar -xf - -C ~/.local/bin

    def supports_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL"""
        return any(re.match(pattern, url) for pattern in self.YOUTUBE_PATTERNS)

    async def get_video_info(self, url: str) -> Optional[VideoInfo]:
        """Get video metadata including all available formats.
        Uses yt-dlp CLI via subprocess to ensure extractor_args (like player_client=tv)
        are properly applied - the Python API has a bug where player variants don't work."""
        import subprocess
        import json
        try:
            # Build CLI command - use sys.executable's directory to find venv yt-dlp
            import sys
            venv_bin = str(Path(sys.executable).parent / 'yt-dlp')
            ytdlp_cmd = venv_bin if Path(venv_bin).exists() else 'yt-dlp'
            cmd = [
                ytdlp_cmd, '--dump-json', '--no-download', '--no-warnings',
                '--no-check-certificates', '--no-cache-dir',
                '--extractor-args', 'youtube:player_client=tv',
            ]

            # Use cookies-from-browser for freshest cookies (file-based cookies
            # get rotated by YouTube when browser tabs are open)
            # Fall back to cookie file if available
            cmd.extend(['--cookies-from-browser', 'chrome'])

            # Add proxy if available
            if self.ydl_opts_base.get('proxy'):
                cmd.extend(['--proxy', self.ydl_opts_base['proxy']])

            cmd.append(url)

            logger.info(f"get_video_info CLI: {' '.join(cmd)}")

            def extract():
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
                if result.returncode != 0:
                    logger.error(f"yt-dlp CLI failed: {result.stderr[:500]}")
                    return None
                return json.loads(result.stdout)

            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, extract)

            if not info:
                return None

            upload_date = None
            if info.get('upload_date'):
                upload_date = datetime.strptime(info['upload_date'], '%Y%m%d')

            # Parse available formats to determine best quality
            formats = info.get('formats', [])
            available_qualities = self._parse_available_qualities(formats)

            # Get best thumbnail
            thumbnails = info.get('thumbnails', [])
            best_thumbnail = self._get_best_thumbnail(thumbnails, info.get('thumbnail', ''))

            return VideoInfo(
                video_id=info.get('id', ''),
                title=info.get('title', ''),
                description=info.get('description', ''),
                duration=info.get('duration', 0),
                thumbnail_url=best_thumbnail,
                uploader=info.get('uploader', ''),
                upload_date=upload_date,
                view_count=info.get('view_count', 0),
                like_count=info.get('like_count', 0),
                tags=info.get('tags', []),
                platform='youtube',
                original_url=url,
                extra={
                    'available_qualities': available_qualities,
                    'best_quality': available_qualities[0] if available_qualities else '720p',
                    'thumbnails': thumbnails,
                }
            )
        except Exception as e:
            logger.error(f"Failed to get YouTube video info: {e}")
            return None

    def _parse_available_qualities(self, formats: List[dict]) -> List[str]:
        """Parse available video qualities from format list"""
        heights = set()
        for fmt in formats:
            height = fmt.get('height')
            vcodec = fmt.get('vcodec', 'none')
            if height and vcodec != 'none':
                heights.add(height)

        # Map heights to quality labels
        quality_labels = []
        height_to_label = {
            2160: "2160p",
            1440: "1440p",
            1080: "1080p",
            720: "720p",
            480: "480p",
            360: "360p",
            240: "240p",
            144: "144p",
        }

        for height in sorted(heights, reverse=True):
            for h, label in height_to_label.items():
                if height >= h:
                    if label not in quality_labels:
                        quality_labels.append(label)
                    break

        return quality_labels if quality_labels else ["720p"]

    def _get_best_thumbnail(self, thumbnails: List[dict], fallback: str) -> str:
        """Get highest resolution thumbnail URL"""
        if not thumbnails:
            return fallback

        # Sort by resolution (prefer higher)
        sorted_thumbs = sorted(
            thumbnails,
            key=lambda t: (t.get('width', 0) or 0) * (t.get('height', 0) or 0),
            reverse=True
        )

        # Prefer maxresdefault or hqdefault
        for thumb in sorted_thumbs:
            url = thumb.get('url', '')
            if 'maxresdefault' in url or 'hqdefault' in url:
                return url

        return sorted_thumbs[0].get('url', fallback) if sorted_thumbs else fallback

    async def download_thumbnail(self, url: str, output_path: Path) -> bool:
        """Download video thumbnail to file with retry and fallback logic"""
        # Common headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.youtube.com/',
        }

        # Try multiple thumbnail quality URLs
        urls_to_try = [url]

        # Extract video ID and add fallback URLs
        if 'youtube.com' in url or 'ytimg.com' in url or 'ggpht.com' in url:
            # Try to extract video ID from URL (handle multiple formats)
            # Formats: /vi/VIDEO_ID/, /vi_webp/VIDEO_ID/, or v=VIDEO_ID
            video_id_match = re.search(r'/vi(?:_webp)?/([a-zA-Z0-9_-]{11})(?:[/\?]|$)', url)
            if not video_id_match:
                # Try alternate format like v=VIDEO_ID
                video_id_match = re.search(r'[?&]v=([a-zA-Z0-9_-]{11})', url)
            if video_id_match:
                video_id = video_id_match.group(1)
                # Add fallback thumbnail URLs in order of quality
                urls_to_try = [
                    f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg',
                    f'https://img.youtube.com/vi/{video_id}/sddefault.jpg',
                    f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg',
                    f'https://img.youtube.com/vi/{video_id}/mqdefault.jpg',
                    f'https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg',
                    f'https://i.ytimg.com/vi/{video_id}/hqdefault.jpg',
                    url,  # Original URL as last resort
                ]

        # Create SSL context that doesn't verify certificates (for macOS compatibility)
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
            for try_url in urls_to_try:
                try:
                    async with session.get(try_url) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            # Verify it's actually an image (not a placeholder)
                            if len(content) > 1000:  # Real thumbnails are larger than 1KB
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(output_path, 'wb') as f:
                                    f.write(content)
                                logger.info(f"Downloaded thumbnail ({len(content)} bytes) to: {output_path}")
                                return True
                            else:
                                logger.debug(f"Thumbnail too small ({len(content)} bytes), trying next URL")
                        else:
                            logger.debug(f"Thumbnail URL returned {resp.status}: {try_url}")
                except Exception as e:
                    logger.debug(f"Failed to download from {try_url}: {e}")
                    continue

        logger.error(f"Failed to download thumbnail after trying {len(urls_to_try)} URLs")
        return False

    async def download(
        self,
        url: str,
        quality: str = "1080p",
        format_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        subtitle_language: Optional[str] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> DownloadResult:
        """
        Download YouTube video with specified quality

        Args:
            url: YouTube video URL
            quality: Quality preset (2160p, 1440p, 1080p, 720p, 480p, 360p)
            format_id: Specific format ID (overrides quality)
            progress_callback: Callback function(percent, message) for progress updates
            cancel_check: Optional callable that returns True if cancellation requested

        Returns:
            DownloadResult with video and audio paths
        """
        try:
            video_info = await self.get_video_info(url)
            if not video_info:
                return DownloadResult(
                    success=False,
                    video_path=None,
                    audio_path=None,
                    video_info=None,
                    error="Failed to get video info"
                )

            # Create output filename
            safe_title = re.sub(r'[^\w\s-]', '', video_info.title)[:50]
            video_filename = f"{video_info.video_id}_{safe_title}"
            video_path = self.output_dir / f"{video_filename}.mp4"
            audio_path = self.output_dir / f"{video_filename}.mp3"
            thumbnail_path = self.output_dir / f"{video_filename}_thumb.jpg"

            # Download thumbnail first
            if video_info.thumbnail_url:
                await self.download_thumbnail(video_info.thumbnail_url, thumbnail_path)

            # Determine format string
            if format_id:
                # Use specific format ID
                format_str = f"{format_id}+bestaudio/best"
                logger.info(f"Using specific format ID: {format_id}")
            else:
                # Use quality preset
                quality_config = self.QUALITY_FORMATS.get(quality, self.QUALITY_FORMATS["1080p"])
                format_str = quality_config["format"]
                logger.info(f"Using quality preset {quality}: target height={quality_config['height']}")
                logger.info(f"Format selector: {format_str}")

            # Custom exception for cancellation
            class DownloadCancelled(Exception):
                pass

            # Progress hook with cancellation support
            def progress_hook(d):
                # Check for cancellation on every progress update
                if cancel_check and cancel_check():
                    logger.info("Download cancelled by user")
                    raise DownloadCancelled("用户手动停止")
                
                if d['status'] == 'downloading':
                    if progress_callback:
                        percent = 0
                        if d.get('total_bytes'):
                            percent = int(d.get('downloaded_bytes', 0) / d['total_bytes'] * 100)
                        elif d.get('total_bytes_estimate'):
                            percent = int(d.get('downloaded_bytes', 0) / d['total_bytes_estimate'] * 100)
                        progress_callback(percent, f"下载中: {percent}%")
                elif d['status'] == 'finished':
                    if progress_callback:
                        progress_callback(100, "下载完成，正在处理...")

            # Download video with robust options
            # Note: Subtitle download is done separately to avoid failing the whole download
            video_opts = {
                'quiet': False,
                'no_warnings': False,
                'format': format_str,
                'outtmpl': str(video_path).replace('.mp4', '.%(ext)s'),
                'merge_output_format': 'mp4',
                'ignoreerrors': False,  # Don't ignore errors for video downloads
                'no_check_certificates': True,
                'progress_hooks': [progress_hook],
                # Force re-download even if file exists (to get correct quality)
                'overwrites': True,
                # Postprocessors to ensure mp4 output
                'postprocessors': [
                    {
                        'key': 'FFmpegVideoRemuxer',
                        'preferedformat': 'mp4',
                    },
                ],
                # Prefer free formats (VP9/AV1) for high quality
                'prefer_free_formats': True,
                # Disable subtitle download here - will be done separately
                # This prevents subtitle errors from failing the video download
                'writesubtitles': False,
                'writeautomaticsub': False,
            }

            # Add cookies if available
            if self.ydl_opts_base.get('cookiefile'):
                video_opts['cookiefile'] = self.ydl_opts_base['cookiefile']

            # Add proxy if available
            if self.ydl_opts_base.get('proxy'):
                video_opts['proxy'] = self.ydl_opts_base['proxy']

            # Add headers
            if self.ydl_opts_base.get('http_headers'):
                video_opts['http_headers'] = self.ydl_opts_base['http_headers']

            # Add extractor args
            if self.ydl_opts_base.get('extractor_args'):
                video_opts['extractor_args'] = self.ydl_opts_base['extractor_args']

            # Add remote components for n-challenge solver
            if self.ydl_opts_base.get('remote_components'):
                video_opts['remote_components'] = self.ydl_opts_base['remote_components']

            def download_video():
                with yt_dlp.YoutubeDL(video_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    # Return actual format info
                    return info

            loop = asyncio.get_event_loop()
            try:
                download_info = await loop.run_in_executor(None, download_video)
            except DownloadCancelled:
                return DownloadResult(
                    success=False,
                    video_path=None,
                    audio_path=None,
                    video_info=video_info,
                    error="用户手动停止"
                )
            except Exception as e:
                if "用户手动停止" in str(e):
                    return DownloadResult(
                        success=False,
                        video_path=None,
                        audio_path=None,
                        video_info=video_info,
                        error="用户手动停止"
                    )
                raise

            # Log actual downloaded format
            if download_info:
                actual_format = download_info.get('format', 'unknown')
                actual_height = download_info.get('height', 'unknown')
                actual_width = download_info.get('width', 'unknown')
                logger.info(f"Actually downloaded: {actual_width}x{actual_height}, format: {actual_format}")

            # Download audio separately for transcription
            audio_opts = {
                'quiet': False,
                'no_warnings': False,
                'format': 'bestaudio/best',
                'outtmpl': str(audio_path).replace('.mp3', '.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'keepvideo': True,
                'ignoreerrors': True,
                'no_check_certificates': True,
                'progress_hooks': [progress_hook],  # Add cancellation support
            }

            # Add cookies if available
            if self.ydl_opts_base.get('cookiefile'):
                audio_opts['cookiefile'] = self.ydl_opts_base['cookiefile']

            # Add proxy if available
            if self.ydl_opts_base.get('proxy'):
                audio_opts['proxy'] = self.ydl_opts_base['proxy']

            # Add headers
            if self.ydl_opts_base.get('http_headers'):
                audio_opts['http_headers'] = self.ydl_opts_base['http_headers']

            # Add extractor args
            if self.ydl_opts_base.get('extractor_args'):
                audio_opts['extractor_args'] = self.ydl_opts_base['extractor_args']

            # Add remote components
            if self.ydl_opts_base.get('remote_components'):
                audio_opts['remote_components'] = self.ydl_opts_base['remote_components']

            def download_audio():
                with yt_dlp.YoutubeDL(audio_opts) as ydl:
                    ydl.download([url])

            try:
                await loop.run_in_executor(None, download_audio)
            except DownloadCancelled:
                return DownloadResult(
                    success=False,
                    video_path=None,
                    audio_path=None,
                    video_info=video_info,
                    error="用户手动停止"
                )
            except Exception as e:
                if "用户手动停止" in str(e):
                    return DownloadResult(
                        success=False,
                        video_path=None,
                        audio_path=None,
                        video_info=video_info,
                        error="用户手动停止"
                    )
                # Audio download failure is not critical, continue
                logger.warning(f"Audio download failed: {e}")

            # Try to download subtitles separately (allow failure)
            # This is separate from video download to prevent subtitle errors from failing video
            try:
                # Determine subtitle languages to download
                # If user specified a language, prioritize that
                # Otherwise fall back to source language and common translations
                if subtitle_language:
                    # User selected a specific subtitle language (e.g., "zh" for Chinese auto-translated)
                    subtitle_langs = [subtitle_language]
                    logger.info(f"Downloading user-selected subtitle language: {subtitle_language}")
                else:
                    # Default: download English and Chinese subtitles
                    subtitle_langs = ['en', 'zh']
                    logger.info(f"Downloading default subtitle languages: {subtitle_langs}")

                subtitle_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'skip_download': True,  # Don't re-download video
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': subtitle_langs,
                    'subtitlesformat': 'srt/vtt/best',
                    'outtmpl': str(video_path).replace('.mp4', '.%(ext)s'),
                    'ignoreerrors': True,  # Don't fail on subtitle errors
                    'no_check_certificates': True,
                }
                # Add cookies if available
                if self.ydl_opts_base.get('cookiefile'):
                    subtitle_opts['cookiefile'] = self.ydl_opts_base['cookiefile']
                if self.ydl_opts_base.get('http_headers'):
                    subtitle_opts['http_headers'] = self.ydl_opts_base['http_headers']
                if self.ydl_opts_base.get('extractor_args'):
                    subtitle_opts['extractor_args'] = self.ydl_opts_base['extractor_args']

                def download_subtitles():
                    with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                        ydl.download([url])

                await loop.run_in_executor(None, download_subtitles)
                logger.info("Subtitle download attempted")
            except Exception as sub_error:
                logger.warning(f"Subtitle download failed (non-fatal): {sub_error}")

            # Find actual downloaded files (extension might differ)
            video_files = list(self.output_dir.glob(f"{video_filename}*.mp4"))
            audio_files = list(self.output_dir.glob(f"{video_filename}*.mp3"))

            actual_video_path = video_files[0] if video_files else None
            actual_audio_path = audio_files[0] if audio_files else None

            # Find downloaded subtitle files
            # yt-dlp creates files like: filename.en.srt or filename.en-US.srt
            subtitle_pattern_srt = f"{video_filename}*.srt"
            subtitle_pattern_vtt = f"{video_filename}*.vtt"
            logger.info(f"Looking for subtitles with pattern: {subtitle_pattern_srt}")
            logger.info(f"All files in output_dir: {list(self.output_dir.glob('*'))}")
            subtitle_files = list(self.output_dir.glob(subtitle_pattern_srt)) + \
                           list(self.output_dir.glob(subtitle_pattern_vtt))
            logger.info(f"Found subtitle files: {subtitle_files}")
            actual_subtitle_path = None
            subtitle_lang = None
            if subtitle_files:
                # If user specified a subtitle language, prefer that
                # Otherwise fall back to English, then any available
                preferred_lang = subtitle_language or 'en'

                # First try to find the preferred language
                for sub_file in subtitle_files:
                    # Check for language code in filename (e.g., .zh. or .zh-Hans.)
                    if f'.{preferred_lang}.' in sub_file.name or f'.{preferred_lang}-' in sub_file.name:
                        actual_subtitle_path = sub_file
                        subtitle_lang = preferred_lang
                        logger.info(f"Found preferred subtitle language {preferred_lang}: {sub_file}")
                        break

                # Fall back to any available subtitle if preferred not found
                if not actual_subtitle_path:
                    actual_subtitle_path = subtitle_files[0]
                    # Try to extract language from filename
                    for lang in ['zh', 'zh-Hans', 'zh-Hant', 'en', 'ja', 'ko', 'es', 'fr', 'de']:
                        if f'.{lang}.' in actual_subtitle_path.name or f'.{lang}-' in actual_subtitle_path.name:
                            subtitle_lang = lang
                            break
                    logger.info(f"Using fallback subtitle: {actual_subtitle_path} (lang: {subtitle_lang})")
                else:
                    logger.info(f"Found subtitle file: {actual_subtitle_path} (lang: {subtitle_lang})")

            # Update video_info with thumbnail path if downloaded
            if thumbnail_path.exists():
                video_info.extra = video_info.extra or {}
                video_info.extra['thumbnail_local'] = str(thumbnail_path)

            logger.info(f"Downloaded YouTube video: {video_info.title}")

            return DownloadResult(
                success=True,
                video_path=actual_video_path,
                audio_path=actual_audio_path,
                video_info=video_info,
                subtitle_path=actual_subtitle_path,
                subtitle_language=subtitle_lang,
                extra={
                    'thumbnail_path': str(thumbnail_path) if thumbnail_path.exists() else None,
                    'requested_quality': quality,
                    'actual_height': download_info.get('height') if download_info else None,
                }
            )

        except Exception as e:
            logger.error(f"Failed to download YouTube video: {e}")
            return DownloadResult(
                success=False,
                video_path=None,
                audio_path=None,
                video_info=None,
                error=str(e)
            )

    async def get_trending(self, count: int = 10, region: str = "US") -> List[VideoInfo]:
        """Get trending YouTube videos"""
        try:
            trending_url = f"https://www.youtube.com/feed/trending"

            opts = {
                **self.ydl_opts_base,
                'extract_flat': True,
                'playlistend': count,
            }

            def extract_trending():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(trending_url, download=False)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, extract_trending)

            videos = []
            if result and 'entries' in result:
                for entry in result['entries'][:count]:
                    if entry:
                        video_url = f"https://www.youtube.com/watch?v={entry.get('id', '')}"
                        video_info = await self.get_video_info(video_url)
                        if video_info:
                            videos.append(video_info)

            return videos

        except Exception as e:
            logger.error(f"Failed to get YouTube trending: {e}")
            return []

    async def search(self, query: str, count: int = 10) -> List[VideoInfo]:
        """Search YouTube videos"""
        try:
            search_url = f"ytsearch{count}:{query}"

            opts = {
                **self.ydl_opts_base,
                'extract_flat': True,
            }

            def search_videos():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(search_url, download=False)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, search_videos)

            videos = []
            if result and 'entries' in result:
                for entry in result['entries']:
                    if entry and entry.get('id'):
                        videos.append(VideoInfo(
                            video_id=entry.get('id', ''),
                            title=entry.get('title', ''),
                            description='',
                            duration=entry.get('duration', 0),
                            thumbnail_url=entry.get('thumbnail', ''),
                            uploader=entry.get('uploader', ''),
                            upload_date=None,
                            view_count=entry.get('view_count', 0),
                            like_count=0,
                            tags=[],
                            platform='youtube',
                            original_url=f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                        ))

            return videos

        except Exception as e:
            logger.error(f"Failed to search YouTube: {e}")
            return []

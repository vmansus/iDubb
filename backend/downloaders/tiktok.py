"""
TikTok Video Downloader using yt-dlp
"""
import asyncio
import re
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import yt_dlp
from loguru import logger

from .base import BaseDownloader, VideoInfo, DownloadResult


class TikTokDownloader(BaseDownloader):
    """TikTok video downloader"""

    TIKTOK_PATTERNS = [
        r'(https?://)?(www\.)?tiktok\.com/@[\w.-]+/video/\d+',
        r'(https?://)?(vm\.)?tiktok\.com/[\w]+',
        r'(https?://)?vt\.tiktok\.com/[\w]+',
    ]

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.ydl_opts_base = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            # TikTok specific options
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }

    def supports_url(self, url: str) -> bool:
        """Check if URL is a TikTok URL"""
        return any(re.match(pattern, url) for pattern in self.TIKTOK_PATTERNS)

    async def get_video_info(self, url: str) -> Optional[VideoInfo]:
        """Get TikTok video metadata"""
        try:
            opts = {**self.ydl_opts_base, 'skip_download': True}

            def extract():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(url, download=False)

            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, extract)

            if not info:
                return None

            upload_date = None
            if info.get('timestamp'):
                upload_date = datetime.fromtimestamp(info['timestamp'])

            return VideoInfo(
                video_id=info.get('id', ''),
                title=info.get('title', '') or info.get('description', '')[:100],
                description=info.get('description', ''),
                duration=info.get('duration', 0),
                thumbnail_url=info.get('thumbnail', ''),
                uploader=info.get('uploader', '') or info.get('creator', ''),
                upload_date=upload_date,
                view_count=info.get('view_count', 0),
                like_count=info.get('like_count', 0),
                tags=info.get('tags', []),
                platform='tiktok',
                original_url=url,
            )
        except Exception as e:
            logger.error(f"Failed to get TikTok video info: {e}")
            return None

    async def download(self, url: str, quality: str = "1080p", **kwargs) -> DownloadResult:
        """Download TikTok video"""
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

            # Create output filename (no spaces, sanitized)
            safe_title = re.sub(r'[^\w-]', '', video_info.title.replace(' ', '_'))[:50]
            video_filename = f"tiktok_{video_info.video_id}_{safe_title}"
            video_path = self.output_dir / f"{video_filename}.mp4"
            audio_path = self.output_dir / f"{video_filename}.mp3"

            # TikTok usually has single format, download best available
            # Use .%(ext)s to let yt-dlp determine the extension
            video_outtmpl = str(self.output_dir / f"{video_filename}.%(ext)s")
            video_opts = {
                'quiet': False,  # Show output for debugging
                'no_warnings': False,
                'extract_flat': False,
                'http_headers': self.ydl_opts_base.get('http_headers', {}),
                # Avoid HEVC/bytevc1 which may have issues, prefer h264
                'format': 'best[vcodec^=avc]/best[ext=mp4]/best',
                'outtmpl': video_outtmpl,
                'merge_output_format': 'mp4',
                # Force remux to mp4 container
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
            }

            download_error = None
            def download_video():
                nonlocal download_error
                try:
                    with yt_dlp.YoutubeDL(video_opts) as ydl:
                        result = ydl.download([url])
                        logger.debug(f"yt-dlp download result: {result}")
                except Exception as e:
                    download_error = str(e)
                    logger.error(f"yt-dlp video download error: {e}")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, download_video)
            
            # Small delay to ensure file is written
            await asyncio.sleep(0.5)
            
            # List what files exist right after download
            post_download_files = list(self.output_dir.glob(f"{video_filename}*"))
            logger.debug(f"Files matching {video_filename}* right after download: {[f.name for f in post_download_files]}")
            
            if download_error:
                return DownloadResult(
                    success=False,
                    video_path=None,
                    audio_path=None,
                    video_info=video_info,
                    error=f"视频下载失败: {download_error}"
                )

            # Find the downloaded video file first
            post_download_video = list(self.output_dir.glob(f"{video_filename}*.mp4"))
            if not post_download_video:
                post_download_video = list(self.output_dir.glob(f"{video_filename}*.webm"))
            if not post_download_video:
                post_download_video = list(self.output_dir.glob(f"{video_filename}*.mkv"))
            
            downloaded_video_path = post_download_video[0] if post_download_video else None
            
            # Extract audio from downloaded video using ffmpeg directly (don't re-download)
            if downloaded_video_path and downloaded_video_path.exists():
                import subprocess
                audio_output = self.output_dir / f"{video_filename}.mp3"
                try:
                    subprocess.run([
                        'ffmpeg', '-i', str(downloaded_video_path),
                        '-vn', '-acodec', 'libmp3lame', '-q:a', '2',
                        '-y', str(audio_output)
                    ], capture_output=True, check=True)
                    logger.debug(f"Extracted audio to: {audio_output}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to extract audio: {e}")
            else:
                logger.warning(f"Cannot extract audio - video file not found: {downloaded_video_path}")

            # Find actual downloaded files (check multiple video formats)
            logger.debug(f"Output dir: {self.output_dir}")
            logger.debug(f"Video outtmpl was: {video_outtmpl}")
            logger.debug(f"Looking for video files matching: {video_filename}*")
            
            # Also check if the exact file exists
            expected_video_path = self.output_dir / f"{video_filename}.mp4"
            logger.debug(f"Expected video path: {expected_video_path}, exists: {expected_video_path.exists()}")
            video_files = []
            for ext in ['mp4', 'webm', 'mkv', 'mov']:
                found = list(self.output_dir.glob(f"{video_filename}*.{ext}"))
                logger.debug(f"  Found {len(found)} .{ext} files")
                video_files.extend(found)
            audio_files = list(self.output_dir.glob(f"{video_filename}*.mp3"))
            
            # Also list all files in directory for debugging
            all_files = list(self.output_dir.glob("tiktok_*"))
            logger.debug(f"All tiktok files in output dir: {[f.name for f in all_files[:10]]}")
            
            # List ALL files including mp4
            all_mp4 = list(self.output_dir.glob("*.mp4"))
            logger.debug(f"All mp4 files in output dir: {[f.name for f in all_mp4[:10]]}")

            actual_video_path = video_files[0] if video_files else None
            actual_audio_path = audio_files[0] if audio_files else None

            if not actual_video_path:
                logger.error(f"TikTok video file not found after download: {video_filename}")
                return DownloadResult(
                    success=False,
                    video_path=None,
                    audio_path=None,
                    video_info=video_info,
                    error="视频下载失败，可能是地区限制或视频已删除"
                )

            logger.info(f"Downloaded TikTok video: {video_info.title} -> {actual_video_path}")

            return DownloadResult(
                success=True,
                video_path=actual_video_path,
                audio_path=actual_audio_path,
                video_info=video_info,
            )

        except Exception as e:
            logger.error(f"Failed to download TikTok video: {e}")
            return DownloadResult(
                success=False,
                video_path=None,
                audio_path=None,
                video_info=None,
                error=str(e)
            )

    async def get_trending(self, count: int = 10) -> List[VideoInfo]:
        """
        Get trending TikTok videos
        Note: TikTok doesn't have a public trending API,
        this would require web scraping or unofficial APIs
        """
        logger.warning("TikTok trending requires authentication or scraping")
        return []

    async def get_user_videos(self, username: str, count: int = 10) -> List[VideoInfo]:
        """Get videos from a TikTok user"""
        try:
            user_url = f"https://www.tiktok.com/@{username}"

            opts = {
                **self.ydl_opts_base,
                'extract_flat': True,
                'playlistend': count,
            }

            def extract_user_videos():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(user_url, download=False)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, extract_user_videos)

            videos = []
            if result and 'entries' in result:
                for entry in result['entries'][:count]:
                    if entry:
                        video_info = await self.get_video_info(entry.get('url', ''))
                        if video_info:
                            videos.append(video_info)

            return videos

        except Exception as e:
            logger.error(f"Failed to get TikTok user videos: {e}")
            return []

"""
Video Source Information Parser
解析视频源的详细信息：可用格式、清晰度、音轨、字幕等
"""
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import yt_dlp
from loguru import logger


@dataclass
class VideoFormat:
    """Video format option"""
    format_id: str
    ext: str
    resolution: str  # e.g., "1920x1080", "1280x720"
    width: int
    height: int
    fps: Optional[float] = None
    vcodec: str = ""
    acodec: str = ""
    filesize: Optional[int] = None  # bytes
    tbr: Optional[float] = None  # total bitrate
    quality_label: str = ""  # e.g., "1080p", "720p"
    has_audio: bool = False
    has_video: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_id": self.format_id,
            "ext": self.ext,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "vcodec": self.vcodec,
            "acodec": self.acodec,
            "filesize": self.filesize,
            "filesize_mb": round(self.filesize / 1024 / 1024, 2) if self.filesize else None,
            "tbr": self.tbr,
            "quality_label": self.quality_label,
            "has_audio": self.has_audio,
            "has_video": self.has_video,
        }


@dataclass
class AudioTrack:
    """Audio track option"""
    format_id: str
    ext: str
    acodec: str
    abr: Optional[float] = None  # audio bitrate
    asr: Optional[int] = None  # audio sample rate
    language: Optional[str] = None
    language_name: Optional[str] = None
    filesize: Optional[int] = None
    is_original: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_id": self.format_id,
            "ext": self.ext,
            "acodec": self.acodec,
            "abr": self.abr,
            "asr": self.asr,
            "language": self.language,
            "language_name": self.language_name,
            "filesize": self.filesize,
            "filesize_mb": round(self.filesize / 1024 / 1024, 2) if self.filesize else None,
            "is_original": self.is_original,
        }


@dataclass
class SubtitleTrack:
    """Subtitle track option"""
    language: str
    language_name: str
    ext: str = "vtt"
    url: Optional[str] = None
    is_auto_generated: bool = False
    is_translatable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "language_name": self.language_name,
            "ext": self.ext,
            "is_auto_generated": self.is_auto_generated,
            "is_translatable": self.is_translatable,
        }


@dataclass
class DetailedVideoInfo:
    """Detailed video source information"""
    video_id: str
    title: str
    description: str
    duration: int  # seconds
    thumbnail_url: str
    uploader: str
    platform: str
    original_url: str

    # Available options
    formats: List[VideoFormat] = field(default_factory=list)
    audio_tracks: List[AudioTrack] = field(default_factory=list)
    subtitles: List[SubtitleTrack] = field(default_factory=list)

    # Recommended options
    recommended_format: Optional[str] = None
    best_quality: Optional[str] = None

    # Additional metadata
    view_count: int = 0
    like_count: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "description": self.description,
            "duration": self.duration,
            "thumbnail_url": self.thumbnail_url,
            "uploader": self.uploader,
            "platform": self.platform,
            "original_url": self.original_url,
            "formats": [f.to_dict() for f in self.formats],
            "audio_tracks": [a.to_dict() for a in self.audio_tracks],
            "subtitles": [s.to_dict() for s in self.subtitles],
            "recommended_format": self.recommended_format,
            "best_quality": self.best_quality,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "tags": self.tags,
        }


class VideoInfoParser:
    """Parse detailed video source information"""

    # Quality labels by height
    QUALITY_LABELS = {
        2160: "4K",
        1440: "2K",
        1080: "1080p",
        720: "720p",
        480: "480p",
        360: "360p",
        240: "240p",
        144: "144p",
    }

    def __init__(self, cookies_file: Optional[Path] = None):
        self.cookies_file = cookies_file
        self.ydl_opts_base = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'ignoreerrors': True,
            'no_check_certificates': True,
        }

        # Add cookies if available
        if cookies_file and cookies_file.exists():
            self.ydl_opts_base['cookiefile'] = str(cookies_file)
            logger.info(f"Using provided cookies file: {cookies_file}")
        else:
            # Try multiple possible locations for cookies file
            possible_paths = [
                Path('/app/data/youtube_cookies.txt'),  # Docker
                Path(__file__).parent.parent.parent / 'data' / 'youtube_cookies.txt',  # Local dev (project root)
                Path.home() / '.idubb' / 'youtube_cookies.txt',  # User home
            ]
            for cookies_path in possible_paths:
                if cookies_path.exists():
                    self.ydl_opts_base['cookiefile'] = str(cookies_path)
                    logger.info(f"Found cookies file at: {cookies_path}")
                    break
            else:
                logger.warning("No YouTube cookies file found. Some videos may not be accessible.")

        # Add headers
        self.ydl_opts_base['http_headers'] = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        }

        # Add extractor args
        # YouTube is rolling out SABR streaming which breaks HLS/web_safari downloads
        # Note: android/ios clients don't support cookies, causing download failures
        # Use mweb and web clients which support cookies
        # See: https://github.com/yt-dlp/yt-dlp/issues/12482
        self.ydl_opts_base['extractor_args'] = {
            'youtube': {
                'player_client': ['mweb', 'web'],
            }
        }

    def _get_quality_label(self, height: int) -> str:
        """Get quality label from height"""
        for h, label in sorted(self.QUALITY_LABELS.items(), reverse=True):
            if height >= h:
                return label
        return f"{height}p"

    def _parse_formats(self, formats_data: List[Dict]) -> tuple:
        """Parse format list into video formats and audio tracks"""
        video_formats = []
        audio_tracks = []
        seen_resolutions = set()
        seen_audio = set()

        for fmt in formats_data or []:
            format_id = fmt.get('format_id', '')
            ext = fmt.get('ext', '')
            vcodec = fmt.get('vcodec', 'none')
            acodec = fmt.get('acodec', 'none')
            width = fmt.get('width') or 0
            height = fmt.get('height') or 0

            has_video = vcodec != 'none' and vcodec is not None
            has_audio = acodec != 'none' and acodec is not None

            # Video format
            if has_video and height > 0:
                quality_label = self._get_quality_label(height)
                resolution_key = f"{width}x{height}"

                # Avoid duplicates for same resolution
                if resolution_key not in seen_resolutions:
                    seen_resolutions.add(resolution_key)
                    video_formats.append(VideoFormat(
                        format_id=format_id,
                        ext=ext,
                        resolution=resolution_key,
                        width=width,
                        height=height,
                        fps=fmt.get('fps'),
                        vcodec=vcodec,
                        acodec=acodec if has_audio else '',
                        filesize=fmt.get('filesize') or fmt.get('filesize_approx'),
                        tbr=fmt.get('tbr'),
                        quality_label=quality_label,
                        has_audio=has_audio,
                        has_video=has_video,
                    ))

            # Audio-only format
            elif has_audio and not has_video:
                audio_key = f"{acodec}_{fmt.get('abr', 0)}"
                if audio_key not in seen_audio:
                    seen_audio.add(audio_key)
                    audio_tracks.append(AudioTrack(
                        format_id=format_id,
                        ext=ext,
                        acodec=acodec,
                        abr=fmt.get('abr'),
                        asr=fmt.get('asr'),
                        language=fmt.get('language'),
                        filesize=fmt.get('filesize') or fmt.get('filesize_approx'),
                    ))

        # Sort by quality
        video_formats.sort(key=lambda x: (x.height, x.width), reverse=True)
        audio_tracks.sort(key=lambda x: x.abr or 0, reverse=True)

        return video_formats, audio_tracks

    def _parse_subtitles(self, subtitles_data: Dict, automatic_captions: Dict) -> List[SubtitleTrack]:
        """Parse subtitle information"""
        subtitles = []

        # Manual subtitles
        for lang_code, sub_list in (subtitles_data or {}).items():
            if sub_list:
                # Get language name
                lang_name = self._get_language_name(lang_code)
                subtitles.append(SubtitleTrack(
                    language=lang_code,
                    language_name=lang_name,
                    ext=sub_list[0].get('ext', 'vtt') if sub_list else 'vtt',
                    url=sub_list[0].get('url') if sub_list else None,
                    is_auto_generated=False,
                    is_translatable=False,
                ))

        # Auto-generated captions
        for lang_code, sub_list in (automatic_captions or {}).items():
            if sub_list and not any(s.language == lang_code and not s.is_auto_generated for s in subtitles):
                lang_name = self._get_language_name(lang_code)
                subtitles.append(SubtitleTrack(
                    language=lang_code,
                    language_name=f"{lang_name} (自动生成)",
                    ext=sub_list[0].get('ext', 'vtt') if sub_list else 'vtt',
                    url=sub_list[0].get('url') if sub_list else None,
                    is_auto_generated=True,
                    is_translatable=True,
                ))

        return subtitles

    def _get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name"""
        language_names = {
            'en': 'English',
            'en-US': 'English (US)',
            'en-GB': 'English (UK)',
            'zh': '中文',
            'zh-CN': '中文 (简体)',
            'zh-TW': '中文 (繁體)',
            'zh-Hans': '中文 (简体)',
            'zh-Hant': '中文 (繁體)',
            'ja': '日本語',
            'ko': '한국어',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'ru': 'Русский',
            'pt': 'Português',
            'it': 'Italiano',
            'ar': 'العربية',
            'hi': 'हिन्दी',
            'vi': 'Tiếng Việt',
            'th': 'ไทย',
        }

        # Try exact match first
        if lang_code in language_names:
            return language_names[lang_code]

        # Try base language
        base_lang = lang_code.split('-')[0]
        if base_lang in language_names:
            return language_names[base_lang]

        return lang_code

    async def get_detailed_info(self, url: str) -> Optional[DetailedVideoInfo]:
        """
        Get detailed video information including all available formats, audio tracks, and subtitles

        Args:
            url: Video URL (YouTube, TikTok, etc.)

        Returns:
            DetailedVideoInfo with all available options
        """
        try:
            opts = {
                **self.ydl_opts_base,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['all'],
                'listsubtitles': False,
            }

            def extract():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(url, download=False)

            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, extract)

            if not info:
                return None

            # Parse formats
            video_formats, audio_tracks = self._parse_formats(info.get('formats', []))

            # Parse subtitles
            subtitles = self._parse_subtitles(
                info.get('subtitles', {}),
                info.get('automatic_captions', {})
            )

            # Determine platform
            platform = 'unknown'
            if 'youtube' in url.lower() or 'youtu.be' in url.lower():
                platform = 'youtube'
            elif 'tiktok' in url.lower():
                platform = 'tiktok'
            elif 'bilibili' in url.lower():
                platform = 'bilibili'

            # Determine recommended format
            recommended = None
            best_quality = None

            for fmt in video_formats:
                if fmt.height >= 1080:
                    if not recommended:
                        recommended = fmt.format_id
                    if not best_quality:
                        best_quality = fmt.quality_label
                    break

            if not recommended and video_formats:
                recommended = video_formats[0].format_id
                best_quality = video_formats[0].quality_label

            return DetailedVideoInfo(
                video_id=info.get('id', ''),
                title=info.get('title', ''),
                description=info.get('description', ''),
                duration=info.get('duration', 0),
                thumbnail_url=info.get('thumbnail', ''),
                uploader=info.get('uploader', ''),
                platform=platform,
                original_url=url,
                formats=video_formats,
                audio_tracks=audio_tracks,
                subtitles=subtitles,
                recommended_format=recommended,
                best_quality=best_quality,
                view_count=info.get('view_count', 0),
                like_count=info.get('like_count', 0),
                tags=info.get('tags', []) or [],
            )

        except Exception as e:
            logger.error(f"Failed to get detailed video info for {url}: {e}")
            import traceback
            logger.debug(f"Detailed video info traceback: {traceback.format_exc()}")
            return None

    async def download_subtitle(self, url: str, language: str, output_path: Path) -> bool:
        """
        Download subtitle for a specific language

        Args:
            url: Video URL
            language: Language code (e.g., 'en', 'zh-CN')
            output_path: Path to save subtitle file

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            opts = {
                **self.ydl_opts_base,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': [language],
                'subtitlesformat': 'srt/vtt/best',
                'outtmpl': str(output_path.with_suffix('')),
            }

            def download():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, download)

            # Check if subtitle was downloaded
            for ext in ['.srt', '.vtt', f'.{language}.srt', f'.{language}.vtt']:
                check_path = output_path.with_suffix(ext)
                if check_path.exists():
                    # Rename to expected path if needed
                    if check_path != output_path:
                        check_path.rename(output_path)
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to download subtitle: {e}")
            return False

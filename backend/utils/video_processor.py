"""
Video Processing Utilities using FFmpeg
"""
import asyncio
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
from loguru import logger


class VideoOrientation(Enum):
    """Video orientation based on aspect ratio"""
    HORIZONTAL = "horizontal"  # Landscape (16:9, 4:3, etc.)
    VERTICAL = "vertical"      # Portrait (9:16, TikTok, Shorts)
    SQUARE = "square"          # 1:1 (Instagram posts)


@dataclass
class VideoInfo:
    """Video file information"""
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width / height)"""
        if self.height == 0:
            return 0
        return self.width / self.height

    @property
    def orientation(self) -> VideoOrientation:
        """Determine video orientation based on aspect ratio"""
        ratio = self.aspect_ratio
        if ratio < 0.9:  # Less than 9:10, considered vertical
            return VideoOrientation.VERTICAL
        elif ratio > 1.1:  # Greater than 10:9, considered horizontal
            return VideoOrientation.HORIZONTAL
        else:  # Between 0.9 and 1.1, considered square
            return VideoOrientation.SQUARE

    @property
    def is_vertical(self) -> bool:
        """Check if video is vertical (portrait orientation)"""
        return self.orientation == VideoOrientation.VERTICAL

    @property
    def is_horizontal(self) -> bool:
        """Check if video is horizontal (landscape orientation)"""
        return self.orientation == VideoOrientation.HORIZONTAL

    @property
    def is_square(self) -> bool:
        """Check if video is square"""
        return self.orientation == VideoOrientation.SQUARE

    @property
    def is_short_form(self) -> bool:
        """Check if video is likely short-form content (TikTok, Shorts, Reels)"""
        # Vertical video under 3 minutes is likely short-form
        return self.is_vertical and self.duration < 180

    def get_orientation_info(self) -> dict:
        """Get detailed orientation information"""
        return {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 2),
            "orientation": self.orientation.value,
            "is_vertical": self.is_vertical,
            "is_short_form": self.is_short_form,
        }


@dataclass
class ProcessResult:
    """Processing result"""
    success: bool
    output_path: Optional[Path]
    error: Optional[str] = None


class VideoProcessor:
    """Video processing utilities using FFmpeg"""

    def __init__(self):
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not found. Video processing may fail.")

    async def get_video_info(self, video_path: Path) -> Optional[VideoInfo]:
        """Get video file information"""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path)
            ]

            def run_probe():
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.stdout

            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, run_probe)

            import json
            data = json.loads(output)

            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if not video_stream:
                return None

            format_info = data.get("format", {})

            return VideoInfo(
                duration=float(format_info.get("duration", 0)),
                width=int(video_stream.get("width", 0)),
                height=int(video_stream.get("height", 0)),
                fps=eval(video_stream.get("r_frame_rate", "0/1")),
                codec=video_stream.get("codec_name", ""),
                bitrate=int(format_info.get("bit_rate", 0)),
            )

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None

    async def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        format: str = "mp3"
    ) -> ProcessResult:
        """Extract audio from video"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "libmp3lame" if format == "mp3" else "aac",
                "-q:a", "2",
                str(output_path)
            ]

            def run_extract():
                return subprocess.run(cmd, capture_output=True, text=True)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_extract)

            if result.returncode != 0:
                return ProcessResult(
                    success=False,
                    output_path=None,
                    error=result.stderr
                )

            return ProcessResult(success=True, output_path=output_path)

        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return ProcessResult(success=False, output_path=None, error=str(e))

    async def merge_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        replace_audio: bool = True,
        audio_volume: float = 1.0
    ) -> ProcessResult:
        """Merge audio with video"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if replace_audio:
                # Replace original audio entirely
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(audio_path),
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    str(output_path)
                ]
            else:
                # Mix with original audio
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(audio_path),
                    "-filter_complex",
                    f"[0:a]volume=0.3[a0];[1:a]volume={audio_volume}[a1];[a0][a1]amix=inputs=2:duration=first[aout]",
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "[aout]",
                    str(output_path)
                ]

            def run_merge():
                return subprocess.run(cmd, capture_output=True, text=True)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_merge)

            if result.returncode != 0:
                return ProcessResult(
                    success=False,
                    output_path=None,
                    error=result.stderr
                )

            logger.info(f"Merged audio/video: {output_path}")
            return ProcessResult(success=True, output_path=output_path)

        except Exception as e:
            logger.error(f"Audio/video merge failed: {e}")
            return ProcessResult(success=False, output_path=None, error=str(e))

    async def resize_video(
        self,
        video_path: Path,
        output_path: Path,
        width: int,
        height: int,
        keep_aspect: bool = True
    ) -> ProcessResult:
        """Resize video"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if keep_aspect:
                scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
            else:
                scale_filter = f"scale={width}:{height}"

            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", scale_filter,
                "-c:a", "copy",
                str(output_path)
            ]

            def run_resize():
                return subprocess.run(cmd, capture_output=True, text=True)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_resize)

            if result.returncode != 0:
                return ProcessResult(
                    success=False,
                    output_path=None,
                    error=result.stderr
                )

            return ProcessResult(success=True, output_path=output_path)

        except Exception as e:
            logger.error(f"Video resize failed: {e}")
            return ProcessResult(success=False, output_path=None, error=str(e))

    async def convert_for_platform(
        self,
        video_path: Path,
        output_path: Path,
        platform: str
    ) -> ProcessResult:
        """
        Convert video to platform-specific format

        Args:
            video_path: Input video
            output_path: Output path
            platform: Target platform (bilibili, douyin, xiaohongshu)
        """
        # Platform-specific settings
        platform_settings = {
            "bilibili": {
                "max_width": 1920,
                "max_height": 1080,
                "max_bitrate": "6M",
                "codec": "h264",
            },
            "douyin": {
                "max_width": 1080,
                "max_height": 1920,  # Vertical video
                "max_bitrate": "4M",
                "codec": "h264",
            },
            "xiaohongshu": {
                "max_width": 1080,
                "max_height": 1920,  # Vertical preferred
                "max_bitrate": "4M",
                "codec": "h264",
            },
        }

        settings = platform_settings.get(platform, platform_settings["bilibili"])

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "28",  # Changed from 23 for smaller file size (~40% reduction)
                "-maxrate", settings["max_bitrate"],
                "-bufsize", settings["max_bitrate"],
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_path)
            ]

            def run_convert():
                return subprocess.run(cmd, capture_output=True, text=True)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_convert)

            if result.returncode != 0:
                return ProcessResult(
                    success=False,
                    output_path=None,
                    error=result.stderr
                )

            logger.info(f"Converted video for {platform}: {output_path}")
            return ProcessResult(success=True, output_path=output_path)

        except Exception as e:
            logger.error(f"Platform conversion failed: {e}")
            return ProcessResult(success=False, output_path=None, error=str(e))

    async def concatenate_audio_segments(
        self,
        audio_files: List[Path],
        output_path: Path,
        timings: List[tuple] = None  # List of (start, end) in seconds
    ) -> ProcessResult:
        """
        Concatenate multiple audio files with optional timing/gaps

        Args:
            audio_files: List of audio file paths
            output_path: Output file path
            timings: Optional list of (start, end) times for each segment
        """
        try:
            if not audio_files:
                return ProcessResult(
                    success=False,
                    output_path=None,
                    error="No audio files provided"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create concat file
            concat_file = output_path.parent / "concat_list.txt"
            with open(concat_file, "w") as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path)
            ]

            def run_concat():
                result = subprocess.run(cmd, capture_output=True, text=True)
                # Clean up concat file
                concat_file.unlink(missing_ok=True)
                return result

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_concat)

            if result.returncode != 0:
                return ProcessResult(
                    success=False,
                    output_path=None,
                    error=result.stderr
                )

            return ProcessResult(success=True, output_path=output_path)

        except Exception as e:
            logger.error(f"Audio concatenation failed: {e}")
            return ProcessResult(success=False, output_path=None, error=str(e))

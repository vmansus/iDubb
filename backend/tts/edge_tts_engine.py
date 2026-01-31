"""
Microsoft Edge TTS Engine (Free, High Quality)
"""
import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional
import edge_tts
from loguru import logger

from .base import BaseTTSEngine, TTSResult, TTSSegment


class EdgeTTSEngine(BaseTTSEngine):
    """
    Microsoft Edge TTS - Free, high-quality neural voices
    Supports many languages including Chinese
    """

    # Popular Chinese voices
    CHINESE_VOICES = {
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",  # Female, warm
        "xiaoyi": "zh-CN-XiaoyiNeural",  # Female, gentle
        "yunjian": "zh-CN-YunjianNeural",  # Male, professional
        "yunxi": "zh-CN-YunxiNeural",  # Male, young
        "yunxia": "zh-CN-YunxiaNeural",  # Male, child
        "yunyang": "zh-CN-YunyangNeural",  # Male, news
    }

    # Popular English voices
    ENGLISH_VOICES = {
        "jenny": "en-US-JennyNeural",
        "guy": "en-US-GuyNeural",
        "aria": "en-US-AriaNeural",
        "davis": "en-US-DavisNeural",
    }

    def __init__(self, default_voice: str = "zh-CN-XiaoxiaoNeural"):
        self.default_voice = default_voice
        logger.info(f"Initialized Edge TTS with voice: {default_voice}")

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str = None,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ) -> TTSResult:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            output_path: Output audio file path
            voice: Voice name (e.g., zh-CN-XiaoxiaoNeural)
            rate: Speech rate (e.g., +10%, -20%)
            volume: Volume adjustment
            pitch: Pitch adjustment

        Returns:
            TTSResult
        """
        try:
            if not text or not text.strip():
                return TTSResult(
                    success=False,
                    audio_path=None,
                    duration=0,
                    error="Empty text"
                )

            voice = voice or self.default_voice

            # Create communicate instance
            communicate = edge_tts.Communicate(
                text,
                voice,
                rate=rate,
                volume=volume,
                pitch=pitch
            )

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save audio
            await communicate.save(str(output_path))

            # Get actual duration using FFprobe
            actual_duration = await self._get_audio_duration(output_path)

            logger.debug(f"Synthesized TTS: {output_path} ({actual_duration:.2f}s)")

            return TTSResult(
                success=True,
                audio_path=output_path,
                duration=actual_duration,
            )

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error=str(e)
            )

    async def get_available_voices(self, language: str = None) -> List[dict]:
        """
        Get available voices

        Args:
            language: Filter by language code (e.g., 'zh', 'en')

        Returns:
            List of voice info dicts
        """
        try:
            voices = await edge_tts.list_voices()

            if language:
                # Filter by language
                voices = [v for v in voices if v.get('Locale', '').startswith(language)]

            return [
                {
                    "name": v.get("ShortName"),
                    "display_name": v.get("FriendlyName"),
                    "gender": v.get("Gender"),
                    "locale": v.get("Locale"),
                }
                for v in voices
            ]

        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []

    async def synthesize_segments(
        self,
        segments: List[TTSSegment],
        output_dir: Path,
        voice: str = None,
        rate: str = "+0%"
    ) -> List[TTSSegment]:
        """
        Synthesize multiple segments to separate audio files

        Args:
            segments: List of TTSSegment with text and timing
            output_dir: Directory to save audio files
            voice: Voice to use
            rate: Speech rate

        Returns:
            List of TTSSegment with audio_path filled in
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        voice = voice or self.default_voice

        results = []
        for i, seg in enumerate(segments):
            output_path = output_dir / f"segment_{i:04d}.mp3"

            result = await self.synthesize(
                seg.text,
                output_path,
                voice=voice,
                rate=rate
            )

            if result.success:
                results.append(TTSSegment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    audio_path=result.audio_path
                ))
            else:
                results.append(seg)  # Keep original without audio

        return results

    async def synthesize_with_subtitles(
        self,
        text: str,
        output_path: Path,
        voice: str = None,
        rate: str = "+0%"
    ) -> tuple:
        """
        Synthesize speech and get word-level timing for subtitles

        Returns:
            Tuple of (TTSResult, List of timing data)
        """
        try:
            voice = voice or self.default_voice
            output_path.parent.mkdir(parents=True, exist_ok=True)

            communicate = edge_tts.Communicate(text, voice, rate=rate)

            timing_data = []
            with open(output_path, "wb") as audio_file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_file.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        timing_data.append({
                            "text": chunk["text"],
                            "offset": chunk["offset"] / 10000000,  # Convert to seconds
                            "duration": chunk["duration"] / 10000000
                        })

            return TTSResult(
                success=True,
                audio_path=output_path,
                duration=sum(t["duration"] for t in timing_data) if timing_data else 0
            ), timing_data

        except Exception as e:
            logger.error(f"TTS with subtitles failed: {e}")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error=str(e)
            ), []

    async def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration using FFprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()

            return float(stdout.decode().strip())
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
            # Fallback to estimate
            return 0.0

    @classmethod
    def get_chinese_voices(cls) -> dict:
        """Get predefined Chinese voices"""
        return cls.CHINESE_VOICES.copy()

    @classmethod
    def get_english_voices(cls) -> dict:
        """Get predefined English voices"""
        return cls.ENGLISH_VOICES.copy()

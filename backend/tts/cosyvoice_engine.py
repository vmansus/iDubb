"""
CosyVoice Engine - Alibaba's Open Source Voice Cloning TTS
Supports voice cloning with zero-shot or few-shot learning
"""
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Optional
from loguru import logger

from .base import BaseTTSEngine, TTSResult, TTSSegment


class CosyVoiceEngine(BaseTTSEngine):
    """
    CosyVoice - Alibaba's open source voice cloning TTS
    Supports:
    - Zero-shot voice cloning
    - Cross-lingual voice cloning
    - Instruct mode for emotion/style control
    """

    # Available modes
    MODES = {
        "preset": "Use predefined speaker presets",
        "zero_shot": "Clone voice from reference audio (zero-shot)",
        "cross_lingual": "Cross-lingual voice cloning",
        "instruct": "Instruction-based TTS with emotion control",
    }

    # Default preset speakers
    PRESET_SPEAKERS = [
        "中文女", "中文男", "英文女", "英文男",
        "日语男", "粤语女", "韩语女",
    ]

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50000,
        mode: str = "preset",
        default_speaker: str = "中文女",
        ref_audio_path: Optional[Path] = None,
        ref_text: Optional[str] = None,
    ):
        """
        Initialize CosyVoice engine

        Args:
            host: CosyVoice server host
            port: CosyVoice server port
            mode: TTS mode (preset, zero_shot, cross_lingual, instruct)
            default_speaker: Default preset speaker
            ref_audio_path: Path to reference audio for zero-shot/cross-lingual
            ref_text: Transcript of reference audio
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.mode = mode
        self.default_speaker = default_speaker
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Initialized CosyVoice engine at {self.base_url} (mode: {mode})")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def check_health(self) -> bool:
        """Check if CosyVoice server is available"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/", timeout=5) as resp:
                return resp.status in [200, 404]  # Server is up
        except Exception as e:
            logger.warning(f"CosyVoice server not available: {e}")
            return False

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str = None,
        rate: str = "+0%",
        ref_audio: Optional[Path] = None,
        ref_text: Optional[str] = None,
        instruct_text: Optional[str] = None,
    ) -> TTSResult:
        """
        Synthesize speech from text using CosyVoice

        Args:
            text: Text to synthesize
            output_path: Output audio file path
            voice: Speaker preset name (for preset mode)
            rate: Speech rate (post-processing)
            ref_audio: Reference audio path (for zero-shot/cross-lingual)
            ref_text: Reference audio transcript
            instruct_text: Instruction for emotion/style (for instruct mode)

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

            # Check server availability
            if not await self.check_health():
                return TTSResult(
                    success=False,
                    audio_path=None,
                    duration=0,
                    error="CosyVoice server not available. Please start the server."
                )

            session = await self._get_session()

            # Select endpoint and build request based on mode
            if self.mode == "preset":
                endpoint = f"{self.base_url}/api/inference/sft"
                data = aiohttp.FormData()
                data.add_field("tts_text", text)
                data.add_field("spk_id", voice or self.default_speaker)

            elif self.mode == "zero_shot":
                endpoint = f"{self.base_url}/api/inference/zero-shot"
                ref_path = ref_audio or self.ref_audio_path
                prompt_text = ref_text or self.ref_text

                if not ref_path or not ref_path.exists():
                    return TTSResult(
                        success=False,
                        audio_path=None,
                        duration=0,
                        error="Reference audio required for zero-shot mode"
                    )

                data = aiohttp.FormData()
                data.add_field("tts_text", text)
                data.add_field("prompt_text", prompt_text or "")
                data.add_field(
                    "prompt_wav",
                    open(ref_path, "rb"),
                    filename=ref_path.name,
                    content_type="audio/wav"
                )

            elif self.mode == "cross_lingual":
                endpoint = f"{self.base_url}/api/inference/cross-lingual"
                ref_path = ref_audio or self.ref_audio_path

                if not ref_path or not ref_path.exists():
                    return TTSResult(
                        success=False,
                        audio_path=None,
                        duration=0,
                        error="Reference audio required for cross-lingual mode"
                    )

                data = aiohttp.FormData()
                data.add_field("tts_text", text)
                data.add_field(
                    "prompt_wav",
                    open(ref_path, "rb"),
                    filename=ref_path.name,
                    content_type="audio/wav"
                )

            elif self.mode == "instruct":
                endpoint = f"{self.base_url}/api/inference/instruct"
                data = aiohttp.FormData()
                data.add_field("tts_text", text)
                data.add_field("spk_id", voice or self.default_speaker)
                data.add_field("instruct_text", instruct_text or "用温柔的语气说")

            else:
                return TTSResult(
                    success=False,
                    audio_path=None,
                    duration=0,
                    error=f"Unknown mode: {self.mode}"
                )

            # Make request
            async with session.post(endpoint, data=data, timeout=120) as resp:
                if resp.status == 200:
                    # Ensure output directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save audio
                    audio_data = await resp.read()
                    with open(output_path, "wb") as f:
                        f.write(audio_data)

                    # Estimate duration
                    duration = max(0, (len(audio_data) - 44)) / (22050 * 2)

                    logger.info(f"CosyVoice synthesized: {output_path}")

                    return TTSResult(
                        success=True,
                        audio_path=output_path,
                        duration=duration,
                    )
                else:
                    error_text = await resp.text()
                    logger.error(f"CosyVoice error: {error_text}")
                    return TTSResult(
                        success=False,
                        audio_path=None,
                        duration=0,
                        error=f"CosyVoice error: {error_text}"
                    )

        except asyncio.TimeoutError:
            logger.error("CosyVoice request timed out")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error="CosyVoice request timed out"
            )
        except Exception as e:
            logger.error(f"CosyVoice synthesis failed: {e}")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error=str(e)
            )

    async def get_available_voices(self, language: str = None) -> List[dict]:
        """Get available voice presets"""
        voices = []

        for speaker in self.PRESET_SPEAKERS:
            # Determine locale based on speaker name
            if "中文" in speaker or "粤语" in speaker:
                locale = "zh-CN"
            elif "英文" in speaker:
                locale = "en-US"
            elif "日语" in speaker:
                locale = "ja-JP"
            elif "韩语" in speaker:
                locale = "ko-KR"
            else:
                locale = "unknown"

            # Determine gender
            gender = "female" if "女" in speaker else "male"

            if language is None or locale.startswith(language):
                voices.append({
                    "name": speaker,
                    "display_name": speaker,
                    "gender": gender,
                    "locale": locale,
                })

        # Add custom voice option
        voices.append({
            "name": "custom",
            "display_name": "Custom (Voice Clone)",
            "gender": "unknown",
            "locale": "any",
        })

        return voices

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
            voice: Voice preset to use
            rate: Speech rate (post-processing)

        Returns:
            List of TTSSegment with audio_path filled in
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, seg in enumerate(segments):
            output_path = output_dir / f"segment_{i:04d}.wav"

            result = await self.synthesize(
                seg.text,
                output_path,
                voice=voice,
            )

            if result.success:
                results.append(TTSSegment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    audio_path=result.audio_path
                ))
            else:
                logger.warning(f"Failed to synthesize segment {i}: {result.error}")
                results.append(seg)

        return results

    async def set_reference_audio(
        self,
        audio_path: Path,
        transcript: Optional[str] = None
    ) -> bool:
        """
        Set reference audio for voice cloning

        Args:
            audio_path: Path to reference audio file (3-10 seconds recommended)
            transcript: Transcript of the reference audio

        Returns:
            True if successful
        """
        if not audio_path.exists():
            logger.error(f"Reference audio not found: {audio_path}")
            return False

        self.ref_audio_path = audio_path
        self.ref_text = transcript
        self.mode = "zero_shot"
        logger.info(f"Set reference audio: {audio_path}")
        return True

    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()

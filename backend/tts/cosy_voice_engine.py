"""
CosyVoice Engine - Alibaba's Advanced Voice Cloning TTS (v2 Compatible)
Supports zero-shot voice cloning, cross-lingual synthesis, and instruction-based TTS
"""
import asyncio
import aiohttp
import io
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from .base import BaseTTSEngine, TTSResult, TTSSegment


class CosyVoiceEngine(BaseTTSEngine):
    """
    CosyVoice 2.0 - Advanced voice cloning TTS engine
    Supports multiple synthesis modes:
    - SFT: Speaker fine-tuned voices
    - Zero-shot: Voice cloning from reference audio
    - Cross-lingual: Maintain voice across languages
    - Instruct: Text instruction-based style control

    Requires CosyVoice server running locally
    """

    # Synthesis modes
    MODES = {
        "sft": "Speaker fine-tuned synthesis",
        "zero_shot": "Zero-shot voice cloning",
        "cross_lingual": "Cross-lingual voice cloning",
        "instruct": "Instruction-based synthesis (v2)",
        "instruct2": "Enhanced instruction synthesis (v2)",
    }

    # Default SFT speakers
    DEFAULT_SPEAKERS = [
        "中文女", "中文男", "日语男", "粤语女", "英文女", "英文男", "韩语女"
    ]

    # Supported languages
    LANGUAGES = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "yue": "Cantonese",
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50000,
        mode: str = "zero_shot",
        default_speaker: str = "中文女",
        ref_audio_path: Optional[Path] = None,
        ref_text: Optional[str] = None,
        api_version: str = "v2",
    ):
        """
        Initialize CosyVoice engine

        Args:
            host: CosyVoice server host
            port: CosyVoice server port
            mode: Synthesis mode (sft, zero_shot, cross_lingual, instruct, instruct2)
            default_speaker: Default SFT speaker name
            ref_audio_path: Path to reference audio for zero-shot/cross-lingual
            ref_text: Reference text (transcript of ref audio)
            api_version: API version (v1, v2)
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.mode = mode
        self.default_speaker = default_speaker
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.api_version = api_version
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Initialized CosyVoice engine at {self.base_url} (mode: {mode})")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=180, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def check_health(self) -> bool:
        """Check if CosyVoice server is available"""
        try:
            session = await self._get_session()

            # Try health endpoint
            async with session.get(f"{self.base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    logger.info("CosyVoice server is healthy")
                    return True

            # Fallback: try docs endpoint (FastAPI)
            async with session.get(f"{self.base_url}/docs", timeout=5) as resp:
                if resp.status == 200:
                    return True

            return False
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
        target_language: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """
        Synthesize speech from text using CosyVoice

        Args:
            text: Text to synthesize
            output_path: Output audio file path
            voice: Speaker name (for SFT mode)
            rate: Speech rate (converted to speed)
            ref_audio: Reference audio path (for zero-shot/cross-lingual)
            ref_text: Reference text transcript
            instruct_text: Style instruction (for instruct mode)
            target_language: Target language code (for cross-lingual)
            speed: Speech speed multiplier

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

            # Choose synthesis method based on mode
            if self.mode == "sft":
                return await self._synthesize_sft(text, output_path, voice, speed)
            elif self.mode == "zero_shot":
                return await self._synthesize_zero_shot(
                    text, output_path, ref_audio, ref_text, speed
                )
            elif self.mode == "cross_lingual":
                return await self._synthesize_cross_lingual(
                    text, output_path, ref_audio, target_language, speed
                )
            elif self.mode == "instruct":
                return await self._synthesize_instruct(
                    text, output_path, voice, instruct_text, speed
                )
            elif self.mode == "instruct2":
                return await self._synthesize_instruct2(
                    text, output_path, voice, instruct_text, ref_audio, speed
                )
            else:
                # Default to SFT
                return await self._synthesize_sft(text, output_path, voice, speed)

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

    async def _synthesize_sft(
        self,
        text: str,
        output_path: Path,
        voice: str = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """SFT mode - Use pre-trained speaker voices"""
        session = await self._get_session()

        data = aiohttp.FormData()
        data.add_field("tts_text", text)
        data.add_field("spk_id", voice or self.default_speaker)
        if speed != 1.0:
            data.add_field("speed", str(speed))

        endpoint = f"{self.base_url}/inference_sft"

        async with session.post(endpoint, data=data) as resp:
            return await self._handle_response(resp, output_path)

    async def _synthesize_zero_shot(
        self,
        text: str,
        output_path: Path,
        ref_audio: Optional[Path] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Zero-shot mode - Clone voice from reference audio"""
        session = await self._get_session()

        ref_path = ref_audio or self.ref_audio_path
        ref_transcript = ref_text or self.ref_text

        if not ref_path or not ref_path.exists():
            logger.error("Reference audio required for zero-shot mode")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error="Reference audio required for zero-shot mode"
            )

        if not ref_transcript:
            logger.warning("Reference text not provided, quality may be reduced")
            ref_transcript = ""

        data = aiohttp.FormData()
        data.add_field("tts_text", text)
        data.add_field("prompt_text", ref_transcript)

        with open(ref_path, "rb") as f:
            data.add_field(
                "prompt_wav",
                f.read(),
                filename=ref_path.name,
                content_type="audio/wav"
            )

        if speed != 1.0:
            data.add_field("speed", str(speed))

        endpoint = f"{self.base_url}/inference_zero_shot"

        async with session.post(endpoint, data=data) as resp:
            return await self._handle_response(resp, output_path)

    async def _synthesize_cross_lingual(
        self,
        text: str,
        output_path: Path,
        ref_audio: Optional[Path] = None,
        target_language: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Cross-lingual mode - Maintain voice across languages"""
        session = await self._get_session()

        ref_path = ref_audio or self.ref_audio_path

        if not ref_path or not ref_path.exists():
            logger.error("Reference audio required for cross-lingual mode")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error="Reference audio required for cross-lingual mode"
            )

        data = aiohttp.FormData()
        data.add_field("tts_text", text)

        with open(ref_path, "rb") as f:
            data.add_field(
                "prompt_wav",
                f.read(),
                filename=ref_path.name,
                content_type="audio/wav"
            )

        if target_language:
            data.add_field("target_lang", target_language)

        if speed != 1.0:
            data.add_field("speed", str(speed))

        endpoint = f"{self.base_url}/inference_cross_lingual"

        async with session.post(endpoint, data=data) as resp:
            return await self._handle_response(resp, output_path)

    async def _synthesize_instruct(
        self,
        text: str,
        output_path: Path,
        voice: str = None,
        instruct_text: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Instruct mode - Text instruction-based style control"""
        session = await self._get_session()

        data = aiohttp.FormData()
        data.add_field("tts_text", text)
        data.add_field("spk_id", voice or self.default_speaker)

        if instruct_text:
            data.add_field("instruct_text", instruct_text)

        if speed != 1.0:
            data.add_field("speed", str(speed))

        endpoint = f"{self.base_url}/inference_instruct"

        async with session.post(endpoint, data=data) as resp:
            return await self._handle_response(resp, output_path)

    async def _synthesize_instruct2(
        self,
        text: str,
        output_path: Path,
        voice: str = None,
        instruct_text: Optional[str] = None,
        ref_audio: Optional[Path] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Instruct2 mode - Enhanced instruction with reference audio (v2)"""
        session = await self._get_session()

        data = aiohttp.FormData()
        data.add_field("tts_text", text)
        data.add_field("spk_id", voice or self.default_speaker)

        if instruct_text:
            data.add_field("instruct_text", instruct_text)

        ref_path = ref_audio or self.ref_audio_path
        if ref_path and ref_path.exists():
            with open(ref_path, "rb") as f:
                data.add_field(
                    "prompt_wav",
                    f.read(),
                    filename=ref_path.name,
                    content_type="audio/wav"
                )

        if speed != 1.0:
            data.add_field("speed", str(speed))

        endpoint = f"{self.base_url}/inference_instruct2"

        try:
            async with session.post(endpoint, data=data) as resp:
                if resp.status == 404:
                    # Fallback to instruct mode
                    return await self._synthesize_instruct(
                        text, output_path, voice, instruct_text, speed
                    )
                return await self._handle_response(resp, output_path)
        except aiohttp.ClientError:
            return await self._synthesize_instruct(
                text, output_path, voice, instruct_text, speed
            )

    async def _handle_response(self, resp: aiohttp.ClientResponse, output_path: Path) -> TTSResult:
        """Handle API response and save audio"""
        if resp.status == 200:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save audio
            audio_data = await resp.read()
            with open(output_path, "wb") as f:
                f.write(audio_data)

            # Estimate duration from WAV data
            # CosyVoice outputs 22050 Hz 16-bit mono WAV
            sample_rate = 22050
            duration = max(0, (len(audio_data) - 44)) / (sample_rate * 2)

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

    async def get_available_voices(self, language: str = None) -> List[dict]:
        """Get available speaker voices"""
        # Try to get from server
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/speakers", timeout=5) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug(f"Could not get speakers from server: {e}")

        # Default speakers
        speakers = []
        for spk in self.DEFAULT_SPEAKERS:
            # Parse language from speaker name
            if "中文" in spk or "粤语" in spk:
                locale = "zh-CN"
            elif "英文" in spk:
                locale = "en-US"
            elif "日语" in spk:
                locale = "ja-JP"
            elif "韩语" in spk:
                locale = "ko-KR"
            else:
                locale = "zh-CN"

            gender = "female" if "女" in spk else "male"

            speakers.append({
                "name": spk,
                "display_name": spk,
                "gender": gender,
                "locale": locale,
            })

        # Filter by language if specified
        if language:
            lang_map = {
                "zh": "zh-CN",
                "en": "en-US",
                "ja": "ja-JP",
                "ko": "ko-KR",
            }
            target_locale = lang_map.get(language, language)
            speakers = [s for s in speakers if s["locale"].startswith(target_locale.split("-")[0])]

        return speakers

    async def synthesize_segments(
        self,
        segments: List[TTSSegment],
        output_dir: Path,
        voice: str = None,
        rate: str = "+0%",
        ref_audio: Optional[Path] = None,
        ref_text: Optional[str] = None,
        instruct_text: Optional[str] = None,
        concurrent: int = 2,
    ) -> List[TTSSegment]:
        """
        Synthesize multiple segments with optional concurrency

        Args:
            segments: List of TTSSegment with text and timing
            output_dir: Directory to save audio files
            voice: Voice preset to use
            rate: Speech rate
            ref_audio: Reference audio for zero-shot
            ref_text: Reference text
            instruct_text: Style instruction
            concurrent: Number of concurrent requests (lower for CosyVoice)

        Returns:
            List of TTSSegment with audio_path filled in
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        async def synthesize_one(i: int, seg: TTSSegment) -> TTSSegment:
            output_path = output_dir / f"segment_{i:04d}.wav"
            result = await self.synthesize(
                seg.text,
                output_path,
                voice=voice,
                ref_audio=ref_audio,
                ref_text=ref_text,
                instruct_text=instruct_text,
            )

            if result.success:
                return TTSSegment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    audio_path=result.audio_path
                )
            else:
                logger.warning(f"Failed to synthesize segment {i}: {result.error}")
                return seg

        # Process with semaphore for concurrency control
        # CosyVoice is heavier, so lower concurrency
        semaphore = asyncio.Semaphore(concurrent)

        async def limited_synthesize(i: int, seg: TTSSegment) -> TTSSegment:
            async with semaphore:
                return await synthesize_one(i, seg)

        tasks = [limited_synthesize(i, seg) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def set_reference_audio(self, audio_path: Path, text: str = None) -> bool:
        """
        Set global reference audio for voice cloning

        Args:
            audio_path: Path to reference audio file (3-10 seconds recommended)
            text: Transcript of the reference audio (improves quality)

        Returns:
            True if successful
        """
        if not audio_path.exists():
            logger.error(f"Reference audio not found: {audio_path}")
            return False

        self.ref_audio_path = audio_path
        if text:
            self.ref_text = text
        self.mode = "zero_shot"
        logger.info(f"Set reference audio: {audio_path}")
        return True

    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()

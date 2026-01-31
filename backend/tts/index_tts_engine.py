"""
IndexTTS Engine - Local Voice Cloning TTS via Gradio Client
Supports voice cloning with reference audio and emotion control
Uses gradio_client for communication with IndexTTS2 server
"""
import asyncio
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

try:
    from gradio_client import Client, handle_file
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    logger.warning("gradio_client not installed. Install with: pip install gradio_client")

from .base import BaseTTSEngine, TTSResult, TTSSegment


class IndexTTSEngine(BaseTTSEngine):
    """
    IndexTTS 2 - Local voice cloning TTS engine with emotion control
    Uses gradio_client for communication with IndexTTS2 WebUI server
    Requires IndexTTS server running locally (python webui.py --port 9880)
    """

    # Emotion control modes (matches Gradio UI)
    EMO_MODES = {
        "same_as_voice": "与音色参考音频相同",      # Mode 0
        "from_ref_audio": "使用情感参考音频",       # Mode 1
        "from_vector": "使用情感向量控制",          # Mode 2
        "from_text": "使用情感描述文本控制",        # Mode 3
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9880,
        ref_audio_path: Optional[Path] = None,
        emo_audio_path: Optional[Path] = None,
        emo_mode: str = "same_as_voice",
        emo_weight: float = 0.65,
        max_tokens_per_segment: int = 120,
    ):
        """
        Initialize IndexTTS engine

        Args:
            host: IndexTTS server host
            port: IndexTTS server port
            ref_audio_path: Path to reference audio for voice cloning
            emo_audio_path: Path to emotion reference audio (for emo_mode="from_ref_audio")
            emo_mode: Emotion control mode (same_as_voice, from_ref_audio, from_vector, from_text)
            emo_weight: Emotion weight 0.0-1.0
            max_tokens_per_segment: Max tokens per generation segment
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ref_audio_path = ref_audio_path
        self.emo_audio_path = emo_audio_path
        self.emo_mode = emo_mode
        self.emo_weight = emo_weight
        self.max_tokens_per_segment = max_tokens_per_segment
        self._client: Optional[Client] = None
        # Increase workers for parallel TTS generation
        # Note: Actual parallelism depends on IndexTTS server capacity
        self._executor = ThreadPoolExecutor(max_workers=8)
        logger.info(f"Initialized IndexTTS engine at {self.base_url} (Gradio Client)")

    def _get_client(self) -> Optional[Client]:
        """Get or create Gradio client"""
        if not GRADIO_CLIENT_AVAILABLE:
            logger.error("gradio_client not available")
            return None

        if self._client is None:
            try:
                self._client = Client(self.base_url, verbose=False)
                logger.debug(f"Connected to IndexTTS server at {self.base_url}")
            except Exception as e:
                logger.error(f"Failed to connect to IndexTTS server: {e}")
                return None
        return self._client

    async def check_health(self) -> bool:
        """Check if IndexTTS Gradio server is available"""
        try:
            # Run synchronous client creation in thread pool
            loop = asyncio.get_event_loop()
            client = await loop.run_in_executor(self._executor, self._get_client)
            if client:
                logger.info("IndexTTS Gradio server is healthy")
                return True
            return False
        except Exception as e:
            logger.warning(f"IndexTTS server not available: {e}")
            return False

    async def get_server_info(self) -> Dict[str, Any]:
        """Get Gradio server info"""
        try:
            client = self._get_client()
            if client:
                return {"type": "gradio", "url": self.base_url}
        except Exception as e:
            logger.debug(f"Could not get server info: {e}")
        return {"type": "gradio", "version": "unknown"}

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str = None,
        rate: str = "+0%",
        ref_audio: Optional[Path] = None,
        emo_alpha: float = None,
        emo_audio: Optional[Path] = None,
        emo_vector: Optional[List[float]] = None,
        emotion_preset: str = None,
        streaming: bool = False,
    ) -> TTSResult:
        """
        Synthesize speech from text using IndexTTS via gradio_client

        Args:
            text: Text to synthesize
            output_path: Output audio file path
            voice: Not used (for API compatibility)
            rate: Not used (for API compatibility)
            ref_audio: Reference audio path (for voice cloning)
            emo_alpha: Emotion strength 0.0-1.0
            emo_audio: Emotion audio prompt path
            emo_vector: Direct emotion vector [8 floats]
            emotion_preset: Not used
            streaming: Not used

        Returns:
            TTSResult
        """
        try:
            if not GRADIO_CLIENT_AVAILABLE:
                return TTSResult(
                    success=False,
                    audio_path=None,
                    duration=0,
                    error="gradio_client not installed. Run: pip install gradio_client"
                )

            if not text or not text.strip():
                return TTSResult(
                    success=False,
                    audio_path=None,
                    duration=0,
                    error="Empty text"
                )

            # Use provided ref_audio or default
            ref_path = ref_audio or self.ref_audio_path
            if not ref_path or not ref_path.exists():
                return TTSResult(
                    success=False,
                    audio_path=None,
                    duration=0,
                    error="Reference audio required for voice cloning"
                )

            # Emotion audio - use ref_audio if not specified
            emo_path = emo_audio or self.emo_audio_path or ref_path

            # Call the synthesis in thread pool (gradio_client is synchronous)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._synthesize_sync,
                text,
                str(ref_path),
                str(emo_path),
                emo_alpha if emo_alpha is not None else self.emo_weight,
                output_path
            )

            return result

        except asyncio.TimeoutError:
            logger.error("IndexTTS request timed out")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error="IndexTTS request timed out"
            )
        except Exception as e:
            logger.error(f"IndexTTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error=str(e)
            )

    def _synthesize_sync(
        self,
        text: str,
        ref_audio_path: str,
        emo_audio_path: str,
        emo_weight: float,
        output_path: Path
    ) -> TTSResult:
        """
        Synchronous synthesis using gradio_client

        This runs in a thread pool to not block the event loop
        """
        try:
            client = self._get_client()
            if not client:
                return TTSResult(
                    success=False,
                    audio_path=None,
                    duration=0,
                    error="Failed to connect to IndexTTS server"
                )

            # Get emotion mode string
            emo_mode_str = self.EMO_MODES.get(self.emo_mode, self.EMO_MODES["same_as_voice"])

            # Call gen_single API
            result = client.predict(
                emo_control_method=emo_mode_str,
                prompt=handle_file(ref_audio_path),
                text=text,
                emo_ref_path=handle_file(emo_audio_path),
                emo_weight=emo_weight,
                max_text_tokens_per_segment=self.max_tokens_per_segment,
                api_name="/gen_single"
            )

            # Extract audio path from result
            if isinstance(result, dict):
                audio_path = result.get("value")
            else:
                audio_path = result

            if not audio_path:
                return TTSResult(
                    success=False,
                    audio_path=None,
                    duration=0,
                    error="No audio path in response"
                )

            # Copy to output path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_path, output_path)

            # Get duration
            duration = self._get_audio_duration_sync(output_path)

            logger.debug(f"IndexTTS synthesized: {output_path} ({duration:.2f}s)")

            return TTSResult(
                success=True,
                audio_path=output_path,
                duration=duration,
            )

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error=str(e)
            )

    def _get_audio_duration_sync(self, audio_path: Path) -> float:
        """Get audio duration using FFprobe (synchronous)"""
        import subprocess
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
            return 0.0

    async def get_available_voices(self, language: str = None) -> List[dict]:
        """
        IndexTTS uses reference audio for voice cloning,
        not preset voices. Returns info about the current setup.
        """
        return [
            {"name": "custom", "display_name": "Custom (Reference Audio)", "gender": "unknown", "locale": "any"},
        ]

    async def get_emotion_modes(self) -> Dict[str, str]:
        """Get available emotion control modes"""
        return self.EMO_MODES

    async def synthesize_segments(
        self,
        segments: List[TTSSegment],
        output_dir: Path,
        voice: str = None,
        rate: str = "+0%",
        emo_alpha: float = 1.0,
        emotion_preset: str = None,
        concurrent: int = 3,
    ) -> List[TTSSegment]:
        """
        Synthesize multiple segments with optional concurrency

        Args:
            segments: List of TTSSegment with text and timing
            output_dir: Directory to save audio files
            voice: Voice preset to use
            rate: Speech rate (post-processing)
            emo_alpha: Emotion strength
            emotion_preset: Emotion preset (v2.5)
            concurrent: Number of concurrent requests

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
                emo_alpha=emo_alpha,
                emotion_preset=emotion_preset,
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
        semaphore = asyncio.Semaphore(concurrent)

        async def limited_synthesize(i: int, seg: TTSSegment) -> TTSSegment:
            async with semaphore:
                return await synthesize_one(i, seg)

        tasks = [limited_synthesize(i, seg) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def set_reference_audio(self, audio_path: Path) -> bool:
        """
        Set global reference audio for voice cloning

        Args:
            audio_path: Path to reference audio file (3-10 seconds recommended)

        Returns:
            True if successful
        """
        if not audio_path.exists():
            logger.error(f"Reference audio not found: {audio_path}")
            return False

        self.ref_audio_path = audio_path
        logger.info(f"Set reference audio: {audio_path}")
        return True

    async def set_emotion_audio(self, audio_path: Path) -> bool:
        """
        Set emotion audio prompt for expressive synthesis

        Args:
            audio_path: Path to emotion audio file

        Returns:
            True if successful
        """
        if not audio_path.exists():
            logger.error(f"Emotion audio not found: {audio_path}")
            return False

        self.emo_audio_path = audio_path
        self.emo_mode = "from_ref_audio"
        logger.info(f"Set emotion audio: {audio_path}")
        return True

    async def close(self):
        """Close resources"""
        if self._executor:
            self._executor.shutdown(wait=False)
        self._client = None

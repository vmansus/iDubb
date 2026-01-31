"""
Qwen3-TTS Engine - Alibaba's Open-Source TTS with Voice Cloning
Supports: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
"""
import asyncio
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from loguru import logger

from .base import BaseTTSEngine, TTSResult, TTSSegment

# Lazy import to avoid loading heavy dependencies at startup
_model = None
_model_name = None


def _get_model(model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"):
    """Lazy load the Qwen3-TTS model"""
    global _model, _model_name

    if _model is None or _model_name != model_name:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            # Determine device and dtype
            if torch.cuda.is_available():
                device_map = "cuda:0"
                dtype = torch.bfloat16
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_map = "mps"
                dtype = torch.float16  # MPS doesn't support bfloat16
            else:
                device_map = "cpu"
                dtype = torch.float32

            # Select best attention implementation
            # - CUDA: flash_attention_2 (fastest, requires flash-attn package)
            # - MPS/CPU: sdpa (PyTorch 2.0+ native optimized attention)
            if torch.cuda.is_available():
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                except ImportError:
                    attn_impl = "sdpa"  # Fallback to SDPA
            else:
                # MPS and CPU use SDPA (faster than eager)
                attn_impl = "sdpa"

            logger.info(f"Loading Qwen3-TTS model: {model_name} on {device_map} (attention: {attn_impl})")
            _model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
            _model_name = model_name
            logger.info(f"Qwen3-TTS model loaded successfully")

        except ImportError as e:
            logger.error(f"qwen-tts package not installed. Run: pip install qwen-tts")
            raise ImportError("qwen-tts package required. Install with: pip install qwen-tts") from e
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS model: {e}")
            raise

    return _model


class Qwen3TTSEngine(BaseTTSEngine):
    """
    Qwen3-TTS Engine - High-quality neural TTS with voice cloning

    Features:
    - 10 language support
    - 9 premium built-in voices
    - Voice cloning from 3 seconds of audio
    - Natural language voice design
    - Streaming support
    """

    # Built-in premium voices
    VOICES = {
        # Chinese/English bilingual voices
        "vivian": {"name": "Vivian", "gender": "female", "languages": ["zh", "en"], "style": "warm, professional"},
        "serena": {"name": "Serena", "gender": "female", "languages": ["zh", "en"], "style": "gentle, soothing"},
        "dylan": {"name": "Dylan", "gender": "male", "languages": ["zh", "en"], "style": "young, energetic"},
        "eric": {"name": "Eric", "gender": "male", "languages": ["zh", "en"], "style": "professional, mature"},
        "ryan": {"name": "Ryan", "gender": "male", "languages": ["zh", "en"], "style": "friendly, casual"},
        "aiden": {"name": "Aiden", "gender": "male", "languages": ["zh", "en"], "style": "youthful, bright"},
        "uncle_fu": {"name": "Uncle_Fu", "gender": "male", "languages": ["zh"], "style": "warm, elder"},
        # Japanese voice
        "ono_anna": {"name": "Ono_Anna", "gender": "female", "languages": ["ja", "en"], "style": "cute, anime"},
        # Korean voice
        "sohee": {"name": "Sohee", "gender": "female", "languages": ["ko", "en"], "style": "sweet, youthful"},
    }

    # Language code mapping
    LANGUAGE_MAP = {
        "zh": "Chinese",
        "zh-CN": "Chinese",
        "zh-TW": "Chinese",
        "en": "English",
        "en-US": "English",
        "en-GB": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }

    def __init__(
        self,
        default_voice: str = "vivian",
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        default_language: str = "Chinese",
    ):
        """
        Initialize Qwen3-TTS Engine

        Args:
            default_voice: Default voice name (vivian, serena, dylan, etc.)
            model_name: HuggingFace model name
            default_language: Default language for synthesis
        """
        self.default_voice = default_voice
        self.model_name = model_name
        self.default_language = default_language
        self._model = None  # Lazy loaded
        # Voice cloning reference audio
        self.ref_audio_path: Optional[Path] = None
        self.ref_text: Optional[str] = None
        self._clone_model = None  # Separate model for voice cloning
        logger.info(f"Initialized Qwen3-TTS with voice: {default_voice}, language: {default_language}")

    def _ensure_model(self):
        """Ensure model is loaded"""
        if self._model is None:
            self._model = _get_model(self.model_name)
        return self._model

    def _get_language(self, lang_code: Optional[str]) -> str:
        """Convert language code to Qwen3-TTS language name"""
        if not lang_code:
            return self.default_language
        return self.LANGUAGE_MAP.get(lang_code, self.default_language)

    async def set_reference_audio(
        self,
        audio_path: Path,
        transcript: Optional[str] = None
    ) -> bool:
        """
        Set reference audio for voice cloning

        Args:
            audio_path: Path to reference audio file (3+ seconds recommended)
            transcript: Transcript of the reference audio (optional, will use Whisper if not provided)

        Returns:
            True if successful
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Reference audio not found: {audio_path}")
            return False

        # Preprocess audio for voice cloning (convert to WAV, correct sample rate, clip duration)
        try:
            processed_path = await self._preprocess_reference_audio(audio_path)
            self.ref_audio_path = processed_path
        except Exception as e:
            logger.error(f"Failed to preprocess reference audio: {e}")
            return False

        # If transcript not provided, try to transcribe with Whisper
        if transcript:
            self.ref_text = transcript
        else:
            # Try to get transcript using Whisper
            try:
                import whisper
                logger.info(f"Transcribing reference audio: {self.ref_audio_path}")
                model = whisper.load_model("base")
                result = model.transcribe(str(self.ref_audio_path))
                self.ref_text = result.get("text", "").strip()
                if self.ref_text:
                    logger.info(f"Reference audio transcribed: '{self.ref_text[:50]}...'")
                else:
                    # Use a generic placeholder
                    self.ref_text = "这是一段参考音频样本。"
                    logger.warning("Could not transcribe reference audio, using placeholder text")
            except Exception as e:
                logger.warning(f"Failed to transcribe reference audio: {e}, using placeholder")
                self.ref_text = "这是一段参考音频样本。"

        logger.info(f"Set reference audio for voice cloning: {self.ref_audio_path}")
        return True

    async def _preprocess_reference_audio(self, audio_path: Path) -> Path:
        """
        Preprocess reference audio for voice cloning:
        - Convert to WAV format
        - Resample to 16kHz
        - Convert to mono
        - Clip to 3-10 seconds (optimal for voice cloning)

        Returns:
            Path to processed audio file
        """
        import tempfile
        import subprocess

        # Create temp file for processed audio
        temp_dir = Path(tempfile.gettempdir()) / "qwen3_tts_ref"
        temp_dir.mkdir(parents=True, exist_ok=True)
        processed_path = temp_dir / f"ref_{audio_path.stem}.wav"

        # Use FFmpeg to convert and preprocess
        # - Extract first 8 seconds (optimal for voice cloning)
        # - Convert to 16kHz mono WAV
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-t", "8",  # Limit to 8 seconds
            "-ss", "0",  # Start from beginning
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",  # Mono
            "-acodec", "pcm_s16le",  # 16-bit PCM
            str(processed_path)
        ]

        logger.info(f"Preprocessing reference audio: {audio_path} -> {processed_path}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg preprocessing failed: {stderr.decode()}")
            raise Exception(f"Audio preprocessing failed: {stderr.decode()}")

        if not processed_path.exists():
            raise Exception("Processed audio file not created")

        logger.info(f"Reference audio preprocessed successfully: {processed_path}")
        return processed_path

    def _ensure_clone_model(self):
        """Ensure voice cloning model is loaded (uses Base model)"""
        if self._clone_model is None:
            self._clone_model = _get_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        return self._clone_model

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str = None,
        rate: str = "+0%",
        language: str = None,
        instruct: str = None,
    ) -> TTSResult:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            output_path: Output audio file path (.wav)
            voice: Voice name (vivian, serena, dylan, etc.)
            rate: Speech rate (not directly supported, use instruct instead)
            language: Language code (zh, en, ja, ko, de, fr, ru, pt, es, it)
            instruct: Style instruction (e.g., "speak slowly and calmly")

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
            voice_info = self.VOICES.get(voice.lower(), self.VOICES.get(self.default_voice))
            speaker = voice_info["name"] if voice_info else "Vivian"

            language_name = self._get_language(language)

            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure .wav extension
            if output_path.suffix.lower() != '.wav':
                output_path = output_path.with_suffix('.wav')

            import soundfile as sf
            loop = asyncio.get_event_loop()

            # Check if voice cloning is enabled (reference audio is set)
            if self.ref_audio_path and self.ref_audio_path.exists() and self.ref_text:
                logger.debug(f"Using voice cloning with reference: {self.ref_audio_path}")
                # Use voice cloning model (Base model)
                clone_model = self._ensure_clone_model()
                wavs, sr = await loop.run_in_executor(
                    None,
                    lambda: clone_model.generate_voice_clone(
                        text=text,
                        language=language_name,
                        ref_audio=str(self.ref_audio_path),
                        ref_text=self.ref_text
                    )
                )
            else:
                # Run standard synthesis in thread pool to avoid blocking
                wavs, sr = await loop.run_in_executor(
                    None,
                    self._synthesize_sync,
                    text,
                    language_name,
                    speaker,
                    instruct
                )

            # Save audio
            sf.write(str(output_path), wavs[0], sr)

            # Calculate duration
            duration = len(wavs[0]) / sr

            clone_mode = " (voice clone)" if self.ref_audio_path else ""
            logger.debug(f"Qwen3-TTS synthesized{clone_mode}: {output_path} ({duration:.2f}s)")

            return TTSResult(
                success=True,
                audio_path=output_path,
                duration=duration,
            )

        except Exception as e:
            logger.error(f"Qwen3-TTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                audio_path=None,
                duration=0,
                error=str(e)
            )

    def _synthesize_sync(
        self,
        text: str,
        language: str,
        speaker: str,
        instruct: Optional[str]
    ):
        """Synchronous synthesis (called in executor)"""
        model = self._ensure_model()
        return model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct or ""
        )

    async def synthesize_with_clone(
        self,
        text: str,
        output_path: Path,
        ref_audio: Union[str, Path],
        ref_text: str,
        language: str = None,
    ) -> TTSResult:
        """
        Synthesize speech using voice cloning

        Args:
            text: Text to synthesize
            output_path: Output audio file path (.wav)
            ref_audio: Reference audio file for voice cloning (3+ seconds)
            ref_text: Transcription of the reference audio
            language: Language code

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

            language_name = self._get_language(language)

            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix.lower() != '.wav':
                output_path = output_path.with_suffix('.wav')

            # Need the voice cloning model
            clone_model = _get_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

            loop = asyncio.get_event_loop()
            wavs, sr = await loop.run_in_executor(
                None,
                lambda: clone_model.generate_voice_clone(
                    text=text,
                    language=language_name,
                    ref_audio=str(ref_audio),
                    ref_text=ref_text
                )
            )

            import soundfile as sf
            sf.write(str(output_path), wavs[0], sr)

            duration = len(wavs[0]) / sr

            logger.debug(f"Qwen3-TTS voice clone: {output_path} ({duration:.2f}s)")

            return TTSResult(
                success=True,
                audio_path=output_path,
                duration=duration,
            )

        except Exception as e:
            logger.error(f"Qwen3-TTS voice cloning failed: {e}")
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
            language: Filter by language code (e.g., 'zh', 'en', 'ja')

        Returns:
            List of voice info dicts
        """
        voices = []
        for voice_id, info in self.VOICES.items():
            # Filter by language if specified
            if language:
                lang_prefix = language.split('-')[0].lower()
                if lang_prefix not in info["languages"]:
                    continue

            # Convert languages list to locale string for API compatibility
            locale = ", ".join(info["languages"])
            voices.append({
                "name": voice_id,
                "display_name": f"{info['name']} ({info['style']})",
                "gender": info["gender"],
                "locale": locale,
            })

        return voices

    async def synthesize_segments(
        self,
        segments: List[TTSSegment],
        output_dir: Path,
        voice: str = None,
        language: str = None,
    ) -> List[TTSSegment]:
        """
        Synthesize multiple segments to separate audio files

        Args:
            segments: List of TTSSegment with text and timing
            output_dir: Directory to save audio files
            voice: Voice to use
            language: Language code

        Returns:
            List of TTSSegment with audio_path filled in
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        voice = voice or self.default_voice

        results = []
        for i, seg in enumerate(segments):
            output_path = output_dir / f"segment_{i:04d}.wav"

            result = await self.synthesize(
                seg.text,
                output_path,
                voice=voice,
                language=language
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
                results.append(seg)  # Keep original without audio

        return results

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages"""
        return list(cls.LANGUAGE_MAP.keys())

    @classmethod
    def get_voice_info(cls, voice_id: str) -> Optional[dict]:
        """Get info for a specific voice"""
        return cls.VOICES.get(voice_id.lower())

    async def check_health(self) -> bool:
        """
        Check if Qwen3-TTS server is available
        Since Qwen3-TTS can run locally or via server, check both
        """
        import aiohttp

        # Check if running as server on port 50001
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:50001/health", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass

        # Check if model can be loaded locally
        try:
            import importlib.util
            if importlib.util.find_spec("qwen_tts") is not None:
                return True
        except Exception:
            pass

        return False

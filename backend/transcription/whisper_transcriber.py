"""
Local Whisper Transcription Module
"""
import asyncio
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import torch
import whisper
from loguru import logger


@dataclass
class TranscriptSegment:
    """A single transcription segment with timing"""
    start: float  # seconds
    end: float  # seconds
    text: str


@dataclass
class TranscriptionResult:
    """Full transcription result"""
    success: bool
    language: str
    segments: List[TranscriptSegment]
    full_text: str
    error: Optional[str] = None
    cancelled: bool = False


class WhisperTranscriber:
    """Local Whisper-based transcription"""

    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        Initialize Whisper transcriber

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to run on (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.device = self._validate_device(device)
        self._model = None
        logger.info(f"Initializing Whisper transcriber with model: {model_name}, device: {self.device}")

    def _validate_device(self, requested_device: str) -> str:
        """Validate and return the best available device"""
        requested_device = requested_device.lower()

        # Auto-detect best device
        if requested_device == "auto":
            device = self.get_best_device()
            logger.info(f"Auto-detected device: {device}")
            return device

        if requested_device == "cuda":
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"

        elif requested_device == "mps":
            # MPS can have issues with Whisper (NaN values) but user explicitly requested it
            if torch.backends.mps.is_available():
                logger.warning("MPS requested - note: MPS may have stability issues with Whisper (NaN values). If you experience errors, try CPU.")
                return "mps"
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"

        return "cpu"

    @classmethod
    def get_best_device(cls) -> str:
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        # Note: MPS disabled due to NaN issues with Whisper
        # elif torch.backends.mps.is_available():
        #     return "mps"
        return "cpu"

    @property
    def model(self):
        """Lazy load the Whisper model"""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Whisper model loaded successfully")
        return self._model

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        cancel_check: Optional[callable] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file

        Args:
            audio_path: Path to audio file
            language: Source language code (e.g., 'en', 'zh') or None for auto-detect
            task: 'transcribe' or 'translate' (translate to English)
            cancel_check: Optional callable that returns True if cancellation requested

        Returns:
            TranscriptionResult with segments and full text
        """
        try:
            # Check for cancellation before starting
            if cancel_check and cancel_check():
                logger.info("Transcription cancelled before starting")
                return TranscriptionResult(
                    success=False,
                    language="",
                    segments=[],
                    full_text="",
                    error="用户取消",
                    cancelled=True
                )

            if not audio_path.exists():
                return TranscriptionResult(
                    success=False,
                    language="",
                    segments=[],
                    full_text="",
                    error=f"Audio file not found: {audio_path}"
                )

            logger.info(f"Transcribing: {audio_path}")

            # Run transcription in executor to not block
            def do_transcribe():
                # Set torch to use deterministic algorithms for stability
                torch.use_deterministic_algorithms(False)

                # Adjust no_speech_threshold for Asian languages (they have more pauses)
                is_asian_language = language in ('ja', 'zh', 'ko', 'vi', 'th')
                no_speech_thresh = 0.8 if is_asian_language else 0.6

                if is_asian_language:
                    logger.info(f"Using optimized settings for {language}: no_speech_threshold={no_speech_thresh}")

                # Ensure we're using float32 for CPU stability
                with torch.no_grad():
                    options = {
                        "task": task,
                        "verbose": False,
                        "fp16": False,  # Use fp32 to avoid NaN issues
                        "temperature": 0,  # Disable sampling to avoid Categorical distribution issues
                        "best_of": 1,  # Don't use beam search to avoid distribution errors
                        "beam_size": 1,  # Single beam to avoid Categorical issues
                        "patience": None,
                        "compression_ratio_threshold": 2.4,
                        "logprob_threshold": -1.0,
                        "no_speech_threshold": no_speech_thresh,
                        "condition_on_previous_text": True,
                    }
                    if language:
                        options["language"] = language

                    return self.model.transcribe(str(audio_path), **options)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, do_transcribe)

            # Parse segments
            segments = []
            for seg in result.get("segments", []):
                segments.append(TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip()
                ))

            detected_language = result.get("language", "unknown")
            full_text = result.get("text", "").strip()

            # Resegment long subtitles for better readability
            original_count = len(segments)
            segments = self._resegment_by_length(segments, max_chars=80, language=detected_language)
            if len(segments) != original_count:
                logger.info(f"Resegmented: {original_count} -> {len(segments)} segments for better readability")
                full_text = " ".join(seg.text for seg in segments)

            logger.info(f"Transcription complete. Language: {detected_language}, Segments: {len(segments)}")

            return TranscriptionResult(
                success=True,
                language=detected_language,
                segments=segments,
                full_text=full_text,
            )

        except Exception as e:
            logger.error(f"Transcription failed with primary method: {e}")
            # Try fallback with even more conservative settings
            try:
                logger.info("Attempting fallback transcription with conservative settings...")
                result = await self._transcribe_fallback(audio_path, language, task)
                if result.success:
                    return result
            except Exception as fallback_error:
                logger.error(f"Fallback transcription also failed: {fallback_error}")

            return TranscriptionResult(
                success=False,
                language="",
                segments=[],
                full_text="",
                error=str(e)
            )

    async def _transcribe_fallback(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> TranscriptionResult:
        """Fallback transcription with most conservative settings"""
        def do_transcribe_fallback():
            with torch.no_grad():
                # Most conservative settings to avoid any distribution errors
                options = {
                    "task": task,
                    "verbose": False,
                    "fp16": False,
                    "temperature": 0,  # Greedy decoding, no sampling
                    "best_of": 1,
                    "beam_size": 1,
                    "without_timestamps": False,
                    "word_timestamps": False,
                }
                if language:
                    options["language"] = language
                else:
                    # Force English detection if auto-detect fails
                    options["language"] = "en"

                return self.model.transcribe(str(audio_path), **options)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, do_transcribe_fallback)

        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip()
            ))

        # Resegment long subtitles for better readability
        detected_language = result.get("language", "en")
        original_count = len(segments)
        segments = self._resegment_by_length(segments, max_chars=80, language=detected_language)
        if len(segments) != original_count:
            logger.info(f"Resegmented: {original_count} -> {len(segments)} segments for better readability")

        return TranscriptionResult(
            success=True,
            language=detected_language,
            segments=segments,
            full_text=result.get("text", "").strip(),
        )

    def _resegment_by_length(
        self,
        segments: List[TranscriptSegment],
        max_chars: int = 80,
        language: str = "en"
    ) -> List[TranscriptSegment]:
        """
        Resegment long subtitles for better readability.
        
        Split at sentence boundaries first, then clause boundaries.
        Uses proportional timing since we don't have word-level timestamps.
        """
        result = []
        
        for seg in segments:
            text = seg.text.strip()
            
            # Short enough - keep as is
            if len(text) <= max_chars:
                result.append(seg)
                continue
            
            # Split long segment
            split_segs = self._split_segment_by_sentences(seg, max_chars, language)
            result.extend(split_segs)
        
        return result
    
    def _split_segment_by_sentences(
        self,
        seg: TranscriptSegment,
        max_chars: int,
        language: str
    ) -> List[TranscriptSegment]:
        """Split a long segment at sentence/clause boundaries with proportional timing."""
        text = seg.text.strip()
        total_chars = len(text)
        duration = seg.end - seg.start
        
        # Punctuation marks
        is_cjk = language in ('zh', 'ja', 'ko', 'yue')
        if is_cjk:
            sentence_ends = ['。', '！', '？', '.', '!', '?']
            clause_seps = ['，', '、', '；', '：', ',', ';', ':']
        else:
            sentence_ends = ['.', '!', '?']
            clause_seps = [',', ';', ':', ' ']
        
        result = []
        start_idx = 0
        
        while start_idx < total_chars:
            remaining = text[start_idx:]
            
            if len(remaining) <= max_chars:
                # Last piece
                seg_start = seg.start + duration * start_idx / total_chars
                result.append(TranscriptSegment(
                    start=seg_start,
                    end=seg.end,
                    text=remaining.strip()
                ))
                break
            
            # Find best split point
            search_text = remaining[:max_chars]
            best_split = -1
            
            # First, look for sentence-ending punctuation
            for punct in sentence_ends:
                pos = search_text.rfind(punct)
                if pos >= 20:  # At least 20 chars
                    best_split = max(best_split, pos + 1)
            
            # If no sentence end, look for clause separator
            if best_split < max_chars // 2:
                for punct in clause_seps:
                    pos = search_text.rfind(punct)
                    if pos >= max_chars // 3:
                        best_split = max(best_split, pos + 1)
            
            # Fallback: hard split at max_chars
            if best_split <= 0:
                best_split = max_chars
            
            chunk = remaining[:best_split].strip()
            
            # Calculate proportional timing
            seg_start = seg.start + duration * start_idx / total_chars
            char_end = start_idx + best_split
            seg_end = seg.start + duration * char_end / total_chars
            
            # Ensure minimum duration of 1 second
            if seg_end - seg_start < 1.0:
                seg_end = min(seg_start + 1.5, seg.end)
            
            if chunk:
                result.append(TranscriptSegment(
                    start=seg_start,
                    end=seg_end,
                    text=chunk
                ))
            
            start_idx += best_split
        
        return result if result else [seg]

    def _get_segment_value(self, seg, key: str):
        """Get value from segment, supporting both dict and object formats"""
        if isinstance(seg, dict):
            return seg.get(key, "")
        return getattr(seg, key, "")

    def generate_srt(self, segments: List) -> str:
        """Generate SRT subtitle format from segments (supports dict or object)"""
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start_time = self._format_srt_time(self._get_segment_value(seg, "start") or 0)
            end_time = self._format_srt_time(self._get_segment_value(seg, "end") or 0)
            text = self._get_segment_value(seg, "text") or ""
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between entries

        return "\n".join(srt_lines)

    def generate_vtt(self, segments: List) -> str:
        """Generate WebVTT subtitle format from segments (supports dict or object)"""
        vtt_lines = ["WEBVTT", ""]
        for seg in segments:
            start_time = self._format_vtt_time(self._get_segment_value(seg, "start") or 0)
            end_time = self._format_vtt_time(self._get_segment_value(seg, "end") or 0)
            text = self._get_segment_value(seg, "text") or ""
            vtt_lines.append(f"{start_time} --> {end_time}")
            vtt_lines.append(text)
            vtt_lines.append("")

        return "\n".join(vtt_lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_time(self, seconds: float) -> str:
        """Format seconds to WebVTT time format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    async def save_subtitles(
        self,
        segments: List[TranscriptSegment],
        output_path: Path,
        format: str = "srt"
    ) -> bool:
        """Save subtitles to file"""
        try:
            if format.lower() == "srt":
                content = self.generate_srt(segments)
            elif format.lower() == "vtt":
                content = self.generate_vtt(segments)
            else:
                logger.error(f"Unsupported subtitle format: {format}")
                return False

            output_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved subtitles to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save subtitles: {e}")
            return False

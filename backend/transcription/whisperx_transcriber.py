"""
WhisperX Transcription Module

Enhanced transcription with word-level alignment using WhisperX.
Provides precise word timing for better subtitle segmentation.

Based on VideoLingo's approach to word-level alignment.
"""
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from loguru import logger

# Standard imports
import torch


@dataclass
class WordSegment:
    """A single word with precise timing"""
    word: str
    start: float  # seconds
    end: float    # seconds
    score: float = 1.0  # confidence score

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptSegmentWithWords:
    """A transcription segment with word-level timing"""
    start: float  # seconds
    end: float    # seconds
    text: str
    words: List[WordSegment] = field(default_factory=list)
    speaker: Optional[str] = None  # For diarization

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.words)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [{"word": w.word, "start": w.start, "end": w.end, "score": w.score} for w in self.words],
            "speaker": self.speaker,
        }


@dataclass
class WhisperXResult:
    """Full WhisperX transcription result"""
    success: bool
    language: str
    segments: List[TranscriptSegmentWithWords]
    full_text: str
    word_count: int = 0
    has_word_alignment: bool = False
    has_diarization: bool = False
    error: Optional[str] = None

    def to_basic_segments(self) -> List["TranscriptSegment"]:
        """Convert to basic segments without word timing (for compatibility)"""
        from .whisper_transcriber import TranscriptSegment
        return [
            TranscriptSegment(start=s.start, end=s.end, text=s.text)
            for s in self.segments
        ]


class WhisperXTranscriber:
    """
    WhisperX-based transcription with word-level alignment.

    Features:
    - Word-level timestamps via forced alignment
    - Better VAD (Voice Activity Detection)
    - Optional speaker diarization
    - Improved accuracy over standard Whisper

    Requires: whisperx package
    Install: pip install whisperx

    Note: For GPU acceleration, ensure CUDA is properly configured.
    """

    # Supported alignment languages
    ALIGNMENT_LANGUAGES = {
        "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
        "ar", "hi", "nl", "pl", "tr", "vi", "th", "id", "ms", "tl",
    }

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        enable_diarization: bool = False,
        hf_token: Optional[str] = None
    ):
        """
        Initialize WhisperX transcriber.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to run on (auto, cpu, cuda)
            compute_type: Compute type (float16, float32, int8)
            enable_diarization: Enable speaker diarization (requires HF token)
            hf_token: HuggingFace token for diarization model access
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.compute_type = self._get_compute_type(compute_type)
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token

        self._model = None
        self._align_model = None
        self._diarize_model = None
        self._whisperx_available = self._check_whisperx()

        logger.info(
            f"Initialized WhisperXTranscriber: model={model_name}, "
            f"device={self.device}, compute_type={self.compute_type}, "
            f"whisperx_available={self._whisperx_available}"
        )

    def _check_whisperx(self) -> bool:
        """Check if WhisperX is available"""
        try:
            import whisperx
            return True
        except ImportError:
            logger.warning(
                "WhisperX not installed. Install with: pip install whisperx\n"
                "Falling back to standard Whisper when transcribing."
            )
            return False

    def _get_device(self, requested: str) -> str:
        """Get the best available device"""
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"

        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"

        return requested

    def _get_compute_type(self, requested: str) -> str:
        """Get appropriate compute type for device"""
        if self.device == "cpu":
            return "float32"  # CPU doesn't support float16 well
        return requested

    @property
    def model(self):
        """Lazy load the WhisperX model"""
        if self._model is None:
            if not self._whisperx_available:
                raise RuntimeError("WhisperX not installed. Install with: pip install whisperx")

            import whisperx

            logger.info(f"Loading WhisperX model: {self.model_name}")
            self._model = whisperx.load_model(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("WhisperX model loaded successfully")

        return self._model

    def _load_align_model(self, language: str):
        """Load alignment model for a specific language"""
        if not self._whisperx_available:
            return None, None

        import whisperx

        # Check if language supports alignment
        lang_code = language[:2].lower() if language else "en"
        if lang_code not in self.ALIGNMENT_LANGUAGES:
            logger.warning(f"Language '{lang_code}' may not have alignment model, using best effort")

        try:
            logger.info(f"Loading alignment model for: {lang_code}")
            model_a, metadata = whisperx.load_align_model(
                language_code=lang_code,
                device=self.device
            )
            return model_a, metadata
        except Exception as e:
            logger.warning(f"Failed to load alignment model: {e}")
            return None, None

    def _load_diarize_model(self):
        """Load speaker diarization model"""
        if not self._whisperx_available or not self.enable_diarization:
            return None

        if not self.hf_token:
            logger.warning("Diarization requires HuggingFace token (HF_TOKEN env var)")
            return None

        try:
            import whisperx
            from pyannote.audio import Pipeline

            logger.info("Loading diarization model...")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
            return diarize_model
        except Exception as e:
            logger.warning(f"Failed to load diarization model: {e}")
            return None

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        align_words: bool = True,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        cancel_check: Optional[callable] = None
    ) -> WhisperXResult:
        """
        Transcribe audio with word-level alignment.

        Args:
            audio_path: Path to audio file
            language: Source language code or None for auto-detect
            task: 'transcribe' or 'translate'
            align_words: Whether to perform word-level alignment
            min_speakers: Minimum speakers for diarization
            max_speakers: Maximum speakers for diarization

        Returns:
            WhisperXResult with word-level segments
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            return WhisperXResult(
                success=False,
                language="",
                segments=[],
                full_text="",
                error=f"Audio file not found: {audio_path}"
            )

        # Fall back to standard Whisper if WhisperX not available
        if not self._whisperx_available:
            return await self._fallback_transcribe(audio_path, language, task)

        try:
            logger.info(f"Transcribing with WhisperX: {audio_path}")

            # Run transcription in executor
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._do_transcribe(
                    audio_path, language, task, align_words,
                    min_speakers, max_speakers, cancel_check
                )
            )

            return result

        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            # Try fallback
            logger.info("Attempting fallback to standard Whisper...")
            return await self._fallback_transcribe(audio_path, language, task)

    def _do_transcribe(
        self,
        audio_path: Path,
        language: Optional[str],
        task: str,
        align_words: bool,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        cancel_check: Optional[callable] = None
    ) -> WhisperXResult:
        """Synchronous transcription worker"""
        import whisperx

        # Load audio
        audio = whisperx.load_audio(str(audio_path))

        # Transcribe
        transcribe_options = {"task": task}
        if language:
            transcribe_options["language"] = language

        result = self.model.transcribe(audio, batch_size=16, **transcribe_options)
        detected_language = result.get("language", language or "en")

        segments = result.get("segments", [])
        has_word_alignment = False
        has_diarization = False

        # Word-level alignment
        if align_words and segments:
            try:
                model_a, metadata = self._load_align_model(detected_language)
                if model_a is not None:
                    logger.info("Performing word-level alignment...")
                    result = whisperx.align(
                        segments,
                        model_a,
                        metadata,
                        audio,
                        self.device,
                        return_char_alignments=False
                    )
                    segments = result.get("segments", segments)
                    has_word_alignment = True
                    logger.info("Word alignment complete")
            except Exception as e:
                logger.warning(f"Word alignment failed: {e}")

        # Speaker diarization
        if self.enable_diarization:
            try:
                diarize_model = self._load_diarize_model()
                if diarize_model is not None:
                    logger.info("Performing speaker diarization...")
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    segments = result.get("segments", segments)
                    has_diarization = True
                    logger.info("Speaker diarization complete")
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")

        # Convert to our format
        output_segments = []
        total_words = 0

        for seg in segments:
            words = []
            if "words" in seg:
                for w in seg["words"]:
                    if "word" in w and "start" in w and "end" in w:
                        words.append(WordSegment(
                            word=w["word"].strip(),
                            start=w["start"],
                            end=w["end"],
                            score=w.get("score", 1.0)
                        ))
                total_words += len(words)

            output_segments.append(TranscriptSegmentWithWords(
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=seg.get("text", "").strip(),
                words=words,
                speaker=seg.get("speaker")
            ))

        full_text = " ".join(seg.text for seg in output_segments)

        # Resegment long subtitles by character length for better readability
        if has_word_alignment:
            original_count = len(output_segments)
            output_segments = self._resegment_by_length(
                output_segments, 
                max_chars=80,  # Max ~80 chars per segment (about 2 lines)
                detected_language=detected_language
            )
            if len(output_segments) != original_count:
                logger.info(f"Resegmented: {original_count} -> {len(output_segments)} segments")
                # Recalculate total words
                total_words = sum(len(seg.words) for seg in output_segments)

        logger.info(
            f"WhisperX complete: {len(output_segments)} segments, "
            f"{total_words} words, language={detected_language}"
        )

        return WhisperXResult(
            success=True,
            language=detected_language,
            segments=output_segments,
            full_text=full_text,
            word_count=total_words,
            has_word_alignment=has_word_alignment,
            has_diarization=has_diarization
        )

    def _resegment_by_length(
        self,
        segments: List[TranscriptSegmentWithWords],
        max_chars: int = 80,
        detected_language: str = "en"
    ) -> List[TranscriptSegmentWithWords]:
        """
        Resegment transcription for better subtitle readability.
        
        Strategy:
        1. First split at sentence boundaries (. ! ? 。！？)
        2. If a sentence is still too long, split at clause boundaries (, ; : ，；：)
        3. If still too long, split at word boundaries
        4. Merge short segments if they fit together
        
        Args:
            segments: Original segments with word-level timing
            max_chars: Maximum characters per segment (default 80, ~2 lines)
            detected_language: Language code for CJK detection
            
        Returns:
            New list of segments with shorter, complete sentences
        """
        is_cjk = detected_language in ('zh', 'ja', 'ko', 'yue')
        result_segments = []
        
        for seg in segments:
            text = seg.text.strip()
            
            # Short enough - keep as is
            if len(text) <= max_chars:
                result_segments.append(seg)
                continue
            
            # No word timestamps - can't split precisely, keep as is
            if not seg.words:
                result_segments.append(seg)
                continue
            
            # Split at sentence/clause boundaries
            new_segs = self._split_by_sentences(seg, max_chars, is_cjk)
            result_segments.extend(new_segs)
        
        # Merge short consecutive segments if they fit
        result_segments = self._merge_short_segments(result_segments, max_chars, is_cjk)
        
        return result_segments
    
    def _split_by_sentences(
        self,
        seg: TranscriptSegmentWithWords,
        max_chars: int,
        is_cjk: bool
    ) -> List[TranscriptSegmentWithWords]:
        """
        Split segment at sentence/clause boundaries using word timestamps.
        
        Priority:
        1. Sentence-ending punctuation (. ! ? 。！？)
        2. Clause-separating punctuation (, ; : ，；：、)
        3. Word boundaries (only if sentence is still too long)
        """
        if not seg.words:
            return [seg]
        
        # Define punctuation marks
        if is_cjk:
            sentence_end = {'。', '！', '？', '!', '?', '.'}
            clause_sep = {'，', '、', '；', '：', ',', ';', ':'}
        else:
            sentence_end = {'.', '!', '?'}
            clause_sep = {',', ';', ':', '-', '—'}
        
        result = []
        current_words: List[WordSegment] = []
        current_text = ""
        
        for word in seg.words:
            word_text = word.word.strip()
            
            # Build test text
            if is_cjk:
                test_text = current_text + word_text
            else:
                test_text = (current_text + " " + word_text).strip() if current_text else word_text
            
            # Check if this word ends with sentence punctuation
            ends_sentence = any(word_text.rstrip().endswith(p) for p in sentence_end)
            ends_clause = any(word_text.rstrip().endswith(p) for p in clause_sep)
            
            # Decide whether to split
            should_split = False
            
            if len(test_text) > max_chars and current_words:
                # Too long - must split
                should_split = True
            elif ends_sentence and len(current_text) >= 20:
                # Natural sentence end and we have enough content
                current_words.append(word)
                current_text = test_text
                should_split = True
                word = None  # Already added
            elif ends_clause and len(test_text) > max_chars * 0.7:
                # Clause boundary and getting long - good split point
                current_words.append(word)
                current_text = test_text
                should_split = True
                word = None
            
            if should_split and current_words:
                result.append(TranscriptSegmentWithWords(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=current_text.strip(),
                    words=current_words.copy(),
                    speaker=seg.speaker
                ))
                current_words = []
                current_text = ""
            
            # Add current word to new segment (if not already added)
            if word is not None:
                current_words.append(word)
                current_text = test_text
        
        # Don't forget the last segment
        if current_words:
            result.append(TranscriptSegmentWithWords(
                start=current_words[0].start,
                end=current_words[-1].end,
                text=current_text.strip(),
                words=current_words,
                speaker=seg.speaker
            ))
        
        return result if result else [seg]
    
    def _merge_short_segments(
        self, 
        segments: List[TranscriptSegmentWithWords], 
        max_chars: int,
        is_cjk: bool
    ) -> List[TranscriptSegmentWithWords]:
        """Merge consecutive short segments if they fit together."""
        if len(segments) <= 1:
            return segments
        
        min_chars = max_chars // 3  # Segments shorter than this should be merged
        result = []
        current = None
        
        for seg in segments:
            if current is None:
                current = TranscriptSegmentWithWords(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=list(seg.words),
                    speaker=seg.speaker
                )
                continue
            
            current_len = len(current.text)
            
            # Try to merge if current is short
            if current_len < min_chars:
                if is_cjk:
                    merged_text = current.text + seg.text
                else:
                    merged_text = current.text + " " + seg.text
                
                if len(merged_text) <= max_chars:
                    # Merge
                    current = TranscriptSegmentWithWords(
                        start=current.start,
                        end=seg.end,
                        text=merged_text.strip(),
                        words=current.words + list(seg.words),
                        speaker=seg.speaker
                    )
                    continue
            
            # Can't merge - save current and start new
            result.append(current)
            current = TranscriptSegmentWithWords(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=list(seg.words),
                speaker=seg.speaker
            )
        
        if current:
            result.append(current)
        
        return result

    async def _fallback_transcribe(
        self,
        audio_path: Path,
        language: Optional[str],
        task: str
    ) -> WhisperXResult:
        """Fallback to standard Whisper transcription"""
        from .whisper_transcriber import WhisperTranscriber

        logger.info("Using standard Whisper as fallback")
        fallback = WhisperTranscriber(model_name=self.model_name, device=self.device)
        result = await fallback.transcribe(audio_path, language, task)

        if not result.success:
            return WhisperXResult(
                success=False,
                language="",
                segments=[],
                full_text="",
                error=result.error
            )

        # Convert to our format (without word alignment)
        segments = [
            TranscriptSegmentWithWords(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=[]  # No word-level timing in fallback
            )
            for seg in result.segments
        ]

        return WhisperXResult(
            success=True,
            language=result.language,
            segments=segments,
            full_text=result.full_text,
            word_count=0,
            has_word_alignment=False,
            has_diarization=False
        )

    def generate_srt(
        self,
        segments: List[TranscriptSegmentWithWords],
        include_speaker: bool = False
    ) -> str:
        """Generate SRT subtitle format"""
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start_time = self._format_srt_time(seg.start)
            end_time = self._format_srt_time(seg.end)
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")

            text = seg.text
            if include_speaker and seg.speaker:
                text = f"[{seg.speaker}] {text}"

            srt_lines.append(text)
            srt_lines.append("")

        return "\n".join(srt_lines)

    def generate_word_level_srt(
        self,
        segments: List[TranscriptSegmentWithWords],
        words_per_segment: int = 5
    ) -> str:
        """
        Generate SRT with word-level precision.

        Groups words into segments of specified size for finer subtitle timing.
        """
        srt_lines = []
        segment_idx = 1

        for seg in segments:
            if not seg.words:
                # No word timing, use segment as-is
                start_time = self._format_srt_time(seg.start)
                end_time = self._format_srt_time(seg.end)
                srt_lines.extend([str(segment_idx), f"{start_time} --> {end_time}", seg.text, ""])
                segment_idx += 1
                continue

            # Group words into smaller segments
            word_groups = [
                seg.words[i:i + words_per_segment]
                for i in range(0, len(seg.words), words_per_segment)
            ]

            for group in word_groups:
                if not group:
                    continue

                start_time = self._format_srt_time(group[0].start)
                end_time = self._format_srt_time(group[-1].end)
                text = " ".join(w.word for w in group)

                srt_lines.extend([str(segment_idx), f"{start_time} --> {end_time}", text, ""])
                segment_idx += 1

        return "\n".join(srt_lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    async def save_subtitles(
        self,
        segments: List[TranscriptSegmentWithWords],
        output_path: Path,
        format: str = "srt",
        word_level: bool = False,
        words_per_segment: int = 5
    ) -> bool:
        """
        Save subtitles to file.

        Args:
            segments: Transcription segments
            output_path: Output file path
            format: 'srt' or 'vtt'
            word_level: Use word-level timing for finer segments
            words_per_segment: Words per subtitle segment (if word_level=True)
        """
        try:
            if word_level:
                content = self.generate_word_level_srt(segments, words_per_segment)
            else:
                content = self.generate_srt(segments)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved subtitles to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save subtitles: {e}")
            return False

    def resegment_by_words(
        self,
        segments: List[TranscriptSegmentWithWords],
        max_chars: int = 42,
        max_words: int = 8
    ) -> List[TranscriptSegmentWithWords]:
        """
        Re-segment transcription using word boundaries for better subtitle compliance.

        This is a key VideoLingo feature - using word timing to create
        natural subtitle breaks that comply with Netflix standards.

        Args:
            segments: Original segments with word timing
            max_chars: Maximum characters per subtitle (Netflix: 42)
            max_words: Maximum words per subtitle

        Returns:
            New segments with improved timing boundaries
        """
        new_segments = []

        for seg in segments:
            if not seg.words:
                # No word timing, keep as-is
                new_segments.append(seg)
                continue

            current_words = []
            current_chars = 0

            for word in seg.words:
                word_chars = len(word.word) + 1  # +1 for space

                # Check if adding this word would exceed limits
                would_exceed_chars = current_chars + word_chars > max_chars
                would_exceed_words = len(current_words) >= max_words

                if current_words and (would_exceed_chars or would_exceed_words):
                    # Create segment from current words
                    new_segments.append(TranscriptSegmentWithWords(
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        text=" ".join(w.word for w in current_words),
                        words=current_words.copy(),
                        speaker=seg.speaker
                    ))
                    current_words = []
                    current_chars = 0

                current_words.append(word)
                current_chars += word_chars

            # Add remaining words
            if current_words:
                new_segments.append(TranscriptSegmentWithWords(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=" ".join(w.word for w in current_words),
                    words=current_words,
                    speaker=seg.speaker
                ))

        logger.info(f"Re-segmented: {len(segments)} -> {len(new_segments)} segments")
        return new_segments


# Convenience function to check WhisperX availability
def is_whisperx_available() -> bool:
    """Check if WhisperX is installed and available"""
    try:
        import whisperx
        return True
    except ImportError:
        return False

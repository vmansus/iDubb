"""
Faster Whisper Transcriber

Uses faster-whisper (CTranslate2) which is significantly faster than OpenAI Whisper:
- 4-8x faster on CPU
- Supports INT8 quantization for even faster inference
- Better memory efficiency
- Works well on Apple Silicon
"""
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class TranscriptionSegment:
    """A transcription segment with timing"""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Result of transcription"""
    success: bool
    language: str
    segments: List[Dict[str, Any]]
    text: str
    error: Optional[str] = None
    cancelled: bool = False



def _get_seg_val(seg, key, default=None):
    """Get segment value, supporting both dict and object formats"""
    if isinstance(seg, dict):
        return seg.get(key, default) if default is not None else seg[key]
    return getattr(seg, key, default) if default is not None else getattr(seg, key)

class FasterWhisperTranscriber:
    """
    Transcriber using faster-whisper (CTranslate2 backend).

    Significantly faster than standard Whisper:
    - 4-8x faster on CPU
    - INT8 quantization available
    - Lower memory usage
    """

    # Model size to compute type mapping for best performance
    COMPUTE_TYPES = {
        "tiny": "int8",
        "base": "int8",
        "small": "int8",
        "medium": "int8",
        "large": "int8",  # INT8 works well on Apple Silicon
        "large-v2": "int8",
        "large-v3": "int8",
    }

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto"
    ):
        """
        Initialize faster-whisper transcriber.

        Args:
            model_name: Model size (tiny, base, small, medium, large-v3)
            device: Device (auto, cpu, cuda)
            compute_type: Compute type (auto, int8, float16, float32)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.compute_type = compute_type if compute_type != "auto" else self.COMPUTE_TYPES.get(model_name, "int8")
        self._model = None

        logger.info(
            f"Initializing FasterWhisper: model={model_name}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )

    def _get_device(self, requested: str) -> str:
        """Get best available device for faster-whisper (CTranslate2)"""
        # CTranslate2 only supports CPU and CUDA, not MPS
        if requested == "mps":
            logger.info("MPS not supported by faster-whisper, using CPU (still fast with INT8)")
            return "cpu"

        if requested not in ("auto", "cpu", "cuda"):
            return "cpu"

        if requested == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"

        return requested

    @property
    def model(self):
        """Lazy load the model"""
        if self._model is None:
            from faster_whisper import WhisperModel
            import time

            logger.info(f"Loading faster-whisper model: {self.model_name} (this may take a moment...)")
            start_time = time.time()

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=8,  # Use more CPU threads on M4 Pro
                num_workers=2,
            )

            load_time = time.time() - start_time
            logger.info(f"faster-whisper model loaded in {load_time:.1f}s")

        return self._model

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        cancel_check: Optional[callable] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            task: 'transcribe' or 'translate'
            cancel_check: Optional callable that returns True if cancellation requested

        Returns:
            TranscriptionResult with segments
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing with faster-whisper: {audio_path}")

        # Run in thread pool to not block async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_path,
            language,
            task,
            cancel_check
        )

        return result

    def _transcribe_sync(
        self,
        audio_path: Path,
        language: Optional[str],
        task: str,
        cancel_check: Optional[callable] = None
    ) -> TranscriptionResult:
        """Synchronous transcription with cancellation support"""

        logger.info(f"Starting faster-whisper transcription: {audio_path.name}")

        # Check for cancellation before starting
        if cancel_check and cancel_check():
            logger.info("Transcription cancelled before starting")
            return TranscriptionResult(
                success=False,
                language="",
                segments=[],
                text="",
                error="用户取消",
                cancelled=True
            )

        # Adjust parameters based on language
        # Japanese, Chinese, Korean tend to have more natural pauses and softer speech
        is_asian_language = language in ('ja', 'zh', 'ko', 'vi', 'th')

        # Use less aggressive filtering for Asian languages
        no_speech_thresh = 0.8 if is_asian_language else 0.6
        use_vad = not is_asian_language  # Disable VAD for Asian languages as it can miss speech

        if is_asian_language:
            logger.info(f"Using optimized settings for {language}: no_speech_threshold={no_speech_thresh}, vad_filter={use_vad}")

        # Transcribe with faster-whisper
        segments_generator, info = self.model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            beam_size=5,
            best_of=5,
            patience=1.0,
            length_penalty=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=no_speech_thresh,
            condition_on_previous_text=True,
            initial_prompt=None,
            word_timestamps=True,  # Enable word-level timestamps for better segmentation
            vad_filter=use_vad,  # Disable VAD for Asian languages
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,  # Add padding around detected speech
            ) if use_vad else None,
        )

        duration = info.duration
        logger.info(f"Audio duration: {duration:.1f}s ({duration/60:.1f} min), detected language: {info.language}")
        logger.info(f"Transcription config: model={self.model_name}, device={self.device}, compute_type={self.compute_type}")

        # Estimate processing time based on model and device
        if self.device == "cpu":
            # Rough estimates for CPU with INT8
            speed_factors = {"tiny": 15, "base": 10, "small": 5, "medium": 2, "large": 0.8, "large-v2": 0.8, "large-v3": 0.8}
            speed = speed_factors.get(self.model_name, 1)
            est_time = duration / speed / 60
            logger.info(f"Estimated time on CPU: ~{est_time:.0f} min (large models are slow on CPU, consider 'base' or 'small')")

        logger.info("Processing segments (progress updates every 5% or 30 seconds)...")

        # Convert generator to list and extract segments with progress
        segments = []
        full_text = []
        last_progress = 0
        last_log_time = __import__('time').time()
        segment_count = 0
        cancelled = False

        for segment in segments_generator:
            # Check for cancellation after each segment
            if cancel_check and cancel_check():
                logger.info(f"Transcription cancelled at segment {segment_count} ({segment.end:.1f}s/{duration:.1f}s)")
                cancelled = True
                break

            segment_count += 1
            # Extract word-level timestamps if available
            words = []
            if hasattr(segment, 'words') and segment.words:
                for w in segment.words:
                    if hasattr(w, 'word') and hasattr(w, 'start') and hasattr(w, 'end'):
                        words.append({
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                        })
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": words,
            })
            full_text.append(segment.text.strip())

            # Log progress every 5% or every 30 seconds
            current_time = __import__('time').time()
            if duration > 0:
                progress = int((segment.end / duration) * 100)
                time_since_last_log = current_time - last_log_time

                if progress >= last_progress + 5 or time_since_last_log >= 30:
                    last_progress = progress
                    last_log_time = current_time
                    logger.info(f"Transcription: {progress}% ({segment.end:.1f}s/{duration:.1f}s) - {segment_count} segments")

        if cancelled:
            return TranscriptionResult(
                success=False,
                language=info.language,
                segments=segments,  # Return partial results
                text=" ".join(full_text),
                error="用户取消",
                cancelled=True
            )

        detected_language = info.language
        
        # Resegment long subtitles for better readability
        original_count = len(segments)
        segments = self._resegment_by_length(segments, max_chars=80, language=detected_language)
        if len(segments) != original_count:
            logger.info(f"Resegmented: {original_count} -> {len(segments)} segments for better readability")
            # Rebuild full text
            full_text = [seg["text"] for seg in segments]
        
        logger.info(
            f"Transcription complete: language={detected_language}, "
            f"segments={len(segments)}, duration={duration:.1f}s"
        )

        return TranscriptionResult(
            success=True,
            language=detected_language,
            segments=segments,
            text=" ".join(full_text)
        )

    def _resegment_by_length(
        self,
        segments: List[Dict[str, Any]],
        max_chars: int = 80,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Resegment transcription for better subtitle readability.
        
        Strategy:
        1. First split at sentence boundaries (. ! ? 。！？)
        2. If a sentence is still too long, split at clause boundaries (, ; : ，；：)
        3. If still too long, split at word boundaries
        4. Merge short segments if they fit together
        
        Args:
            segments: Original segments with optional word-level timing
            max_chars: Maximum characters per segment (default 80, ~2 lines)
            language: Language code for CJK detection
            
        Returns:
            New list of segments with shorter, complete sentences
        """
        is_cjk = language in ('zh', 'ja', 'ko', 'yue')
        result = []
        
        for seg in segments:
            text = _get_seg_val(seg, "text", "").strip()
            
            # Short enough - keep as is
            if len(text) <= max_chars:
                result.append(seg)
                continue
            
            words = seg.get("words", [])
            
            # Has word timestamps - split intelligently
            if words:
                new_segs = self._split_by_sentences(seg, max_chars, is_cjk)
                result.extend(new_segs)
            else:
                # No word timestamps - split by character with proportional timing
                new_segs = self._split_by_chars(seg, max_chars, is_cjk)
                result.extend(new_segs)
        
        # Merge short consecutive segments if they fit
        result = self._merge_short_segments(result, max_chars, is_cjk)
        
        return result
    
    def _split_by_sentences(self, seg: Dict[str, Any], max_chars: int, is_cjk: bool) -> List[Dict[str, Any]]:
        """
        Split segment at sentence/clause boundaries using word timestamps.
        
        Priority:
        1. Sentence-ending punctuation (. ! ? 。！？)
        2. Clause-separating punctuation (, ; : ，；：、)
        3. Word boundaries (only if sentence is still too long)
        """
        words = seg.get("words", [])
        if not words:
            return [seg]
        
        # Define punctuation marks
        if is_cjk:
            sentence_end = {'。', '！', '？', '!', '?', '.'}
            clause_sep = {'，', '、', '；', '：', ',', ';', ':'}
        else:
            sentence_end = {'.', '!', '?'}
            clause_sep = {',', ';', ':', '-', '—'}
        
        result = []
        current_words = []
        current_text = ""
        
        for i, word in enumerate(words):
            word_text = word.get("word", "").strip()
            
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
                # Add this word first, then split
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
                result.append({
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                    "text": current_text.strip(),
                    "words": current_words.copy(),
                })
                current_words = []
                current_text = ""
            
            # Add current word to new segment (if not already added)
            if word is not None:
                current_words.append(word)
                current_text = test_text
        
        # Don't forget the last segment
        if current_words:
            result.append({
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": current_text.strip(),
                "words": current_words,
            })
        
        return result if result else [seg]
    
    def _merge_short_segments(
        self, 
        segments: List[Dict[str, Any]], 
        max_chars: int,
        is_cjk: bool
    ) -> List[Dict[str, Any]]:
        """Merge consecutive short segments if they fit together."""
        if len(segments) <= 1:
            return segments
        
        min_chars = max_chars // 3  # Segments shorter than this should be merged
        result = []
        current = None
        
        for seg in segments:
            if current is None:
                current = seg.copy()
                current["words"] = list(seg.get("words", []))
                continue
            
            current_len = len(current.get("text", ""))
            seg_len = len(seg.get("text", ""))
            
            # Try to merge if current is short
            if current_len < min_chars:
                if is_cjk:
                    merged_text = current.get("text", "") + seg.get("text", "")
                else:
                    merged_text = current.get("text", "") + " " + seg.get("text", "")
                
                if len(merged_text) <= max_chars:
                    # Merge
                    current["text"] = merged_text.strip()
                    current["end"] = seg.get("end", current.get("end"))
                    current["words"] = current.get("words", []) + seg.get("words", [])
                    continue
            
            # Can't merge - save current and start new
            result.append(current)
            current = seg.copy()
            current["words"] = list(seg.get("words", []))
        
        if current:
            result.append(current)
        
        return result
    
    def _split_by_chars(self, seg: Dict[str, Any], max_chars: int, is_cjk: bool) -> List[Dict[str, Any]]:
        """Split segment by character count with proportional timing (fallback when no word timestamps)."""
        text = _get_seg_val(seg, "text", "").strip()
        total_chars = len(text)
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start
        
        # Define punctuation for splitting
        if is_cjk:
            sentence_end = ['。', '！', '？', '!', '?', '.']
            clause_sep = ['，', '、', '；', '：', ',', ';', ':']
        else:
            sentence_end = ['.', '!', '?']
            clause_sep = [',', ';', ':', ' ']
        
        result = []
        start_idx = 0
        
        while start_idx < total_chars:
            end_idx = min(start_idx + max_chars, total_chars)
            
            # Try to find a good break point
            if end_idx < total_chars:
                best_break = -1
                
                # First, look for sentence-ending punctuation
                for punct in sentence_end:
                    punct_pos = text.rfind(punct, start_idx, end_idx)
                    if punct_pos > start_idx + 10:  # At least 10 chars
                        best_break = max(best_break, punct_pos + 1)
                
                # If no sentence end, look for clause separator
                if best_break < start_idx + max_chars // 2:
                    for punct in clause_sep:
                        punct_pos = text.rfind(punct, start_idx, end_idx)
                        if punct_pos > start_idx + max_chars // 3:
                            best_break = max(best_break, punct_pos + 1)
                
                if best_break > start_idx:
                    end_idx = best_break
            
            chunk_text = text[start_idx:end_idx].strip()
            if chunk_text:
                # Proportional timing
                ratio_start = start_idx / total_chars
                ratio_end = end_idx / total_chars
                chunk_start = start + duration * ratio_start
                chunk_end = start + duration * ratio_end
                
                result.append({
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": chunk_text,
                    "words": [],
                })
            
            start_idx = end_idx
        
        return result if result else [seg]

    async def save_subtitles(
        self,
        segments: List[Dict[str, Any]],
        output_path: Path,
        format: str = "srt"
    ) -> None:
        """Save segments to subtitle file"""
        output_path = Path(output_path)

        if format == "srt":
            content = self._to_srt(segments)
        elif format == "vtt":
            content = self._to_vtt(segments)
        else:
            raise ValueError(f"Unknown format: {format}")

        output_path.write_text(content, encoding="utf-8")
        logger.info(f"Saved subtitles to: {output_path}")

    def _to_srt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert segments to SRT format"""
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_timestamp_srt(_get_seg_val(seg, "start"))
            end = self._format_timestamp_srt(_get_seg_val(seg, "end"))
            text = _get_seg_val(seg, "text", "")
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    def _to_vtt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert segments to VTT format"""
        lines = ["WEBVTT\n"]
        for seg in segments:
            start = self._format_timestamp_vtt(_get_seg_val(seg, "start"))
            end = self._format_timestamp_vtt(_get_seg_val(seg, "end"))
            text = _get_seg_val(seg, "text", "")
            lines.append(f"{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    def _format_timestamp_srt(self, seconds: float) -> str:
        """Format timestamp for SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format timestamp for VTT (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

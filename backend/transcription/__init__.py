"""
Transcription Package

Provides:
- WhisperTranscriber: Standard Whisper transcription
- FasterWhisperTranscriber: Faster transcription using CTranslate2 (4-8x faster)
- WhisperXTranscriber: Enhanced transcription with word-level alignment
- TranscriptSegment, TranscriptionResult, Transcription: Data structures for transcription results
- WordSegment, TranscriptSegmentWithWords: Word-level timing data structures
"""
from dataclasses import dataclass
from typing import List, Optional

from .whisper_transcriber import WhisperTranscriber, TranscriptSegment, TranscriptionResult
from .faster_transcriber import FasterWhisperTranscriber
from .whisperx_transcriber import (
    WhisperXTranscriber,
    WhisperXResult,
    WordSegment,
    TranscriptSegmentWithWords,
)


@dataclass
class Transcription:
    """Generic transcription result for internal use"""
    text: str  # Full transcribed text
    segments: List[TranscriptSegment]
    language: str
    success: bool = True
    error: Optional[str] = None
    
    # For compatibility with TranscriptionResult
    @property
    def full_text(self) -> str:
        return self.text


__all__ = [
    # Standard Whisper
    "WhisperTranscriber",
    # Faster Whisper (recommended - 4-8x faster)
    "FasterWhisperTranscriber",
    # WhisperX with word-level alignment
    "WhisperXTranscriber",
    "WhisperXResult",
    "WordSegment",
    "TranscriptSegmentWithWords",
    # Data classes
    "TranscriptSegment",
    "TranscriptionResult",
    "Transcription",
]

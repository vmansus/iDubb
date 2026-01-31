"""
Transcription Package

Provides:
- WhisperTranscriber: Standard Whisper transcription
- FasterWhisperTranscriber: Faster transcription using CTranslate2 (4-8x faster)
- WhisperXTranscriber: Enhanced transcription with word-level alignment
- WordSegment, TranscriptSegmentWithWords: Word-level timing data structures
"""
from .whisper_transcriber import WhisperTranscriber
from .faster_transcriber import FasterWhisperTranscriber
from .whisperx_transcriber import (
    WhisperXTranscriber,
    WhisperXResult,
    WordSegment,
    TranscriptSegmentWithWords,
)

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
]

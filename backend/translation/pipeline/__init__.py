"""
VideoLingo-style Translation Pipeline

Supports two modes:
1. Three-Step Mode: Translate → Reflect → Adapt (more thorough)
2. Two-Step Mode: Translate → Expressiveness (more efficient, VideoLingo-style)

Also includes subtitle alignment for synchronization.

Based on VideoLingo's proven translation methodology.
"""
from .three_step_translator import ThreeStepTranslator, TranslationConfig, ThreeStepResult
from .translate_step import TranslateStep
from .reflect_step import ReflectStep
from .adapt_step import AdaptStep
from .expressiveness_step import ExpressivenessStep, ExpressivenessResult
from .align_step import SubtitleAligner, AlignmentResult, AlignedSegment

__all__ = [
    # Main translators
    "ThreeStepTranslator",
    "TranslationConfig",
    "ThreeStepResult",
    # Individual steps
    "TranslateStep",
    "ReflectStep", 
    "AdaptStep",
    # Two-step mode (VideoLingo-style)
    "ExpressivenessStep",
    "ExpressivenessResult",
    # Subtitle alignment
    "SubtitleAligner",
    "AlignmentResult",
    "AlignedSegment",
]

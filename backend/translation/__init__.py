"""
Translation Package

Provides:
- Standard translation via multiple engines (Google, DeepL, GPT, Claude, DeepSeek)
- VideoLingo-style 3-step translation (Translate-Reflect-Adapt)
- Terminology management for consistent translations
- Enhanced translator combining all features
- Token-optimized translator (VideoLingo approach)
"""
from .translator import Translator, TranslationResult
from .pipeline import ThreeStepTranslator, TranslationConfig
from .terminology import TerminologyManager, Glossary
from .enhanced_translator import EnhancedTranslator, EnhancedTranslationConfig
from .summarizer import VideoSummarizer, VideoContext, OptimizedTranslator

__all__ = [
    # Standard translation
    "Translator",
    "TranslationResult",
    # 3-step pipeline (VideoLingo-style)
    "ThreeStepTranslator",
    "TranslationConfig",
    # Terminology management
    "TerminologyManager",
    "Glossary",
    # Enhanced translator
    "EnhancedTranslator",
    "EnhancedTranslationConfig",
    # Token-optimized translator (recommended for GPT/Claude)
    "VideoSummarizer",
    "VideoContext",
    "OptimizedTranslator",
]

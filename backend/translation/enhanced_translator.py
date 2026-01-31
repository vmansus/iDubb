"""
Enhanced Translator

Convenience wrapper that combines:
- 3-step translation pipeline
- Terminology management
- Netflix subtitle validation

This is the recommended entry point for high-quality translation.
"""
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from .translator import Translator, TranslationResult
from .pipeline import ThreeStepTranslator, TranslationConfig, ThreeStepResult
from .terminology import TerminologyManager


@dataclass
class EnhancedTranslationConfig:
    """Configuration for enhanced translation"""
    # Translation engine
    engine: str = "google"  # google, gpt, claude, deepseek, deepl
    api_key: Optional[str] = None
    model: Optional[str] = None

    # 3-step pipeline settings
    enable_three_step: bool = True
    enable_reflection: bool = True
    enable_adaptation: bool = True
    quality_threshold: float = 0.7

    # Terminology settings
    use_terminology: bool = True
    glossary_names: Optional[List[str]] = None  # None = use all

    # Subtitle validation
    validate_netflix: bool = True
    max_line_length: int = 42
    max_lines: int = 2

    # Context window for translation
    context_window: int = 2


class EnhancedTranslator:
    """
    Enhanced translator with VideoLingo-style features.

    Combines:
    - Multi-engine translation support
    - 3-step translation pipeline (Translate-Reflect-Adapt)
    - Terminology management for consistency
    - Netflix subtitle validation

    Usage:
        translator = EnhancedTranslator(config)
        results = await translator.translate_subtitles(segments, "en", "zh-CN")
    """

    def __init__(
        self,
        config: Optional[EnhancedTranslationConfig] = None,
        terminology_dir: Optional[Path] = None
    ):
        """
        Initialize enhanced translator.

        Args:
            config: Translation configuration
            terminology_dir: Directory for terminology glossaries
        """
        self.config = config or EnhancedTranslationConfig()

        # Initialize base translator
        self.translator = Translator(
            engine=self.config.engine,
            api_key=self.config.api_key,
            model=self.config.model
        )

        # Initialize terminology manager
        self.terminology_manager = TerminologyManager(
            glossaries_dir=terminology_dir,
            auto_load=True
        )

        # Initialize 3-step translator if enabled
        self.three_step_translator = None
        if self.config.enable_three_step:
            pipeline_config = TranslationConfig(
                enable_reflection=self.config.enable_reflection,
                enable_adaptation=self.config.enable_adaptation,
                quality_threshold=self.config.quality_threshold,
                context_window=self.config.context_window,
                max_line_length=self.config.max_line_length,
            )
            self.three_step_translator = ThreeStepTranslator(
                engine=self.translator._engine,
                config=pipeline_config
            )

        logger.info(
            f"Initialized EnhancedTranslator: "
            f"engine={self.config.engine}, "
            f"three_step={self.config.enable_three_step}, "
            f"terminology={self.config.use_terminology}"
        )

    def update_terminology(
        self,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, str]:
        """
        Get and update terminology for a language pair.

        Args:
            source_lang: Source language
            target_lang: Target language

        Returns:
            Dictionary of term -> translation
        """
        terminology = self.terminology_manager.get_all_terms(source_lang, target_lang)

        # Update in 3-step translator if available
        if self.three_step_translator:
            self.three_step_translator.update_terminology(terminology)

        return terminology

    async def translate_text(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-CN"
    ) -> str:
        """
        Translate a single text segment.

        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language

        Returns:
            Translated text
        """
        # Update terminology
        if self.config.use_terminology:
            self.update_terminology(source_lang, target_lang)

        # Use 3-step if enabled
        if self.three_step_translator:
            result = await self.three_step_translator.translate_segment(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang
            )
            return result.final_translation

        # Fall back to standard translation
        result = await self.translator.translate(text, source_lang, target_lang)
        return result.translated_text

    async def translate_subtitles(
        self,
        segments: List[Dict[str, Any]],
        source_lang: str = "en",
        target_lang: str = "zh-CN",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Translate subtitle segments with enhanced features.

        Args:
            segments: List of subtitle segments with 'text', 'start', 'end'
            source_lang: Source language
            target_lang: Target language
            progress_callback: Optional callback(current, total, segment)

        Returns:
            Dictionary with:
            - segments: Translated segments
            - report: Translation report
            - validation: Netflix validation results (if enabled)
        """
        if not segments:
            return {"segments": [], "report": {}, "validation": None}

        # Update terminology
        if self.config.use_terminology:
            terminology = self.update_terminology(source_lang, target_lang)
            logger.info(f"Using {len(terminology)} terminology entries")

        # Extract text for translation
        texts = [seg.get("text", "") for seg in segments]
        total = len(texts)

        # Translate using 3-step or standard
        translated_texts = []
        translation_results = []

        if self.three_step_translator:
            logger.info(f"Using 3-step translation for {total} segments")
            results = await self.three_step_translator.translate_segments(
                segments=texts,
                source_lang=source_lang,
                target_lang=target_lang,
                progress_callback=progress_callback
            )
            translated_texts = self.three_step_translator.get_final_translations(results)
            translation_results = results
            report = self.three_step_translator.get_translation_report(results)
        else:
            logger.info(f"Using standard translation for {total} segments")
            for i, text in enumerate(texts):
                result = await self.translator.translate(text, source_lang, target_lang)
                translated_texts.append(result.translated_text)
                if progress_callback:
                    progress_callback(i + 1, total, text)
                await asyncio.sleep(0.05)

            report = {
                "segments": total,
                "steps_used": {"translate": total},
                "quality": {"average": 0.8},
            }

        # Build translated segments
        translated_segments = []
        for i, seg in enumerate(segments):
            translated_seg = {
                "index": i + 1,
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": translated_texts[i] if i < len(translated_texts) else seg.get("text", ""),
                "original_text": seg.get("text", ""),
            }
            translated_segments.append(translated_seg)

        # Validate against Netflix standards if enabled
        validation = None
        if self.config.validate_netflix:
            from ..subtitles import NetflixValidator
            validator = NetflixValidator()
            validation = validator.validate_subtitles(translated_segments)

            if not validation.is_valid:
                logger.warning(
                    f"Netflix validation: {validation.error_count} errors, "
                    f"{validation.warning_count} warnings"
                )

        return {
            "segments": translated_segments,
            "report": report,
            "validation": validation.to_dict() if validation else None,
        }

    async def extract_and_suggest_terms(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-CN"
    ) -> Dict[str, str]:
        """
        Extract potential terms from text and suggest translations.

        Args:
            text: Text to analyze
            source_lang: Source language
            target_lang: Target language

        Returns:
            Dictionary of term -> suggested translation
        """
        # Extract potential terms
        terms = await self.terminology_manager.extract_terms_from_text(
            text=text,
            source_lang=source_lang
        )

        # Filter out terms we already have
        existing = self.terminology_manager.get_all_terms(source_lang, target_lang)
        new_terms = [t for t in terms if t not in existing]

        if not new_terms:
            return {}

        # Suggest translations using AI
        suggestions = await self.terminology_manager.suggest_translations(
            terms=new_terms,
            ai_engine=self.translator._engine,
            source_lang=source_lang,
            target_lang=target_lang
        )

        return suggestions

    def add_term(
        self,
        source_term: str,
        target_term: str,
        source_lang: str = "en",
        target_lang: str = "zh-CN"
    ) -> None:
        """Add a term to the terminology"""
        self.terminology_manager.add_term(
            source_term=source_term,
            target_term=target_term,
            glossary_name="custom",
            source_lang=source_lang,
            target_lang=target_lang
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get translator statistics"""
        return {
            "engine": self.config.engine,
            "three_step_enabled": self.config.enable_three_step,
            "terminology": self.terminology_manager.get_stats(),
        }

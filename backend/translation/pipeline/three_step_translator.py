"""
VideoLingo-style 3-Step Translation Pipeline

Orchestrates the Translate-Reflect-Adapt workflow for high-quality translations.
This is the main entry point for the enhanced translation system.
"""
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger

from .translate_step import TranslateStep, TranslateResult
from .reflect_step import ReflectStep, ReflectResult
from .adapt_step import AdaptStep, AdaptResult
from .expressiveness_step import ExpressivenessStep, ExpressivenessResult
from .align_step import SubtitleAligner, AlignmentResult


@dataclass
class TranslationConfig:
    """Configuration for the translation pipeline"""
    # Pipeline mode
    use_two_step_mode: bool = True  # True = VideoLingo-style (Translate → Expressiveness)
                                     # False = Three-step (Translate → Reflect → Adapt)
    
    # Enable/disable steps (for three-step mode)
    enable_reflection: bool = True
    enable_adaptation: bool = True

    # Quality thresholds
    quality_threshold: float = 0.7  # Below this triggers adaptation
    skip_adaptation_threshold: float = 0.9  # Above this skips adaptation

    # Context settings
    context_window: int = 2  # Segments before/after for context

    # Performance settings
    batch_size: int = 10
    max_concurrent: int = 3

    # Subtitle constraints (Netflix standards)
    max_line_length: int = 42
    max_lines_per_subtitle: int = 2
    
    # Alignment
    enable_alignment: bool = False  # Enable subtitle alignment post-translation


@dataclass
class ThreeStepResult:
    """Complete result from 3-step translation"""
    original: str
    final_translation: str

    # Step results
    translate_result: Optional[TranslateResult] = None
    reflect_result: Optional[ReflectResult] = None
    adapt_result: Optional[AdaptResult] = None

    # Metadata
    steps_executed: List[str] = field(default_factory=list)
    total_quality_score: float = 0.0
    issues_found: int = 0
    issues_fixed: int = 0


class ThreeStepTranslator:
    """
    VideoLingo-style 3-Step Translation Pipeline

    Workflow:
    1. TRANSLATE: Initial translation with context and terminology
    2. REFLECT: Quality assessment and issue identification
    3. ADAPT: Refinement based on reflection feedback

    This approach significantly improves translation quality by:
    - Using context for coherent translations
    - Self-critiquing to identify issues
    - Iteratively improving based on feedback
    """

    def __init__(
        self,
        engine: Any,  # Base translation engine
        config: Optional[TranslationConfig] = None,
        terminology: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the translator.

        Args:
            engine: Translation engine (GPT, Claude, etc.)
            config: Pipeline configuration
            terminology: Term -> translation mappings
        """
        self.engine = engine
        self.config = config or TranslationConfig()
        self.terminology = terminology or {}

        # Initialize translation step (used in both modes)
        self.translate_step = TranslateStep(engine, terminology)
        
        # Initialize mode-specific steps
        if self.config.use_two_step_mode:
            # Two-step mode: Translate → Expressiveness (VideoLingo-style)
            self.expressiveness_step = ExpressivenessStep(
                engine, terminology, self.config.quality_threshold
            )
            self.reflect_step = None
            self.adapt_step = None
        else:
            # Three-step mode: Translate → Reflect → Adapt
            self.expressiveness_step = None
            self.reflect_step = ReflectStep(engine, terminology, self.config.quality_threshold)
            self.adapt_step = AdaptStep(engine, terminology, self.config.max_line_length)
        
        # Initialize aligner if enabled
        self.aligner = SubtitleAligner(engine) if self.config.enable_alignment else None

        mode_name = "Two-Step (VideoLingo)" if self.config.use_two_step_mode else "Three-Step"
        logger.info(
            f"Initialized Translator: mode={mode_name}, "
            f"alignment={self.config.enable_alignment}"
        )

    def update_terminology(self, terminology: Dict[str, str]) -> None:
        """
        Update terminology dictionary.

        Args:
            terminology: New term -> translation mappings
        """
        self.terminology = terminology
        self.translate_step.terminology = terminology
        
        if self.config.use_two_step_mode:
            if self.expressiveness_step:
                self.expressiveness_step.terminology = terminology
        else:
            if self.reflect_step:
                self.reflect_step.terminology = terminology
            if self.adapt_step:
                self.adapt_step.terminology = terminology
                
        logger.info(f"Updated terminology with {len(terminology)} terms")

    async def translate_segment(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        prev_context: Optional[List[str]] = None,
        next_context: Optional[List[str]] = None
    ) -> ThreeStepResult:
        """
        Translate a single segment using the configured pipeline.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            prev_context: Previous segments for context
            next_context: Next segments for context

        Returns:
            ThreeStepResult with complete translation data
        """
        steps_executed = []
        issues_found = 0
        issues_fixed = 0

        # Step 1: Direct Translation
        logger.debug(f"Step 1: Translating '{text[:50]}...'")
        translate_result = await self.translate_step.translate_segment(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            prev_context=prev_context,
            next_context=next_context
        )
        steps_executed.append("translate")
        current_translation = translate_result.translated

        reflect_result = None
        adapt_result = None
        quality_score = 0.8  # Default score

        if self.config.use_two_step_mode:
            # Two-Step Mode: Expressiveness (VideoLingo-style)
            logger.debug(f"Step 2: Expressiveness (reflect + adapt combined)")
            expr_result = await self.expressiveness_step.improve_translation(
                original=text,
                direct_translation=current_translation,
                source_lang=source_lang,
                target_lang=target_lang,
                prev_context=prev_context,
                next_context=next_context
            )
            steps_executed.append("expressiveness")
            current_translation = expr_result.free_translation
            quality_score = expr_result.quality_score
            issues_fixed = len(expr_result.changes_made)
            
        else:
            # Three-Step Mode: Reflect → Adapt
            # Step 2: Reflect (if enabled)
            if self.config.enable_reflection:
                logger.debug(f"Step 2: Reflecting on translation")
                reflect_result = await self.reflect_step.reflect_on_translation(
                    original=text,
                    translated=current_translation,
                    source_lang=source_lang,
                    target_lang=target_lang
                )
                steps_executed.append("reflect")
                quality_score = reflect_result.quality_score
                issues_found = len(reflect_result.issues)

                # Step 3: Adapt (if needed and enabled)
                if (self.config.enable_adaptation and
                    reflect_result.needs_adaptation and
                    quality_score < self.config.skip_adaptation_threshold):

                    logger.debug(f"Step 3: Adapting translation (quality={quality_score:.2f})")
                    adapt_result = await self.adapt_step.adapt_translation(
                        original=text,
                        translated=current_translation,
                        issues=reflect_result.issues,
                        source_lang=source_lang,
                        target_lang=target_lang
                    )
                    steps_executed.append("adapt")
                    current_translation = adapt_result.adapted_translation
                    quality_score = adapt_result.final_quality_score
                    issues_fixed = len(adapt_result.changes_made)

        return ThreeStepResult(
            original=text,
            final_translation=current_translation,
            translate_result=translate_result,
            reflect_result=reflect_result,
            adapt_result=adapt_result,
            steps_executed=steps_executed,
            total_quality_score=quality_score,
            issues_found=issues_found,
            issues_fixed=issues_fixed
        )

    async def translate_segments(
        self,
        segments: List[str],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[callable] = None
    ) -> List[ThreeStepResult]:
        """
        Translate multiple segments using the 3-step pipeline.

        Args:
            segments: List of text segments
            source_lang: Source language code
            target_lang: Target language code
            progress_callback: Optional callback(current, total, segment)

        Returns:
            List of ThreeStepResult objects
        """
        results = []
        total = len(segments)

        logger.info(f"Starting 3-step translation of {total} segments")

        for i, segment in enumerate(segments):
            # Get context window
            prev_context = segments[max(0, i - self.config.context_window):i]
            next_context = segments[i + 1:i + 1 + self.config.context_window]

            # Translate with 3-step pipeline
            result = await self.translate_segment(
                text=segment,
                source_lang=source_lang,
                target_lang=target_lang,
                prev_context=prev_context if prev_context else None,
                next_context=next_context if next_context else None
            )
            results.append(result)

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total, segment)

            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Translated {i + 1}/{total} segments")

            # Rate limiting between segments
            await asyncio.sleep(0.05)

        # Summary statistics
        total_issues = sum(r.issues_found for r in results)
        total_fixed = sum(r.issues_fixed for r in results)
        avg_quality = sum(r.total_quality_score for r in results) / len(results) if results else 0

        logger.info(
            f"3-step translation complete: "
            f"{total} segments, "
            f"{total_issues} issues found, "
            f"{total_fixed} issues fixed, "
            f"avg quality: {avg_quality:.2f}"
        )

        return results

    def get_final_translations(self, results: List[ThreeStepResult]) -> List[str]:
        """
        Extract final translations from results.

        Args:
            results: List of ThreeStepResult objects

        Returns:
            List of final translated strings
        """
        return [r.final_translation for r in results]

    def get_translation_report(self, results: List[ThreeStepResult]) -> Dict[str, Any]:
        """
        Generate a summary report of the translation process.

        Args:
            results: List of ThreeStepResult objects

        Returns:
            Dictionary with translation statistics
        """
        if not results:
            return {"segments": 0}

        total_issues = sum(r.issues_found for r in results)
        total_fixed = sum(r.issues_fixed for r in results)
        adaptations = sum(1 for r in results if r.adapt_result and r.adapt_result.adaptation_applied)

        quality_scores = [r.total_quality_score for r in results]

        return {
            "segments": len(results),
            "steps_used": {
                "translate": len(results),
                "reflect": sum(1 for r in results if "reflect" in r.steps_executed),
                "adapt": sum(1 for r in results if "adapt" in r.steps_executed),
            },
            "quality": {
                "average": sum(quality_scores) / len(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores),
                "below_threshold": sum(1 for s in quality_scores if s < self.config.quality_threshold),
            },
            "issues": {
                "found": total_issues,
                "fixed": total_fixed,
                "adaptations_applied": adaptations,
            },
            "config": {
                "reflection_enabled": self.config.enable_reflection,
                "adaptation_enabled": self.config.enable_adaptation,
                "quality_threshold": self.config.quality_threshold,
                "context_window": self.config.context_window,
            }
        }

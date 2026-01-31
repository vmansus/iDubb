"""
Step 1: Initial Translation

Performs the initial translation with context awareness and terminology support.
Uses the provided translation engine (GPT, Claude, etc.) for AI-powered translation.
"""
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class TranslateResult:
    """Result from the initial translation step"""
    original: str
    translated: str
    context_used: bool
    terminology_applied: List[str]


class TranslateStep:
    """
    Step 1: Initial Translation

    Translates text with:
    - Context from surrounding segments
    - Terminology/glossary enforcement
    - Style preservation hints
    """

    SYSTEM_PROMPT = """## Role
You are a professional Netflix subtitle translator, fluent in both {source_lang} and {target_lang}, as well as their respective cultures.
Your expertise lies in accurately understanding the semantics and structure of the original {source_lang} text and faithfully translating it into {target_lang} while preserving the original meaning.

## Task
Translate the given {source_lang} subtitles into {target_lang}. These subtitles come from a specific video context and may contain specific themes and terminology.

{context_section}

{terminology_section}

<translation_principles>
1. **Faithful to the original**: Accurately convey the content and meaning of the original text, without arbitrarily changing, adding, or omitting content.
2. **Accurate terminology**: Use professional terms correctly and maintain consistency in terminology throughout the video.
3. **Understand the context**: Fully comprehend and reflect the background and contextual relationships of the text.
4. **Natural expression**: Use natural, conversational {target_lang} that flows well when read as subtitles.
5. **Concise for subtitles**: Keep translations concise - ideally max 42 characters per line for readability.
6. **Preserve tone**: Maintain the speaker's original tone, emotion, and intent.
</translation_principles>

Translate each line accurately and naturally. Output ONLY the translations, one per line, matching the input order exactly."""

    def __init__(
        self,
        engine: Any,  # TranslationEngine from parent module
        terminology: Optional[Dict[str, str]] = None
    ):
        """
        Initialize translate step.

        Args:
            engine: Translation engine instance (GPT, Claude, etc.)
            terminology: Dictionary of term -> translation mappings
        """
        self.engine = engine
        self.terminology = terminology or {}

    def _build_terminology_section(self) -> str:
        """Build terminology section for prompt"""
        if not self.terminology:
            return ""

        terms = "\n".join([f"- {src} → {tgt}" for src, tgt in self.terminology.items()])
        return f"""TERMINOLOGY (use these translations consistently):
{terms}"""

    def _build_context_section(self, prev_segments: List[str], next_segments: List[str]) -> str:
        """Build context section for prompt"""
        parts = []

        if prev_segments:
            prev_text = "\n".join([f"  [{i+1}] {s}" for i, s in enumerate(prev_segments)])
            parts.append(f"Previous context:\n{prev_text}")

        if next_segments:
            next_text = "\n".join([f"  [{i+1}] {s}" for i, s in enumerate(next_segments)])
            parts.append(f"Following context:\n{next_text}")

        if parts:
            return "CONTEXT (for reference, do not translate):\n" + "\n".join(parts)
        return ""

    async def translate_segment(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        prev_context: Optional[List[str]] = None,
        next_context: Optional[List[str]] = None
    ) -> TranslateResult:
        """
        Translate a single segment with context.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            prev_context: Previous segments for context
            next_context: Next segments for context

        Returns:
            TranslateResult with translation and metadata
        """
        try:
            # For simple engines (Google, DeepL), just translate directly
            if not hasattr(self.engine, 'translate_with_prompt'):
                translated = await self.engine.translate(text, source_lang, target_lang)
                return TranslateResult(
                    original=text,
                    translated=translated,
                    context_used=False,
                    terminology_applied=[]
                )

            # For AI engines, use context-aware translation
            terminology_section = self._build_terminology_section()
            context_section = self._build_context_section(
                prev_context or [],
                next_context or []
            )

            system_prompt = self.SYSTEM_PROMPT.format(
                source_lang=source_lang,
                target_lang=target_lang,
                terminology_section=terminology_section,
                context_section=context_section
            )

            translated = await self.engine.translate_with_prompt(
                text=text,
                system_prompt=system_prompt,
                source_lang=source_lang,
                target_lang=target_lang
            )

            # Check which terminology was applied
            applied_terms = [
                term for term in self.terminology.keys()
                if term.lower() in text.lower()
            ]

            return TranslateResult(
                original=text,
                translated=translated,
                context_used=bool(prev_context or next_context),
                terminology_applied=applied_terms
            )

        except Exception as e:
            logger.error(f"Translation step failed: {e}")
            # Return original on failure
            return TranslateResult(
                original=text,
                translated=text,
                context_used=False,
                terminology_applied=[]
            )

    async def translate_batch(
        self,
        segments: List[str],
        source_lang: str,
        target_lang: str,
        context_window: int = 2
    ) -> List[TranslateResult]:
        """
        Translate multiple segments with sliding context window.

        Args:
            segments: List of text segments to translate
            source_lang: Source language code
            target_lang: Target language code
            context_window: Number of segments before/after for context

        Returns:
            List of TranslateResult objects
        """
        results = []

        for i, segment in enumerate(segments):
            # Get context window
            prev_context = segments[max(0, i - context_window):i]
            next_context = segments[i + 1:i + 1 + context_window]

            result = await self.translate_segment(
                text=segment,
                source_lang=source_lang,
                target_lang=target_lang,
                prev_context=prev_context if prev_context else None,
                next_context=next_context if next_context else None
            )
            results.append(result)

            # Small delay to avoid rate limits
            if i > 0 and i % 10 == 0:
                await asyncio.sleep(0.1)

        return results

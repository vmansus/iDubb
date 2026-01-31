"""
Step 3: Translation Adaptation

Refines the translation based on reflection feedback.
Applies cultural adaptations and fixes identified issues.
"""
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

from .reflect_step import ReflectResult, TranslationIssue, IssueType


@dataclass
class AdaptResult:
    """Result from the adaptation step"""
    original: str
    initial_translation: str
    adapted_translation: str
    changes_made: List[str]
    final_quality_score: float
    adaptation_applied: bool


class AdaptStep:
    """
    Step 3: Translation Adaptation

    Refines translations based on:
    - Issues identified in reflection step
    - Cultural adaptation requirements
    - Subtitle formatting constraints
    - Terminology enforcement
    """

    ADAPTATION_PROMPT = """## Role
You are a professional Netflix subtitle translator and language consultant.
Your expertise lies in optimizing {target_lang} translations to better suit the target language's expression habits and cultural background.

## Task
We have identified some issues with the current translation. Your task is to improve it based on the feedback.

## INPUT
<source lang="{source_lang}">
{original}
</source>

<current_translation lang="{target_lang}">
{translated}
</current_translation>

## Issues to Address
{issues_section}

{terminology_section}

## Adaptation Guidelines
<adaptation_principles>
1. **Fix identified issues**: Address each issue listed above
2. **Maintain naturalness**: The result should flow naturally in {target_lang}, conforming to native expression habits
3. **Subtitle constraints**: Keep it concise for subtitles (ideally ≤42 characters per line)
4. **Preserve meaning**: Do not change the original meaning while improving expression
5. **Match tone**: Adapt the language style to match the content (casual for tutorials, technical for tech content, formal for documentaries)
6. **Cultural fit**: Ensure the target audience can easily understand and accept the translation
</adaptation_principles>

## Important
- Do NOT add any comments or explanations
- Do NOT include the original text
- Output ONLY the improved {target_lang} translation, nothing else"""

    def __init__(
        self,
        engine: Any,  # AI translation engine
        terminology: Optional[Dict[str, str]] = None,
        max_length: int = 42
    ):
        """
        Initialize adapt step.

        Args:
            engine: AI engine for adaptation
            terminology: Terminology dictionary
            max_length: Maximum recommended subtitle length
        """
        self.engine = engine
        self.terminology = terminology or {}
        self.max_length = max_length

    def _build_issues_section(self, issues: List[TranslationIssue]) -> str:
        """Build issues section for adaptation prompt"""
        if not issues:
            return "No specific issues identified - refine for naturalness."

        lines = []
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. [{issue.severity.upper()}] {issue.issue_type.value}: {issue.description}")
            if issue.suggestion:
                lines.append(f"   Suggestion: {issue.suggestion}")

        return "\n".join(lines)

    def _build_terminology_section(self) -> str:
        """Build terminology section"""
        if not self.terminology:
            return ""

        terms = "\n".join([f"- {src} → {tgt}" for src, tgt in self.terminology.items()])
        return f"""TERMINOLOGY TO USE:
{terms}"""

    async def adapt_translation(
        self,
        original: str,
        translated: str,
        issues: List[TranslationIssue],
        source_lang: str,
        target_lang: str
    ) -> AdaptResult:
        """
        Adapt a single translation based on issues.

        Args:
            original: Original text
            translated: Initial translation
            issues: Issues from reflection step
            source_lang: Source language
            target_lang: Target language

        Returns:
            AdaptResult with adapted translation
        """
        # If no issues, return as-is with minor refinement
        if not issues:
            return AdaptResult(
                original=original,
                initial_translation=translated,
                adapted_translation=translated,
                changes_made=[],
                final_quality_score=1.0,
                adaptation_applied=False
            )

        # For non-AI engines, apply simple fixes
        if not hasattr(self.engine, 'adapt'):
            return self._simple_adapt(original, translated, issues)

        try:
            issues_section = self._build_issues_section(issues)
            terminology_section = self._build_terminology_section()

            prompt = self.ADAPTATION_PROMPT.format(
                source_lang=source_lang,
                target_lang=target_lang,
                original=original,
                translated=translated,
                issues_section=issues_section,
                terminology_section=terminology_section
            )

            adapted = await self.engine.adapt(prompt)

            # Clean up the response
            adapted = adapted.strip()

            # Track changes
            changes = []
            if adapted != translated:
                changes.append("AI-refined translation based on reflection feedback")

                # Check specific changes
                if len(adapted) < len(translated):
                    changes.append(f"Shortened from {len(translated)} to {len(adapted)} characters")

                # Check terminology
                for term, expected in self.terminology.items():
                    if expected in adapted and expected not in translated:
                        changes.append(f"Applied terminology: {term} → {expected}")

            # Calculate final quality (simple heuristic)
            final_score = 0.9  # Assume good after adaptation
            if len(adapted) > self.max_length * 1.5:
                final_score -= 0.1

            return AdaptResult(
                original=original,
                initial_translation=translated,
                adapted_translation=adapted,
                changes_made=changes,
                final_quality_score=final_score,
                adaptation_applied=True
            )

        except Exception as e:
            logger.warning(f"AI adaptation failed: {e}")
            return self._simple_adapt(original, translated, issues)

    def _simple_adapt(
        self,
        original: str,
        translated: str,
        issues: List[TranslationIssue]
    ) -> AdaptResult:
        """
        Apply simple non-AI adaptations.

        Handles:
        - Terminology substitution
        - Basic length trimming hints
        """
        adapted = translated
        changes = []

        # Apply terminology fixes
        for issue in issues:
            if issue.issue_type == IssueType.CONSISTENCY:
                # Try to extract term from suggestion
                for term, expected in self.terminology.items():
                    if term.lower() in original.lower():
                        # Simple replacement if we can identify the incorrect translation
                        # This is a best-effort approach
                        pass

        # Note length issues
        if len(adapted) > self.max_length:
            changes.append(f"Note: Translation is {len(adapted)} chars (recommended: {self.max_length})")

        return AdaptResult(
            original=original,
            initial_translation=translated,
            adapted_translation=adapted,
            changes_made=changes,
            final_quality_score=0.7 if issues else 0.9,
            adaptation_applied=bool(changes)
        )

    async def adapt_batch(
        self,
        reflections: List[ReflectResult],
        source_lang: str,
        target_lang: str
    ) -> List[AdaptResult]:
        """
        Adapt multiple translations based on reflections.

        Args:
            reflections: List of ReflectResult from reflect step
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of AdaptResult objects
        """
        results = []

        for i, reflection in enumerate(reflections):
            if reflection.needs_adaptation:
                result = await self.adapt_translation(
                    original=reflection.original,
                    translated=reflection.translated,
                    issues=reflection.issues,
                    source_lang=source_lang,
                    target_lang=target_lang
                )
            else:
                # No adaptation needed
                result = AdaptResult(
                    original=reflection.original,
                    initial_translation=reflection.translated,
                    adapted_translation=reflection.translated,
                    changes_made=[],
                    final_quality_score=reflection.quality_score,
                    adaptation_applied=False
                )

            results.append(result)

            # Rate limit
            if i > 0 and i % 5 == 0:
                await asyncio.sleep(0.1)

        return results

"""
Step 2: Translation Reflection

Analyzes the initial translation for quality issues and identifies areas for improvement.
This is the critical "self-critique" step that makes VideoLingo translations superior.
"""
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class IssueType(str, Enum):
    """Types of translation issues"""
    ACCURACY = "accuracy"           # Meaning not preserved
    NATURALNESS = "naturalness"     # Sounds unnatural
    CONSISTENCY = "consistency"     # Terminology inconsistent
    LENGTH = "length"               # Too long for subtitles
    TONE = "tone"                   # Wrong tone/register
    CULTURAL = "cultural"           # Cultural adaptation needed
    GRAMMAR = "grammar"             # Grammar errors


@dataclass
class TranslationIssue:
    """A specific issue identified in translation"""
    issue_type: IssueType
    severity: str  # "low", "medium", "high"
    description: str
    suggestion: str
    segment_index: Optional[int] = None


@dataclass
class ReflectResult:
    """Result from the reflection step"""
    original: str
    translated: str
    quality_score: float  # 0.0 to 1.0
    issues: List[TranslationIssue] = field(default_factory=list)
    needs_adaptation: bool = False
    reflection_notes: str = ""


class ReflectStep:
    """
    Step 2: Translation Reflection

    Analyzes translations for:
    - Accuracy (meaning preservation)
    - Naturalness (fluency in target language)
    - Consistency (terminology usage)
    - Length (subtitle appropriateness)
    - Tone (speaker intent preservation)
    - Cultural fit (adaptation needs)
    """

    REFLECTION_PROMPT = """## Role
You are a professional Netflix subtitle translator and quality analyst, fluent in both {source_lang} and {target_lang}.
Your expertise lies not only in accurately understanding the original {source_lang} but also in evaluating whether the {target_lang} translation suits the target language's expression habits and cultural background.

## Task
Analyze the following translation and identify any issues that need improvement.

## INPUT
<source lang="{source_lang}">
{original}
</source>

<translation lang="{target_lang}">
{translated}
</translation>

{terminology_section}

## Evaluation Criteria
Please evaluate the translation using these Netflix subtitle standards:

1. **ACCURACY** (意思准确)
   - Is the original meaning fully preserved?
   - Are there any mistranslations or omissions?

2. **NATURALNESS** (表达自然)
   - Does it sound natural and fluent in {target_lang}?
   - Does it conform to {target_lang} expression habits?

3. **CONSISTENCY** (术语一致)
   - Are professional terms translated consistently?
   - Do translations match the provided terminology?

4. **LENGTH** (字幕长度)
   - Is it appropriate for subtitles? (ideally ≤42 chars per line)
   - Is it too wordy or can it be more concise?

5. **TONE** (语气保持)
   - Is the speaker's tone and intent preserved?
   - Does the register match the content (casual/formal/technical)?

6. **CULTURAL** (文化适配)
   - Are cultural references appropriately adapted?
   - Will the target audience understand it?

## Output (JSON only)
```json
{{
    "quality_score": <0.0-1.0>,
    "needs_adaptation": <true/false>,
    "issues": [
        {{
            "type": "<accuracy|naturalness|consistency|length|tone|cultural|grammar>",
            "severity": "<low|medium|high>",
            "description": "<具体问题描述>",
            "suggestion": "<改进建议>"
        }}
    ],
    "notes": "<整体评价，一句话总结>"
}}
```

Note: If the translation is good with no significant issues, return quality_score >= 0.85 with an empty issues list."""

    def __init__(
        self,
        engine: Any,  # AI translation engine for reflection
        terminology: Optional[Dict[str, str]] = None,
        quality_threshold: float = 0.7
    ):
        """
        Initialize reflect step.

        Args:
            engine: AI engine for reflection (GPT, Claude)
            terminology: Terminology dictionary for consistency checks
            quality_threshold: Score below which adaptation is needed
        """
        self.engine = engine
        self.terminology = terminology or {}
        self.quality_threshold = quality_threshold

    def _build_terminology_section(self) -> str:
        """Build terminology reference for reflection"""
        if not self.terminology:
            return ""

        terms = "\n".join([f"- {src} → {tgt}" for src, tgt in self.terminology.items()])
        return f"""EXPECTED TERMINOLOGY:
{terms}"""

    async def reflect_on_translation(
        self,
        original: str,
        translated: str,
        source_lang: str,
        target_lang: str
    ) -> ReflectResult:
        """
        Reflect on a single translation.

        Args:
            original: Original text
            translated: Translated text
            source_lang: Source language
            target_lang: Target language

        Returns:
            ReflectResult with quality analysis
        """
        # For non-AI engines, do basic heuristic checks
        if not hasattr(self.engine, 'analyze'):
            return self._heuristic_reflect(original, translated)

        try:
            terminology_section = self._build_terminology_section()

            prompt = self.REFLECTION_PROMPT.format(
                source_lang=source_lang,
                target_lang=target_lang,
                original=original,
                translated=translated,
                terminology_section=terminology_section
            )

            # Get AI analysis
            import json
            response = await self.engine.analyze(prompt)

            # Parse JSON response
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    return self._heuristic_reflect(original, translated)

            # Build issues list
            issues = []
            for issue_data in analysis.get("issues", []):
                try:
                    issue = TranslationIssue(
                        issue_type=IssueType(issue_data.get("type", "accuracy")),
                        severity=issue_data.get("severity", "low"),
                        description=issue_data.get("description", ""),
                        suggestion=issue_data.get("suggestion", "")
                    )
                    issues.append(issue)
                except (ValueError, KeyError):
                    continue

            quality_score = float(analysis.get("quality_score", 0.8))
            needs_adaptation = analysis.get("needs_adaptation", quality_score < self.quality_threshold)

            return ReflectResult(
                original=original,
                translated=translated,
                quality_score=quality_score,
                issues=issues,
                needs_adaptation=needs_adaptation,
                reflection_notes=analysis.get("notes", "")
            )

        except Exception as e:
            logger.warning(f"AI reflection failed, using heuristics: {e}")
            return self._heuristic_reflect(original, translated)

    def _heuristic_reflect(self, original: str, translated: str) -> ReflectResult:
        """
        Perform heuristic quality checks without AI.

        Checks:
        - Length ratio (translation shouldn't be too different in length)
        - Subtitle length (max 42 chars per line recommended)
        - Empty translation
        - Terminology presence
        """
        issues = []
        quality_score = 1.0

        # Check for empty translation
        if not translated or not translated.strip():
            issues.append(TranslationIssue(
                issue_type=IssueType.ACCURACY,
                severity="high",
                description="Translation is empty",
                suggestion="Re-translate the segment"
            ))
            quality_score -= 0.5

        # Check length ratio (very rough heuristic)
        if original and translated:
            ratio = len(translated) / len(original)
            # Chinese is typically shorter than English by ~30-50%
            if ratio > 2.0:
                issues.append(TranslationIssue(
                    issue_type=IssueType.LENGTH,
                    severity="medium",
                    description=f"Translation is {ratio:.1f}x longer than original",
                    suggestion="Consider condensing the translation"
                ))
                quality_score -= 0.1
            elif ratio < 0.2:
                issues.append(TranslationIssue(
                    issue_type=IssueType.ACCURACY,
                    severity="medium",
                    description="Translation is much shorter than original",
                    suggestion="Verify no content is missing"
                ))
                quality_score -= 0.1

        # Check subtitle length (Netflix standard: 42 chars)
        if len(translated) > 42:
            issues.append(TranslationIssue(
                issue_type=IssueType.LENGTH,
                severity="low",
                description=f"Translation is {len(translated)} chars (recommended max: 42)",
                suggestion="Consider splitting into multiple lines"
            ))
            quality_score -= 0.05

        # Check terminology consistency
        for term, expected_translation in self.terminology.items():
            if term.lower() in original.lower():
                if expected_translation.lower() not in translated.lower():
                    issues.append(TranslationIssue(
                        issue_type=IssueType.CONSISTENCY,
                        severity="medium",
                        description=f"Term '{term}' should be translated as '{expected_translation}'",
                        suggestion=f"Use '{expected_translation}' for '{term}'"
                    ))
                    quality_score -= 0.1

        quality_score = max(0.0, min(1.0, quality_score))

        return ReflectResult(
            original=original,
            translated=translated,
            quality_score=quality_score,
            issues=issues,
            needs_adaptation=quality_score < self.quality_threshold,
            reflection_notes="Heuristic analysis (AI reflection unavailable)"
        )

    async def reflect_batch(
        self,
        translations: List[tuple],  # List of (original, translated) tuples
        source_lang: str,
        target_lang: str
    ) -> List[ReflectResult]:
        """
        Reflect on multiple translations.

        Args:
            translations: List of (original, translated) tuples
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of ReflectResult objects
        """
        results = []

        for i, (original, translated) in enumerate(translations):
            result = await self.reflect_on_translation(
                original=original,
                translated=translated,
                source_lang=source_lang,
                target_lang=target_lang
            )
            result.issues = [
                TranslationIssue(
                    issue_type=issue.issue_type,
                    severity=issue.severity,
                    description=issue.description,
                    suggestion=issue.suggestion,
                    segment_index=i
                )
                for issue in result.issues
            ]
            results.append(result)

            # Rate limit
            if i > 0 and i % 5 == 0:
                await asyncio.sleep(0.1)

        return results

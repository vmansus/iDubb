"""
Step 2: Expressiveness (VideoLingo-style)

Combines reflection and adaptation into a single step:
1. Analyze the direct translation, identify issues
2. Provide improved "free translation" based on analysis

This is more efficient than separate reflect + adapt steps.
"""
import asyncio
import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ExpressivenessResult:
    """Result from the expressiveness step"""
    original: str
    direct_translation: str
    reflection: str  # Analysis of issues
    free_translation: str  # Improved translation
    quality_score: float
    changes_made: List[str] = field(default_factory=list)


class ExpressivenessStep:
    """
    Step 2: Expressiveness (VideoLingo-style)
    
    Combines reflection and free translation into one LLM call:
    - Analyzes the direct translation for issues
    - Provides improved translation based on analysis
    
    This is more efficient and produces better results than
    separate reflect + adapt steps.
    """

    EXPRESSIVENESS_PROMPT = """## Role
You are a professional Netflix subtitle translator and language consultant.
Your expertise lies not only in accurately understanding the original {source_lang} but also in optimizing the {target_lang} translation to better suit the target language's expression habits and cultural background.

## Task
We already have a direct translation of the original {source_lang} subtitles.
Your task is to reflect on and improve this direct translation to create more natural and fluent {target_lang} subtitles.

## INPUT
<source lang="{source_lang}">
{original}
</source>

<direct_translation lang="{target_lang}">
{translated}
</direct_translation>

{terminology_section}

{context_section}

## Translation Analysis Steps
Please use a two-step thinking process:

### Step 1: Direct Translation Reflection
- Evaluate language fluency: Does it sound natural?
- Check language style consistency with the original
- Check subtitle conciseness: Is it too wordy? (ideally ≤42 chars)
- Verify terminology usage
- Assess tone preservation

### Step 2: Free Translation
Based on your reflection, provide an improved translation that:
- Flows naturally and smoothly in {target_lang}
- Conforms to {target_lang} expression habits
- Is easy for {target_lang} audience to understand
- Matches the content style (casual for vlogs, technical for tutorials, formal for documentaries)
- Stays concise for subtitle readability

## Output (JSON only)
```json
{{
    "reflection": "Brief analysis of issues found in the direct translation (1-2 sentences)",
    "free_translation": "Your improved {target_lang} translation",
    "quality_score": <0.0-1.0 rating of your free translation>,
    "changes": ["list", "of", "improvements", "made"]
}}
```

## Important
- Do NOT add comments or explanations in the free_translation
- The free_translation should be ready for subtitle display
- If the direct translation is already good, you may keep it with minor refinements

Note: Start your answer with ```json and end with ```, do not add any other text."""

    def __init__(
        self,
        engine: Any,
        terminology: Optional[Dict[str, str]] = None,
        quality_threshold: float = 0.7
    ):
        """
        Initialize expressiveness step.
        
        Args:
            engine: AI engine (GPT, Claude, etc.)
            terminology: Term -> translation mappings
            quality_threshold: Minimum acceptable quality score
        """
        self.engine = engine
        self.terminology = terminology or {}
        self.quality_threshold = quality_threshold

    def _build_terminology_section(self) -> str:
        """Build terminology section for prompt"""
        if not self.terminology:
            return ""
        
        terms = "\n".join([f"- {src} → {tgt}" for src, tgt in self.terminology.items()])
        return f"""## Terminology (use consistently)
{terms}"""

    def _build_context_section(
        self, 
        prev_segments: Optional[List[str]] = None,
        next_segments: Optional[List[str]] = None
    ) -> str:
        """Build context section for prompt"""
        parts = []
        
        if prev_segments:
            prev_text = " | ".join(prev_segments[-2:])  # Last 2 segments
            parts.append(f"Previous: {prev_text}")
        
        if next_segments:
            next_text = " | ".join(next_segments[:2])  # Next 2 segments
            parts.append(f"Following: {next_text}")
        
        if parts:
            return "## Context\n" + "\n".join(parts)
        return ""

    async def improve_translation(
        self,
        original: str,
        direct_translation: str,
        source_lang: str,
        target_lang: str,
        prev_context: Optional[List[str]] = None,
        next_context: Optional[List[str]] = None
    ) -> ExpressivenessResult:
        """
        Improve a direct translation with reflection and free translation.
        
        Args:
            original: Original text
            direct_translation: Initial direct translation
            source_lang: Source language
            target_lang: Target language
            prev_context: Previous translated segments for context
            next_context: Next original segments for context
            
        Returns:
            ExpressivenessResult with improved translation
        """
        # For non-AI engines, return as-is
        if not hasattr(self.engine, 'analyze') and not hasattr(self.engine, 'complete'):
            return ExpressivenessResult(
                original=original,
                direct_translation=direct_translation,
                reflection="No AI engine available",
                free_translation=direct_translation,
                quality_score=0.8,
                changes_made=[]
            )

        try:
            terminology_section = self._build_terminology_section()
            context_section = self._build_context_section(prev_context, next_context)
            
            prompt = self.EXPRESSIVENESS_PROMPT.format(
                source_lang=source_lang,
                target_lang=target_lang,
                original=original,
                translated=direct_translation,
                terminology_section=terminology_section,
                context_section=context_section
            )

            # Call AI
            if hasattr(self.engine, 'analyze'):
                response = await self.engine.analyze(prompt)
            elif hasattr(self.engine, 'complete'):
                response = await self.engine.complete(prompt)
            else:
                response = await self._fallback_call(prompt)

            # Parse response
            result = self._parse_response(response, original, direct_translation)
            return result

        except Exception as e:
            logger.warning(f"Expressiveness step failed: {e}")
            return ExpressivenessResult(
                original=original,
                direct_translation=direct_translation,
                reflection=f"Error: {str(e)}",
                free_translation=direct_translation,
                quality_score=0.7,
                changes_made=[]
            )

    async def _fallback_call(self, prompt: str) -> str:
        """Fallback method to call the engine"""
        if hasattr(self.engine, 'translate_with_prompt'):
            return await self.engine.translate_with_prompt(
                text=prompt,
                system_prompt="You are a translation quality analyst.",
                source_lang="en",
                target_lang="zh"
            )
        return ""

    def _parse_response(
        self, 
        response: str, 
        original: str, 
        direct_translation: str
    ) -> ExpressivenessResult:
        """Parse AI response into ExpressivenessResult"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # Try parsing the whole response as JSON
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            return ExpressivenessResult(
                original=original,
                direct_translation=direct_translation,
                reflection=data.get("reflection", ""),
                free_translation=data.get("free_translation", direct_translation),
                quality_score=float(data.get("quality_score", 0.85)),
                changes_made=data.get("changes", [])
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse expressiveness response: {e}")
            # Return direct translation if parsing fails
            return ExpressivenessResult(
                original=original,
                direct_translation=direct_translation,
                reflection="Failed to parse AI response",
                free_translation=direct_translation,
                quality_score=0.75,
                changes_made=[]
            )

    async def improve_batch(
        self,
        translations: List[tuple],  # List of (original, direct_translation)
        source_lang: str,
        target_lang: str,
        all_translations: Optional[List[str]] = None  # For context
    ) -> List[ExpressivenessResult]:
        """
        Improve multiple translations.
        
        Args:
            translations: List of (original, direct_translation) tuples
            source_lang: Source language
            target_lang: Target language
            all_translations: All translations for context building
            
        Returns:
            List of ExpressivenessResult objects
        """
        results = []
        
        for i, (original, direct_trans) in enumerate(translations):
            # Build context from surrounding translations
            prev_context = None
            next_context = None
            
            if all_translations:
                if i > 0:
                    prev_context = all_translations[max(0, i-2):i]
                if i < len(translations) - 1:
                    next_context = [t[0] for t in translations[i+1:i+3]]  # Next originals
            
            result = await self.improve_translation(
                original=original,
                direct_translation=direct_trans,
                source_lang=source_lang,
                target_lang=target_lang,
                prev_context=prev_context,
                next_context=next_context
            )
            results.append(result)
            
            # Rate limiting
            if i > 0 and i % 5 == 0:
                await asyncio.sleep(0.1)
        
        return results

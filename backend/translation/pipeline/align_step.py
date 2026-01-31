"""
Subtitle Alignment Step

Ensures translated subtitles align properly with source subtitles.
When source subtitles are split (e.g., for timing), the translation
should be split correspondingly to maintain synchronization.

Based on VideoLingo's alignment approach.
"""
import asyncio
import json
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class AlignedSegment:
    """A single aligned subtitle segment"""
    source: str
    translation: str
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass 
class AlignmentResult:
    """Result from subtitle alignment"""
    success: bool
    segments: List[AlignedSegment]
    original_source: str
    original_translation: str
    split_count: int
    error: Optional[str] = None


class SubtitleAligner:
    """
    Aligns translated subtitles with source subtitle splits.
    
    When a source subtitle is split into multiple parts (for timing/length),
    the translation should be split correspondingly while maintaining meaning.
    
    Example:
        Source: "Hello everyone, welcome to my channel"
        Split: ["Hello everyone,", "welcome to my channel"]
        Translation: "大家好，欢迎来到我的频道"
        Aligned: ["大家好，", "欢迎来到我的频道"]
    """

    ALIGN_PROMPT = """## Role
You are a Netflix subtitle alignment expert fluent in both {source_lang} and {target_lang}.

## Task
We have {source_lang} and {target_lang} subtitles for a video, and the {source_lang} subtitle has been split into multiple parts for timing purposes.
Your task is to split the {target_lang} subtitle to match the {source_lang} splits while preserving meaning.

## INPUT
<source_original>
{source_original}
</source_original>

<translation_original>
{translation_original}
</translation_original>

<source_splits>
{source_splits}
</source_splits>

## Alignment Rules
1. Split the {target_lang} translation into exactly {num_parts} parts
2. Each part should correspond semantically to the matching {source_lang} part
3. Never leave empty parts - if hard to split by meaning, you may slightly rewrite
4. Maintain natural sentence flow in each part
5. Keep splits roughly proportional in length to source splits

## Output (JSON only)
```json
{{
    "analysis": "Brief analysis of how the splits should align semantically",
    "aligned_parts": [
        {{
            "source": "source part 1",
            "translation": "corresponding translation part 1"
        }},
        {{
            "source": "source part 2", 
            "translation": "corresponding translation part 2"
        }}
    ]
}}
```

Note: The number of aligned_parts must exactly match the number of source_splits ({num_parts}).
Start your answer with ```json and end with ```, do not add any other text."""

    def __init__(self, engine: Any):
        """
        Initialize subtitle aligner.
        
        Args:
            engine: AI engine for alignment (GPT, Claude, etc.)
        """
        self.engine = engine

    async def align_subtitle(
        self,
        source_original: str,
        translation_original: str,
        source_splits: List[str],
        source_lang: str = "en",
        target_lang: str = "zh"
    ) -> AlignmentResult:
        """
        Align a translated subtitle with source splits.
        
        Args:
            source_original: Original full source text
            translation_original: Full translation
            source_splits: Source text split into parts
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            AlignmentResult with aligned segments
        """
        num_parts = len(source_splits)
        
        # If only one part, no alignment needed
        if num_parts <= 1:
            return AlignmentResult(
                success=True,
                segments=[AlignedSegment(
                    source=source_original,
                    translation=translation_original
                )],
                original_source=source_original,
                original_translation=translation_original,
                split_count=1
            )

        # For non-AI engines, use simple proportional split
        if not hasattr(self.engine, 'analyze') and not hasattr(self.engine, 'complete'):
            return self._simple_align(
                source_original, translation_original, source_splits
            )

        try:
            # Format source splits for prompt
            splits_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(source_splits)])
            
            prompt = self.ALIGN_PROMPT.format(
                source_lang=source_lang,
                target_lang=target_lang,
                source_original=source_original,
                translation_original=translation_original,
                source_splits=splits_text,
                num_parts=num_parts
            )

            # Call AI
            if hasattr(self.engine, 'analyze'):
                response = await self.engine.analyze(prompt)
            elif hasattr(self.engine, 'complete'):
                response = await self.engine.complete(prompt)
            else:
                return self._simple_align(
                    source_original, translation_original, source_splits
                )

            # Parse response
            result = self._parse_response(
                response, source_original, translation_original, source_splits
            )
            return result

        except Exception as e:
            logger.warning(f"Alignment failed: {e}, using simple split")
            return self._simple_align(
                source_original, translation_original, source_splits
            )

    def _parse_response(
        self,
        response: str,
        source_original: str,
        translation_original: str,
        source_splits: List[str]
    ) -> AlignmentResult:
        """Parse AI response into AlignmentResult"""
        try:
            # Extract JSON
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found")

            aligned_parts = data.get("aligned_parts", [])
            
            # Validate we got the right number of parts
            if len(aligned_parts) != len(source_splits):
                logger.warning(
                    f"Alignment mismatch: expected {len(source_splits)}, "
                    f"got {len(aligned_parts)}"
                )
                return self._simple_align(
                    source_original, translation_original, source_splits
                )

            segments = []
            for i, part in enumerate(aligned_parts):
                segments.append(AlignedSegment(
                    source=part.get("source", source_splits[i]),
                    translation=part.get("translation", "")
                ))

            return AlignmentResult(
                success=True,
                segments=segments,
                original_source=source_original,
                original_translation=translation_original,
                split_count=len(segments)
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse alignment response: {e}")
            return self._simple_align(
                source_original, translation_original, source_splits
            )

    def _simple_align(
        self,
        source_original: str,
        translation_original: str,
        source_splits: List[str]
    ) -> AlignmentResult:
        """
        Simple proportional alignment without AI.
        Splits translation based on relative lengths of source splits.
        """
        num_parts = len(source_splits)
        
        # Calculate proportional lengths
        source_total = sum(len(s) for s in source_splits)
        if source_total == 0:
            source_total = 1
        
        proportions = [len(s) / source_total for s in source_splits]
        
        # Try to split by punctuation first
        translation_parts = self._split_by_punctuation(
            translation_original, num_parts
        )
        
        if len(translation_parts) == num_parts:
            segments = [
                AlignedSegment(source=src, translation=trans)
                for src, trans in zip(source_splits, translation_parts)
            ]
        else:
            # Fall back to proportional character split
            translation_parts = self._proportional_split(
                translation_original, proportions
            )
            segments = [
                AlignedSegment(source=src, translation=trans)
                for src, trans in zip(source_splits, translation_parts)
            ]

        return AlignmentResult(
            success=True,
            segments=segments,
            original_source=source_original,
            original_translation=translation_original,
            split_count=num_parts
        )

    def _split_by_punctuation(self, text: str, num_parts: int) -> List[str]:
        """Try to split text by punctuation marks"""
        # Common Chinese and English punctuation
        punctuation = r'[，。！？；：,.!?;:\n]'
        
        # Find all punctuation positions
        parts = re.split(f'({punctuation})', text)
        
        # Recombine parts with their punctuation
        combined = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if i + 1 < len(parts) and re.match(punctuation, parts[i + 1]):
                part += parts[i + 1]
                i += 2
            else:
                i += 1
            if part.strip():
                combined.append(part.strip())
        
        # If we got the right number of parts, return them
        if len(combined) == num_parts:
            return combined
        
        # Try to merge or split to get the right number
        if len(combined) > num_parts:
            # Merge smallest adjacent parts
            while len(combined) > num_parts:
                min_idx = 0
                min_len = float('inf')
                for i in range(len(combined) - 1):
                    total_len = len(combined[i]) + len(combined[i + 1])
                    if total_len < min_len:
                        min_len = total_len
                        min_idx = i
                combined[min_idx] = combined[min_idx] + combined[min_idx + 1]
                combined.pop(min_idx + 1)
            return combined
        
        # Not enough parts, return empty to trigger proportional split
        return []

    def _proportional_split(
        self, 
        text: str, 
        proportions: List[float]
    ) -> List[str]:
        """Split text proportionally by character count"""
        total_len = len(text)
        parts = []
        start = 0
        
        for i, prop in enumerate(proportions[:-1]):
            end = start + int(total_len * prop)
            
            # Try to find a good split point (space or punctuation)
            best_split = end
            for offset in range(min(10, end - start)):
                check_pos = end - offset
                if check_pos > start and check_pos < total_len:
                    char = text[check_pos]
                    if char in ' ，。！？；：,.!?;:\n':
                        best_split = check_pos + 1
                        break
            
            parts.append(text[start:best_split].strip())
            start = best_split
        
        # Last part gets the rest
        parts.append(text[start:].strip())
        
        return parts

    async def align_batch(
        self,
        items: List[Tuple[str, str, List[str]]],  # (source, translation, splits)
        source_lang: str = "en",
        target_lang: str = "zh"
    ) -> List[AlignmentResult]:
        """
        Align multiple subtitle pairs.
        
        Args:
            items: List of (source_original, translation_original, source_splits)
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            List of AlignmentResult objects
        """
        results = []
        
        for i, (source, translation, splits) in enumerate(items):
            result = await self.align_subtitle(
                source_original=source,
                translation_original=translation,
                source_splits=splits,
                source_lang=source_lang,
                target_lang=target_lang
            )
            results.append(result)
            
            # Rate limiting
            if i > 0 and i % 10 == 0:
                await asyncio.sleep(0.1)
        
        return results

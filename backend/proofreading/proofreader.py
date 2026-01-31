"""
Main subtitle proofreader that orchestrates all validation checks
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from .models import (
    ProofreadingResult, SegmentProofreadResult, ProofreadingConfig,
    ProofreadingIssue, IssueSeverity
)
from .validators import (
    TimingValidator, FormatValidator, TerminologyValidator, AIValidator
)


class SubtitleProofreader:
    """
    Main proofreader class that orchestrates subtitle validation.

    Performs the following checks:
    1. Timing validation (speech rate, gaps, overlaps)
    2. Format validation (encoding, empty segments, line length)
    3. Terminology consistency
    4. AI-powered translation quality check
    """

    def __init__(self, config: Optional[ProofreadingConfig] = None):
        self.config = config or ProofreadingConfig()
        self.timing_validator = TimingValidator(self.config)
        self.format_validator = FormatValidator(self.config)
        self.terminology_validator = TerminologyValidator(self.config)
        self.ai_validator = AIValidator(self.config)

    async def proofread(
        self,
        original_segments: List[Dict[str, Any]],
        translated_segments: List[Dict[str, Any]],
        source_lang: str = "en",
        target_lang: str = "zh"
    ) -> ProofreadingResult:
        """
        Proofread subtitles and return detailed results.

        Args:
            original_segments: Original language subtitle segments
            translated_segments: Translated subtitle segments
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            ProofreadingResult with all issues and confidence scores
        """
        logger.info(f"Starting proofreading: {len(original_segments)} segments")

        all_issues: List[ProofreadingIssue] = []
        terminology_score = 1.0

        # Ensure segments match
        if len(original_segments) != len(translated_segments):
            logger.warning(
                f"Segment count mismatch: {len(original_segments)} vs {len(translated_segments)}"
            )
            # Pad shorter list
            while len(translated_segments) < len(original_segments):
                translated_segments.append({"text": "", "start": 0, "end": 0})
            while len(original_segments) < len(translated_segments):
                original_segments.append({"text": "", "start": 0, "end": 0})

        # Run timing validation
        if self.config.check_timing:
            logger.debug("Running timing validation...")
            timing_issues = self.timing_validator.validate(
                original_segments, translated_segments
            )
            all_issues.extend(timing_issues)
            logger.info(f"Timing validation found {len(timing_issues)} issues")

        # Run format validation
        if self.config.check_formatting:
            logger.debug("Running format validation...")
            format_issues = self.format_validator.validate(
                original_segments, translated_segments
            )
            all_issues.extend(format_issues)
            logger.info(f"Format validation found {len(format_issues)} issues")

        # Run terminology validation
        if self.config.check_terminology:
            logger.debug("Running terminology validation...")
            term_issues, terminology_score = self.terminology_validator.validate(
                original_segments, translated_segments
            )
            all_issues.extend(term_issues)
            logger.info(
                f"Terminology validation found {len(term_issues)} issues, "
                f"consistency score: {terminology_score:.2f}"
            )

        # Run AI validation (async)
        if self.config.check_grammar and self.config.use_ai_validation:
            logger.debug("Running AI validation...")
            ai_issues = await self.ai_validator.validate(
                original_segments, translated_segments,
                source_lang, target_lang
            )
            all_issues.extend(ai_issues)
            logger.info(f"AI validation found {len(ai_issues)} issues")

        # Build segment results
        segment_results = self._build_segment_results(
            original_segments, translated_segments, all_issues
        )

        # Calculate statistics
        avg_cps = self._calculate_avg_chars_per_second(
            original_segments, translated_segments
        )

        # Build final result
        result = ProofreadingResult.from_segments(segment_results)
        result.avg_chars_per_second = avg_cps
        result.terminology_consistency_score = terminology_score

        # Check if should pause based on config threshold
        if result.overall_confidence < self.config.min_confidence_threshold:
            result.should_pause = True
            if not result.pause_reason:
                result.pause_reason = (
                    f"置信度 ({result.overall_confidence:.1%}) 低于阈值 "
                    f"({self.config.min_confidence_threshold:.1%})"
                )

        logger.info(
            f"Proofreading complete: {result.total_issues} issues, "
            f"confidence: {result.overall_confidence:.2f}, "
            f"should_pause: {result.should_pause}"
        )

        return result

    def _build_segment_results(
        self,
        original_segments: List[Dict[str, Any]],
        translated_segments: List[Dict[str, Any]],
        all_issues: List[ProofreadingIssue]
    ) -> List[SegmentProofreadResult]:
        """Build per-segment results with issues"""
        # Group issues by segment index
        issues_by_segment: Dict[int, List[ProofreadingIssue]] = {}
        for issue in all_issues:
            idx = issue.segment_index
            if idx not in issues_by_segment:
                issues_by_segment[idx] = []
            issues_by_segment[idx].append(issue)

        # Build results
        results = []
        for i, (orig, trans) in enumerate(zip(original_segments, translated_segments)):
            segment_issues = issues_by_segment.get(i, [])

            # Calculate segment confidence
            confidence = self._calculate_segment_confidence(segment_issues)

            results.append(SegmentProofreadResult(
                index=i,
                original_text=orig.get("text", ""),
                translated_text=trans.get("text", ""),
                start_time=orig.get("start", 0),
                end_time=orig.get("end", 0),
                issues=segment_issues,
                confidence=confidence,
            ))

        return results

    def _calculate_segment_confidence(
        self,
        issues: List[ProofreadingIssue]
    ) -> float:
        """Calculate confidence score for a segment based on its issues"""
        if not issues:
            return 1.0

        # Penalty weights by severity
        penalties = {
            IssueSeverity.CRITICAL: 0.4,
            IssueSeverity.ERROR: 0.2,
            IssueSeverity.WARNING: 0.1,
            IssueSeverity.INFO: 0.02,
        }

        total_penalty = 0.0
        for issue in issues:
            total_penalty += penalties.get(issue.severity, 0.05)

        # Confidence is 1 minus penalty, but at least 0
        return max(0.0, 1.0 - total_penalty)

    def _calculate_avg_chars_per_second(
        self,
        original_segments: List[Dict[str, Any]],
        translated_segments: List[Dict[str, Any]]
    ) -> float:
        """Calculate average characters per second across all segments"""
        total_chars = 0
        total_duration = 0.0

        for orig, trans in zip(original_segments, translated_segments):
            duration = orig.get("end", 0) - orig.get("start", 0)
            if duration > 0:
                trans_text = trans.get("text", "")
                total_chars += len(trans_text.replace(" ", ""))
                total_duration += duration

        if total_duration > 0:
            return total_chars / total_duration
        return 0.0

    async def quick_check(
        self,
        original_segments: List[Dict[str, Any]],
        translated_segments: List[Dict[str, Any]]
    ) -> bool:
        """
        Quick check to determine if proofreading should pause the task.
        Returns True if there are critical issues.
        """
        # Only run fast checks
        issues = []

        if self.config.check_formatting:
            issues.extend(self.format_validator.validate(
                original_segments, translated_segments
            ))

        if self.config.check_timing:
            issues.extend(self.timing_validator.validate(
                original_segments, translated_segments
            ))

        # Check for critical or error severity
        has_serious_issues = any(
            i.severity in [IssueSeverity.CRITICAL, IssueSeverity.ERROR]
            for i in issues
        )

        return has_serious_issues


# Global instance for convenience
_proofreader: Optional[SubtitleProofreader] = None


def get_proofreader(config: Optional[ProofreadingConfig] = None) -> SubtitleProofreader:
    """Get or create the global proofreader instance"""
    global _proofreader
    if _proofreader is None or config is not None:
        _proofreader = SubtitleProofreader(config)
    return _proofreader


async def proofread_subtitles(
    original_segments: List[Dict[str, Any]],
    translated_segments: List[Dict[str, Any]],
    source_lang: str = "en",
    target_lang: str = "zh",
    config: Optional[ProofreadingConfig] = None
) -> ProofreadingResult:
    """
    Convenience function to proofread subtitles.

    Args:
        original_segments: Original subtitle segments with text, start, end
        translated_segments: Translated subtitle segments
        source_lang: Source language code
        target_lang: Target language code
        config: Optional proofreading configuration

    Returns:
        ProofreadingResult with all findings
    """
    proofreader = get_proofreader(config)
    return await proofreader.proofread(
        original_segments, translated_segments,
        source_lang, target_lang
    )

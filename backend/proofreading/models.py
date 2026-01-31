"""
Proofreading data models
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class IssueSeverity(Enum):
    """Severity levels for proofreading issues"""
    INFO = "info"           # Minor suggestion
    WARNING = "warning"     # Should be reviewed
    ERROR = "error"         # Likely needs correction
    CRITICAL = "critical"   # Must be fixed before proceeding


class IssueType(Enum):
    """Types of proofreading issues"""
    # Translation issues
    GRAMMAR_ERROR = "grammar_error"
    UNNATURAL_PHRASING = "unnatural_phrasing"
    MISTRANSLATION = "mistranslation"
    MISSING_TRANSLATION = "missing_translation"

    # Terminology issues
    INCONSISTENT_TERM = "inconsistent_term"
    UNKNOWN_TERM = "unknown_term"

    # Timing issues
    SPEECH_TOO_FAST = "speech_too_fast"
    SPEECH_TOO_SLOW = "speech_too_slow"
    OVERLAP_DETECTED = "overlap_detected"
    GAP_TOO_LONG = "gap_too_long"

    # Format issues
    ENCODING_ERROR = "encoding_error"
    EMPTY_SEGMENT = "empty_segment"
    LINE_TOO_LONG = "line_too_long"


@dataclass
class ProofreadingIssue:
    """A single proofreading issue"""
    segment_index: int
    issue_type: IssueType
    severity: IssueSeverity
    message: str
    original_text: str = ""
    translated_text: str = ""
    suggestion: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    auto_fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "suggestion": self.suggestion,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "auto_fixable": self.auto_fixable,
        }


@dataclass
class SegmentProofreadResult:
    """Proofreading result for a single segment"""
    index: int
    original_text: str
    translated_text: str
    start_time: float
    end_time: float
    issues: List[ProofreadingIssue] = field(default_factory=list)
    confidence: float = 1.0  # 0.0 to 1.0
    corrected_text: Optional[str] = None

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)

    @property
    def has_errors(self) -> bool:
        return any(i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL] for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "issues": [i.to_dict() for i in self.issues],
            "confidence": self.confidence,
            "corrected_text": self.corrected_text,
        }


@dataclass
class ProofreadingResult:
    """Complete proofreading result for all segments"""
    segments: List[SegmentProofreadResult]
    overall_confidence: float = 1.0
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0

    # Statistics
    avg_chars_per_second: float = 0.0
    terminology_consistency_score: float = 1.0

    # Recommendation
    should_pause: bool = False
    pause_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "overall_confidence": self.overall_confidence,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "error_issues": self.error_issues,
            "warning_issues": self.warning_issues,
            "avg_chars_per_second": self.avg_chars_per_second,
            "terminology_consistency_score": self.terminology_consistency_score,
            "should_pause": self.should_pause,
            "pause_reason": self.pause_reason,
        }

    @classmethod
    def from_segments(cls, segments: List[SegmentProofreadResult]) -> "ProofreadingResult":
        """Create result from segment results"""
        total_issues = sum(len(s.issues) for s in segments)
        critical = sum(1 for s in segments for i in s.issues if i.severity == IssueSeverity.CRITICAL)
        errors = sum(1 for s in segments for i in s.issues if i.severity == IssueSeverity.ERROR)
        warnings = sum(1 for s in segments for i in s.issues if i.severity == IssueSeverity.WARNING)

        # Calculate overall confidence
        if segments:
            overall_confidence = sum(s.confidence for s in segments) / len(segments)
        else:
            overall_confidence = 1.0

        # Determine if should pause
        should_pause = critical > 0 or overall_confidence < 0.6
        pause_reason = ""
        if critical > 0:
            pause_reason = f"发现 {critical} 个严重问题需要人工确认"
        elif overall_confidence < 0.6:
            pause_reason = f"整体置信度过低 ({overall_confidence:.1%})，建议人工检查"

        return cls(
            segments=segments,
            overall_confidence=overall_confidence,
            total_issues=total_issues,
            critical_issues=critical,
            error_issues=errors,
            warning_issues=warnings,
            should_pause=should_pause,
            pause_reason=pause_reason,
        )


@dataclass
class ProofreadingConfig:
    """Configuration for proofreading"""
    # Enable/disable checks
    check_grammar: bool = True
    check_terminology: bool = True
    check_timing: bool = True
    check_formatting: bool = True

    # Thresholds
    min_confidence_threshold: float = 0.6  # Below this, pause task
    max_chars_per_second: float = 25.0     # Chinese characters per second
    min_chars_per_second: float = 2.0
    max_line_length: int = 40              # Characters per line

    # AI settings
    use_ai_validation: bool = True
    ai_model: str = "gpt-4o-mini"  # or claude-3-haiku

    # Auto-fix settings
    auto_fix_minor_issues: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_grammar": self.check_grammar,
            "check_terminology": self.check_terminology,
            "check_timing": self.check_timing,
            "check_formatting": self.check_formatting,
            "min_confidence_threshold": self.min_confidence_threshold,
            "max_chars_per_second": self.max_chars_per_second,
            "min_chars_per_second": self.min_chars_per_second,
            "max_line_length": self.max_line_length,
            "use_ai_validation": self.use_ai_validation,
            "ai_model": self.ai_model,
            "auto_fix_minor_issues": self.auto_fix_minor_issues,
        }

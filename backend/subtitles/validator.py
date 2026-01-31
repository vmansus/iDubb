"""
Netflix Subtitle Standards Validator

Validates subtitles against Netflix Timed Text Style Guide:
https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617

Key Standards:
- Maximum 42 characters per line
- Maximum 2 lines per subtitle
- Minimum 1 second display duration
- Maximum reading speed: 17 characters/second (children), 20 chars/sec (adult)
- Minimum gap between subtitles: 2 frames (approx 83ms at 24fps)
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class IssueSeverity(str, Enum):
    """Severity levels for validation issues"""
    ERROR = "error"      # Must fix
    WARNING = "warning"  # Should fix
    INFO = "info"        # Suggestion


class IssueType(str, Enum):
    """Types of subtitle validation issues"""
    LINE_LENGTH = "line_length"
    LINE_COUNT = "line_count"
    DURATION_SHORT = "duration_short"
    DURATION_LONG = "duration_long"
    READING_SPEED = "reading_speed"
    GAP_TOO_SHORT = "gap_too_short"
    EMPTY_SUBTITLE = "empty_subtitle"
    TIMING_OVERLAP = "timing_overlap"
    LINE_BREAK = "line_break"


@dataclass
class ValidationIssue:
    """A single validation issue"""
    issue_type: IssueType
    severity: IssueSeverity
    message: str
    segment_index: Optional[int] = None
    line_number: Optional[int] = None
    current_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class SubtitleValidationResult:
    """Result of subtitle validation"""
    is_valid: bool
    total_segments: int
    issues: List[ValidationIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    @property
    def summary(self) -> str:
        """Get human-readable summary"""
        status = "✓ Valid" if self.is_valid else "✗ Invalid"
        return (
            f"{status}: {self.total_segments} segments, "
            f"{self.error_count} errors, {self.warning_count} warnings"
        )


@dataclass
class NetflixConfig:
    """Netflix subtitle validation configuration"""
    # Character limits
    max_chars_per_line: int = 42
    max_lines_per_subtitle: int = 2

    # Timing limits
    min_duration_seconds: float = 0.833  # 20 frames at 24fps (Netflix minimum)
    max_duration_seconds: float = 7.0
    min_gap_seconds: float = 0.083  # 2 frames at 24fps

    # Reading speed (characters per second)
    max_reading_speed_adult: float = 20.0
    max_reading_speed_children: float = 17.0

    # Content type
    is_children_content: bool = False

    # Strictness
    strict_mode: bool = True  # Treat warnings as errors


class NetflixValidator:
    """
    Validates subtitles against Netflix Timed Text Style Guide.

    Usage:
        validator = NetflixValidator()
        result = validator.validate_subtitles(segments)
        if not result.is_valid:
            for issue in result.issues:
                print(f"{issue.severity}: {issue.message}")
    """

    def __init__(self, config: Optional[NetflixConfig] = None):
        """
        Initialize validator.

        Args:
            config: Validation configuration (uses Netflix defaults if not provided)
        """
        self.config = config or NetflixConfig()

    def validate_segment(
        self,
        text: str,
        start_time: float,
        end_time: float,
        segment_index: int,
        prev_end_time: Optional[float] = None
    ) -> List[ValidationIssue]:
        """
        Validate a single subtitle segment.

        Args:
            text: Subtitle text
            start_time: Start time in seconds
            end_time: End time in seconds
            segment_index: Index of this segment
            prev_end_time: End time of previous segment (for gap check)

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        # Check for empty subtitle
        if not text or not text.strip():
            issues.append(ValidationIssue(
                issue_type=IssueType.EMPTY_SUBTITLE,
                severity=IssueSeverity.ERROR,
                message="Empty subtitle text",
                segment_index=segment_index
            ))
            return issues

        # Split into lines
        lines = text.strip().split('\n')

        # Check line count
        if len(lines) > self.config.max_lines_per_subtitle:
            issues.append(ValidationIssue(
                issue_type=IssueType.LINE_COUNT,
                severity=IssueSeverity.ERROR,
                message=f"Too many lines: {len(lines)} (max: {self.config.max_lines_per_subtitle})",
                segment_index=segment_index,
                current_value=len(lines),
                expected_value=self.config.max_lines_per_subtitle,
                suggestion="Split into multiple subtitle segments"
            ))

        # Check each line length
        for line_idx, line in enumerate(lines):
            line_length = len(line.strip())
            if line_length > self.config.max_chars_per_line:
                issues.append(ValidationIssue(
                    issue_type=IssueType.LINE_LENGTH,
                    severity=IssueSeverity.ERROR,
                    message=f"Line {line_idx + 1} too long: {line_length} chars (max: {self.config.max_chars_per_line})",
                    segment_index=segment_index,
                    line_number=line_idx + 1,
                    current_value=line_length,
                    expected_value=self.config.max_chars_per_line,
                    suggestion=self._suggest_line_break(line)
                ))

        # Check duration
        duration = end_time - start_time

        if duration < self.config.min_duration_seconds:
            issues.append(ValidationIssue(
                issue_type=IssueType.DURATION_SHORT,
                severity=IssueSeverity.WARNING,
                message=f"Duration too short: {duration:.2f}s (min: {self.config.min_duration_seconds:.2f}s)",
                segment_index=segment_index,
                current_value=duration,
                expected_value=self.config.min_duration_seconds,
                suggestion="Extend duration or merge with adjacent subtitle"
            ))

        if duration > self.config.max_duration_seconds:
            issues.append(ValidationIssue(
                issue_type=IssueType.DURATION_LONG,
                severity=IssueSeverity.WARNING,
                message=f"Duration too long: {duration:.2f}s (max: {self.config.max_duration_seconds:.2f}s)",
                segment_index=segment_index,
                current_value=duration,
                expected_value=self.config.max_duration_seconds,
                suggestion="Split into multiple subtitles"
            ))

        # Check reading speed
        total_chars = sum(len(line.strip()) for line in lines)
        if duration > 0:
            reading_speed = total_chars / duration
            max_speed = (self.config.max_reading_speed_children
                        if self.config.is_children_content
                        else self.config.max_reading_speed_adult)

            if reading_speed > max_speed:
                issues.append(ValidationIssue(
                    issue_type=IssueType.READING_SPEED,
                    severity=IssueSeverity.WARNING,
                    message=f"Reading speed too fast: {reading_speed:.1f} chars/sec (max: {max_speed:.1f})",
                    segment_index=segment_index,
                    current_value=reading_speed,
                    expected_value=max_speed,
                    suggestion="Extend duration or shorten text"
                ))

        # Check gap from previous subtitle
        if prev_end_time is not None:
            gap = start_time - prev_end_time
            if gap < 0:
                issues.append(ValidationIssue(
                    issue_type=IssueType.TIMING_OVERLAP,
                    severity=IssueSeverity.ERROR,
                    message=f"Overlaps with previous subtitle by {abs(gap):.3f}s",
                    segment_index=segment_index,
                    current_value=gap,
                    expected_value=0
                ))
            elif gap < self.config.min_gap_seconds:
                issues.append(ValidationIssue(
                    issue_type=IssueType.GAP_TOO_SHORT,
                    severity=IssueSeverity.INFO,
                    message=f"Gap too short: {gap:.3f}s (min: {self.config.min_gap_seconds:.3f}s)",
                    segment_index=segment_index,
                    current_value=gap,
                    expected_value=self.config.min_gap_seconds
                ))

        return issues

    def validate_subtitles(
        self,
        segments: List[Dict[str, Any]]
    ) -> SubtitleValidationResult:
        """
        Validate a list of subtitle segments.

        Args:
            segments: List of dicts with 'text', 'start', 'end' keys

        Returns:
            SubtitleValidationResult with all issues
        """
        all_issues = []
        prev_end_time = None

        for i, segment in enumerate(segments):
            text = segment.get('text', '')
            start = segment.get('start', 0)
            end = segment.get('end', 0)

            issues = self.validate_segment(
                text=text,
                start_time=start,
                end_time=end,
                segment_index=i,
                prev_end_time=prev_end_time
            )
            all_issues.extend(issues)
            prev_end_time = end

        # Count by severity
        error_count = sum(1 for i in all_issues if i.severity == IssueSeverity.ERROR)
        warning_count = sum(1 for i in all_issues if i.severity == IssueSeverity.WARNING)
        info_count = sum(1 for i in all_issues if i.severity == IssueSeverity.INFO)

        # Determine validity
        is_valid = error_count == 0
        if self.config.strict_mode:
            is_valid = is_valid and warning_count == 0

        return SubtitleValidationResult(
            is_valid=is_valid,
            total_segments=len(segments),
            issues=all_issues,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count
        )

    def _suggest_line_break(self, line: str) -> str:
        """Suggest where to break a long line"""
        if len(line) <= self.config.max_chars_per_line:
            return line

        # Try to break at natural points
        words = line.split()
        mid_point = len(line) // 2

        best_break = mid_point
        current_pos = 0

        for i, word in enumerate(words):
            word_end = current_pos + len(word)
            if abs(word_end - mid_point) < abs(best_break - mid_point):
                best_break = word_end
            current_pos = word_end + 1  # +1 for space

        # Create suggested break
        first_half = []
        second_half = []
        current_pos = 0

        for word in words:
            if current_pos + len(word) <= best_break:
                first_half.append(word)
            else:
                second_half.append(word)
            current_pos += len(word) + 1

        return f"Suggest: '{' '.join(first_half)}\\n{' '.join(second_half)}'"

    def fix_line_breaks(self, text: str) -> str:
        """
        Automatically fix line breaks to comply with Netflix standards.

        Args:
            text: Subtitle text

        Returns:
            Text with corrected line breaks
        """
        lines = text.strip().split('\n')

        # If already valid, return as-is
        if all(len(line) <= self.config.max_chars_per_line for line in lines):
            if len(lines) <= self.config.max_lines_per_subtitle:
                return text

        # Combine all text and re-break
        full_text = ' '.join(line.strip() for line in lines)
        words = full_text.split()

        if not words:
            return text

        # Greedy line breaking
        result_lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_len = len(word)

            # Would this word fit on current line?
            new_length = current_length + word_len + (1 if current_line else 0)

            if new_length <= self.config.max_chars_per_line:
                current_line.append(word)
                current_length = new_length
            else:
                # Start new line
                if current_line:
                    result_lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_len

                # Check if we've exceeded max lines
                if len(result_lines) >= self.config.max_lines_per_subtitle:
                    # Can't fit - return best effort
                    break

        # Add final line
        if current_line:
            result_lines.append(' '.join(current_line))

        return '\n'.join(result_lines[:self.config.max_lines_per_subtitle])

    def calculate_reading_speed(self, text: str, duration: float) -> float:
        """
        Calculate reading speed in characters per second.

        Args:
            text: Subtitle text
            duration: Duration in seconds

        Returns:
            Reading speed (chars/second)
        """
        if duration <= 0:
            return float('inf')

        char_count = sum(len(line.strip()) for line in text.split('\n'))
        return char_count / duration

    def suggest_duration(self, text: str) -> float:
        """
        Suggest appropriate duration for a subtitle.

        Args:
            text: Subtitle text

        Returns:
            Suggested duration in seconds
        """
        char_count = sum(len(line.strip()) for line in text.split('\n'))
        max_speed = (self.config.max_reading_speed_children
                    if self.config.is_children_content
                    else self.config.max_reading_speed_adult)

        # Calculate minimum duration for comfortable reading
        min_duration = char_count / max_speed

        # Apply minimum duration constraint
        return max(min_duration, self.config.min_duration_seconds)

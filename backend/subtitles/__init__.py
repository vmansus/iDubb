"""
Subtitle Processing Module

Provides:
- Netflix-compliant subtitle validation
- Subtitle parsing (SRT, VTT)
- Subtitle formatting and segmentation
- Reading speed calculation
"""
from .validator import NetflixValidator, SubtitleValidationResult, ValidationIssue
from .parser import SubtitleParser, SubtitleSegment
from .formatter import SubtitleFormatter

__all__ = [
    "NetflixValidator",
    "SubtitleValidationResult",
    "ValidationIssue",
    "SubtitleParser",
    "SubtitleSegment",
    "SubtitleFormatter",
]

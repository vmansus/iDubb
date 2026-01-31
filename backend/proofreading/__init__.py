"""
Proofreading Package

Provides AI-powered subtitle proofreading capabilities:
- Translation quality validation
- Timing and speech rate checks
- Terminology consistency
- Format validation
- AI-powered subtitle optimization
"""
from .models import (
    ProofreadingResult,
    SegmentProofreadResult,
    ProofreadingIssue,
    ProofreadingConfig,
    IssueSeverity,
    IssueType,
)
from .proofreader import (
    SubtitleProofreader,
    get_proofreader,
    proofread_subtitles,
)
from .optimizer import (
    SubtitleOptimizer,
    OptimizationConfig,
    OptimizationResult,
    optimize_subtitles,
)

__all__ = [
    # Models
    "ProofreadingResult",
    "SegmentProofreadResult",
    "ProofreadingIssue",
    "ProofreadingConfig",
    "IssueSeverity",
    "IssueType",
    # Proofreader
    "SubtitleProofreader",
    "get_proofreader",
    "proofread_subtitles",
    # Optimizer
    "SubtitleOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "optimize_subtitles",
]

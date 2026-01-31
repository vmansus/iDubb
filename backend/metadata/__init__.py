"""
Metadata Generation Module
AI-powered video metadata generation for titles, descriptions, and keywords
"""
from .metadata_generator import MetadataGenerator, VideoMetadataResult
from .preset_matcher import PresetMatcher, PresetMatch, MatchResult, preset_matcher, select_preset_for_task

__all__ = [
    "MetadataGenerator",
    "VideoMetadataResult",
    "PresetMatcher",
    "PresetMatch",
    "MatchResult",
    "preset_matcher",
    "select_preset_for_task",
]

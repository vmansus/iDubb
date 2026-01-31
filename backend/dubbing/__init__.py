"""
Dubbing module for synchronized voice-over generation
"""
from .dubbing_processor import DubbingProcessor, DubbingResult, DubbingSegment, parse_srt_segments

__all__ = ['DubbingProcessor', 'DubbingResult', 'DubbingSegment', 'parse_srt_segments']

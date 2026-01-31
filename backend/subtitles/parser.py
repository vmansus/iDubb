"""
Subtitle Parser Module

Parses SRT and VTT subtitle formats into a unified structure.
Consolidates duplicate parsing code from pipeline.py and dubbing_processor.py.
"""
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger


@dataclass
class SubtitleSegment:
    """A single subtitle segment with timing and text"""
    index: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str
    original_text: Optional[str] = None  # For dual subtitles

    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        return self.end - self.start

    @property
    def char_count(self) -> int:
        """Get total character count"""
        return sum(len(line.strip()) for line in self.text.split('\n'))

    @property
    def reading_speed(self) -> float:
        """Get reading speed (chars/second)"""
        if self.duration <= 0:
            return float('inf')
        return self.char_count / self.duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "original_text": self.original_text,
            "duration": self.duration,
        }


class SubtitleParser:
    """
    Unified subtitle parser for SRT and VTT formats.

    Usage:
        parser = SubtitleParser()
        segments = parser.parse_file("subtitles.srt")
        # or
        segments = parser.parse_srt(srt_content)
    """

    # SRT timestamp pattern: 00:00:00,000 or 00:00:00.000
    SRT_TIME_PATTERN = re.compile(
        r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})'
    )

    # VTT timestamp pattern: 00:00:00.000 or 00:00.000
    VTT_TIME_PATTERN = re.compile(
        r'(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{3})'
    )

    def parse_file(self, file_path: Path) -> List[SubtitleSegment]:
        """
        Parse a subtitle file (auto-detects format).

        Args:
            file_path: Path to subtitle file

        Returns:
            List of SubtitleSegment objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"Subtitle file not found: {file_path}")
            return []

        content = file_path.read_text(encoding='utf-8')

        if file_path.suffix.lower() == '.vtt':
            return self.parse_vtt(content)
        else:
            return self.parse_srt(content)

    def parse_srt(self, content: str) -> List[SubtitleSegment]:
        """
        Parse SRT format subtitles.

        Args:
            content: SRT file content

        Returns:
            List of SubtitleSegment objects
        """
        segments = []

        # Split into blocks (separated by blank lines)
        blocks = re.split(r'\n\s*\n', content.strip())

        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split('\n')

            if len(lines) < 2:
                continue

            try:
                # First line should be index
                try:
                    index = int(lines[0].strip())
                except ValueError:
                    # Some SRT files don't have proper indices
                    index = len(segments) + 1
                    lines = [str(index)] + lines

                # Second line should be timing
                timing_line = lines[1] if len(lines) > 1 else ""

                # Parse timing: 00:00:00,000 --> 00:00:00,000
                timing_match = re.match(
                    r'(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})',
                    timing_line
                )

                if not timing_match:
                    continue

                start = self._parse_srt_time(timing_match.group(1))
                end = self._parse_srt_time(timing_match.group(2))

                # Rest is text
                text = '\n'.join(lines[2:]).strip()

                if text:
                    segments.append(SubtitleSegment(
                        index=index,
                        start=start,
                        end=end,
                        text=text
                    ))

            except Exception as e:
                logger.warning(f"Failed to parse SRT block: {e}")
                continue

        logger.info(f"Parsed {len(segments)} segments from SRT")
        return segments

    def parse_vtt(self, content: str) -> List[SubtitleSegment]:
        """
        Parse WebVTT format subtitles.

        Args:
            content: VTT file content

        Returns:
            List of SubtitleSegment objects
        """
        segments = []

        # Remove WEBVTT header
        if content.startswith('WEBVTT'):
            content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.DOTALL)

        # Split into cues
        blocks = re.split(r'\n\s*\n', content.strip())

        for i, block in enumerate(blocks):
            if not block.strip():
                continue

            lines = block.strip().split('\n')

            try:
                # Check if first line is a cue identifier
                timing_line_idx = 0
                if '-->' not in lines[0]:
                    timing_line_idx = 1

                if timing_line_idx >= len(lines):
                    continue

                timing_line = lines[timing_line_idx]

                # Parse timing: 00:00:00.000 --> 00:00:00.000
                timing_match = re.match(
                    r'([\d:.]+)\s*-->\s*([\d:.]+)',
                    timing_line
                )

                if not timing_match:
                    continue

                start = self._parse_vtt_time(timing_match.group(1))
                end = self._parse_vtt_time(timing_match.group(2))

                # Rest is text (skip timing line)
                text_lines = lines[timing_line_idx + 1:]
                text = '\n'.join(text_lines).strip()

                # Remove VTT styling tags
                text = re.sub(r'<[^>]+>', '', text)

                if text:
                    segments.append(SubtitleSegment(
                        index=len(segments) + 1,
                        start=start,
                        end=end,
                        text=text
                    ))

            except Exception as e:
                logger.warning(f"Failed to parse VTT cue: {e}")
                continue

        logger.info(f"Parsed {len(segments)} segments from VTT")
        return segments

    def _parse_srt_time(self, time_str: str) -> float:
        """Parse SRT timestamp to seconds"""
        # Normalize comma to dot
        time_str = time_str.replace(',', '.')

        match = self.SRT_TIME_PATTERN.match(time_str)
        if not match:
            return 0.0

        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        milliseconds = int(match.group(4))

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

    def _parse_vtt_time(self, time_str: str) -> float:
        """Parse VTT timestamp to seconds"""
        # VTT can be HH:MM:SS.mmm or MM:SS.mmm
        parts = time_str.split(':')

        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            sec_ms = parts[2]
        elif len(parts) == 2:
            hours = 0
            minutes = int(parts[0])
            sec_ms = parts[1]
        else:
            return 0.0

        if '.' in sec_ms:
            sec_parts = sec_ms.split('.')
            seconds = int(sec_parts[0])
            milliseconds = int(sec_parts[1].ljust(3, '0')[:3])
        else:
            seconds = int(sec_ms)
            milliseconds = 0

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

    def segments_to_text_list(self, segments: List[SubtitleSegment]) -> List[str]:
        """Extract text from segments as a list"""
        return [seg.text for seg in segments]

    def merge_segments(
        self,
        original_segments: List[SubtitleSegment],
        translated_texts: List[str]
    ) -> List[SubtitleSegment]:
        """
        Merge translated texts back into segments.

        Args:
            original_segments: Original subtitle segments
            translated_texts: List of translated text strings

        Returns:
            New segments with translated text and original preserved
        """
        if len(original_segments) != len(translated_texts):
            logger.warning(
                f"Segment count mismatch: {len(original_segments)} vs {len(translated_texts)}"
            )

        merged = []
        for i, segment in enumerate(original_segments):
            translated = translated_texts[i] if i < len(translated_texts) else segment.text

            merged.append(SubtitleSegment(
                index=segment.index,
                start=segment.start,
                end=segment.end,
                text=translated,
                original_text=segment.text
            ))

        return merged

"""
Subtitle Formatter Module

Formats subtitles for output in SRT and VTT formats.
Supports dual subtitles (original + translated).
"""
from typing import List, Optional
from pathlib import Path
from loguru import logger

from .parser import SubtitleSegment


class SubtitleFormatter:
    """
    Formats subtitle segments into SRT or VTT files.

    Supports:
    - Standard SRT/VTT output
    - Dual subtitles (original + translated)
    - Line break normalization
    """

    def __init__(self, max_line_length: int = 42):
        """
        Initialize formatter.

        Args:
            max_line_length: Maximum characters per line (Netflix: 42)
        """
        self.max_line_length = max_line_length

    def format_srt(
        self,
        segments: List[SubtitleSegment],
        dual_subtitles: bool = False
    ) -> str:
        """
        Format segments as SRT content.

        Args:
            segments: List of subtitle segments
            dual_subtitles: Include original text below translated

        Returns:
            SRT formatted string
        """
        lines = []

        for segment in segments:
            # Index
            lines.append(str(segment.index))

            # Timing
            start_ts = self._format_srt_timestamp(segment.start)
            end_ts = self._format_srt_timestamp(segment.end)
            lines.append(f"{start_ts} --> {end_ts}")

            # Text
            if dual_subtitles and segment.original_text:
                lines.append(segment.text)
                lines.append(segment.original_text)
            else:
                lines.append(segment.text)

            # Blank line separator
            lines.append("")

        return "\n".join(lines)

    def format_vtt(
        self,
        segments: List[SubtitleSegment],
        dual_subtitles: bool = False
    ) -> str:
        """
        Format segments as WebVTT content.

        Args:
            segments: List of subtitle segments
            dual_subtitles: Include original text below translated

        Returns:
            VTT formatted string
        """
        lines = ["WEBVTT", ""]

        for segment in segments:
            # Timing
            start_ts = self._format_vtt_timestamp(segment.start)
            end_ts = self._format_vtt_timestamp(segment.end)
            lines.append(f"{start_ts} --> {end_ts}")

            # Text
            if dual_subtitles and segment.original_text:
                lines.append(segment.text)
                lines.append(segment.original_text)
            else:
                lines.append(segment.text)

            # Blank line separator
            lines.append("")

        return "\n".join(lines)

    def save_srt(
        self,
        segments: List[SubtitleSegment],
        output_path: Path,
        dual_subtitles: bool = False
    ) -> bool:
        """
        Save segments to SRT file.

        Args:
            segments: List of subtitle segments
            output_path: Output file path
            dual_subtitles: Include original text

        Returns:
            True if successful
        """
        try:
            content = self.format_srt(segments, dual_subtitles)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding='utf-8')
            logger.info(f"Saved SRT: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save SRT: {e}")
            return False

    def save_vtt(
        self,
        segments: List[SubtitleSegment],
        output_path: Path,
        dual_subtitles: bool = False
    ) -> bool:
        """
        Save segments to VTT file.

        Args:
            segments: List of subtitle segments
            output_path: Output file path
            dual_subtitles: Include original text

        Returns:
            True if successful
        """
        try:
            content = self.format_vtt(segments, dual_subtitles)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding='utf-8')
            logger.info(f"Saved VTT: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save VTT: {e}")
            return False

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds * 1000) % 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_timestamp(self, seconds: float) -> str:
        """Format seconds to VTT timestamp (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds * 1000) % 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def normalize_line_breaks(self, text: str) -> str:
        """
        Normalize line breaks to comply with max line length.

        Args:
            text: Subtitle text

        Returns:
            Text with normalized line breaks
        """
        lines = text.strip().split('\n')

        # Check if already valid
        if all(len(line) <= self.max_line_length for line in lines):
            return text

        # Rebuild with proper breaks
        full_text = ' '.join(line.strip() for line in lines)
        words = full_text.split()

        result_lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_len = len(word)
            new_length = current_length + word_len + (1 if current_line else 0)

            if new_length <= self.max_line_length:
                current_line.append(word)
                current_length = new_length
            else:
                if current_line:
                    result_lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_len

        if current_line:
            result_lines.append(' '.join(current_line))

        return '\n'.join(result_lines)

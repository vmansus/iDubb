"""
Dubbing Processor - VideoLingo-style synchronized dubbing

Key improvements over simple TTS:
1. Per-segment TTS generation (each subtitle line gets its own audio)
2. Speed adjustment to match subtitle duration
3. Silence insertion between segments for timing sync
4. Final audio merge with proper timeline alignment
"""
import asyncio
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from loguru import logger

from tts.base import BaseTTSEngine, TTSSegment


@dataclass
class DubbingSegment:
    """A single dubbing segment with timing and audio info"""
    index: int
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    audio_path: Optional[Path] = None
    audio_duration: float = 0.0
    speed_factor: float = 1.0
    adjusted_audio_path: Optional[Path] = None

    @property
    def target_duration(self) -> float:
        """Target duration based on subtitle timing"""
        return self.end_time - self.start_time

    @property
    def needs_speed_adjustment(self) -> bool:
        """Check if audio needs speed adjustment"""
        if self.audio_duration <= 0:
            return False
        # Allow 20% tolerance before speed adjustment kicks in
        return abs(self.audio_duration - self.target_duration) > self.target_duration * 0.2


@dataclass
class DubbingResult:
    """Result of dubbing process"""
    success: bool
    audio_path: Optional[Path] = None
    segments: List[DubbingSegment] = field(default_factory=list)
    total_duration: float = 0.0
    error: Optional[str] = None


class DubbingProcessor:
    """
    Processor for creating synchronized dubbing/voice-over

    Workflow:
    1. Parse subtitle segments with timing
    2. Generate TTS for each segment (parallel)
    3. Calculate speed factors
    4. Adjust audio speed to match subtitle duration
    5. Insert silence between segments
    6. Merge all audio into final track
    """

    # Maximum speed adjustment - keep conservative to sound natural
    # 1.3x faster or 0.8x slower max (beyond this sounds unnatural)
    MAX_SPEED = 1.3
    MIN_SPEED = 0.8

    # Tolerance for duration matching (seconds)
    DURATION_TOLERANCE = 0.6

    # Concurrent TTS requests
    # Increase for faster processing if TTS server supports it
    MAX_CONCURRENT_TTS = 8

    # Text patterns to skip (music, sound effects, etc.)
    SKIP_PATTERNS = {
        '🎵', '🎶', '♪', '♫', '♬',  # Music symbols
        '[音乐]', '[Music]', '[music]',
        '...', '…',  # Only ellipsis
    }

    # Patterns to clean from text before TTS (music symbols, etc.)
    CLEAN_PATTERNS = ['♪', '♫', '♬', '🎵', '🎶', '[音乐]', '[Music]', '[music]']

    def __init__(self, tts_engine: BaseTTSEngine):
        self.tts_engine = tts_engine

    def _should_skip_text(self, text: str) -> bool:
        """Check if text should be skipped (music, sound effects, etc.)"""
        text = text.strip()
        if not text:
            return True
        # Skip if text is only skip patterns or whitespace
        clean = text
        for pattern in self.SKIP_PATTERNS:
            clean = clean.replace(pattern, '')
        clean = clean.strip()
        # Skip if nothing meaningful remains (allow single characters like 啊, 好, 嗯)
        return len(clean) == 0

    def _clean_text_for_tts(self, text: str) -> str:
        """Remove music symbols and other non-speech patterns from text"""
        clean = text
        for pattern in self.CLEAN_PATTERNS:
            clean = clean.replace(pattern, '')
        # Remove extra whitespace
        clean = ' '.join(clean.split())
        return clean.strip()

    async def process(
        self,
        segments: List[Tuple[float, float, str]],  # (start, end, text)
        output_path: Path,
        voice: str = None,
        rate: str = "+0%",
        temp_dir: Path = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> DubbingResult:
        """
        Process dubbing for all segments

        Args:
            segments: List of (start_time, end_time, translated_text)
            output_path: Path for final merged audio
            voice: TTS voice to use
            rate: Base TTS rate
            temp_dir: Directory for temporary files
            cancel_check: Optional callable that returns True if cancellation requested

        Returns:
            DubbingResult with merged audio path
        """
        if not segments:
            return DubbingResult(success=False, error="No segments provided")

        # Check for cancellation before starting
        if cancel_check and cancel_check():
            logger.info("Dubbing cancelled before starting")
            return DubbingResult(success=False, error="用户手动停止")

        # Create temp directory if not provided
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="dubbing_"))
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Create DubbingSegment objects
            dubbing_segments = [
                DubbingSegment(
                    index=i,
                    text=text.strip(),
                    start_time=start,
                    end_time=end
                )
                for i, (start, end, text) in enumerate(segments)
                if text.strip()  # Skip empty segments
            ]

            if not dubbing_segments:
                return DubbingResult(success=False, error="No valid segments after filtering")

            logger.info(f"Processing {len(dubbing_segments)} dubbing segments")

            # Step 2: Generate TTS for each segment
            await self._generate_tts_segments(dubbing_segments, temp_dir, voice, rate, cancel_check)

            # Check for cancellation after TTS generation
            if cancel_check and cancel_check():
                logger.info("Dubbing cancelled after TTS generation")
                return DubbingResult(success=False, error="用户手动停止")

            # Step 3: Calculate and apply speed adjustments
            await self._adjust_speeds(dubbing_segments, temp_dir)

            # Check for cancellation after speed adjustment
            if cancel_check and cancel_check():
                logger.info("Dubbing cancelled after speed adjustment")
                return DubbingResult(success=False, error="用户手动停止")

            # Step 4: Merge all audio with silence gaps
            total_duration = await self._merge_audio(dubbing_segments, output_path, temp_dir)

            return DubbingResult(
                success=True,
                audio_path=output_path,
                segments=dubbing_segments,
                total_duration=total_duration
            )

        except Exception as e:
            logger.error(f"Dubbing process failed: {e}")
            return DubbingResult(success=False, error=str(e))

    async def _generate_tts_segments(
        self,
        segments: List[DubbingSegment],
        temp_dir: Path,
        voice: str,
        rate: str,
        cancel_check: Optional[Callable[[], bool]] = None
    ):
        """Generate TTS audio for each segment (parallel processing)"""
        # Filter segments that need TTS
        segments_to_process = [
            seg for seg in segments
            if seg.text and not self._should_skip_text(seg.text)
        ]

        skipped = len(segments) - len(segments_to_process)
        if skipped > 0:
            logger.info(f"Skipping {skipped} segments (music/sound effects)")

        total_segments = len(segments_to_process)
        logger.info(f"Generating TTS for {total_segments} segments (parallel, max {self.MAX_CONCURRENT_TTS} concurrent)")

        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_TTS)
        completed_count = [0]  # Use list for mutable counter in closure
        cancelled = [False]  # Track cancellation state
        import time
        start_time = time.time()

        async def process_segment(seg: DubbingSegment):
            # Check for cancellation before processing
            if cancelled[0] or (cancel_check and cancel_check()):
                cancelled[0] = True
                return

            async with semaphore:
                # Check again inside semaphore
                if cancelled[0] or (cancel_check and cancel_check()):
                    cancelled[0] = True
                    return
                audio_path = temp_dir / f"segment_{seg.index:04d}.mp3"
                try:
                    # Clean text before TTS (remove music symbols etc.)
                    clean_text = self._clean_text_for_tts(seg.text)
                    if not clean_text:
                        logger.debug(f"Segment {seg.index}: skipped after cleaning (original: '{seg.text[:30]}...')")
                        completed_count[0] += 1
                        return

                    result = await self.tts_engine.synthesize(
                        text=clean_text,
                        output_path=audio_path,
                        voice=voice,
                        rate=rate
                    )

                    completed_count[0] += 1
                    # Use the actual output path from result (some engines may change extension)
                    actual_path = Path(result.audio_path) if result.audio_path else audio_path
                    if result.success and actual_path.exists():
                        seg.audio_path = actual_path
                        seg.audio_duration = result.duration
                        # Progress logging every few segments
                        if completed_count[0] % 5 == 0 or completed_count[0] == total_segments:
                            elapsed = time.time() - start_time
                            rate_per_min = (completed_count[0] / elapsed) * 60 if elapsed > 0 else 0
                            logger.info(f"TTS progress: {completed_count[0]}/{total_segments} ({rate_per_min:.1f} segments/min)")
                        logger.debug(f"Segment {seg.index}: {seg.audio_duration:.2f}s for '{clean_text[:30]}...'")
                    else:
                        logger.warning(f"TTS failed for segment {seg.index}: {result.error}")

                except Exception as e:
                    completed_count[0] += 1
                    logger.warning(f"TTS error for segment {seg.index}: {e}")

        # Process all segments in parallel
        await asyncio.gather(*[process_segment(seg) for seg in segments_to_process])

        # Check if cancelled during processing
        if cancelled[0]:
            logger.info(f"TTS generation cancelled after {completed_count[0]}/{total_segments} segments")
            raise Exception("用户手动停止")

        elapsed = time.time() - start_time
        logger.info(f"TTS generation completed: {total_segments} segments in {elapsed:.1f}s ({total_segments/elapsed*60:.1f} segments/min)")

    async def _adjust_speeds(
        self,
        segments: List[DubbingSegment],
        temp_dir: Path
    ):
        """Adjust audio speed to match subtitle timing"""
        logger.info("Adjusting audio speeds...")

        for seg in segments:
            if not seg.audio_path or seg.audio_duration <= 0:
                continue

            target = seg.target_duration
            actual = seg.audio_duration

            if target <= 0:
                continue

            # Calculate speed factor
            speed = actual / target

            # Clamp to reasonable range - beyond this sounds unnatural
            original_speed = speed
            speed = max(self.MIN_SPEED, min(self.MAX_SPEED, speed))
            seg.speed_factor = speed

            # Warn if we're hitting limits (audio will be out of sync)
            if original_speed != speed:
                logger.warning(
                    f"Segment {seg.index}: speed clamped from {original_speed:.2f}x to {speed:.2f}x "
                    f"(TTS: {actual:.1f}s, target: {target:.1f}s) - may be out of sync"
                )

            # Only adjust if needed - use 15% tolerance
            if abs(speed - 1.0) < 0.15:
                seg.adjusted_audio_path = seg.audio_path
                continue

            # Apply speed adjustment using FFmpeg
            adjusted_path = temp_dir / f"segment_{seg.index:04d}_adjusted.mp3"

            try:
                await self._adjust_audio_speed(
                    seg.audio_path,
                    adjusted_path,
                    speed
                )

                if adjusted_path.exists():
                    seg.adjusted_audio_path = adjusted_path
                    # Verify new duration
                    new_duration = await self._get_audio_duration(adjusted_path)
                    logger.debug(
                        f"Segment {seg.index}: {actual:.2f}s -> {new_duration:.2f}s "
                        f"(target: {target:.2f}s, speed: {speed:.2f}x)"
                    )
                else:
                    seg.adjusted_audio_path = seg.audio_path

            except Exception as e:
                logger.warning(f"Speed adjustment failed for segment {seg.index}: {e}")
                seg.adjusted_audio_path = seg.audio_path

    async def _adjust_audio_speed(
        self,
        input_path: Path,
        output_path: Path,
        speed: float
    ):
        """
        Adjust audio speed using FFmpeg atempo filter

        Note: atempo only accepts values between 0.5 and 2.0,
        so we chain multiple filters for extreme adjustments
        """
        # Build atempo filter chain for extreme speeds
        filters = []
        remaining_speed = speed

        while remaining_speed > 2.0:
            filters.append("atempo=2.0")
            remaining_speed /= 2.0
        while remaining_speed < 0.5:
            filters.append("atempo=0.5")
            remaining_speed /= 0.5

        if remaining_speed != 1.0:
            filters.append(f"atempo={remaining_speed:.4f}")

        if not filters:
            # No adjustment needed, just copy
            filters = ["acopy"]

        filter_str = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-filter:a", filter_str,
            "-vn",  # No video
            "-ar", "44100",  # Sample rate
            "-ac", "1",  # Mono
            "-b:a", "128k",  # Bitrate
            str(output_path)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

    async def _merge_audio(
        self,
        segments: List[DubbingSegment],
        output_path: Path,
        temp_dir: Path
    ) -> float:
        """
        Merge all audio segments with silence gaps

        Creates silence between segments to maintain timeline alignment
        """
        logger.info("Merging audio segments...")

        # Build concat file
        concat_file = temp_dir / "concat.txt"
        silence_file = temp_dir / "silence.mp3"

        # Create a 1-second silence file
        await self._create_silence(silence_file, 1.0)

        current_time = 0.0
        concat_entries = []

        for seg in segments:
            audio_path = seg.adjusted_audio_path or seg.audio_path
            if not audio_path or not audio_path.exists():
                continue

            # Add silence before segment if there's a gap
            gap = seg.start_time - current_time
            if gap > 0.1:  # Only add silence for gaps > 100ms
                # Create specific silence file for this gap
                gap_silence = temp_dir / f"silence_{seg.index:04d}.mp3"
                await self._create_silence(gap_silence, gap)
                if gap_silence.exists():
                    concat_entries.append(f"file '{gap_silence}'")

            # Add the segment audio
            concat_entries.append(f"file '{audio_path}'")

            # Update current time
            audio_duration = await self._get_audio_duration(audio_path)
            current_time = seg.start_time + audio_duration

        if not concat_entries:
            raise Exception("No audio segments to merge")

        # Write concat file
        concat_file.write_text("\n".join(concat_entries))

        # Merge using FFmpeg concat demuxer
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:a", "libmp3lame",
            "-ar", "44100",
            "-ac", "1",
            "-b:a", "128k",
            str(output_path)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg merge failed: {stderr.decode()}")
            raise Exception("Audio merge failed")

        # Get final duration
        total_duration = await self._get_audio_duration(output_path)
        logger.info(f"Merged audio: {total_duration:.2f}s, {len(segments)} segments")

        return total_duration

    async def _create_silence(self, output_path: Path, duration: float):
        """Create a silent audio file of specified duration"""
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=mono",
            "-t", str(duration),
            "-c:a", "libmp3lame",
            "-b:a", "32k",
            str(output_path)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

    async def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration using FFprobe"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()

        try:
            return float(stdout.decode().strip())
        except Exception:
            return 0.0


def parse_srt_segments(srt_path: Path) -> List[Tuple[float, float, str]]:
    """
    Parse SRT file and return list of (start, end, text) tuples
    """
    segments = []

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Parse timing line
            timing = lines[1]
            if ' --> ' in timing:
                start_str, end_str = timing.split(' --> ')
                start = parse_srt_time(start_str.strip())
                end = parse_srt_time(end_str.strip())
                text = ' '.join(lines[2:])
                segments.append((start, end, text))

    return segments


def parse_srt_time(time_str: str) -> float:
    """Parse SRT time format (HH:MM:SS,mmm) to seconds"""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')

    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    return 0.0

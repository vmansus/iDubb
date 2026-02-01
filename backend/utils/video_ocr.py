"""
Video OCR - Extract text from video frames using vision models or local OCR
"""
import asyncio
import base64
import re
from loguru import logger
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import tempfile
import json
from difflib import SequenceMatcher

# Use loguru logger


@dataclass
class OCRSegment:
    """A segment of detected text with timing"""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class OCRResult:
    """Result of video OCR"""
    success: bool
    segments: List[OCRSegment]
    error: Optional[str] = None


class VideoOCR:
    """Extract text from video frames"""
    
    def __init__(
        self,
        engine: str = "paddleocr",  # paddleocr (free), openai, anthropic
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        frame_interval: float = 0.5,
        min_text_duration: float = 0.5,
        cancel_check: Optional[Callable[[], bool]] = None,
    ):
        self.engine = engine
        self.api_key = api_key
        self.model = model or self._default_model()
        self.frame_interval = frame_interval
        self.min_text_duration = min_text_duration
        self._paddle_ocr = None
        self.cancel_check = cancel_check
        self._active_processes: List[asyncio.subprocess.Process] = []
        
        # Validate engine configuration early
        self._validate_engine()
    
    def _validate_engine(self):
        """Validate OCR engine is available before starting"""
        logger.info(f"Initializing VideoOCR with engine: {self.engine}")
        
        if self.engine == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                logger.info("PaddleOCR module available")
            except ImportError:
                raise ImportError(
                    "PaddleOCR not installed. Run: pip install paddlepaddle paddleocr"
                )
        elif self.engine == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key required for openai OCR engine")
            logger.info(f"Using OpenAI OCR with model: {self.model}")
        elif self.engine == "anthropic":
            if not self.api_key:
                raise ValueError("Anthropic API key required for anthropic OCR engine")
            logger.info(f"Using Anthropic OCR with model: {self.model}")
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine}")
    
    def _default_model(self) -> str:
        if self.engine == "openai":
            return "gpt-4o"
        elif self.engine == "anthropic":
            return "claude-sonnet-4-20250514"
        return ""
    
    def _get_paddle_ocr(self):
        """Lazy initialize PaddleOCR"""
        if self._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                logger.info("Initializing PaddleOCR (this may take a moment on first run)...")
                self._paddle_ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang='en',
                    
                    
                )
                logger.info("PaddleOCR initialized successfully")
            except ImportError:
                raise ImportError(
                    "PaddleOCR not installed. Run: pip install paddlepaddle paddleocr"
                )
        return self._paddle_ocr
    
    async def extract_text(self, video_path: Path) -> OCRResult:
        """Extract text from video frames"""
        try:
            logger.info(f"Starting OCR extraction from video: {video_path}")
            
            duration = await self._get_video_duration(video_path)
            if duration <= 0:
                logger.error(f"Could not get video duration for: {video_path}")
                return OCRResult(success=False, segments=[], error="Could not get video duration")
            
            logger.info(f"Video duration: {duration:.2f}s, extracting frames every {self.frame_interval}s")
            
            frames = await self._extract_frames(video_path, duration)
            if not frames:
                logger.error("No frames were extracted from the video")
                return OCRResult(success=False, segments=[], error="No frames extracted")
            
            logger.info(f"Extracted {len(frames)} frames, starting OCR...")
            
            frame_texts = await self._ocr_frames(frames)
            
            # Log summary of OCR results
            non_empty_count = sum(1 for _, text in frame_texts if text.strip())
            logger.info(f"OCR completed: {non_empty_count}/{len(frame_texts)} frames had detected text")
            
            # Log first few detected texts for debugging
            detected_samples = [(ts, text) for ts, text in frame_texts if text.strip()][:5]
            if detected_samples:
                logger.info(f"Sample detected texts: {detected_samples}")
            else:
                logger.warning("No text detected in any frame!")
            
            segments = self._build_segments(frame_texts)
            
            # Cleanup
            for frame_path, _ in frames:
                try:
                    frame_path.unlink()
                except:
                    pass
            
            logger.info(f"Built {len(segments)} text segments from OCR results")
            if segments:
                for seg in segments[:3]:
                    logger.debug(f"Segment [{seg.start_time:.1f}s - {seg.end_time:.1f}s]: {seg.text[:50]}...")
            
            return OCRResult(success=True, segments=segments)
            
        except Exception as e:
            # Kill any active processes on error
            self.kill_active_processes()
            
            # Re-raise cancellation exceptions
            if "用户手动停止" in str(e):
                raise
            
            logger.error(f"Video OCR failed: {e}", exc_info=True)
            return OCRResult(success=False, segments=[], error=str(e))
    
    async def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data.get("format", {}).get("duration", 0))
                logger.debug(f"Video duration from ffprobe: {duration}s")
                return duration
            else:
                logger.error(f"ffprobe failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
        return 0
    
    def _check_cancelled(self):
        """Check if cancellation was requested"""
        if self.cancel_check and self.cancel_check():
            raise Exception("用户手动停止")
    
    async def _run_subprocess(self, cmd: List[str], timeout: float = 30) -> Tuple[int, str, str]:
        """Run subprocess with cancellation support"""
        self._check_cancelled()
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        self._active_processes.append(proc)
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode, stdout.decode() if stdout else "", stderr.decode() if stderr else ""
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        finally:
            if proc in self._active_processes:
                self._active_processes.remove(proc)
    
    def kill_active_processes(self):
        """Kill all active subprocesses (called on cancellation)"""
        for proc in self._active_processes:
            try:
                if proc.returncode is None:
                    proc.kill()
                    logger.info(f"Killed OCR subprocess {proc.pid}")
            except Exception as e:
                logger.warning(f"Failed to kill process: {e}")
        self._active_processes.clear()

    async def _extract_frames(self, video_path: Path, duration: float) -> List[Tuple[Path, float]]:
        """Extract frames at regular intervals"""
        frames = []
        temp_dir = Path(tempfile.mkdtemp(prefix="ocr_frames_"))
        logger.debug(f"Extracting frames to temp dir: {temp_dir}")
        
        times = []
        t = 0.0
        while t < duration:
            times.append(t)
            t += self.frame_interval
        
        logger.info(f"Will extract {len(times)} frames at {self.frame_interval}s intervals")
        
        for i, timestamp in enumerate(times):
            # Check for cancellation before each frame
            self._check_cancelled()
            
            frame_path = temp_dir / f"frame_{i:04d}.jpg"
            try:
                returncode, _, stderr = await self._run_subprocess(
                    ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
                     "-frames:v", "1", "-q:v", "2", str(frame_path)],
                    timeout=30
                )
                if returncode == 0 and frame_path.exists():
                    frames.append((frame_path, timestamp))
                else:
                    logger.warning(f"Failed to extract frame at {timestamp}s: {stderr}")
            except Exception as e:
                if "用户手动停止" in str(e):
                    raise
                logger.warning(f"Failed to extract frame at {timestamp}s: {e}")
        
        logger.info(f"Successfully extracted {len(frames)}/{len(times)} frames")
        return frames
    
    async def _ocr_frames(self, frames: List[Tuple[Path, float]]) -> List[Tuple[float, str]]:
        """OCR each frame with image deduplication
        
        Uses perceptual hashing to skip OCR for visually similar consecutive frames.
        This reduces redundant OCR calls and improves consistency.
        """
        results = []
        last_hash = None
        last_text = ""
        skipped_count = 0
        ocr_count = 0
        
        # Image hash distance threshold (0-64 scale)
        # Lower = more strict, Higher = more lenient
        # 5-10 is good for near-duplicate detection
        HASH_DISTANCE_THRESHOLD = 8
        
        for frame_path, timestamp in frames:
            # Check for cancellation
            self._check_cancelled()
            
            # Compute image hash
            current_hash = self._compute_image_hash(frame_path)
            
            # Check if frame is similar to previous
            if last_hash and current_hash:
                distance = self._image_hash_distance(current_hash, last_hash)
                if distance < HASH_DISTANCE_THRESHOLD:
                    # Frame is similar, reuse last OCR result
                    results.append((timestamp, last_text))
                    skipped_count += 1
                    logger.debug(f"Frame at {timestamp:.1f}s: skipped OCR (hash distance={distance}, reusing '{last_text[:30]}...')")
                    continue
            
            # Frame is different, run OCR
            _, text = await self._ocr_single_frame(frame_path, timestamp)
            results.append((timestamp, text))
            ocr_count += 1
            
            # Update last hash and text
            last_hash = current_hash
            last_text = text
            
            # Small delay between OCR calls to avoid overwhelming the system
            if ocr_count % 5 == 0:
                await asyncio.sleep(0.1)
        
        logger.info(f"OCR dedup stats: {ocr_count} frames OCR'd, {skipped_count} skipped (reused)")
        return results
    
    async def _ocr_single_frame(self, frame_path: Path, timestamp: float) -> Tuple[float, str]:
        """OCR a single frame"""
        try:
            if self.engine == "paddleocr":
                text = await self._ocr_paddle(frame_path)
            elif self.engine == "openai":
                text = await self._ocr_openai(frame_path)
            elif self.engine == "anthropic":
                text = await self._ocr_anthropic(frame_path)
            else:
                text = ""
            
            if text.strip():
                logger.debug(f"Frame at {timestamp:.1f}s: detected text '{text[:50]}...'" if len(text) > 50 else f"Frame at {timestamp:.1f}s: detected text '{text}'")
            
            return (timestamp, text.strip())
        except Exception as e:
            logger.warning(f"OCR failed for frame at {timestamp}s: {e}")
            return (timestamp, "")
    
    async def _ocr_paddle(self, frame_path: Path) -> str:
        """OCR using PaddleOCR (free, local)"""
        def _do_ocr():
            ocr = self._get_paddle_ocr()
            result = ocr.ocr(str(frame_path))
            
            if not result:
                logger.debug(f"PaddleOCR returned None for {frame_path}")
                return ""
            if not result[0]:
                logger.debug(f"PaddleOCR returned empty result[0] for {frame_path}")
                return ""
            
            texts = []
            # Handle new PaddleOCR format (list of dicts with 'rec_texts' and 'rec_scores')
            for item in result:
                if hasattr(item, 'rec_texts') and hasattr(item, 'rec_scores'):
                    # New format: object with rec_texts and rec_scores
                    for text, score in zip(item.rec_texts, item.rec_scores):
                        if score > 0.5:
                            logger.debug(f"PaddleOCR detected: '{text}' (confidence: {score:.2f})")
                            texts.append(text)
                elif isinstance(item, dict):
                    # Dict format
                    if 'rec_texts' in item:
                        for text, score in zip(item['rec_texts'], item.get('rec_scores', [1.0]*len(item['rec_texts']))):
                            if score > 0.5:
                                texts.append(text)
                elif isinstance(item, list):
                    # Old format: list of [box, (text, confidence)]
                    for line in item:
                        if line and len(line) >= 2:
                            if isinstance(line[1], tuple) and len(line[1]) == 2:
                                text, confidence = line[1]
                                if confidence > 0.5:
                                    texts.append(text)
            return " ".join(texts)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _do_ocr)
    
    async def _ocr_openai(self, frame_path: Path) -> str:
        """OCR using OpenAI GPT-4o Vision"""
        import httpx
        
        with open(frame_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract ONLY the overlay text/subtitles visible in this video frame. Return ONLY the text, nothing else. If no text, return empty string."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "low"}}
                        ]
                    }],
                    "max_tokens": 200
                }
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return ""
    
    async def _ocr_anthropic(self, frame_path: Path) -> str:
        """OCR using Anthropic Claude Vision"""
        import httpx
        
        with open(frame_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "max_tokens": 200,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                            {"type": "text", "text": "Extract ONLY the overlay text/subtitles visible in this video frame. Return ONLY the text, nothing else. If no text, return empty string."}
                        ]
                    }]
                }
            )
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
            return ""
    
    def _build_segments(self, frame_texts: List[Tuple[float, str]]) -> List[OCRSegment]:
        """Build text segments from frame OCR results"""
        if not frame_texts:
            logger.warning("No frame texts to build segments from")
            return []
        
        segments = []
        current_text = None
        start_time = 0.0
        
        for timestamp, text in frame_texts:
            normalized = self._normalize_text(text)
            
            if current_text is None:
                if normalized:
                    current_text = text
                    start_time = timestamp
            elif self._text_similarity(text, current_text) < 0.7:  # 70% similarity threshold (increased for better grouping)
                if current_text:
                    end_time = timestamp
                    if end_time - start_time >= self.min_text_duration:
                        segments.append(OCRSegment(text=current_text, start_time=start_time, end_time=end_time))
                        logger.debug(f"Built segment: [{start_time:.1f}s - {end_time:.1f}s] '{current_text[:30]}...'")
                
                if normalized:
                    current_text = text
                    start_time = timestamp
                else:
                    current_text = None
        
        if current_text and frame_texts:
            end_time = frame_texts[-1][0] + self.frame_interval
            if end_time - start_time >= self.min_text_duration:
                segments.append(OCRSegment(text=current_text, start_time=start_time, end_time=end_time))
                logger.debug(f"Built final segment: [{start_time:.1f}s - {end_time:.1f}s] '{current_text[:30]}...'")
        
        # Post-process: merge similar adjacent segments
        segments = self._merge_similar_segments(segments)
        
        return segments
    
    def _merge_similar_segments(self, segments: List[OCRSegment]) -> List[OCRSegment]:
        """Post-process to merge adjacent segments with similar text
        
        This catches cases where the same text was split into multiple segments
        due to minor OCR variations between frames.
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        for seg in segments:
            if merged and self._text_similarity(merged[-1].text, seg.text) > 0.8:
                # Similar to previous segment, extend it
                logger.debug(f"Merging segments: '{merged[-1].text[:30]}' + '{seg.text[:30]}' (similarity > 0.8)")
                merged[-1] = OCRSegment(
                    text=merged[-1].text,  # Keep the first text (usually cleaner)
                    start_time=merged[-1].start_time,
                    end_time=seg.end_time,  # Extend end time
                    confidence=merged[-1].confidence
                )
            else:
                merged.append(seg)
        
        if len(merged) < len(segments):
            logger.info(f"Post-merge: {len(segments)} segments -> {len(merged)} segments")
        
        return merged
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        # Remove special chars, keep only alphanumeric and spaces
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return " ".join(normalized.split())
    
    def _compute_image_hash(self, frame_path: Path) -> Optional[str]:
        """Compute perceptual hash of an image for deduplication
        
        Uses average hash (aHash) - fast and good for near-duplicate detection.
        Returns None if hashing fails.
        """
        try:
            from PIL import Image
            import hashlib
            
            # Simple perceptual hash: resize to 8x8, convert to grayscale, compute mean
            with Image.open(frame_path) as img:
                # Resize to 8x8 and convert to grayscale
                small = img.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
                pixels = list(small.getdata())
                avg = sum(pixels) / len(pixels)
                # Create binary hash based on whether each pixel is above/below average
                bits = ''.join('1' if p > avg else '0' for p in pixels)
                # Convert to hex string
                return hex(int(bits, 2))[2:].zfill(16)
        except Exception as e:
            logger.warning(f"Failed to compute image hash: {e}")
            return None
    
    def _image_hash_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two image hashes
        
        Returns number of different bits (0 = identical, 64 = completely different)
        """
        if not hash1 or not hash2:
            return 64
        try:
            n1, n2 = int(hash1, 16), int(hash2, 16)
            xor = n1 ^ n2
            return bin(xor).count('1')
        except:
            return 64
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts (0.0 to 1.0) using edit distance
        
        Uses SequenceMatcher which is more robust to OCR errors like:
        - "Hello World" vs "Hell0 World" (0 instead of o)
        - "Hello World" vs "Hello Wor1d" (1 instead of l)
        """
        if not text1 or not text2:
            return 0.0
        n1, n2 = self._normalize_text(text1), self._normalize_text(text2)
        if not n1 or not n2:
            return 0.0
        # Use SequenceMatcher for edit-distance based similarity
        # More robust to OCR character substitution errors
        return SequenceMatcher(None, n1, n2).ratio()
    
    def segments_to_srt(self, segments: List[OCRSegment]) -> str:
        """Convert segments to SRT format"""
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_srt_time(seg.start_time)
            end = self._format_srt_time(seg.end_time)
            lines.extend([str(i), f"{start} --> {end}", seg.text, ""])
        return "\n".join(lines)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

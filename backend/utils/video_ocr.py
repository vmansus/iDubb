"""
Video OCR - Extract text from video frames using vision models or local OCR
"""
import asyncio
import base64
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import tempfile
import json

logger = logging.getLogger(__name__)


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
    ):
        self.engine = engine
        self.api_key = api_key
        self.model = model or self._default_model()
        self.frame_interval = frame_interval
        self.min_text_duration = min_text_duration
        self._paddle_ocr = None
    
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
                self._paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False,
                    use_gpu=False,
                )
            except ImportError:
                raise ImportError(
                    "PaddleOCR not installed. Run: pip install paddlepaddle paddleocr"
                )
        return self._paddle_ocr
    
    async def extract_text(self, video_path: Path) -> OCRResult:
        """Extract text from video frames"""
        try:
            duration = await self._get_video_duration(video_path)
            if duration <= 0:
                return OCRResult(success=False, segments=[], error="Could not get video duration")
            
            logger.info(f"Video duration: {duration:.2f}s, extracting frames every {self.frame_interval}s")
            
            frames = await self._extract_frames(video_path, duration)
            if not frames:
                return OCRResult(success=False, segments=[], error="No frames extracted")
            
            logger.info(f"Extracted {len(frames)} frames")
            
            frame_texts = await self._ocr_frames(frames)
            segments = self._build_segments(frame_texts)
            
            # Cleanup
            for frame_path, _ in frames:
                try:
                    frame_path.unlink()
                except:
                    pass
            
            logger.info(f"Detected {len(segments)} text segments")
            return OCRResult(success=True, segments=segments)
            
        except Exception as e:
            logger.error(f"Video OCR failed: {e}")
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
                return float(data.get("format", {}).get("duration", 0))
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
        return 0
    
    async def _extract_frames(self, video_path: Path, duration: float) -> List[Tuple[Path, float]]:
        """Extract frames at regular intervals"""
        frames = []
        temp_dir = Path(tempfile.mkdtemp(prefix="ocr_frames_"))
        
        times = []
        t = 0.0
        while t < duration:
            times.append(t)
            t += self.frame_interval
        
        for i, timestamp in enumerate(times):
            frame_path = temp_dir / f"frame_{i:04d}.jpg"
            result = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "2", str(frame_path)],
                capture_output=True, text=True
            )
            if result.returncode == 0 and frame_path.exists():
                frames.append((frame_path, timestamp))
        
        return frames
    
    async def _ocr_frames(self, frames: List[Tuple[Path, float]]) -> List[Tuple[float, str]]:
        """OCR each frame"""
        results = []
        batch_size = 5
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self._ocr_single_frame(fp, ts) for fp, ts in batch]
            )
            results.extend(batch_results)
            if i + batch_size < len(frames):
                await asyncio.sleep(0.5)
        
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
            return (timestamp, text.strip())
        except Exception as e:
            logger.warning(f"OCR failed for frame at {timestamp}s: {e}")
            return (timestamp, "")
    
    async def _ocr_paddle(self, frame_path: Path) -> str:
        """OCR using PaddleOCR (free, local)"""
        def _do_ocr():
            ocr = self._get_paddle_ocr()
            result = ocr.ocr(str(frame_path), cls=True)
            if not result or not result[0]:
                return ""
            texts = []
            for line in result[0]:
                if line and len(line) >= 2:
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
            logger.error(f"OpenAI API error: {response.status_code}")
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
            logger.error(f"Anthropic API error: {response.status_code}")
            return ""
    
    def _build_segments(self, frame_texts: List[Tuple[float, str]]) -> List[OCRSegment]:
        """Build text segments from frame OCR results"""
        if not frame_texts:
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
            elif normalized != self._normalize_text(current_text):
                if current_text:
                    end_time = timestamp
                    if end_time - start_time >= self.min_text_duration:
                        segments.append(OCRSegment(text=current_text, start_time=start_time, end_time=end_time))
                
                if normalized:
                    current_text = text
                    start_time = timestamp
                else:
                    current_text = None
        
        if current_text and frame_texts:
            end_time = frame_texts[-1][0] + self.frame_interval
            if end_time - start_time >= self.min_text_duration:
                segments.append(OCRSegment(text=current_text, start_time=start_time, end_time=end_time))
        
        return segments
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        return " ".join(text.lower().split())
    
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

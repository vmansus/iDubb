"""
Video Processing Pipeline - Core orchestration module

处理流程:
1. 下载视频 (YouTube/TikTok)
2. 提取音频
3. 语音识别 (Whisper)
4. 翻译字幕
5. 生成TTS配音
6. 合成视频 (字幕 + 配音)
7. 上传到各平台

每个步骤都是可中断的，支持单独重试和查看结果。
"""
import asyncio
import os
import signal
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json
import shutil
from loguru import logger

from config import settings
from downloaders import YouTubeDownloader, TikTokDownloader
from transcription import WhisperTranscriber, FasterWhisperTranscriber, WhisperXTranscriber
from translation import Translator
from tts import EdgeTTSEngine, IndexTTSEngine, CosyVoiceEngine, Qwen3TTSEngine, BaseTTSEngine
from dubbing import DubbingProcessor, parse_srt_segments
from utils import VideoProcessor, SubtitleBurner
from utils.subtitle_burner import SubtitleStyle
from uploaders import BilibiliUploader, DouyinUploader, DouyinPlaywrightUploader, XiaohongshuUploader
from uploaders.base import VideoMetadata
from utils.thumbnail_generator import ThumbnailGenerator
import os


class TaskStatus(Enum):
    """Task status enum"""
    PENDING = "pending"
    QUEUED = "queued"  # Task is queued, waiting for execution slot
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    GENERATING_TTS = "generating_tts"
    PROCESSING_VIDEO = "processing_video"
    PENDING_REVIEW = "pending_review"  # Video ready, waiting for metadata review
    PENDING_UPLOAD = "pending_upload"  # Metadata approved, ready for upload
    UPLOADING = "uploading"
    UPLOADED = "uploaded"  # Upload completed (final state)
    COMPLETED = "completed"  # Legacy: kept for backwards compatibility
    FAILED = "failed"
    PAUSED = "paused"  # Paused state for manual intervention


class StepStatus(Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of a single processing step with timing"""
    step_name: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None  # Step duration in seconds
    error: Optional[str] = None
    output_files: Dict[str, str] = field(default_factory=dict)  # name -> path
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "duration_formatted": self._format_duration() if self.duration_seconds else None,
            "error": self.error,
            "output_files": self.output_files,
            "metadata": self.metadata,
        }

    def _format_duration(self) -> str:
        """Format duration as human readable string"""
        if not self.duration_seconds:
            return ""
        total_seconds = int(self.duration_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            return f"{hours}小时{minutes}分{seconds}秒"
        elif minutes > 0:
            return f"{minutes}分{seconds}秒"
        else:
            return f"{seconds}秒"


@dataclass
class ProcessingOptions:
    """Processing options for a video task"""
    # Source
    source_url: str = ""  # URL for online videos (empty for local uploads)
    source_platform: str = "auto"  # auto, youtube, tiktok, local
    local_file_path: Optional[str] = None  # Path to uploaded local video file

    # Processing mode
    processing_mode: str = "full"  # full | subtitle | direct | auto
    # full = complete workflow (transcribe + translate + TTS + compose)
    # subtitle = transcribe + translate + embed subtitles (no TTS)
    # direct = download and upload directly (no processing)
    # auto = AI decides based on content analysis
    
    # Transcription
    source_language: str = "auto"
    target_language: str = "zh-CN"

    # Whisper settings (per-task override)
    whisper_backend: str = "auto"  # auto, faster, openai
    whisper_model: str = "auto"  # auto, tiny, base, small, medium, large-v3
    whisper_device: str = "auto"  # auto, cpu, cuda, mps (mps only for openai backend)

    # OCR settings (for videos with text overlays instead of speech)
    use_ocr: bool = False  # Use OCR to extract text from video frames
    ocr_engine: str = "paddleocr"  # paddleocr (free), openai, anthropic
    ocr_frame_interval: float = 0.5  # Extract frame every N seconds

    # Translation options
    skip_translation: bool = False  # True = subtitles only mode (no translation, no TTS)
    translation_engine: str = "google"  # google, deepl, gpt, claude

    # Subtitle options
    add_subtitles: bool = True
    subtitle_style: str = "default"  # default, minimal, bold
    dual_subtitles: bool = True  # Show both original and translation
    use_existing_subtitles: bool = True  # Use video's existing subtitles if available
    subtitle_language: Optional[str] = None  # Preferred subtitle language to download
    subtitle_preset: Optional[str] = None  # Subtitle preset ID for styling

    # TTS options
    add_tts: bool = True
    tts_service: str = "edge"  # edge, index, cosyvoice (overrides global setting)
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    tts_rate: str = "+0%"
    replace_original_audio: bool = False  # False = mix with original
    tts_ref_audio: Optional[str] = None  # Reference audio for voice cloning
    tts_ref_text: Optional[str] = None  # Transcript of reference audio
    voice_cloning_mode: str = "disabled"  # disabled, video_audio, custom
    original_audio_volume: float = 0.3  # Volume for original audio when mixing (0-1)
    tts_audio_volume: float = 1.0  # Volume for TTS audio (0-1)

    # Output options
    video_quality: str = "1080p"
    format_id: Optional[str] = None  # Specific format ID (overrides video_quality)
    video_quality_label: Optional[str] = None  # Human-readable quality label

    # Upload targets
    upload_bilibili: bool = False
    upload_douyin: bool = False
    upload_xiaohongshu: bool = False
    bilibili_account_uid: Optional[str] = None  # Specific Bilibili account UID (None = default)
    douyin_account_id: Optional[str] = None  # Specific Douyin account ID (None = default)
    xiaohongshu_account_id: Optional[str] = None  # Specific Xiaohongshu account ID (None = default)

    # Metadata
    custom_title: Optional[str] = None
    custom_description: Optional[str] = None
    custom_tags: List[str] = field(default_factory=list)
    metadata_preset_id: Optional[str] = None  # Metadata preset ID for title prefix and signature
    use_ai_preset_selection: bool = False  # Use AI to automatically select best preset

    # Proofreading options
    enable_proofreading: bool = True  # Enable AI proofreading after translation
    proofreading_auto_pause: bool = True  # Auto-pause if confidence is low
    proofreading_min_confidence: float = 0.6  # Minimum confidence threshold
    proofreading_auto_optimize: bool = False  # Auto-optimize subtitles after proofreading
    proofreading_optimization_level: str = "moderate"  # minimal, moderate, aggressive

    # Directory for organizing tasks (immutable after task creation)
    directory: Optional[str] = None  # Directory name for grouping tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        from dataclasses import asdict
        return asdict(self)


@dataclass
class ProcessingTask:
    """A single video processing task with step-by-step tracking and timing"""
    task_id: str
    options: ProcessingOptions
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Step-by-step results
    steps: Dict[str, StepResult] = field(default_factory=dict)
    current_step: Optional[str] = None

    # Task folder name (task_id + sanitized video name)
    task_folder: Optional[str] = None
    # Cached output directory path
    _output_dir: Optional[Path] = field(default=None, repr=False)

    # Intermediate results (file paths)
    video_info: Optional[Dict[str, Any]] = None
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    downloaded_subtitle_path: Optional[Path] = None  # Subtitle downloaded with video
    downloaded_subtitle_language: Optional[str] = None  # Language of downloaded subtitle
    subtitle_path: Optional[Path] = None  # Original subtitle (downloaded or transcribed)
    translated_subtitle_path: Optional[Path] = None
    tts_audio_path: Optional[Path] = None
    final_video_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None  # Local thumbnail path

    # Upload results
    upload_results: Dict[str, Any] = field(default_factory=dict)

    # Proofreading results
    proofreading_result: Optional[Dict[str, Any]] = None

    # Optimization results
    optimization_result: Optional[Dict[str, Any]] = None

    # Error info
    error: Optional[str] = None

    # Cancellation support
    _cancel_requested: bool = field(default=False, repr=False)
    _active_processes: set = field(default_factory=set, repr=False)
    _process_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # Timing statistics
    processing_started_at: Optional[datetime] = None  # When processing started
    total_processing_time: Optional[float] = None  # Total time in seconds

    def request_cancel(self):
        """Request cancellation of this task and kill all active processes"""
        self._cancel_requested = True
        self._kill_active_processes()

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested"""
        return self._cancel_requested

    def clear_cancel(self):
        """Clear cancellation flag (for retry)"""
        self._cancel_requested = False

    def check_cancelled(self):
        """Check if cancelled and raise exception if so"""
        if self._cancel_requested:
            raise Exception("用户手动停止")

    def register_process(self, proc):
        """Register an active subprocess for tracking"""
        self._active_processes.add(proc)
        logger.debug(f"Registered process {proc.pid}, active count: {len(self._active_processes)}")

    def unregister_process(self, proc):
        """Unregister a subprocess when it completes"""
        self._active_processes.discard(proc)
        logger.debug(f"Unregistered process {proc.pid}, active count: {len(self._active_processes)}")

    def _kill_active_processes(self):
        """Kill all active subprocesses"""
        if not self._active_processes:
            return
        
        logger.info(f"Killing {len(self._active_processes)} active processes for task cancellation")
        for proc in list(self._active_processes):
            try:
                if proc.returncode is None:  # Still running
                    # Try to kill the entire process group for subprocess
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        logger.info(f"Sent SIGTERM to process group {pgid}")
                    except (ProcessLookupError, PermissionError, OSError):
                        # Fallback to killing just the process
                        proc.terminate()
                        logger.info(f"Terminated process {proc.pid}")
            except Exception as e:
                logger.warning(f"Failed to kill process {proc.pid}: {e}")
        
        # Give processes a moment to terminate, then force kill
        import time
        time.sleep(0.5)
        
        for proc in list(self._active_processes):
            try:
                if proc.returncode is None:
                    proc.kill()
                    logger.info(f"Force killed process {proc.pid}")
            except Exception as e:
                logger.warning(f"Failed to force kill process {proc.pid}: {e}")
        
        self._active_processes.clear()

    async def run_subprocess(self, cmd: List[str], timeout: float = None, check: bool = False) -> tuple:
        """
        Run a subprocess with cancellation support.
        
        Args:
            cmd: Command and arguments
            timeout: Optional timeout in seconds
            check: If True, raise exception on non-zero return code
            
        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        if self._cancel_requested:
            raise Exception("用户手动停止")
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        self._active_processes.add(proc)
        
        try:
            if timeout:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            else:
                stdout, stderr = await proc.communicate()
            
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            if check and proc.returncode != 0:
                raise Exception(f"Command failed with code {proc.returncode}: {stderr_str}")
            
            return proc.returncode, stdout_str, stderr_str
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise Exception(f"Command timed out after {timeout}s")
        finally:
            self._active_processes.discard(proc)

    def __post_init__(self):
        """Initialize step tracking"""
        step_names = [
            "download",
            "transcribe",
            "translate",
            "proofread",
            "optimize",
            "tts",
            "process_video",
            "upload"
        ]
        for name in step_names:
            if name not in self.steps:
                self.steps[name] = StepResult(step_name=name)

    def get_step(self, name: str) -> StepResult:
        """Get step result by name"""
        return self.steps.get(name, StepResult(step_name=name))

    def get_step_timings(self) -> Dict[str, Any]:
        """Get all step timings as dictionary"""
        timings = {}
        for name, step in self.steps.items():
            if step.duration_seconds is not None:
                timings[name] = {
                    "duration_seconds": step.duration_seconds,
                    "duration_formatted": step._format_duration(),
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                }
        return timings

    def get_total_time_formatted(self) -> Optional[str]:
        """Get formatted total processing time"""
        if not self.total_processing_time:
            return None
        total_seconds = int(self.total_processing_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            return f"{hours}小时{minutes}分{seconds}秒"
        elif minutes > 0:
            return f"{minutes}分{seconds}秒"
        else:
            return f"{seconds}秒"

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API response"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_step": self.current_step,
            "task_folder": self.task_folder,
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
            "video_info": self.video_info,
            "files": self._get_files_dict(),
            "upload_results": self.upload_results,
            "error": self.error,
            "step_timings": self.get_step_timings(),
            "total_processing_time": self.total_processing_time,
            "total_time_formatted": self.get_total_time_formatted(),
        }

    def _get_files_dict(self) -> Dict[str, Optional[str]]:
        """Get all file paths as strings"""
        return {
            "video": str(self.video_path) if self.video_path else None,
            "audio": str(self.audio_path) if self.audio_path else None,
            "original_subtitle": str(self.subtitle_path) if self.subtitle_path else None,
            "translated_subtitle": str(self.translated_subtitle_path) if self.translated_subtitle_path else None,
            "tts_audio": str(self.tts_audio_path) if self.tts_audio_path else None,
            "final_video": str(self.final_video_path) if self.final_video_path else None,
            "thumbnail": str(self.thumbnail_path) if self.thumbnail_path else None,
        }

    async def get_output_dir(self) -> Path:
        """Get the output directory for this task's files"""
        if self._output_dir is not None:
            return self._output_dir

        from utils.storage import get_task_directory
        self._output_dir = await get_task_directory(
            self.task_id,
            self.task_folder,
            self.options.directory
        )
        return self._output_dir

    def set_task_folder(self, video_title: Optional[str] = None):
        """Set the task folder name based on video title"""
        from utils.storage import generate_task_folder_name
        self.task_folder = generate_task_folder_name(self.task_id, video_title)
        # Clear cached output dir to regenerate with new folder name
        self._output_dir = None


class VideoPipeline:
    """Main video processing pipeline with step-by-step control"""

    STEP_PROGRESS = {
        "download": (0, 15),
        "transcribe": (15, 35),
        "translate": (35, 50),
        "tts": (50, 65),
        "process_video": (65, 85),
        "upload": (85, 100),
    }

    def __init__(self):
        # Initialize components
        from settings_store import settings_store

        self.youtube_downloader = YouTubeDownloader(settings.DOWNLOADS_DIR)
        self.tiktok_downloader = TikTokDownloader(settings.DOWNLOADS_DIR)

        # Default transcriber (can be overridden per-task)
        self._default_transcriber = None
        self._transcriber_cache: Dict[str, Any] = {}  # Cache for different model configs
        
        # Single GPU lock to prevent concurrent GPU operations (MPS conflicts)
        # Used by both transcription and local TTS engines
        self._gpu_lock = asyncio.Lock()

        # Load translation settings from settings_store
        global_settings = settings_store.load()
        self.translator = Translator(
            engine=global_settings.translation.engine or settings.TRANSLATION_SERVICE,
            api_key=global_settings.translation.get_api_key(),
            model=global_settings.translation.model
        )
        self.tts_engine = self._create_tts_engine()
        self.video_processor = VideoProcessor()
        self.subtitle_burner = SubtitleBurner()

        # Uploaders (initialized on demand)
        self._bilibili: Optional[BilibiliUploader] = None
        self._douyin = None  # Can be DouyinUploader or DouyinPlaywrightUploader
        self._xiaohongshu: Optional[XiaohongshuUploader] = None

        # Task tracking
        self.tasks: Dict[str, ProcessingTask] = {}

        # Transcription cache for retry
        self._transcription_cache: Dict[str, Any] = {}

    def _parse_srt_file(self, srt_path: Path) -> List[Dict[str, Any]]:
        """
        Parse SRT file back into segments.
        Used to restore transcription from saved files after restart.
        """
        segments = []
        if not srt_path.exists():
            return segments

        content = srt_path.read_text(encoding="utf-8")
        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            try:
                # Parse timestamp line: "00:00:00,000 --> 00:00:05,000"
                time_line = lines[1]
                start_str, end_str = time_line.split(" --> ")

                def parse_time(t: str) -> float:
                    t = t.replace(",", ".")
                    parts = t.split(":")
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

                start = parse_time(start_str)
                end = parse_time(end_str)
                text = "\n".join(lines[2:])

                segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
            except Exception as e:
                logger.warning(f"Failed to parse SRT block: {e}")
                continue

        return segments

    def _restore_transcription_from_srt(self, task: ProcessingTask) -> bool:
        """
        Restore transcription cache from saved SRT file.
        Returns True if successful, False otherwise.
        """
        if not task.subtitle_path or not task.subtitle_path.exists():
            return False

        segments_dicts = self._parse_srt_file(task.subtitle_path)
        if not segments_dicts:
            return False

        # Convert dict segments to TranscriptSegment objects
        from transcription import Transcription, TranscriptSegment
        
        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"]
            )
            for seg in segments_dicts
        ]

        # Detect language from task options or default
        language = task.options.source_language if task.options.source_language != "auto" else "en"

        transcription = Transcription(
            text=" ".join(seg.text for seg in segments),
            segments=segments,
            language=language
        )

        self._transcription_cache[task.task_id] = transcription
        logger.info(f"Restored transcription from SRT for task {task.task_id}: {len(segments)} segments")
        return True

    def _create_tts_engine(self) -> BaseTTSEngine:
        """
        Create TTS engine based on settings.TTS_SERVICE

        Supports:
        - edge: Microsoft Edge TTS (free, online)
        - qwen3: Qwen3-TTS (local, high quality, voice cloning)
        - index: IndexTTS local voice cloning
        - cosyvoice: Alibaba CosyVoice voice cloning
        """
        service = settings.TTS_SERVICE.lower()

        if service == "index":
            logger.info(f"Using IndexTTS engine at {settings.INDEX_TTS_HOST}:{settings.INDEX_TTS_PORT}")
            ref_audio = Path(settings.INDEX_TTS_REF_AUDIO) if settings.INDEX_TTS_REF_AUDIO else None
            return IndexTTSEngine(
                host=settings.INDEX_TTS_HOST,
                port=settings.INDEX_TTS_PORT,
                ref_audio_path=ref_audio,
                emo_mode=settings.INDEX_TTS_EMO_MODE,
                emo_weight=settings.INDEX_TTS_EMO_WEIGHT,
            )
        elif service == "cosyvoice":
            logger.info(f"Using CosyVoice engine at {settings.COSYVOICE_HOST}:{settings.COSYVOICE_PORT}")
            return CosyVoiceEngine(
                host=settings.COSYVOICE_HOST,
                port=settings.COSYVOICE_PORT,
                mode=settings.COSYVOICE_MODE,
                default_speaker=settings.COSYVOICE_SPEAKER,
            )
        elif service == "qwen3":
            logger.info(f"Using Qwen3-TTS engine with voice: {settings.TTS_VOICE}")
            return Qwen3TTSEngine(
                default_voice=settings.TTS_VOICE or "vivian",
                model_name=getattr(settings, 'QWEN3_TTS_MODEL', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'),
                default_language=getattr(settings, 'QWEN3_TTS_LANGUAGE', 'Chinese'),
            )
        else:
            # Default to Edge TTS
            logger.info(f"Using Edge TTS engine with voice: {settings.TTS_VOICE}")
            return EdgeTTSEngine(default_voice=settings.TTS_VOICE)

    def _get_transcriber(self, opts: ProcessingOptions):
        """
        Get transcriber based on task options.

        Supports per-task configuration of whisper backend, model, and device.
        Caches transcribers to avoid reloading models.
        """
        # Determine backend, model and device from options or use defaults
        backend = opts.whisper_backend if opts.whisper_backend != "auto" else settings.WHISPER_BACKEND
        model = opts.whisper_model if opts.whisper_model != "auto" else settings.WHISPER_MODEL
        device = opts.whisper_device if opts.whisper_device != "auto" else settings.WHISPER_DEVICE

        # Parse model_id format like "faster:base" -> backend="faster", model="base"
        if ":" in model:
            parts = model.split(":", 1)
            backend = parts[0]
            model = parts[1]

        # Note: MPS only works with OpenAI Whisper, not faster-whisper/whisperx
        if device == "mps" and backend in ("faster", "whisperx"):
            logger.warning(f"MPS is not supported by {backend}. Using CPU instead.")
            device = "cpu"

        # Create cache key including device
        cache_key = f"{backend}:{model}:{device}"

        # Return cached transcriber if available
        if cache_key in self._transcriber_cache:
            logger.debug(f"Using cached transcriber: {cache_key}")
            return self._transcriber_cache[cache_key]

        # Create new transcriber
        logger.info(f"Creating transcriber: backend={backend}, model={model}, device={device}")

        if backend == "whisperx":
            # WhisperX: word-level alignment, best for subtitle segmentation
            transcriber = WhisperXTranscriber(
                model_name=model,
                device=device
            )
        elif backend == "faster":
            transcriber = FasterWhisperTranscriber(
                model_name=model,
                device=device
            )
        else:
            transcriber = WhisperTranscriber(
                model_name=model,
                device=device
            )

        # Cache for reuse
        self._transcriber_cache[cache_key] = transcriber
        return transcriber

    def _get_downloader(self, url: str, platform: str = "auto"):
        """Get appropriate downloader for URL"""
        if platform == "youtube" or (platform == "auto" and self.youtube_downloader.supports_url(url)):
            return self.youtube_downloader
        elif platform == "tiktok" or (platform == "auto" and self.tiktok_downloader.supports_url(url)):
            return self.tiktok_downloader
        else:
            raise ValueError(f"Unsupported URL or platform: {url}")

    async def _try_auto_refresh_youtube_cookies(self) -> bool:
        """
        Attempt to auto-refresh YouTube cookies from the default browser.
        Returns True if cookies were successfully refreshed.
        """
        from utils.cookie_extractor import extract_youtube_cookies as extract_cookies, get_available_browsers

        # Check if running in Docker (can't access host browser)
        in_docker = os.path.exists('/.dockerenv')
        if in_docker:
            logger.info("Running in Docker, cannot auto-refresh cookies")
            return False

        # Get available browsers
        browsers = get_available_browsers()
        if not browsers:
            logger.warning("No supported browsers found for cookie extraction")
            return False

        # Try Chrome first, then other browsers
        preferred_order = ['chrome', 'edge', 'brave', 'chromium', 'opera', 'vivaldi']
        browsers_to_try = sorted(browsers, key=lambda b: preferred_order.index(b) if b in preferred_order else 99)

        cookie_path = settings.DATA_DIR / "youtube_cookies.txt"

        for browser in browsers_to_try:
            logger.info(f"Attempting to auto-refresh cookies from {browser}...")
            result = extract_cookies(browser, cookie_path)

            if result["success"]:
                # Reinitialize the YouTube downloader with new cookies
                self.youtube_downloader = YouTubeDownloader(settings.DOWNLOADS_DIR)
                logger.info(f"Auto-refreshed YouTube cookies from {browser}: {result['cookie_count']} cookies")
                return True
            else:
                logger.warning(f"Failed to extract cookies from {browser}: {result.get('message', 'Unknown error')}")

        return False

    def _is_cookie_related_error(self, error_message: str) -> bool:
        """Check if an error message indicates a cookie-related issue"""
        error_lower = error_message.lower()
        cookie_related_keywords = [
            'sign in', 'signin', 'login', 'cookie', 'authentication',
            'bot', 'verify', 'confirm you', 'age', 'restricted',
            # Generic errors that are often caused by cookie issues
            'failed to get video info', 'video unavailable', 'private video',
            'this video is not available', 'no video formats',
            # n challenge / format failures often caused by expired cookies
            'requested format is not available', 'only images are available',
            'sig function', 'n challenge'
        ]
        return any(keyword in error_lower for keyword in cookie_related_keywords)

    async def _start_step(self, task: ProcessingTask, step_name: str):
        """Mark a step as started and persist to database"""
        step = task.steps[step_name]
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        step.error = None
        task.current_step = step_name
        task.updated_at = datetime.now()
        # Persist to database
        from database.task_persistence import task_persistence
        await task_persistence.save_step_status(
            task.task_id, step_name, step.status.value
        )

    async def _complete_step(self, task: ProcessingTask, step_name: str, output_files: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Mark a step as completed and persist to database"""
        step = task.steps[step_name]
        step.status = StepStatus.COMPLETED
        step.error = None  # Clear any previous error
        step.completed_at = datetime.now()
        # Calculate duration
        if step.started_at:
            step.duration_seconds = (step.completed_at - step.started_at).total_seconds()
        if output_files:
            step.output_files = output_files
        if metadata:
            step.metadata = metadata
        task.updated_at = datetime.now()
        # Persist to database
        from database.task_persistence import task_persistence
        await task_persistence.save_step_status(
            task.task_id, step_name, step.status.value,
            output_files=output_files, metadata=metadata
        )

    async def _fail_step(self, task: ProcessingTask, step_name: str, error: str):
        """Mark a step as failed and persist to database"""
        step = task.steps[step_name]
        step.status = StepStatus.FAILED
        step.completed_at = datetime.now()
        # Calculate duration even for failed steps
        if step.started_at:
            step.duration_seconds = (step.completed_at - step.started_at).total_seconds()
        step.error = error
        task.updated_at = datetime.now()
        # Persist to database
        from database.task_persistence import task_persistence
        await task_persistence.save_step_status(
            task.task_id, step_name, step.status.value, error=error
        )

    async def _skip_step(self, task: ProcessingTask, step_name: str, reason: str = ""):
        """Mark a step as skipped and persist to database"""
        step = task.steps[step_name]
        step.status = StepStatus.SKIPPED
        step.completed_at = datetime.now()
        step.duration_seconds = 0  # Skipped steps have no duration
        step.metadata["skip_reason"] = reason
        task.updated_at = datetime.now()
        # Persist to database
        from database.task_persistence import task_persistence
        await task_persistence.save_step_status(
            task.task_id, step_name, step.status.value,
            metadata={"skip_reason": reason}
        )

    async def _detect_processing_mode(self, task: ProcessingTask) -> str:
        """
        Auto-detect processing mode based on content analysis.
        
        Uses quick transcription + AI judgment to determine:
        - "direct": No meaningful speech, just upload directly
        - "subtitle": Has speech that needs translation, embed subtitles only
        - "full": Has speech, use full workflow with TTS
        
        Returns:
            Processing mode: "direct", "subtitle", or "full"
        """
        logger.info(f"Auto-detecting processing mode for task {task.task_id}")
        
        try:
            # Quick transcription using tiny model
            await self._update_task(task, progress=15, message="分析视频内容...")
            
            if not task.audio_path or not task.audio_path.exists():
                # Extract audio first
                from utils.video_processor import extract_audio
                task.audio_path = await extract_audio(task.video_path)
            
            # Use tiny model for fast transcription
            transcriber = self._get_transcriber(task.options)
            # Force tiny model for speed
            original_model = task.options.whisper_model
            task.options.whisper_model = "tiny"
            
            try:
                quick_transcription = await transcriber.transcribe(
                    task.audio_path,
                    language=task.options.source_language if task.options.source_language != "auto" else None
                )
            finally:
                task.options.whisper_model = original_model  # Restore
            
            # Check if transcription is empty or very short
            if not quick_transcription or not quick_transcription.get("segments"):
                logger.info("No speech detected, using direct mode")
                return "direct"
            
            # Get transcribed text
            segments = quick_transcription.get("segments", [])
            full_text = " ".join([s.get("text", "").strip() for s in segments])
            
            if len(full_text.strip()) < 10:
                logger.info(f"Very short transcription ({len(full_text)} chars), using direct mode")
                return "direct"
            
            # Use AI to judge if the content needs translation
            await self._update_task(task, progress=18, message="AI 判断内容类型...")
            
            needs_translation = await self._ai_judge_needs_translation(full_text, task)
            
            if needs_translation:
                # Cache the transcription for later use
                self._transcription_cache[task.task_id] = quick_transcription
                logger.info("AI determined content needs translation, using subtitle mode")
                return "subtitle"
            else:
                logger.info("AI determined content doesn't need translation, using direct mode")
                return "direct"
                
        except Exception as e:
            logger.warning(f"Auto-detection failed: {e}, falling back to subtitle mode")
            return "subtitle"  # Safe default
    
    async def _ai_judge_needs_translation(self, text: str, task: ProcessingTask) -> bool:
        """
        Use AI to judge if the transcribed text needs translation.
        
        Args:
            text: Transcribed text
            task: Processing task for context
            
        Returns:
            True if translation is needed, False otherwise
        """
        try:
            from settings_store import settings_store
            settings = settings_store.load()
            
            api_key = settings.translation.get_api_key()
            engine = settings.translation.engine
            
            if not api_key or engine in ["google", "deepl"]:
                # No AI available, assume needs translation if text is substantial
                return len(text.strip()) > 50
            
            prompt = f"""分析以下视频转录内容，判断是否需要翻译成中文。

转录内容：
{text[:500]}

判断标准：
1. 如果是有意义的英文对话、解说、教程等内容 → 需要翻译
2. 如果只是音效、语气词（如 "wow", "oh", "yeah"）、歌词片段、或无意义的声音 → 不需要翻译
3. 如果内容已经是中文 → 不需要翻译

只回答 "是" 或 "否"，不要解释。
是 = 需要翻译
否 = 不需要翻译"""

            if engine in ["gpt", "openai"]:
                import openai
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                answer = response.choices[0].message.content.strip()
            elif engine in ["claude", "anthropic"]:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.content[0].text.strip()
            elif engine == "deepseek":
                import openai
                client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                answer = response.choices[0].message.content.strip()
            else:
                return len(text.strip()) > 50
            
            logger.info(f"AI judgment: '{answer}'")
            return answer.startswith("是") or answer.lower().startswith("yes")
            
        except Exception as e:
            logger.warning(f"AI judgment failed: {e}, assuming needs translation")
            return True  # Safe default

    async def _handle_post_processing(self, task: ProcessingTask):
        """Handle post-processing steps for direct mode (metadata, thumbnail, upload)"""
        # Auto-generate AI thumbnail if enabled
        await self._auto_generate_ai_thumbnail(task)
        
        # Auto-generate AI metadata if enabled  
        await self._auto_generate_metadata(task)
        
        # Calculate total processing time
        if task.processing_started_at:
            task.total_processing_time = (datetime.now() - task.processing_started_at).total_seconds()
            logger.info(f"Task {task.task_id} direct mode completed in {task.get_total_time_formatted()}")
        
        # Check if metadata review is required
        from settings_store import settings_store
        from database.task_persistence import task_persistence
        global_settings = settings_store.load()
        require_review = global_settings.metadata.require_review
        
        if require_review:
            task.status = TaskStatus.PENDING_REVIEW
            task.message = "直接搬运模式，等待元数据审核"
            return
        
        # Auto-approve and upload
        task.status = TaskStatus.PENDING_UPLOAD
        task.message = "准备上传"
        
        opts = task.options
        if opts.upload_bilibili or opts.upload_douyin or opts.upload_xiaohongshu:
            await self.step_upload(task)
            task.current_step = None
            
            successful = [p for p, r in task.upload_results.items() if r.get("success")]
            failed = [p for p, r in task.upload_results.items() if not r.get("success")]
            
            if successful and not failed:
                task.status = TaskStatus.UPLOADED
                task.message = "上传完成"
            elif successful and failed:
                task.status = TaskStatus.UPLOADED
                task.message = f"部分上传完成 (成功: {', '.join(successful)}; 失败: {', '.join(failed)})"
            else:
                task.status = TaskStatus.COMPLETED
                task.message = "处理完成（上传失败）"
        else:
            task.status = TaskStatus.COMPLETED
            task.message = "直接搬运完成"

    async def _update_task(
        self,
        task: ProcessingTask,
        status: TaskStatus = None,
        progress: int = None,
        message: str = None
    ):
        """Update task status"""
        if status:
            task.status = status
        if progress is not None:
            task.progress = progress
        if message:
            task.message = message
        task.updated_at = datetime.now()
        logger.info(f"Task {task.task_id}: {task.status.value} - {task.message} ({task.progress}%)")

    # ==================== Individual Step Methods ====================

    async def step_download(self, task: ProcessingTask) -> bool:
        """Step 1: Download video (or use local uploaded file)"""
        step_name = "download"
        await self._start_step(task, step_name)

        try:
            # Check for cancellation before starting
            task.check_cancelled()

            opts = task.options

            # Check if this is a local file upload
            if opts.local_file_path and opts.source_platform == "local":
                await self._handle_local_file(task, step_name)
            else:
                await self._handle_url_download(task, step_name)

            await self._complete_step(task, step_name,
                output_files={
                    "video": str(task.video_path) if task.video_path else None,
                    "audio": str(task.audio_path) if task.audio_path else None,
                    "thumbnail": str(task.thumbnail_path) if task.thumbnail_path else None,
                },
                metadata=task.video_info
            )
            # Persist video_info and file paths to tasks table immediately
            from database.task_persistence import task_persistence
            if task.video_info:
                await task_persistence.save_video_info(task.task_id, task.video_info)
            await task_persistence.save_task_files(
                task.task_id,
                video_path=str(task.video_path) if task.video_path else None,
                audio_path=str(task.audio_path) if task.audio_path else None,
                thumbnail_path=str(task.thumbnail_path) if task.thumbnail_path else None
            )
            await self._update_task(task, progress=15, message="视频准备完成")
            return True

        except Exception as e:
            await self._fail_step(task, step_name, str(e))
            raise

    async def _handle_local_file(self, task: ProcessingTask, step_name: str):
        """Handle local file upload - copy to task directory and extract audio"""
        opts = task.options
        local_path = Path(opts.local_file_path)

        if not local_path.exists():
            raise Exception(f"本地文件不存在: {local_path}")

        await self._update_task(task, TaskStatus.DOWNLOADING, 5, "处理本地视频文件...")
        logger.info(f"Processing local file: {local_path}")

        # Get video duration and info using ffprobe
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(local_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                import json as json_module
                probe_data = json_module.loads(result.stdout)
                duration = float(probe_data.get("format", {}).get("duration", 0))
            else:
                duration = 0
        except Exception as e:
            logger.warning(f"Failed to probe video: {e}")
            duration = 0

        # Use filename (without extension) as title
        video_title = local_path.stem
        task.video_info = {
            "title": video_title,
            "description": "",
            "tags": [],
            "duration": duration,
            "platform": "local",
        }

        # Set task folder name from video title
        task.set_task_folder(video_title)
        logger.info(f"Task folder set to: {task.task_folder}")

        # Get task directory
        task_dir = await task.get_output_dir()

        # Copy video file to task directory
        new_video_path = task_dir / f"video{local_path.suffix}"
        shutil.copy2(str(local_path), str(new_video_path))
        task.video_path = new_video_path
        logger.info(f"Copied video to: {new_video_path}")

        # Get video dimensions and add to video_info
        try:
            video_dimensions = await self.video_processor.get_video_info(new_video_path)
            if video_dimensions:
                task.video_info["width"] = video_dimensions.width
                task.video_info["height"] = video_dimensions.height
                task.video_info["is_vertical"] = video_dimensions.is_vertical
                logger.info(f"Video dimensions: {video_dimensions.width}x{video_dimensions.height}, is_vertical={video_dimensions.is_vertical}")
        except Exception as e:
            logger.warning(f"Failed to get video dimensions: {e}")

        # Extract audio from video using ffmpeg (skip in transfer-only mode)
        opts = task.options
        if opts.add_subtitles or opts.add_tts:
            await self._update_task(task, TaskStatus.DOWNLOADING, 10, "提取音频...")
            audio_path = task_dir / "audio.mp3"
            try:
                returncode, stdout, stderr = await task.run_subprocess(
                    ["ffmpeg", "-i", str(new_video_path), "-vn", "-acodec", "libmp3lame", "-q:a", "2", "-y", str(audio_path)]
                )
                if returncode == 0 and audio_path.exists():
                    task.audio_path = audio_path
                    logger.info(f"Extracted audio to: {audio_path}")
                else:
                    logger.warning(f"Failed to extract audio: {stderr}")
            except Exception as e:
                if "用户手动停止" in str(e):
                    raise
                logger.warning(f"Failed to extract audio: {e}")
        else:
            logger.info("Skipping audio extraction: transfer-only mode (no subtitles or TTS)")

        # Generate thumbnail from first frame
        thumbnail_path = task_dir / "thumbnail.jpg"
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", str(new_video_path), "-vframes", "1", "-q:v", "2", "-y", str(thumbnail_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and thumbnail_path.exists():
                task.thumbnail_path = thumbnail_path
                logger.info(f"Generated thumbnail: {thumbnail_path}")
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail: {e}")

    async def _handle_url_download(self, task: ProcessingTask, step_name: str):
        """Handle URL-based video download with auto-refresh cookies on failure"""
        await self._update_task(task, TaskStatus.DOWNLOADING, 5, "下载视频中...")

        opts = task.options
        downloader = self._get_downloader(opts.source_url, opts.source_platform)
        is_youtube = self.youtube_downloader.supports_url(opts.source_url)

        logger.info(f"Downloading with quality={opts.video_quality}, format_id={opts.format_id}, subtitle_language={opts.subtitle_language}")
        download_result = await downloader.download(
            opts.source_url,
            quality=opts.video_quality,
            format_id=opts.format_id,
            subtitle_language=opts.subtitle_language if opts.use_existing_subtitles else None,
            cancel_check=task.is_cancelled
        )

        # Check for cancellation after download
        task.check_cancelled()

        # If download failed and it's YouTube, try auto-refreshing cookies
        if not download_result.success and is_youtube:
            error_msg = download_result.error or ""
            if self._is_cookie_related_error(error_msg):
                logger.info(f"Download failed with possible cookie issue: {error_msg}")
                await self._update_task(task, TaskStatus.DOWNLOADING, 5, "尝试自动刷新Cookie...")

                if await self._try_auto_refresh_youtube_cookies():
                    logger.info("Cookies refreshed, retrying download...")
                    await self._update_task(task, TaskStatus.DOWNLOADING, 5, "Cookie已刷新，重新下载中...")

                    # Get fresh downloader reference after cookie refresh
                    downloader = self._get_downloader(opts.source_url, opts.source_platform)
                    download_result = await downloader.download(
                        opts.source_url,
                        quality=opts.video_quality,
                        format_id=opts.format_id,
                        subtitle_language=opts.subtitle_language if opts.use_existing_subtitles else None,
                        cancel_check=task.is_cancelled
                    )
                    task.check_cancelled()
                else:
                    logger.warning("Failed to auto-refresh cookies")

        if not download_result.success:
            raise Exception(f"Download failed: {download_result.error}")

        task.video_path = download_result.video_path
        task.audio_path = download_result.audio_path
        task.downloaded_subtitle_path = download_result.subtitle_path
        task.downloaded_subtitle_language = download_result.subtitle_language
        task.video_info = {
            "title": download_result.video_info.title,
            "description": download_result.video_info.description,
            "tags": download_result.video_info.tags,
            "duration": download_result.video_info.duration,
            "platform": download_result.video_info.platform,
            "thumbnail_url": download_result.video_info.thumbnail_url,
        }

        # Include extra metadata like actual quality and available qualities
        if download_result.extra:
            task.video_info.update(download_result.extra)
            
            # Update video_quality_label with actual downloaded resolution
            actual_height = download_result.extra.get('actual_height')
            if actual_height:
                # Convert height to quality label
                height_to_label = {
                    2160: "2160p", 1440: "1440p", 1080: "1080p",
                    720: "720p", 480: "480p", 360: "360p"
                }
                for h, label in sorted(height_to_label.items(), reverse=True):
                    if actual_height >= h:
                        opts.video_quality_label = label
                        task.video_info['actual_quality'] = label
                        logger.info(f"Updated video quality label: {label} (actual height: {actual_height})")
                        # Also persist the updated options to database
                        from database.task_persistence import task_persistence
                        await task_persistence.save_options(task.task_id, opts.to_dict())
                        break

        # Generate task folder name from video title
        video_title = download_result.video_info.title if download_result.video_info else None
        task.set_task_folder(video_title)
        logger.info(f"Task folder set to: {task.task_folder}")

        # Get task directory and move downloaded files there
        task_dir = await task.get_output_dir()

        # Move video file to task directory
        if task.video_path and task.video_path.exists():
            new_video_path = task_dir / f"video{task.video_path.suffix}"
            shutil.move(str(task.video_path), str(new_video_path))
            task.video_path = new_video_path
            logger.info(f"Moved video to: {new_video_path}")

            # Get video dimensions and add to video_info
            try:
                video_dimensions = await self.video_processor.get_video_info(new_video_path)
                if video_dimensions:
                    task.video_info["width"] = video_dimensions.width
                    task.video_info["height"] = video_dimensions.height
                    task.video_info["is_vertical"] = video_dimensions.is_vertical
                    logger.info(f"Video dimensions: {video_dimensions.width}x{video_dimensions.height}, is_vertical={video_dimensions.is_vertical}")
            except Exception as e:
                logger.warning(f"Failed to get video dimensions: {e}")

        # Move audio file to task directory (skip in transfer-only mode)
        if task.audio_path and task.audio_path.exists():
            if opts.add_subtitles or opts.add_tts:
                new_audio_path = task_dir / f"audio{task.audio_path.suffix}"
                shutil.move(str(task.audio_path), str(new_audio_path))
                task.audio_path = new_audio_path
                logger.info(f"Moved audio to: {new_audio_path}")
            else:
                # Transfer-only mode: delete temp audio file, not needed
                task.audio_path.unlink(missing_ok=True)
                task.audio_path = None
                logger.info("Deleted temp audio: transfer-only mode (no subtitles or TTS)")

        # Move downloaded subtitle to task directory
        if task.downloaded_subtitle_path and task.downloaded_subtitle_path.exists():
            new_subtitle_path = task_dir / f"downloaded_subtitle{task.downloaded_subtitle_path.suffix}"
            shutil.move(str(task.downloaded_subtitle_path), str(new_subtitle_path))
            task.downloaded_subtitle_path = new_subtitle_path
            logger.info(f"Moved subtitle to: {new_subtitle_path} (lang: {task.downloaded_subtitle_language})")
        elif task.downloaded_subtitle_path:
            logger.info(f"Downloaded subtitle path set but file not found: {task.downloaded_subtitle_path}")

        # Handle thumbnail - first try to use already-downloaded thumbnail, then fall back to download
        thumbnail_path = task_dir / "thumbnail.jpg"
        thumbnail_set = False
        try:
            # First check if thumbnail was already downloaded during video download
            existing_thumbnail = task.video_info.get('thumbnail_path') if isinstance(task.video_info, dict) else None
            if existing_thumbnail:
                existing_thumb_path = Path(existing_thumbnail)
                if existing_thumb_path.exists():
                    shutil.move(str(existing_thumb_path), str(thumbnail_path))
                    task.thumbnail_path = thumbnail_path
                    thumbnail_set = True
                    logger.info(f"Moved existing thumbnail to: {thumbnail_path}")

            # If no existing thumbnail, download from URL
            if not thumbnail_set:
                thumbnail_url = task.video_info.get('thumbnail_url') if isinstance(task.video_info, dict) else None
                if thumbnail_url and hasattr(downloader, 'download_thumbnail'):
                    thumbnail_success = await downloader.download_thumbnail(thumbnail_url, thumbnail_path)
                    if thumbnail_success and thumbnail_path.exists():
                        task.thumbnail_path = thumbnail_path
                        thumbnail_set = True
                        logger.info(f"Thumbnail downloaded: {thumbnail_path}")

            if not thumbnail_set:
                logger.warning("No thumbnail available for this video")
        except Exception as thumb_error:
            logger.warning(f"Failed to handle thumbnail: {thumb_error}")
            # Non-fatal error, continue without thumbnail

    async def step_transcribe(self, task: ProcessingTask) -> bool:
        """Step 2: Transcribe audio"""
        step_name = "transcribe"
        opts = task.options
        
        # Skip transcription entirely if no subtitles or TTS needed (direct mode)
        if not opts.add_subtitles and not opts.add_tts:
            logger.info(f"Skipping transcription: add_subtitles={opts.add_subtitles}, add_tts={opts.add_tts}")
            await self._skip_step(task, step_name, "仅搬运模式，跳过语音识别")
            return True
        
        await self._start_step(task, step_name)

        try:
            # Check if we should use existing subtitles
            use_existing = opts.use_existing_subtitles and task.downloaded_subtitle_path and task.downloaded_subtitle_path.exists()

            if use_existing:
                # Use downloaded subtitles instead of AI transcription
                logger.info(f"Using existing subtitles: {task.downloaded_subtitle_path}")
                await self._update_task(task, TaskStatus.TRANSCRIBING, 20, "使用视频自带字幕...")

                # Copy downloaded subtitle to task directory
                task_dir = await task.get_output_dir()
                task.subtitle_path = task_dir / "original.srt"

                # Convert VTT to SRT if needed
                if task.downloaded_subtitle_path.suffix.lower() == '.vtt':
                    await self._convert_vtt_to_srt(task.downloaded_subtitle_path, task.subtitle_path)
                else:
                    import shutil
                    shutil.copy(task.downloaded_subtitle_path, task.subtitle_path)

                # Parse subtitles to create transcription cache for translation step
                transcription = await self._parse_srt_to_transcription(task.subtitle_path, task.downloaded_subtitle_language or opts.source_language)
                self._transcription_cache[task.task_id] = transcription

                await self._complete_step(task, step_name,
                    output_files={
                        "original_subtitle": str(task.subtitle_path),
                    },
                    metadata={
                        "language": task.downloaded_subtitle_language or opts.source_language,
                        "segments_count": len(transcription.segments) if transcription else 0,
                        "source": "downloaded",  # Indicate subtitle was downloaded, not transcribed
                    }
                )
                # Persist subtitle path immediately
                from database.task_persistence import task_persistence
                await task_persistence.save_task_files(
                    task.task_id,
                    subtitle_path=str(task.subtitle_path) if task.subtitle_path else None
                )
                await self._update_task(task, progress=35, message="使用视频自带字幕完成")
                return True

            # Check if OCR mode is enabled (for videos with text overlays, no speech)
            if opts.use_ocr:
                logger.info("Using OCR mode to extract text from video frames")
                await self._update_task(task, TaskStatus.TRANSCRIBING, 20, "OCR识别画面文字中...")
                
                from utils.video_ocr import VideoOCR
                
                # Get API key for paid engines
                api_key = None
                if opts.ocr_engine in ["openai", "anthropic"]:
                    from settings_store import settings_store
                    global_settings = settings_store.load()
                    if opts.ocr_engine == "openai":
                        api_key = global_settings.translation.openai_api_key
                    else:
                        api_key = global_settings.translation.anthropic_api_key
                    if not api_key:
                        raise Exception(f"API key not configured for OCR engine: {opts.ocr_engine}")
                
                ocr = VideoOCR(
                    engine=opts.ocr_engine,
                    api_key=api_key,
                    frame_interval=opts.ocr_frame_interval,
                    cancel_check=task.is_cancelled,
                )
                
                try:
                    result = await ocr.extract_text(task.video_path)
                except Exception as e:
                    # Ensure processes are killed on any error
                    ocr.kill_active_processes()
                    raise
                
                if not result.success:
                    raise Exception(f"OCR failed: {result.error}")
                
                task_dir = await task.get_output_dir()
                task.subtitle_path = task_dir / "original.srt"
                
                if result.segments:
                    srt_content = ocr.segments_to_srt(result.segments)
                    task.subtitle_path.write_text(srt_content, encoding="utf-8")
                    
                    # Create transcription cache for translation step
                    from transcription import Transcription, TranscriptSegment
                    segments = [
                        TranscriptSegment(start=seg.start_time, end=seg.end_time, text=seg.text)
                        for seg in result.segments
                    ]
                    transcription = Transcription(
                        text=" ".join(seg.text for seg in result.segments),
                        segments=segments,
                        language=opts.source_language
                    )
                    self._transcription_cache[task.task_id] = transcription
                else:
                    task.subtitle_path.write_text("")
                    logger.warning("OCR found no text in video")
                
                await self._complete_step(task, step_name,
                    output_files={"original_subtitle": str(task.subtitle_path)},
                    metadata={"language": opts.source_language, "segments_count": len(result.segments), "source": "ocr"}
                )
                from database.task_persistence import task_persistence
                await task_persistence.save_task_files(task.task_id, subtitle_path=str(task.subtitle_path))
                await self._update_task(task, progress=35, message=f"OCR识别完成，检测到 {len(result.segments)} 段文字")
                return True

            # Fall back to AI transcription (Whisper)
            logger.info("No existing subtitles or use_existing_subtitles=False, using AI transcription")
            await self._update_task(task, TaskStatus.TRANSCRIBING, 20, "AI语音识别中...")

            # Extract audio if not available
            if not task.audio_path or not task.audio_path.exists():
                if not task.video_path or not task.video_path.exists():
                    raise Exception("No video or audio file available")
                task_dir = await task.get_output_dir()
                task.audio_path = task_dir / "audio.mp3"
                await self.video_processor.extract_audio(task.video_path, task.audio_path)

            # Get transcriber based on task options (supports per-task model selection)
            logger.info(f"Task whisper settings: backend={opts.whisper_backend}, model={opts.whisper_model}")
            transcriber = self._get_transcriber(opts)

            # Use GPU lock to prevent concurrent transcription (MPS conflicts)
            async with self._gpu_lock:
                logger.debug("Acquired GPU lock for transcription")
                # Pass cancellation check to transcriber
                transcription = await transcriber.transcribe(
                    task.audio_path,
                    language=opts.source_language if opts.source_language != "auto" else None,
                    cancel_check=task.is_cancelled
                )
                logger.debug("Released GPU lock")

            # Check if cancelled
            if getattr(transcription, 'cancelled', False):
                raise Exception("用户手动停止")

            if not transcription.success:
                raise Exception(f"Transcription failed: {transcription.error}")

            # Cache transcription for potential retry of later steps
            self._transcription_cache[task.task_id] = transcription

            # Save original subtitles to task directory
            task_dir = await task.get_output_dir()
            task.subtitle_path = task_dir / "original.srt"
            await transcriber.save_subtitles(
                transcription.segments,
                task.subtitle_path
            )

            # Get duration from last segment (handle both dataclass and dict)
            duration = 0
            if transcription.segments:
                last_seg = transcription.segments[-1]
                duration = last_seg.end if hasattr(last_seg, 'end') else last_seg.get('end', 0)

            await self._complete_step(task, step_name,
                output_files={
                    "original_subtitle": str(task.subtitle_path),
                },
                metadata={
                    "language": transcription.language,
                    "segments_count": len(transcription.segments),
                    "duration": duration,
                    "source": "whisper",  # Indicate AI transcription was used
                }
            )
            # Persist subtitle path immediately
            from database.task_persistence import task_persistence
            await task_persistence.save_task_files(
                task.task_id,
                subtitle_path=str(task.subtitle_path) if task.subtitle_path else None
            )
            await self._update_task(task, progress=35, message="语音识别完成")
            return True

        except Exception as e:
            await self._fail_step(task, step_name, str(e))
            raise

    async def step_translate(self, task: ProcessingTask) -> bool:
        """Step 3: Translate subtitles"""
        step_name = "translate"
        opts = task.options

        # Check if translation is needed
        if not opts.add_subtitles and not opts.add_tts:
            await self._skip_step(task, step_name, "No subtitles or TTS requested")
            return True

        # Skip translation in subtitles-only mode
        if opts.skip_translation:
            await self._skip_step(task, step_name, "Subtitles-only mode (no translation)")
            return True

        await self._start_step(task, step_name)
        await self._update_task(task, TaskStatus.TRANSLATING, 40, "翻译字幕中...")

        try:
            # Check for cancellation before starting
            task.check_cancelled()

            # Get transcription from cache or restore from SRT file
            transcription = self._transcription_cache.get(task.task_id)
            if not transcription:
                # Try to restore from saved SRT file first (fast, no re-processing needed)
                if task.subtitle_path and task.subtitle_path.exists():
                    logger.info(f"Restoring transcription from SRT file: {task.subtitle_path}")
                    if self._restore_transcription_from_srt(task):
                        transcription = self._transcription_cache.get(task.task_id)

                # If still no transcription, we need to re-transcribe (slow fallback)
                if not transcription:
                    if not task.audio_path or not task.audio_path.exists():
                        raise Exception("Original subtitles and audio not found. Please run transcription step first.")
                    logger.warning("Re-transcribing audio (this is slower, SRT file was not available)")
                    transcriber = self._get_transcriber(opts)
                    transcription = await transcriber.transcribe(
                        task.audio_path,
                        language=opts.source_language if opts.source_language != "auto" else None
                    )
                    self._transcription_cache[task.task_id] = transcription

            # Get translator for this task (respects task-specific engine setting)
            translator, fast_mode = self._get_translator_for_task(opts)
            logger.info(f"Using translation engine: {opts.translation_engine}")

            # Check if using OptimizedTranslator (has fast_mode parameter)
            from translation import OptimizedTranslator
            if isinstance(translator, OptimizedTranslator):
                translated_segments = await translator.translate_segments(
                    transcription.segments,
                    source_lang=transcription.language,
                    target_lang=opts.target_language,
                    fast_mode=fast_mode,
                    cancel_check=task.is_cancelled  # Pass cancellation check
                )
            else:
                translated_segments = await translator.translate_segments(
                    transcription.segments,
                    source_lang=transcription.language,
                    target_lang=opts.target_language,
                    cancel_check=task.is_cancelled  # Pass cancellation check
                )

            # Check for cancellation after translation
            task.check_cancelled()

            # Save translated subtitles to task directory
            task_dir = await task.get_output_dir()
            task.translated_subtitle_path = task_dir / "translated.srt"
            transcriber = self._get_transcriber(opts)
            await transcriber.save_subtitles(
                translated_segments,
                task.translated_subtitle_path
            )

            # Cache translated segments for TTS
            self._transcription_cache[f"{task.task_id}_translated"] = translated_segments

            await self._complete_step(task, step_name,
                output_files={
                    "translated_subtitle": str(task.translated_subtitle_path),
                },
                metadata={
                    "source_lang": transcription.language,
                    "target_lang": opts.target_language,
                    "translation_engine": opts.translation_engine,
                    "segments_count": len(translated_segments),
                }
            )
            # Persist translated subtitle path immediately
            from database.task_persistence import task_persistence
            await task_persistence.save_task_files(
                task.task_id,
                translated_subtitle_path=str(task.translated_subtitle_path) if task.translated_subtitle_path else None
            )
            await self._update_task(task, progress=50, message="翻译完成")
            return True

        except Exception as e:
            await self._fail_step(task, step_name, str(e))
            raise

    async def step_proofread(self, task: ProcessingTask) -> bool:
        """Step 3.5: AI proofreading of translated subtitles"""
        step_name = "proofread"
        opts = task.options

        # Check if proofreading is enabled
        if not opts.enable_proofreading:
            await self._skip_step(task, step_name, "Proofreading disabled")
            return True

        # Skip if no translation was done
        if opts.skip_translation:
            await self._skip_step(task, step_name, "No translation to proofread")
            return True

        # Skip if no translated subtitles
        if not task.translated_subtitle_path or not task.translated_subtitle_path.exists():
            await self._skip_step(task, step_name, "No translated subtitles available")
            return True

        await self._start_step(task, step_name)
        await self._update_task(task, progress=52, message="校对字幕中...")

        try:
            # Check for cancellation
            task.check_cancelled()

            from proofreading import proofread_subtitles, ProofreadingConfig

            # Get original and translated segments from cache
            transcription = self._transcription_cache.get(task.task_id)
            translated_segments = self._transcription_cache.get(f"{task.task_id}_translated")

            if not transcription or not translated_segments:
                # Try to restore from SRT files (returns dicts already)
                original_segments = []
                if task.subtitle_path and task.subtitle_path.exists():
                    original_segments = self._parse_srt_file(task.subtitle_path)

                translated_segments = []
                if task.translated_subtitle_path and task.translated_subtitle_path.exists():
                    translated_segments = self._parse_srt_file(task.translated_subtitle_path)

                if not original_segments or not translated_segments:
                    await self._skip_step(task, step_name, "Could not load subtitle segments")
                    return True
                # translated_segments are already dicts from _parse_srt_file
            else:
                # Convert transcription segments to dict format
                original_segments = [
                    {"text": seg.text, "start": seg.start, "end": seg.end}
                    for seg in transcription.segments
                ]
                # Convert translated_segments to dict format (they are TranscriptSegment objects)
                translated_segments = [
                    {"text": seg.text, "start": seg.start, "end": seg.end}
                    for seg in translated_segments
                ]

            # Configure proofreading
            config = ProofreadingConfig(
                min_confidence_threshold=opts.proofreading_min_confidence,
                use_ai_validation=True,
            )

            # Run proofreading
            logger.info(f"Proofreading {len(original_segments)} segments...")
            result = await proofread_subtitles(
                original_segments=original_segments,
                translated_segments=translated_segments,
                source_lang=opts.source_language if opts.source_language != "auto" else "en",
                target_lang=opts.target_language,
                config=config
            )

            # Store result
            task.proofreading_result = result.to_dict()

            # Save to database for persistence across restarts
            from database.task_persistence import task_persistence
            await task_persistence.save_proofreading_result(task.task_id, task.proofreading_result)

            # Complete the proofreading step
            has_issues = result.should_pause or result.total_issues > 0
            await self._complete_step(task, step_name,
                metadata={
                    "total_issues": result.total_issues,
                    "critical_issues": result.critical_issues,
                    "overall_confidence": result.overall_confidence,
                    "should_pause": has_issues,
                    "pause_reason": result.pause_reason if has_issues else None,
                }
            )

            await self._update_task(
                task, progress=54,
                message=f"校对完成 (置信度: {result.overall_confidence:.0%})"
            )

            return True  # Continue to optimize step

        except Exception as e:
            logger.error(f"Proofreading error: {e}")
            # Don't fail the task, just skip proofreading
            await self._skip_step(task, step_name, f"Proofreading error: {str(e)}")
            return True

    async def step_optimize(self, task: ProcessingTask) -> bool:
        """
        Step: AI Subtitle Optimization

        This step handles subtitle optimization after proofreading:
        - If auto_optimize is enabled: run optimization automatically and continue
        - If auto_optimize is disabled: pause and wait for user to optimize/confirm

        Returns:
            True to continue to next step
            False to pause the task (wait for user action)
        """
        step_name = "optimize"
        opts = task.options

        # Skip in direct mode (no subtitles)
        if not opts.add_subtitles and not opts.add_tts:
            await self._skip_step(task, step_name, "仅搬运模式，跳过字幕优化")
            return True

        # Skip if proofreading is not enabled
        if not opts.enable_proofreading:
            await self._skip_step(task, step_name, "Proofreading not enabled")
            return True

        # Check if proofreading was completed
        if not task.proofreading_result:
            await self._skip_step(task, step_name, "No proofreading result available")
            return True

        await self._start_step(task, step_name)

        try:
            # Check for cancellation
            task.check_cancelled()

            # Load global settings to check auto_optimize
            from settings_store import settings_store
            global_settings = settings_store.load()
            auto_optimize = global_settings.proofreading.auto_optimize

            proofreading_result = task.proofreading_result
            has_issues = proofreading_result.get("total_issues", 0) > 0

            if auto_optimize and has_issues:
                # Auto-optimize: run optimization and continue
                await self._update_task(task, progress=56, message="AI自动优化字幕中...")

                await self._run_optimization(task, proofreading_result, opts)

                await self._complete_step(task, step_name,
                    metadata={
                        "auto_optimized": True,
                        "optimized_count": task.optimization_result.get("optimized_count", 0) if task.optimization_result else 0,
                    }
                )

                await self._update_task(task, progress=58, message="AI优化完成，继续处理...")
                logger.info(f"Task {task.task_id} auto-optimized, continuing to next step")
                return True  # Continue to TTS

            elif auto_optimize and not has_issues:
                # No issues, skip optimization
                await self._complete_step(task, step_name,
                    metadata={
                        "auto_optimized": False,
                        "reason": "no_issues",
                    }
                )
                await self._update_task(task, progress=58, message="无需优化，继续处理...")
                return True  # Continue to TTS

            else:
                # Manual mode: pause and wait for user
                await self._update_task(task, progress=55, message="等待用户优化确认...")

                # Complete the step but mark as needing user action
                await self._complete_step(task, step_name,
                    metadata={
                        "auto_optimized": False,
                        "waiting_for_user": True,
                        "has_issues": has_issues,
                    }
                )

                # Pause the task
                if has_issues:
                    task.message = f"字幕校对发现 {proofreading_result.get('total_issues', 0)} 个问题，请优化后确认继续"
                else:
                    confidence = proofreading_result.get('overall_confidence', 0)
                    task.message = f"校对完成 (置信度: {confidence:.0%})，请确认字幕后继续"

                task.status = TaskStatus.PAUSED
                task.current_step = None
                logger.info(f"Task {task.task_id} paused for manual optimization/confirmation")
                return False  # Signal to stop processing

        except Exception as e:
            logger.error(f"Optimization step error: {e}")
            # Don't fail the task, just skip optimization
            await self._skip_step(task, step_name, f"Optimization error: {str(e)}")
            return True

    async def _run_optimization(
        self,
        task: ProcessingTask,
        proofreading_result: Dict[str, Any],
        opts: ProcessingOptions
    ):
        """Run subtitle optimization and save results"""
        try:
            from proofreading.optimizer import optimize_subtitles
            from subtitles.parser import SubtitleParser

            logger.info(f"Running optimization for task {task.task_id}")

            # Parse current subtitles
            parser = SubtitleParser()
            translated_segments = parser.parse_file(task.translated_subtitle_path)
            original_segments = []
            if task.subtitle_path and task.subtitle_path.exists():
                original_segments = parser.parse_file(task.subtitle_path)

            # Build segments list
            segments = []
            for i, trans_seg in enumerate(translated_segments):
                orig_text = ""
                if i < len(original_segments):
                    orig_text = original_segments[i].text

                segments.append({
                    "index": i,
                    "start_time": trans_seg.start,
                    "end_time": trans_seg.end,
                    "original_text": orig_text,
                    "translated_text": trans_seg.text
                })

            # Load optimization level from global settings
            from settings_store import settings_store
            global_settings = settings_store.load()
            optimization_level = global_settings.proofreading.optimization_level

            # Run optimization
            result = await optimize_subtitles(
                segments=segments,
                proofreading_result=proofreading_result,
                source_lang=opts.source_language,
                target_lang=opts.target_language,
                level=optimization_level
            )

            if result.success:
                # Apply optimized text to subtitle file
                optimized_segments = translated_segments.copy()
                for change in result.changes or []:
                    idx = change.get("index", -1)
                    if 0 <= idx < len(optimized_segments):
                        optimized_segments[idx].text = change.get("optimized_text", optimized_segments[idx].text)

                # Save optimized subtitles as SRT format
                def format_srt_time(seconds: float) -> str:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 1000)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

                srt_lines = []
                for i, seg in enumerate(optimized_segments):
                    srt_lines.append(str(i + 1))
                    srt_lines.append(f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}")
                    srt_lines.append(seg.text)
                    srt_lines.append("")

                srt_content = "\n".join(srt_lines)
                task.translated_subtitle_path.write_text(srt_content, encoding="utf-8")
                logger.info(f"Saved optimized subtitles to {task.translated_subtitle_path}")

                # Store optimization result as dict for persistence
                task.optimization_result = result.to_dict()

                # Persist optimization result
                from database.task_persistence import task_persistence
                await task_persistence.save_optimization_result(task.task_id, task.optimization_result)

        except Exception as e:
            logger.error(f"Optimization error: {e}")
            raise

    async def _auto_optimize_subtitles(
        self,
        task: ProcessingTask,
        proofreading_result: Dict[str, Any],
        opts: ProcessingOptions
    ):
        """Auto-optimize subtitles based on proofreading results"""
        try:
            from proofreading.optimizer import optimize_subtitles
            from subtitles.parser import SubtitleParser

            logger.info(f"Auto-optimizing subtitles for task {task.task_id}")
            await self._update_task(task, progress=56, message="AI优化字幕中...")

            # Parse current subtitles
            parser = SubtitleParser()
            translated_segments = parser.parse_file(task.translated_subtitle_path)
            original_segments = []
            if task.subtitle_path and task.subtitle_path.exists():
                original_segments = parser.parse_file(task.subtitle_path)

            # Build segments list
            segments = []
            for i, trans_seg in enumerate(translated_segments):
                orig_text = ""
                if i < len(original_segments):
                    orig_text = original_segments[i].text

                segments.append({
                    "index": i,
                    "start_time": trans_seg.start,
                    "end_time": trans_seg.end,
                    "original_text": orig_text,
                    "translated_text": trans_seg.text
                })

            # Run optimization
            result = await optimize_subtitles(
                segments=segments,
                proofreading_result=proofreading_result,
                source_lang=opts.source_language,
                target_lang=opts.target_language,
                level=opts.proofreading_optimization_level
            )

            if result.success and result.optimized_count > 0:
                # Save optimized subtitles
                def format_srt_time(seconds: float) -> str:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 1000)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

                srt_lines = []
                for seg in result.segments:
                    idx = seg.get("index", 0) + 1
                    start = format_srt_time(seg.get("start_time", 0))
                    end = format_srt_time(seg.get("end_time", 0))
                    text = seg.get("translated_text", "")

                    srt_lines.append(str(idx))
                    srt_lines.append(f"{start} --> {end}")
                    srt_lines.append(text)
                    srt_lines.append("")

                srt_content = "\n".join(srt_lines)
                task.translated_subtitle_path.write_text(srt_content, encoding="utf-8")

                logger.info(f"Auto-optimization complete: {result.optimized_count} segments optimized")
                await self._update_task(
                    task, progress=57,
                    message=f"AI优化完成 ({result.optimized_count}条字幕已优化)"
                )

                # Clear proofreading result since subtitles changed
                task.proofreading_result = None
                from database.task_persistence import task_persistence
                await task_persistence.save_proofreading_result(task.task_id, None)
            else:
                logger.info(f"Auto-optimization: no changes needed or failed")

        except Exception as e:
            logger.error(f"Auto-optimization failed: {e}")
            # Don't fail the task, just log the error

    async def step_tts(self, task: ProcessingTask) -> bool:
        """Step 4: Generate TTS audio"""
        step_name = "tts"
        opts = task.options

        # Check if TTS is needed
        if not opts.add_tts:
            await self._skip_step(task, step_name, "TTS not requested")
            return True

        # Skip TTS in subtitles-only mode (requires translated text)
        if opts.skip_translation:
            await self._skip_step(task, step_name, "Subtitles-only mode (no TTS)")
            return True

        await self._start_step(task, step_name)
        await self._update_task(task, TaskStatus.GENERATING_TTS, 55, "生成配音中...")

        try:
            # Check for cancellation before starting
            task.check_cancelled()

            # Check for translated subtitles
            if not task.translated_subtitle_path or not task.translated_subtitle_path.exists():
                raise Exception("Translated subtitles not found. Please run translation step first.")

            # Parse SRT file to get segments with timing
            segments = parse_srt_segments(task.translated_subtitle_path)

            if not segments:
                raise Exception("No subtitle segments found in translated subtitles")

            logger.info(f"Found {len(segments)} subtitle segments for dubbing")

            # Save TTS audio to task directory
            task_dir = await task.get_output_dir()
            task.tts_audio_path = task_dir / "tts.mp3"
            temp_dir = task_dir / "tts_temp"

            # Select TTS engine based on task options or global settings
            tts_engine = self._get_tts_engine_for_task(opts)

            # Set reference audio for voice cloning engines (IndexTTS, CosyVoice, Qwen3)
            if hasattr(tts_engine, 'set_reference_audio'):
                # Check voice cloning mode - only proceed if not disabled
                voice_cloning_mode = getattr(opts, 'voice_cloning_mode', 'disabled')

                if voice_cloning_mode == 'disabled':
                    logger.info("Voice cloning disabled by user settings")
                    ref_path = None
                elif voice_cloning_mode == 'custom' and opts.tts_ref_audio:
                    # Use user-specified custom reference audio
                    ref_path = Path(opts.tts_ref_audio)
                    logger.info(f"Using custom reference audio: {ref_path}")
                elif voice_cloning_mode == 'video_audio' and task.audio_path and task.audio_path.exists():
                    # Use video's original audio as reference for voice cloning
                    ref_path = task.audio_path
                    logger.info(f"Using video's original audio as reference: {ref_path}")
                else:
                    ref_path = None
                    if voice_cloning_mode != 'disabled':
                        logger.warning(f"Voice cloning mode is '{voice_cloning_mode}' but no reference audio available")

                if ref_path and ref_path.exists():
                    await tts_engine.set_reference_audio(ref_path)
                elif voice_cloning_mode != 'disabled':
                    logger.warning(f"Reference audio path does not exist: {ref_path}")

            # Use DubbingProcessor for synchronized dubbing (VideoLingo-style)
            dubbing_processor = DubbingProcessor(tts_engine)

            # Check if using local GPU-based TTS engine
            local_tts_engines = ('index', 'cosyvoice', 'qwen3')
            is_local_tts = opts.tts_service.lower() in local_tts_engines

            if is_local_tts:
                # Use GPU lock for local TTS to prevent MPS conflicts
                async with self._gpu_lock:
                    logger.debug("Acquired GPU lock for local TTS")
                    dubbing_result = await dubbing_processor.process(
                        segments=segments,
                        output_path=task.tts_audio_path,
                        voice=opts.tts_voice,
                        rate=opts.tts_rate,
                        temp_dir=temp_dir,
                        cancel_check=task.is_cancelled
                    )
                    logger.debug("Released GPU lock")
            else:
                # Edge TTS uses cloud API, can run in parallel
                dubbing_result = await dubbing_processor.process(
                    segments=segments,
                    output_path=task.tts_audio_path,
                    voice=opts.tts_voice,
                    rate=opts.tts_rate,
                    temp_dir=temp_dir,
                    cancel_check=task.is_cancelled
                )

            # Check for cancellation after TTS
            task.check_cancelled()

            if not dubbing_result.success:
                raise Exception(f"Dubbing failed: {dubbing_result.error}")

            # Cleanup temp directory
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

            await self._complete_step(task, step_name,
                output_files={
                    "tts_audio": str(task.tts_audio_path),
                },
                metadata={
                    "engine": opts.tts_service,
                    "voice": opts.tts_voice,
                    "rate": opts.tts_rate,
                    "segments_count": len(segments),
                    "total_duration": dubbing_result.total_duration,
                    "ref_audio": opts.tts_ref_audio,
                    "method": "synchronized_dubbing",  # New method indicator
                }
            )
            # Persist TTS audio path immediately
            from database.task_persistence import task_persistence
            await task_persistence.save_task_files(
                task.task_id,
                tts_audio_path=str(task.tts_audio_path) if task.tts_audio_path else None
            )
            await self._update_task(task, progress=65, message="配音生成完成")
            return True

        except Exception as e:
            await self._fail_step(task, step_name, str(e))
            raise

    async def _convert_vtt_to_srt(self, vtt_path: Path, srt_path: Path):
        """Convert VTT subtitle file to SRT format"""
        import re

        with open(vtt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove VTT header
        content = re.sub(r'^WEBVTT\n', '', content)
        content = re.sub(r'^Kind:.*\n', '', content)
        content = re.sub(r'^Language:.*\n', '', content)

        # Convert timestamps (VTT uses . for milliseconds, SRT uses ,)
        content = re.sub(r'(\d{2}:\d{2}:\d{2})\.(\d{3})', r'\1,\2', content)

        # Add sequence numbers if missing
        blocks = content.strip().split('\n\n')
        srt_blocks = []
        for i, block in enumerate(blocks, 1):
            lines = block.strip().split('\n')
            if lines and '-->' in lines[0]:
                # Block doesn't have sequence number
                srt_blocks.append(f"{i}\n{block}")
            elif len(lines) > 1 and '-->' in lines[1]:
                # Block already has sequence number, renumber it
                srt_blocks.append(f"{i}\n" + '\n'.join(lines[1:]))
            else:
                # Not a valid cue, skip
                continue

        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(srt_blocks))

        logger.info(f"Converted VTT to SRT: {vtt_path} -> {srt_path}")

    async def _parse_srt_to_transcription(self, srt_path: Path, language: str = "en"):
        """Parse SRT file and create a transcription-like object for the translation step"""
        from transcription.whisper_transcriber import TranscriptionResult, TranscriptSegment

        segments = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse SRT format
        import re
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            seq_num, start_time, end_time, text = match

            # Convert timestamp to seconds
            def time_to_seconds(ts):
                h, m, s_ms = ts.split(':')
                s, ms = s_ms.split(',')
                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

            segment = TranscriptSegment(
                start=time_to_seconds(start_time),
                end=time_to_seconds(end_time),
                text=text.strip().replace('\n', ' ')
            )
            segments.append(segment)

        # Combine all segment texts for full_text
        full_text = ' '.join(seg.text for seg in segments)

        return TranscriptionResult(
            success=True,
            language=language,
            segments=segments,
            full_text=full_text
        )

    def _get_tts_engine_for_task(self, opts: ProcessingOptions) -> BaseTTSEngine:
        """
        Get TTS engine for a specific task based on options

        If task specifies a different engine than global setting, create a new instance.
        Otherwise use the global engine.
        """
        task_service = opts.tts_service.lower()
        global_service = settings.TTS_SERVICE.lower()

        logger.info(f"TTS engine selection: task_service={task_service}, global_service={global_service}")

        # If task service matches global, use existing engine
        if task_service == global_service:
            logger.info(f"Using global TTS engine: {global_service}")
            return self.tts_engine

        # Create task-specific engine
        if task_service == "index":
            # Reference audio will be set in step_tts from video's original audio
            logger.info(f"Using task-specific IndexTTS engine")
            return IndexTTSEngine(
                host=settings.INDEX_TTS_HOST,
                port=settings.INDEX_TTS_PORT,
                ref_audio_path=None,  # Will be set from video's audio in step_tts
                emo_mode=settings.INDEX_TTS_EMO_MODE,
                emo_weight=settings.INDEX_TTS_EMO_WEIGHT,
            )
        elif task_service == "cosyvoice":
            logger.info(f"Using task-specific CosyVoice engine")
            return CosyVoiceEngine(
                host=settings.COSYVOICE_HOST,
                port=settings.COSYVOICE_PORT,
                mode=settings.COSYVOICE_MODE,
                default_speaker=settings.COSYVOICE_SPEAKER,
            )
        elif task_service == "qwen3":
            logger.info(f"Using task-specific Qwen3-TTS engine")
            return Qwen3TTSEngine(
                default_voice=opts.tts_voice or "vivian",
                model_name=getattr(settings, 'QWEN3_TTS_MODEL', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'),
                default_language=getattr(settings, 'QWEN3_TTS_LANGUAGE', 'Chinese'),
            )
        else:
            # Default to Edge TTS
            logger.info(f"Using task-specific Edge TTS engine")
            return EdgeTTSEngine(default_voice=opts.tts_voice)

    def _get_translator_for_task(self, opts: ProcessingOptions):
        """
        Get translator for a specific task based on options

        Uses OptimizedTranslator for GPT/Claude to reduce token usage.
        Falls back to standard Translator for free engines (google).
        """
        from settings_store import settings_store
        from translation import OptimizedTranslator

        task_engine = opts.translation_engine.lower()

        # Get API key from global settings based on engine type
        global_settings = settings_store.load()
        api_key = global_settings.translation.api_keys.get_key_for_engine(task_engine)
        # Fallback to legacy api_key if api_keys not configured
        if not api_key:
            api_key = global_settings.translation.api_key
        model = global_settings.translation.model

        # Check if we should use optimized translator (for paid APIs)
        use_optimized = global_settings.translation.use_optimized
        fast_mode = global_settings.translation.fast_mode
        paid_engines = ["gpt", "claude", "deepseek"]

        if use_optimized and task_engine in paid_engines and api_key:
            # Set appropriate model for engine if not specified
            if not model or model == "gpt-4":
                engine_models = {
                    "gpt": "gpt-4",
                    "claude": "claude-3-sonnet-20240229",
                    "deepseek": "deepseek-chat",
                }
                model = engine_models.get(task_engine, model)

            logger.info(
                f"Using OptimizedTranslator: engine={task_engine}, model={model}, "
                f"fast_mode={fast_mode} (VideoLingo-style, reduced tokens)"
            )
            return OptimizedTranslator(
                api_key=api_key,
                model=model,
                engine=task_engine,  # Pass engine for correct API endpoint
                chunk_size=8,  # 8 lines per chunk
                context_before=3,
                context_after=2,
                enable_reflection=not fast_mode
            ), fast_mode

        # Use standard translator for free engines or when optimization is disabled
        logger.info(f"Using standard Translator: engine={task_engine}")
        return Translator(engine=task_engine, api_key=api_key, model=model), False

    async def step_process_video(self, task: ProcessingTask) -> bool:
        """Step 5: Process video (burn subtitles, merge audio)"""
        step_name = "process_video"
        opts = task.options

        # Skip entirely in transfer-only mode (no subtitles, no TTS)
        if not opts.add_subtitles and not opts.add_tts:
            logger.info("Skipping video processing: transfer-only mode")
            # Use original video as final video (no processing needed)
            task.final_video_path = task.video_path
            await self._skip_step(task, step_name, "仅搬运模式，跳过视频处理")
            return True

        await self._start_step(task, step_name)
        await self._update_task(task, TaskStatus.PROCESSING_VIDEO, 70, "处理视频中...")

        try:
            # Check for cancellation before starting
            task.check_cancelled()

            # Check prerequisites
            if not task.video_path or not task.video_path.exists():
                raise Exception("Source video not found. Please run download step first.")

            # Save final video to task directory
            task_dir = await task.get_output_dir()
            task.final_video_path = task_dir / "final.mp4"

            # Load subtitle settings from settings_store
            from settings_store import settings_store
            from database.task_persistence import settings_persistence
            from utils.subtitle_burner import adapt_style_for_vertical
            global_settings = settings_store.load()
            subtitle_settings = global_settings.subtitle

            # Get video dimensions for vertical video detection
            video_dimensions = await self.video_processor.get_video_info(task.video_path)
            is_vertical_video = video_dimensions and video_dimensions.is_vertical
            if is_vertical_video:
                logger.info(f"Detected vertical video: {video_dimensions.width}x{video_dimensions.height} "
                           f"(aspect ratio: {video_dimensions.aspect_ratio:.2f})")

            # Add subtitles
            if opts.add_subtitles:
                # Load preset data once for all subtitle modes
                preset_data = None
                preset_id = opts.subtitle_preset

                # Auto-select vertical preset if video is vertical and no preset specified
                if is_vertical_video and not preset_id:
                    preset_id = "vertical_classic"  # Default vertical preset
                    logger.info(f"Auto-selecting vertical preset: {preset_id}")

                if preset_id:
                    preset_data = await settings_persistence.get_preset_by_id(preset_id)
                    if preset_data:
                        logger.info(f"Using subtitle preset: {preset_data.get('name')} (ID: {preset_id})")
                    else:
                        logger.warning(f"Subtitle preset not found: {preset_id}, using global defaults")

                # Determine subtitle mode from preset or fallback to dual_subtitles option
                subtitle_mode = "dual"  # default
                if preset_data:
                    subtitle_mode = preset_data.get('subtitle_mode', 'dual')
                    logger.info(f"Using subtitle_mode from preset: {subtitle_mode}")
                elif not opts.dual_subtitles:
                    subtitle_mode = "translated_only"
                    logger.info(f"Using subtitle_mode from dual_subtitles option: {subtitle_mode}")

                if subtitle_mode == "dual" and task.translated_subtitle_path and task.translated_subtitle_path.exists():
                    # Dual subtitles mode - use preset or global defaults
                    if preset_data:
                        # Use preset styles
                        original_style = SubtitleStyle.from_settings(preset_data.get('original_style', {}))
                        translated_style = SubtitleStyle.from_settings(preset_data.get('translated_style', {}))
                        logger.debug(f"[DEBUG] Created original_style: font_size={original_style.font_size}, color={original_style.primary_color}, bold={original_style.bold}")
                        logger.debug(f"[DEBUG] Created translated_style: font_size={translated_style.font_size}, color={translated_style.primary_color}, bold={translated_style.bold}")
                    else:
                        # Create SubtitleStyle objects from global settings
                        original_style = SubtitleStyle.from_settings(
                            subtitle_settings.original_style.to_dict()
                        )
                        translated_style = SubtitleStyle.from_settings(
                            subtitle_settings.translated_style.to_dict()
                        )
                        logger.debug(f"[DEBUG] Using global defaults: original font_size={original_style.font_size}, translated font_size={translated_style.font_size}")

                    # Adapt styles for vertical video if not using a vertical preset
                    if is_vertical_video and video_dimensions and not preset_id.startswith("vertical_"):
                        logger.info("Adapting subtitle styles for vertical video dimensions")
                        original_style = adapt_style_for_vertical(original_style, video_dimensions.width, video_dimensions.height)
                        translated_style = adapt_style_for_vertical(translated_style, video_dimensions.width, video_dimensions.height)

                    # Dual subtitles (original + translation) with per-language styling
                    # Pass video dimensions for vertical video optimization
                    await self.subtitle_burner.burn_dual_subtitles(
                        task.video_path,
                        task.subtitle_path,
                        task.translated_subtitle_path,
                        task.final_video_path,
                        original_style=original_style,
                        translated_style=translated_style,
                        chinese_on_top=subtitle_settings.chinese_on_top,
                        video_width=video_dimensions.width if video_dimensions else 0,
                        video_height=video_dimensions.height if video_dimensions else 0
                    )
                elif subtitle_mode == "translated_only" and task.translated_subtitle_path and task.translated_subtitle_path.exists():
                    # Single subtitle (translated only)
                    single_style = None
                    if preset_data:
                        single_style = SubtitleStyle.from_settings(preset_data.get('translated_style', {}))
                        logger.info(f"Using preset translated_style for single subtitle: {preset_data.get('name')}")
                    
                    if single_style:
                        await self.subtitle_burner.burn_subtitles(
                            task.video_path,
                            task.translated_subtitle_path,
                            task.final_video_path,
                            style=single_style,
                            video_width=video_dimensions.width if video_dimensions else 0,
                            video_height=video_dimensions.height if video_dimensions else 0
                        )
                    else:
                        await self.subtitle_burner.burn_subtitles(
                            task.video_path,
                            task.translated_subtitle_path,
                            task.final_video_path,
                            video_width=video_dimensions.width if video_dimensions else 0,
                            video_height=video_dimensions.height if video_dimensions else 0
                        )
                elif subtitle_mode == "original_only" and task.subtitle_path and task.subtitle_path.exists():
                    # Single subtitle (original only)
                    single_style = None
                    if preset_data:
                        single_style = SubtitleStyle.from_settings(preset_data.get('original_style', {}))
                        logger.info(f"Using preset original_style for single subtitle: {preset_data.get('name')}")

                    if single_style:
                        await self.subtitle_burner.burn_subtitles(
                            task.video_path,
                            task.subtitle_path,
                            task.final_video_path,
                            style=single_style,
                            video_width=video_dimensions.width if video_dimensions else 0,
                            video_height=video_dimensions.height if video_dimensions else 0
                        )
                    else:
                        await self.subtitle_burner.burn_subtitles(
                            task.video_path,
                            task.subtitle_path,
                            task.final_video_path,
                            video_width=video_dimensions.width if video_dimensions else 0,
                            video_height=video_dimensions.height if video_dimensions else 0
                        )
                elif task.translated_subtitle_path and task.translated_subtitle_path.exists():
                    # Fallback: translated subtitle available
                    await self.subtitle_burner.burn_subtitles(
                        task.video_path,
                        task.translated_subtitle_path,
                        task.final_video_path,
                        video_width=video_dimensions.width if video_dimensions else 0,
                        video_height=video_dimensions.height if video_dimensions else 0
                    )
                elif task.subtitle_path and task.subtitle_path.exists():
                    # Fallback: original subtitle available
                    await self.subtitle_burner.burn_subtitles(
                        task.video_path,
                        task.subtitle_path,
                        task.final_video_path,
                        video_width=video_dimensions.width if video_dimensions else 0,
                        video_height=video_dimensions.height if video_dimensions else 0
                    )
                else:
                    # No subtitles available, just copy
                    shutil.copy(task.video_path, task.final_video_path)
            else:
                # No subtitles, just copy
                shutil.copy(task.video_path, task.final_video_path)

            # Check for cancellation after subtitle processing
            task.check_cancelled()

            # Add TTS audio
            if opts.add_tts and task.tts_audio_path and task.tts_audio_path.exists():
                temp_video = task.final_video_path.with_suffix(".temp.mp4")
                task.final_video_path.rename(temp_video)

                await self.video_processor.merge_audio_video(
                    temp_video,
                    task.tts_audio_path,
                    task.final_video_path,
                    replace_audio=opts.replace_original_audio
                )

                temp_video.unlink(missing_ok=True)

            # Check for cancellation after video processing
            task.check_cancelled()

            # Get file size
            file_size = task.final_video_path.stat().st_size if task.final_video_path.exists() else 0

            await self._complete_step(task, step_name,
                output_files={
                    "final_video": str(task.final_video_path),
                },
                metadata={
                    "has_subtitles": opts.add_subtitles,
                    "has_tts": opts.add_tts and task.tts_audio_path and task.tts_audio_path.exists(),
                    "dual_subtitles": opts.dual_subtitles,
                    "file_size": file_size,
                }
            )
            # Persist final video path immediately
            from database.task_persistence import task_persistence
            await task_persistence.save_task_files(
                task.task_id,
                final_video_path=str(task.final_video_path) if task.final_video_path else None
            )
            await self._update_task(task, progress=85, message="视频处理完成")
            return True

        except Exception as e:
            await self._fail_step(task, step_name, str(e))
            raise

    async def step_upload(self, task: ProcessingTask) -> bool:
        """Step 6: Upload to platforms"""
        step_name = "upload"
        opts = task.options

        # Check if any upload is requested
        if not opts.upload_bilibili and not opts.upload_douyin and not opts.upload_xiaohongshu:
            await self._skip_step(task, step_name, "No upload platforms selected")
            await self._update_task(task, progress=100, message="处理完成（未上传）")
            return True

        await self._start_step(task, step_name)
        await self._update_task(task, TaskStatus.UPLOADING, 90, "上传中...")

        try:
            # Check prerequisites
            if not task.final_video_path or not task.final_video_path.exists():
                raise Exception("Final video not found. Please run video processing step first.")

            # Load generated metadata from database (this is what user approved)
            from database.task_persistence import task_persistence
            generated_metadata = await task_persistence.load_generated_metadata(task.task_id)

            # Helper function to get metadata for a specific platform
            def get_platform_metadata(platform: str) -> tuple:
                """Get title, description, tags for a specific platform"""
                if generated_metadata:
                    # Check if new format (per-platform) or old format (flat)
                    if platform in generated_metadata:
                        # New format: per-platform metadata
                        pm = generated_metadata[platform]
                        title = pm.get("title") or opts.custom_title or task.video_info.get("title", "Untitled")
                        description = pm.get("description") or opts.custom_description or task.video_info.get("description", "")
                        tags = pm.get("keywords") or opts.custom_tags or task.video_info.get("tags", [])
                        logger.info(f"Using platform-specific metadata for {platform}: {title[:30]}...")
                        return title, description, tags
                    elif "title" in generated_metadata:
                        # Old format: flat metadata (backward compatibility)
                        title = generated_metadata.get("title") or opts.custom_title or task.video_info.get("title", "Untitled")
                        description = generated_metadata.get("description") or opts.custom_description or task.video_info.get("description", "")
                        tags = generated_metadata.get("keywords") or opts.custom_tags or task.video_info.get("tags", [])
                        logger.info(f"Using legacy flat metadata for {platform}: {title[:30]}...")
                        return title, description, tags
                    elif "generic" in generated_metadata:
                        # Generic fallback
                        pm = generated_metadata["generic"]
                        title = pm.get("title") or opts.custom_title or task.video_info.get("title", "Untitled")
                        description = pm.get("description") or opts.custom_description or task.video_info.get("description", "")
                        tags = pm.get("keywords") or opts.custom_tags or task.video_info.get("tags", [])
                        logger.info(f"Using generic metadata for {platform}: {title[:30]}...")
                        return title, description, tags
                
                # Check if we have custom metadata set by user
                if opts.custom_title:
                    title = opts.custom_title
                    description = opts.custom_description or task.video_info.get("description", "")
                    tags = opts.custom_tags if opts.custom_tags else task.video_info.get("tags", [])
                    logger.info(f"Using custom metadata for {platform}: {title[:30]}...")
                    return title, description, tags
                
                # No generated or custom metadata
                logger.error(f"Task {task.task_id}: No metadata available for {platform}!")
                raise Exception(f"没有可用的元数据({platform})，请先生成或手动填写元数据后再上传")

            # Get cover/thumbnail path - check if AI thumbnail should be used
            cover_path = None
            ai_thumbnail_info = await task_persistence.load_ai_thumbnail_info(task.task_id)
            use_ai_thumbnail = ai_thumbnail_info.get("use_ai_thumbnail", False) if ai_thumbnail_info else False
            ai_thumbnail_path = ai_thumbnail_info.get("ai_thumbnail_path") if ai_thumbnail_info else None

            if use_ai_thumbnail and ai_thumbnail_path:
                ai_path = Path(ai_thumbnail_path)
                if ai_path.exists():
                    cover_path = ai_path
                    logger.info(f"Using AI-generated thumbnail: {ai_path}")
                else:
                    logger.warning(f"AI thumbnail selected but file not found: {ai_path}")

            # Fall back to original thumbnail if AI thumbnail not used/available
            if not cover_path:
                cover_path = task.thumbnail_path if task.thumbnail_path and task.thumbnail_path.exists() else None
                if cover_path:
                    logger.info(f"Using original thumbnail: {cover_path}")

            if not cover_path:
                # Try to generate thumbnail from final video
                import subprocess
                logger.info("Generating thumbnail from final video...")
                task_dir = await task.get_output_dir()
                generated_thumbnail = task_dir / "thumbnail_generated.jpg"
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-i", str(task.final_video_path), "-ss", "00:00:01",
                         "-vframes", "1", "-q:v", "2", "-y", str(generated_thumbnail)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0 and generated_thumbnail.exists():
                        task.thumbnail_path = generated_thumbnail
                        cover_path = generated_thumbnail
                        logger.info(f"Generated thumbnail from video: {generated_thumbnail}")
                except Exception as thumb_err:
                    logger.warning(f"Failed to generate thumbnail: {thumb_err}")

            if not cover_path:
                logger.warning("No thumbnail available for upload")

            # Helper to create VideoMetadata for a platform
            def create_metadata_for_platform(platform: str) -> VideoMetadata:
                title, description, tags = get_platform_metadata(platform)
                return VideoMetadata(
                    title=title,
                    description=description,
                    tags=tags,
                    cover_path=cover_path,
                    is_original=False,
                    source_url=opts.source_url if hasattr(opts, 'source_url') else None,
                )

            # Upload to each platform with platform-specific metadata
            upload_tasks = []

            if opts.upload_bilibili:
                bilibili_metadata = create_metadata_for_platform("bilibili")
                upload_tasks.append(self._upload_to_bilibili(task, bilibili_metadata))
            if opts.upload_douyin:
                douyin_metadata = create_metadata_for_platform("douyin")
                upload_tasks.append(self._upload_to_douyin(task, douyin_metadata))
            if opts.upload_xiaohongshu:
                xiaohongshu_metadata = create_metadata_for_platform("xiaohongshu")
                upload_tasks.append(self._upload_to_xiaohongshu(task, xiaohongshu_metadata))

            if upload_tasks:
                upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
                for result in upload_results:
                    if isinstance(result, Exception):
                        logger.error(f"Upload error: {result}")
                    elif result:
                        task.upload_results[result.platform] = {
                            "success": result.success,
                            "video_id": result.video_id,
                            "video_url": result.video_url,
                            "error": result.error
                        }

            # Save upload_results to database upload_results column
            if task.upload_results:
                from database.task_persistence import task_persistence
                await task_persistence.save_upload_results(task.task_id, task.upload_results)

            # Check upload results to determine success/failure
            successful_platforms = [p for p, r in task.upload_results.items() if r.get("success")]
            failed_platforms = [p for p, r in task.upload_results.items() if not r.get("success")]

            if not successful_platforms and failed_platforms:
                # All uploads failed
                error_msgs = [f"{p}: {task.upload_results[p].get('error', 'Unknown error')}" for p in failed_platforms]
                error_detail = "; ".join(error_msgs)
                await self._fail_step(task, step_name, f"所有平台上传失败: {error_detail}")
                raise Exception(f"Upload failed: {error_detail}")
            elif failed_platforms:
                # Some succeeded, some failed - complete with warning
                await self._complete_step(task, step_name,
                    output_files={},
                    metadata={
                        "platforms": list(task.upload_results.keys()),
                        "results": task.upload_results,
                    }
                )
                await self._update_task(task, progress=100,
                    message=f"部分上传完成 (成功: {', '.join(successful_platforms)}; 失败: {', '.join(failed_platforms)})")
            else:
                # All succeeded
                await self._complete_step(task, step_name,
                    output_files={},
                    metadata={
                        "platforms": list(task.upload_results.keys()),
                        "results": task.upload_results,
                    }
                )
                await self._update_task(task, progress=100, message="上传完成")
            return True

        except Exception as e:
            await self._fail_step(task, step_name, str(e))
            raise

    # ==================== AI Thumbnail Generation ====================

    async def _auto_generate_ai_thumbnail(self, task: ProcessingTask):
        """
        Auto-generate AI thumbnail if enabled in settings.
        Called after step_process_video completes.
        """
        try:
            from settings_store import settings_store
            from database.task_persistence import task_persistence

            # Load thumbnail settings
            global_settings = settings_store.load()
            thumb_settings = global_settings.thumbnail

            # Check if auto-generation is enabled
            if not thumb_settings.enabled or not thumb_settings.auto_generate:
                logger.debug("AI thumbnail auto-generation is disabled")
                return

            # Check if original thumbnail exists
            if not task.thumbnail_path or not task.thumbnail_path.exists():
                logger.warning(f"No original thumbnail available for AI generation: {task.task_id}")
                return

            logger.info(f"Auto-generating AI thumbnail for task {task.task_id}")

            # Get task directory for output
            task_dir = await task.get_output_dir()
            ai_thumbnail_path = task_dir / "ai_thumbnail.jpg"

            # Get video info for title generation
            video_title = task.video_info.get("title", "") if task.video_info else ""
            video_description = task.video_info.get("description", "") if task.video_info else ""
            video_tags = task.video_info.get("tags", []) if task.video_info else []

            # Get DeepSeek API key for title generation
            api_key = global_settings.translation.api_keys.deepseek or global_settings.translation.api_key

            # Create thumbnail generator with settings
            generator = ThumbnailGenerator(
                api_key=api_key,
                style=thumb_settings.style,
                font_name=thumb_settings.font_name if thumb_settings.font_name else None,
                font_size=thumb_settings.font_size,
                text_color=thumb_settings.text_color,
                gradient_color=thumb_settings.gradient_color,
                gradient_opacity=thumb_settings.gradient_opacity,
            )

            # Generate thumbnail
            result = await generator.generate(
                original_thumbnail_path=task.thumbnail_path,
                output_path=ai_thumbnail_path,
                video_title=video_title,
                video_description=video_description,
                keywords=video_tags,
            )

            if result.success:
                logger.info(f"AI thumbnail generated: {result.output_path}, title: {result.title}")
                # Save to database
                await task_persistence.save_ai_thumbnail(
                    task_id=task.task_id,
                    ai_thumbnail_path=str(result.output_path),
                    ai_thumbnail_title=result.title
                )
                # Also set default use_ai_thumbnail from settings
                if thumb_settings.default_use_ai:
                    await task_persistence.update_use_ai_thumbnail(task.task_id, True)
            else:
                logger.error(f"AI thumbnail generation failed: {result.error}")

        except Exception as e:
            # Non-fatal error - log and continue
            logger.error(f"Failed to auto-generate AI thumbnail: {e}")

    # ==================== AI Metadata Generation ====================

    async def _auto_generate_metadata(self, task: ProcessingTask):
        """
        Auto-generate AI metadata (title, description, keywords) if enabled in settings.
        Called after step_process_video completes.
        """
        try:
            from settings_store import settings_store
            from database.task_persistence import task_persistence
            from metadata import MetadataGenerator, select_preset_for_task

            # Load metadata settings
            global_settings = settings_store.load()
            metadata_settings = global_settings.metadata

            # Check if auto-generation is enabled
            if not metadata_settings.enabled or not metadata_settings.auto_generate:
                logger.debug("AI metadata auto-generation is disabled")
                return

            # Check if transcription is completed (required for metadata generation)
            if not task.subtitle_path or not task.subtitle_path.exists():
                logger.warning(f"No subtitle available for metadata generation: {task.task_id}")
                return

            logger.info(f"Auto-generating AI metadata for task {task.task_id}")

            # Get translation settings for API key
            engine = global_settings.translation.engine
            api_key = global_settings.translation.get_api_key()
            model = global_settings.translation.model

            if not api_key and engine not in ["google"]:
                logger.warning(f"API key not configured for {engine}, skipping metadata generation")
                return

            # Read transcript from subtitle file
            transcript = task.subtitle_path.read_text(encoding="utf-8")

            # Get original title
            original_title = task.video_info.get("title", "") if task.video_info else ""

            # Get source URL
            source_url = task.options.source_url if metadata_settings.include_source_url else ""

            # Determine which preset to use
            preset_id = task.options.metadata_preset_id
            title_prefix = ""
            custom_signature = ""

            # If AI preset selection is enabled, use it
            if task.options.use_ai_preset_selection:
                logger.info(f"Using AI to select metadata preset for task {task.task_id}")
                try:
                    # Build video_info dict for preset matching
                    video_info_for_match = {
                        "title": original_title,
                        "description": task.video_info.get("description", "") if task.video_info else "",
                        "tags": task.video_info.get("tags", []) if task.video_info else [],
                        "platform": task.video_info.get("platform", "") if task.video_info else "",
                    }
                    ai_result = await select_preset_for_task(
                        video_info=video_info_for_match,
                        transcript=transcript[:500] if transcript else ""
                    )
                    # ai_result is a PresetMatch dataclass with preset_id, confidence, reason
                    if ai_result and ai_result.preset_id:
                        preset_id = ai_result.preset_id
                        logger.info(f"AI selected preset: {preset_id} (confidence: {ai_result.confidence:.2f})")
                except Exception as e:
                    logger.warning(f"AI preset selection failed: {e}, using default preset")

            # Fetch preset (specified, AI-selected, or default)
            try:
                from database.connection import get_session_factory
                from database.repository import MetadataPresetRepository

                session_factory = get_session_factory()
                async with session_factory() as session:
                    preset_repo = MetadataPresetRepository(session)
                    
                    if preset_id:
                        preset = await preset_repo.get(preset_id)
                        if preset:
                            title_prefix = preset.title_prefix or ""
                            custom_signature = preset.custom_signature or ""
                            logger.info(f"Using metadata preset: {preset.name}")
                    
                    # Fall back to default preset if not found
                    if not title_prefix:
                        default_preset = await preset_repo.get_default()
                        if default_preset:
                            title_prefix = default_preset.title_prefix or ""
                            custom_signature = default_preset.custom_signature or ""
                            logger.info(f"Using default metadata preset: {default_preset.name}")
            except Exception as e:
                logger.warning(f"Failed to load preset: {e}")

            # Determine target platforms for metadata generation
            platforms_to_generate = []
            if task.options.upload_douyin:
                platforms_to_generate.append("douyin")
            if task.options.upload_bilibili:
                platforms_to_generate.append("bilibili")
            if task.options.upload_xiaohongshu:
                platforms_to_generate.append("xiaohongshu")
            
            # If no specific platforms, generate generic metadata
            if not platforms_to_generate:
                platforms_to_generate.append("generic")
            
            logger.info(f"Generating metadata for platforms: {platforms_to_generate}")

            # Create generator
            generator = MetadataGenerator(engine=engine, api_key=api_key, model=model)
            
            # Generate metadata for each platform
            metadata_by_platform = {}
            for platform in platforms_to_generate:
                logger.info(f"Generating metadata for platform: {platform}")
                result = await generator.generate(
                    original_title=original_title,
                    transcript=transcript,
                    source_url=source_url,
                    source_language=task.options.source_language,
                    target_language=task.options.target_language,
                    title_prefix=title_prefix,
                    custom_signature=custom_signature,
                    max_keywords=metadata_settings.max_keywords,
                    platform=platform
                )

                if result.success:
                    logger.info(f"AI metadata generated for {platform}: {result.title[:50]}...")
                    metadata_by_platform[platform] = {
                        "title": result.title_translated or result.title,
                        "description": result.description,
                        "keywords": result.keywords,
                    }
                else:
                    logger.error(f"AI metadata generation failed for {platform}: {result.error}")
            
            # Save all platform metadata to database
            if metadata_by_platform:
                await task_persistence.save_generated_metadata(task.task_id, metadata_by_platform)
                logger.info(f"Metadata saved for task {task.task_id}: {list(metadata_by_platform.keys())}")

        except Exception as e:
            # Non-fatal error - log and continue
            logger.error(f"Failed to auto-generate AI metadata: {e}")

    # ==================== Main Process Methods ====================

    async def process(self, task: ProcessingTask) -> ProcessingTask:
        """
        Run full processing pipeline for a task

        Args:
            task: ProcessingTask with options

        Returns:
            Updated ProcessingTask with results
        """
        self.tasks[task.task_id] = task
        # Record processing start time
        task.processing_started_at = datetime.now()
        
        # Set cancel_check on processors so they can be interrupted
        self.video_processor.set_cancel_check(task.is_cancelled)

        try:
            # Run video processing steps in sequence
            await self.step_download(task)
            
            # Handle processing mode
            mode = task.options.processing_mode
            
            # Auto mode: analyze content to determine actual mode
            if mode == "auto":
                mode = await self._detect_processing_mode(task)
                logger.info(f"Auto-detected processing mode: {mode}")
                task.options.processing_mode = mode  # Update for reference
            
            # Direct mode: skip all processing, use original video
            if mode == "direct":
                logger.info(f"Direct mode: skipping transcription, translation, TTS, and composition")
                await self._skip_step(task, "transcribe", "Direct mode (no processing)")
                await self._skip_step(task, "translate", "Direct mode (no processing)")
                # Set final video to original video
                task.final_video_path = task.video_path
                task.files["final_video"] = str(task.video_path)
                task.status = TaskStatus.PENDING_REVIEW
                task.message = "直接搬运模式，准备上传"
                # Skip to upload section
                await self._handle_post_processing(task)
                return task
            
            await self.step_transcribe(task)
            await self.step_translate(task)

            # Proofreading step
            await self.step_proofread(task)

            # Optimization step - may pause the task if manual mode
            optimize_ok = await self.step_optimize(task)
            if not optimize_ok:
                # Task was paused for manual optimization/confirmation
                return task

            # Subtitle mode: skip TTS, only embed subtitles
            if mode == "subtitle":
                logger.info(f"Subtitle mode: skipping TTS, will only embed subtitles")
                await self._skip_step(task, "tts", "Subtitle mode (no TTS)")
            else:
                await self.step_tts(task)
            
            await self.step_process_video(task)

            # Auto-generate AI thumbnail if enabled
            await self._auto_generate_ai_thumbnail(task)

            # Auto-generate AI metadata if enabled
            await self._auto_generate_metadata(task)

            # Calculate total processing time
            if task.processing_started_at:
                task.total_processing_time = (datetime.now() - task.processing_started_at).total_seconds()
                logger.info(f"Task {task.task_id} video processing completed in {task.get_total_time_formatted()}")

            # Check if metadata review is required
            from settings_store import settings_store
            from database.task_persistence import task_persistence
            global_settings = settings_store.load()
            require_review = global_settings.metadata.require_review
            metadata_enabled = global_settings.metadata.enabled and global_settings.metadata.auto_generate

            # If metadata auto-generation is enabled, verify it was actually generated
            metadata_generated = False
            if metadata_enabled:
                generated_metadata = await task_persistence.load_generated_metadata(task.task_id)
                # Check for new format (per-platform) or old format (flat with "title")
                if generated_metadata:
                    # New format: check if any platform has metadata
                    has_platform_metadata = any(
                        isinstance(generated_metadata.get(p), dict) and generated_metadata[p].get("title")
                        for p in ["douyin", "bilibili", "xiaohongshu", "generic"]
                    )
                    # Old format: check for flat "title" key
                    has_flat_metadata = bool(generated_metadata.get("title"))
                    metadata_generated = has_platform_metadata or has_flat_metadata
                if not metadata_generated:
                    logger.warning(f"Task {task.task_id}: Metadata auto-generation enabled but no metadata was generated")

            # Determine if we should wait for review
            # 1. If require_review is True, always wait
            # 2. If metadata was expected but not generated, wait for manual input
            should_wait_for_review = require_review or (metadata_enabled and not metadata_generated)

            if should_wait_for_review:
                # Stop here and wait for user to review/approve metadata
                task.status = TaskStatus.PENDING_REVIEW
                task.current_step = None
                if metadata_enabled and not metadata_generated:
                    task.message = "视频处理完成，元数据生成失败，请手动填写"
                    logger.info(f"Task {task.task_id} waiting for manual metadata input (auto-generation failed)")
                else:
                    task.message = "视频处理完成，等待元数据审核"
                    logger.info(f"Task {task.task_id} waiting for metadata review")
                return task
            else:
                # Auto-approve metadata and continue to upload
                task.status = TaskStatus.PENDING_UPLOAD
                task.message = "元数据已自动审核，准备上传"
                logger.info(f"Task {task.task_id} auto-approved, ready for upload")

                # Check if any upload platform is enabled
                opts = task.options
                if opts.upload_bilibili or opts.upload_douyin or opts.upload_xiaohongshu:
                    await self.step_upload(task)
                    task.current_step = None

                    # Check upload results to set correct status
                    successful = [p for p, r in task.upload_results.items() if r.get("success")]
                    failed = [p for p, r in task.upload_results.items() if not r.get("success")]

                    if successful and not failed:
                        # All uploads succeeded
                        task.status = TaskStatus.UPLOADED
                        task.message = "上传完成"
                        logger.info(f"Task {task.task_id} uploaded successfully")
                    elif successful and failed:
                        # Partial success
                        task.status = TaskStatus.UPLOADED
                        task.message = f"部分上传完成 (成功: {', '.join(successful)}; 失败: {', '.join(failed)})"
                        logger.warning(f"Task {task.task_id} partial upload: success={successful}, failed={failed}")
                    else:
                        # All failed - this should be caught by step_upload exception, but handle anyway
                        task.status = TaskStatus.FAILED
                        task.message = "上传失败"
                        logger.error(f"Task {task.task_id} all uploads failed")
                else:
                    # No upload platforms enabled, stay at pending_upload
                    task.current_step = None
                    task.message = "视频处理完成，无上传平台配置"
                    logger.info(f"Task {task.task_id} completed without upload (no platforms configured)")

            return task

        except Exception as e:
            # Calculate total processing time even for failed tasks
            if task.processing_started_at:
                task.total_processing_time = (datetime.now() - task.processing_started_at).total_seconds()
            # Log detailed error info for debugging
            import traceback
            error_msg = str(e) if str(e) else f"{type(e).__name__}: (no message)"
            logger.error(f"Pipeline error: {error_msg}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            task.status = TaskStatus.FAILED
            task.error = error_msg
            task.message = f"处理失败: {error_msg}"
            return task

    async def retry_step(self, task_id: str, step_name: str) -> ProcessingTask:
        """
        Retry a specific step and continue with subsequent steps

        Args:
            task_id: Task ID
            step_name: Name of step to retry

        Returns:
            Updated ProcessingTask
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Clear any previous cancellation
        task.clear_cancel()

        # Use continue_from_step to retry and continue
        return await self.continue_from_step(task_id, step_name)

    async def stop_task(self, task_id: str) -> ProcessingTask:
        """
        Stop a running task.

        Args:
            task_id: Task ID to stop

        Returns:
            Updated ProcessingTask
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Request cancellation
        task.request_cancel()
        task.status = TaskStatus.PAUSED

        # Mark the current running step as interrupted (so it can be retried)
        interrupted_step = None
        for step_name, step in task.steps.items():
            if step.status == StepStatus.RUNNING:
                step.status = StepStatus.FAILED
                step.error = "用户手动停止"
                interrupted_step = step_name
                logger.info(f"Marked step {step_name} as interrupted for task {task_id}")
                # Persist step failure to database
                from database.task_persistence import task_persistence
                await task_persistence.save_step_status(
                    task_id, step_name, step.status.value, error=step.error
                )

        if interrupted_step:
            task.message = f"任务已停止 (中断于: {interrupted_step})，可修改配置后重试"
        else:
            task.message = "任务已停止，可修改配置后重试"

        task.current_step = None
        logger.info(f"Stop requested for task {task_id}, interrupted step: {interrupted_step}")

        return task

    def update_task_options(self, task_id: str, updates: Dict[str, Any]) -> ProcessingTask:
        """
        Update task options (when completed, paused, or failed).

        Args:
            task_id: Task ID
            updates: Dictionary of option updates

        Returns:
            Updated ProcessingTask
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        allowed_statuses = [
            TaskStatus.COMPLETED, TaskStatus.PAUSED, TaskStatus.FAILED,
            TaskStatus.PENDING_REVIEW, TaskStatus.PENDING_UPLOAD, TaskStatus.UPLOADED
        ]
        if task.status not in allowed_statuses:
            raise ValueError("Can only update options when task is not actively processing")

        # Update options
        for key, value in updates.items():
            if hasattr(task.options, key):
                setattr(task.options, key, value)
                logger.info(f"Updated task {task_id} option: {key} = {value}")
            else:
                logger.warning(f"Unknown option: {key}")

        return task

    async def continue_from_step(self, task_id: str, step_name: str) -> ProcessingTask:
        """
        Continue processing from a specific step

        Args:
            task_id: Task ID
            step_name: Name of step to start from

        Returns:
            Updated ProcessingTask
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        step_order = ["download", "transcribe", "translate", "proofread", "optimize", "tts", "process_video", "upload"]

        try:
            start_index = step_order.index(step_name)
        except ValueError:
            raise ValueError(f"Unknown step: {step_name}")

        step_methods = {
            "download": self.step_download,
            "transcribe": self.step_transcribe,
            "translate": self.step_translate,
            "proofread": self.step_proofread,
            "optimize": self.step_optimize,
            "tts": self.step_tts,
            "process_video": self.step_process_video,
            "upload": self.step_upload,
        }

        try:
            task.status = TaskStatus.PENDING
            task.error = None
            task.clear_cancel()  # Clear any previous cancellation

            # Load settings to check review requirement
            from settings_store import settings_store
            global_settings = settings_store.load()
            require_review = global_settings.metadata.require_review

            # Run remaining steps
            for step in step_order[start_index:]:
                # Check for cancellation before each step
                if task.is_cancelled():
                    task.status = TaskStatus.PAUSED
                    task.message = "任务已停止"
                    logger.info(f"Task {task_id} cancelled before step {step}")
                    return task

                # If we're about to run upload step and review is required, check approval
                if step == "upload" and require_review:
                    # Check if metadata is approved (via database)
                    from database.task_persistence import task_persistence
                    task_data = await task_persistence.load_task(task_id)
                    is_approved = task_data.get("metadata_approved", False) if task_data else False
                    logger.info(f"Task {task_id} upload check: is_approved={is_approved}, task_data exists={task_data is not None}")

                    if not is_approved:
                        task.status = TaskStatus.PENDING_REVIEW
                        task.current_step = None
                        task.message = "视频处理完成，等待元数据审核"
                        logger.info(f"Task {task_id} waiting for metadata review (metadata_approved=False)")
                        return task
                    else:
                        logger.info(f"Task {task_id} metadata approved, proceeding with upload")

                method = step_methods[step]
                result = await method(task)

                # Handle optimize step returning False (needs user confirmation)
                if step == "optimize" and result is False:
                    # Task was paused for manual optimization/confirmation
                    logger.info(f"Task {task_id} paused for manual optimization/confirmation")
                    return task

                # Check for cancellation after each step
                if task.is_cancelled():
                    task.status = TaskStatus.PAUSED
                    task.message = f"任务已停止 (完成到: {step})"
                    logger.info(f"Task {task_id} cancelled after step {step}")
                    return task

                # After process_video, generate AI thumbnail and check if we should stop for review
                if step == "process_video":
                    # Auto-generate AI thumbnail if enabled
                    await self._auto_generate_ai_thumbnail(task)

                    if require_review:
                        task.status = TaskStatus.PENDING_REVIEW
                        task.current_step = None
                        task.message = "视频处理完成，等待元数据审核"
                        logger.info(f"Task {task_id} waiting for metadata review")
                        return task

            # If we completed the upload step, check results to set correct status
            if step_order[-1] == "upload" and start_index <= step_order.index("upload"):
                successful = [p for p, r in task.upload_results.items() if r.get("success")]
                failed = [p for p, r in task.upload_results.items() if not r.get("success")]

                if successful and not failed:
                    # All uploads succeeded
                    task.status = TaskStatus.UPLOADED
                    task.message = "上传完成"
                elif successful and failed:
                    # Partial success
                    task.status = TaskStatus.UPLOADED
                    task.message = f"部分上传完成 (成功: {', '.join(successful)}; 失败: {', '.join(failed)})"
                elif failed:
                    # All uploads failed
                    task.status = TaskStatus.FAILED
                    task.message = f"上传失败: {', '.join(failed)}"
                else:
                    # No upload results - stay at pending_upload
                    task.status = TaskStatus.PENDING_UPLOAD
                    task.message = "等待上传"
            else:
                task.status = TaskStatus.PENDING_UPLOAD
            task.current_step = None
            return task

        except Exception as e:
            logger.error(f"Continue from step error: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return task

    # ==================== Upload Helper Methods ====================

    async def _upload_to_bilibili(self, task: ProcessingTask, metadata: VideoMetadata):
        """Upload to Bilibili"""
        from platform_credentials import platform_credentials
        from config import settings
        from bilibili_accounts import bilibili_account_manager

        # Create a fresh uploader for each upload to support multiple accounts
        uploader = BilibiliUploader()
        
        opts = task.options
        account_uid = opts.bilibili_account_uid
        authenticated = False
        account_name = None
        
        logger.info(f"Bilibili upload - account_uid from options: '{account_uid}'")
        
        # Try multi-account system first
        if account_uid:
            # Specific account requested
            logger.info(f"Looking for specific account: {account_uid}")
            account = await bilibili_account_manager.get_account(account_uid)
            if account:
                logger.info(f"Found account: {account.nickname} (label: {account.label})")
                auth_dict = account.to_auth_dict()
                logger.debug(f"Auth dict keys: {list(auth_dict.keys())}, values present: SESSDATA={bool(auth_dict.get('SESSDATA'))}, bili_jct={bool(auth_dict.get('bili_jct'))}, buvid3={bool(auth_dict.get('buvid3'))}")
                await uploader.authenticate(auth_dict)
                authenticated = uploader.is_authenticated
                account_name = account.display_name
            else:
                logger.warning(f"Account {account_uid} not found!")
        
        if not authenticated:
            # Try default account from multi-account system
            logger.info("No specific account or auth failed, trying default account")
            account = await bilibili_account_manager.get_default_account()
            if account:
                logger.info(f"Using default account: {account.nickname} (label: {account.label})")
                await uploader.authenticate(account.to_auth_dict())
                authenticated = uploader.is_authenticated
                account_name = account.display_name
        
        if not authenticated:
            # Fall back to old single-account system for backward compatibility
            creds = platform_credentials.get_bilibili()
            if creds.is_configured:
                await uploader.authenticate(creds.to_auth_dict())
                authenticated = uploader.is_authenticated
            elif settings.BILIBILI_SESSDATA:
                await uploader.authenticate({
                    "SESSDATA": settings.BILIBILI_SESSDATA,
                    "bili_jct": settings.BILIBILI_BILI_JCT,
                    "buvid3": settings.BILIBILI_BUVID3,
                })
                authenticated = uploader.is_authenticated

        if not authenticated:
            from uploaders.base import UploadResult
            return UploadResult(
                success=False,
                platform="bilibili",
                error="未配置Bilibili凭证，请在设置中配置或扫码登录"
            )
        
        if account_name:
            logger.info(f"Uploading to Bilibili as: {account_name}")

        return await uploader.upload(task.final_video_path, metadata)

    async def _upload_to_douyin(self, task: ProcessingTask, metadata: VideoMetadata):
        """Upload to Douyin using Playwright browser automation"""
        from platform_credentials import platform_credentials
        from config import settings
        from douyin_accounts import douyin_account_manager

        # Use Playwright uploader for more stability
        if not self._douyin:
            self._douyin = DouyinPlaywrightUploader()
            logger.info("Using Playwright-based Douyin uploader")

        # Authenticate if not already authenticated
        if not self._douyin.is_authenticated:
            # Check if specific account is requested
            account = None
            if task.options.douyin_account_id:
                account = await douyin_account_manager.get_account(task.options.douyin_account_id)
                if account:
                    logger.info(f"Using specified Douyin account: {account.nickname}")
            
            # Fall back to primary account
            if not account:
                account = await douyin_account_manager.get_primary()
                if account:
                    logger.info(f"Using Douyin primary account: {account.nickname}")
            
            if account:
                await self._douyin.authenticate(account.to_auth_dict())

            # Fall back to single-account system
            if not self._douyin.is_authenticated:
                creds = platform_credentials.get_douyin()
                logger.debug(f"Douyin single-account: is_configured={creds.is_configured}, cookies_len={len(creds.cookies) if creds.cookies else 0}")
                if creds.is_configured:
                    auth_dict = creds.to_auth_dict()
                    logger.info(f"Using Douyin single-account credentials (cookies length: {len(auth_dict.get('cookies', ''))})")
                    await self._douyin.authenticate(auth_dict)

            # Fall back to env vars
            if not self._douyin.is_authenticated and settings.DOUYIN_COOKIES:
                logger.info("Using Douyin env var credentials")
                await self._douyin.authenticate({"cookies": settings.DOUYIN_COOKIES})

        if not self._douyin.is_authenticated:
            from uploaders.base import UploadResult
            return UploadResult(
                success=False,
                platform="douyin",
                error="未配置抖音凭证，请在设置中配置"
            )

        return await self._douyin.upload(task.final_video_path, metadata)

    async def _upload_to_xiaohongshu(self, task: ProcessingTask, metadata: VideoMetadata):
        """Upload to Xiaohongshu"""
        from platform_credentials import platform_credentials
        from config import settings
        from xiaohongshu_accounts import xiaohongshu_account_manager

        if not self._xiaohongshu:
            self._xiaohongshu = XiaohongshuUploader()

        # Authenticate if not already authenticated
        if not self._xiaohongshu.is_authenticated:
            # Check if specific account is requested
            account = None
            if task.options.xiaohongshu_account_id:
                account = await xiaohongshu_account_manager.get_account(task.options.xiaohongshu_account_id)
                if account:
                    logger.info(f"Using specified Xiaohongshu account: {account.nickname}")
            
            # Fall back to primary account
            if not account:
                account = await xiaohongshu_account_manager.get_primary()
                if account:
                    logger.info(f"Using Xiaohongshu primary account: {account.nickname}")
            
            if account:
                await self._xiaohongshu.authenticate(account.to_auth_dict())

            # Fall back to single-account system
            if not self._xiaohongshu.is_authenticated:
                creds = platform_credentials.get_xiaohongshu()
                if creds.is_configured:
                    logger.info("Using Xiaohongshu single-account credentials")
                    await self._xiaohongshu.authenticate(creds.to_auth_dict())

            # Fall back to env vars
            if not self._xiaohongshu.is_authenticated and settings.XHS_COOKIES:
                logger.info("Using Xiaohongshu env var credentials")
                await self._xiaohongshu.authenticate({"cookies": settings.XHS_COOKIES})

        if not self._xiaohongshu.is_authenticated:
            from uploaders.base import UploadResult
            return UploadResult(
                success=False,
                platform="xiaohongshu",
                error="未配置小红书凭证，请在设置中配置"
            )

        return await self._xiaohongshu.upload(task.final_video_path, metadata)


# Global pipeline instance
pipeline = VideoPipeline()

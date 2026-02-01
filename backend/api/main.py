# -*- coding: utf-8 -*-
"""
FastAPI Backend for iDubb

Web管理界面后端API
支持步骤级别的任务追踪、重试和文件下载
支持任务和配置的持久化存储
"""
import os

# 禁用 uvloop，使用标准 asyncio 事件循环
# uvloop 在 macOS ARM64 上存在稳定性问题，可能导致段错误
os.environ.setdefault("UVLOOP_DISABLE", "1")
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
import mimetypes

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from pipeline import VideoPipeline, ProcessingTask, ProcessingOptions, TaskStatus, StepStatus
from task_executor import task_executor
from downloaders import VideoInfoParser
from settings_store import settings_store, GlobalSettings
from database import init_db, close_db, CookieRepository, MetadataPresetRepository
from database.repository import SubscriptionRepository, DirectoryRepository, TaskRepository, TrendingRepository
from database.connection import get_session_factory
from database.task_persistence import task_persistence, settings_persistence
from api.transcription_api import (
    get_whisper_models,
    get_model_by_id,
    estimate_transcription_time,
    get_recommended_model,
    get_all_estimates,
)
from subscriptions import (
    SubscriptionScheduler,
    get_fetcher,
    SUPPORTED_PLATFORMS,
    ChannelInfo,
    VideoInfo as SubscriptionVideoInfo,
)
from subscriptions.scheduler import subscription_scheduler
from trending import trending_scheduler
from tiktok import tiktok_scheduler


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - handle startup and shutdown"""
    # Startup
    logger.info("Starting iDubb API...")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Restore cookies from database
    await restore_cookies_from_db()

    # Start task persistence worker
    await task_persistence.start()
    logger.info("Task persistence worker started")

    # Load existing tasks from database into memory
    await load_tasks_from_db()
    logger.info("Tasks loaded from database")

    # Sync directory task counts
    await sync_directory_counts_on_startup()

    # Initialize and start task executor
    task_executor.set_process_function(process_task_with_persistence)
    await task_executor.start()
    logger.info(f"Task executor started (max_concurrent={task_executor.max_concurrent})")

    # Initialize and start subscription scheduler
    await init_subscription_scheduler()
    logger.info("Subscription scheduler started")

    # Initialize and start trending scheduler
    await init_trending_scheduler()
    logger.info("Trending scheduler started")

    # Initialize and start TikTok scheduler
    await init_tiktok_scheduler()
    logger.info("TikTok scheduler started")

    yield

    # Shutdown
    logger.info("Shutting down iDubb API...")
    await trending_scheduler.stop()
    await tiktok_scheduler.stop()
    await subscription_scheduler.stop()
    await task_executor.stop()
    await task_persistence.stop()
    await close_db()
    logger.info("Shutdown complete")


async def restore_cookies_from_db():
    """Restore cookie files from database on startup"""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            cookie_repo = CookieRepository(session)

            # Restore YouTube cookies if available
            youtube_cookie_path = settings.DATA_DIR / "youtube_cookies.txt"
            success = await cookie_repo.write_cookie_file("youtube", youtube_cookie_path)
            if success:
                logger.info("Restored YouTube cookies from database")
            else:
                logger.info("No YouTube cookies found in database")

    except Exception as e:
        logger.warning(f"Failed to restore cookies from database: {e}")


async def sync_directory_counts_on_startup():
    """Sync directory task counts on startup"""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            dir_repo = DirectoryRepository(session)
            task_repo = TaskRepository(session)
            updated = await dir_repo.sync_task_counts(task_repo)
            await session.commit()
            if updated:
                logger.info(f"Synced directory task counts: {updated}")
    except Exception as e:
        logger.warning(f"Failed to sync directory counts: {e}")


async def init_subscription_scheduler():
    """Initialize the subscription scheduler with callbacks"""
    from datetime import timedelta

    async def get_due_subscriptions():
        """Get subscriptions that are due for checking"""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = SubscriptionRepository(session)
            return await repo.get_due_subscriptions()

    async def update_subscription(subscription_id: str, updates: dict):
        """Update subscription data"""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = SubscriptionRepository(session)
            
            # Handle error_count increment when last_error is set
            if "last_error" in updates and updates.get("last_error"):
                subscription = await repo.get(subscription_id)
                if subscription:
                    updates["error_count"] = (subscription.error_count or 0) + 1
            
            await repo.update(subscription_id, **updates)
            await session.commit()

    async def on_new_videos(subscription_id: str, videos: list):
        """Handle new videos found for a subscription"""
        if not videos:
            return

        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = SubscriptionRepository(session)
            subscription = await repo.get(subscription_id)

            if not subscription or not subscription.auto_process:
                return

            # Create a task for each new video
            for video in videos:
                try:
                    result = await create_task_from_subscription(subscription, video)
                    if result is not None:
                        logger.info(f"Created task for new video: {video.title}")
                    else:
                        logger.info(f"Skipped duplicate video: {video.title}")
                except Exception as e:
                    logger.error(f"Failed to create task for video {video.video_id}: {e}")

    subscription_scheduler.set_callbacks(
        get_due_subscriptions=get_due_subscriptions,
        update_subscription=update_subscription,
        on_new_videos=on_new_videos,
    )

    await subscription_scheduler.start()


async def init_trending_scheduler():
    """Initialize the trending scheduler with database callbacks"""

    async def get_trending_settings():
        """Get trending settings from settings store (includes API key for internal use)"""
        settings = settings_store.load()
        return settings.trending.to_dict(mask_secrets=False)

    async def update_trending_settings(updates: dict):
        """Update trending settings"""
        settings_store.update({"trending": updates})

    async def save_trending_videos(category: str, videos: list):
        """Save trending videos to database (handles individual failures gracefully)"""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = TrendingRepository(session)
            saved_count = 0
            for video_data in videos:
                try:
                    await repo.upsert_video(category=category, **video_data)
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save trending video {video_data.get('video_id', 'unknown')}: {e}")
                    # Continue with next video instead of failing all
            await session.commit()
            logger.info(f"Saved {saved_count}/{len(videos)} trending videos for {category}")

    trending_scheduler.set_callbacks(
        get_settings=get_trending_settings,
        update_settings=update_trending_settings,
        save_videos=save_trending_videos,
    )

    await trending_scheduler.start()


async def init_tiktok_scheduler():
    """Initialize the TikTok scheduler with database callbacks"""

    async def get_tiktok_settings():
        """Get TikTok settings from settings store"""
        settings = settings_store.load()
        return settings.tiktok.to_dict()

    async def update_tiktok_settings(updates: dict):
        """Update TikTok settings"""
        settings_store.update({"tiktok": updates})

    async def save_tiktok_videos(category: str, videos: list):
        """Save TikTok videos to database"""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = TrendingRepository(session)
            saved_count = 0
            for video_data in videos:
                try:
                    await repo.upsert_video(category=category, **video_data)
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save TikTok video {video_data.get('video_id', 'unknown')}: {e}")
            await session.commit()
            logger.info(f"Saved {saved_count}/{len(videos)} TikTok videos for {category}")

    tiktok_scheduler.set_callbacks(
        get_settings=get_tiktok_settings,
        update_settings=update_tiktok_settings,
        save_videos=save_tiktok_videos,
    )

    await tiktok_scheduler.start()


async def create_task_from_subscription(subscription, video: SubscriptionVideoInfo):
    """Create a processing task from a subscription video"""
    
    # Check for duplicate task (same URL already pending/processing)
    session_factory = get_session_factory()
    async with session_factory() as session:
        task_repo = TaskRepository(session)
        if await task_repo.exists_pending_by_url(video.url):
            logger.info(f"Skipping duplicate task for video: {video.url}")
            return None
    
    task_id = str(uuid.uuid4())[:8]

    # Get default options from subscription or use defaults
    process_options = subscription.process_options or {}
    
    # Interpret processing_mode to set proper flags
    processing_mode = process_options.get("processing_mode", "full")
    logger.info(f"Subscription process_options: {process_options}")
    logger.info(f"Subscription processing_mode: {processing_mode}")
    
    # Set values based on processing_mode (these OVERRIDE saved options for mode-specific flags)
    if processing_mode == "direct":
        # Direct repost - no transcription, translation, or TTS
        # Force these values regardless of saved options
        force_skip_translation = True
        force_add_subtitles = False
        force_add_tts = False
        force_dual_subtitles = False
    elif processing_mode == "subtitle":
        # Subtitle only - no TTS
        force_skip_translation = False
        force_add_subtitles = True
        force_add_tts = False
        force_dual_subtitles = process_options.get("dual_subtitles", True)
    elif processing_mode == "auto":
        # Auto mode - use saved options with sensible defaults
        force_skip_translation = process_options.get("skip_translation", False)
        force_add_subtitles = process_options.get("add_subtitles", True)
        force_add_tts = process_options.get("add_tts", False)
        force_dual_subtitles = process_options.get("dual_subtitles", True)
    else:  # "full" or default
        force_skip_translation = process_options.get("skip_translation", False)
        force_add_subtitles = process_options.get("add_subtitles", True)
        force_add_tts = process_options.get("add_tts", True)
        force_dual_subtitles = process_options.get("dual_subtitles", True)
    
    logger.info(f"Task flags: skip_translation={force_skip_translation}, add_subtitles={force_add_subtitles}, add_tts={force_add_tts}")

    options = ProcessingOptions(
        source_url=video.url,
        source_platform=subscription.platform,
        # Transcription
        source_language=process_options.get("source_language", "auto"),
        target_language=process_options.get("target_language", "zh-CN"),
        whisper_backend=process_options.get("whisper_backend", "auto"),
        whisper_model=process_options.get("whisper_model", "auto"),
        whisper_device=process_options.get("whisper_device", "auto"),
        # Translation
        skip_translation=force_skip_translation,
        translation_engine=process_options.get("translation_engine", "google"),
        # Subtitles
        add_subtitles=force_add_subtitles,
        subtitle_style=process_options.get("subtitle_style", "default"),
        dual_subtitles=force_dual_subtitles,
        use_existing_subtitles=process_options.get("use_existing_subtitles", True),
        subtitle_language=process_options.get("subtitle_language"),
        subtitle_preset=process_options.get("subtitle_preset"),
        # TTS
        add_tts=force_add_tts,
        tts_service=process_options.get("tts_service", "edge"),
        tts_voice=process_options.get("tts_voice", "zh-CN-XiaoxiaoNeural"),
        tts_rate=process_options.get("tts_rate", "+0%"),
        voice_cloning_mode=process_options.get("voice_cloning_mode", "disabled"),
        tts_ref_audio=process_options.get("tts_ref_audio"),
        tts_ref_text=process_options.get("tts_ref_text"),
        original_audio_volume=process_options.get("original_audio_volume", 0.3),
        tts_audio_volume=process_options.get("tts_audio_volume", 1.0),
        replace_original_audio=process_options.get("replace_original_audio", False),
        # Video quality
        video_quality=process_options.get("video_quality", "1080p"),
        format_id=process_options.get("format_id"),
        video_quality_label=process_options.get("video_quality_label"),
        # Upload targets
        upload_bilibili=process_options.get("upload_bilibili", False),
        upload_douyin=process_options.get("upload_douyin", False),
        upload_xiaohongshu=process_options.get("upload_xiaohongshu", False),
        bilibili_account_uid=process_options.get("bilibili_account_uid"),
        # Metadata
        custom_title=process_options.get("custom_title"),
        custom_description=process_options.get("custom_description"),
        custom_tags=process_options.get("custom_tags", []),
        metadata_preset_id=process_options.get("metadata_preset_id"),
        use_ai_preset_selection=process_options.get("use_ai_preset_selection", False),
        # Proofreading
        enable_proofreading=process_options.get("enable_proofreading", True),
        proofreading_auto_pause=process_options.get("proofreading_auto_pause", True),
        proofreading_min_confidence=process_options.get("proofreading_min_confidence", 0.6),
        proofreading_auto_optimize=process_options.get("proofreading_auto_optimize", False),
        proofreading_optimization_level=process_options.get("proofreading_optimization_level", "moderate"),
        # Directory
        directory=subscription.directory,
    )

    task = ProcessingTask(task_id=task_id, options=options)
    task.message = f"从订阅自动创建: {subscription.channel_name}"

    # Store task
    pipeline.tasks[task_id] = task

    # Persist to database with directory
    options_dict = {
        "source_url": options.source_url,
        "source_platform": options.source_platform,
        # Transcription
        "source_language": options.source_language,
        "target_language": options.target_language,
        "whisper_backend": options.whisper_backend,
        "whisper_model": options.whisper_model,
        "whisper_device": options.whisper_device,
        # Translation
        "skip_translation": options.skip_translation,
        "translation_engine": options.translation_engine,
        # Subtitles
        "add_subtitles": options.add_subtitles,
        "subtitle_style": options.subtitle_style,
        "dual_subtitles": options.dual_subtitles,
        "use_existing_subtitles": options.use_existing_subtitles,
        "subtitle_language": options.subtitle_language,
        "subtitle_preset": options.subtitle_preset,
        # TTS
        "add_tts": options.add_tts,
        "tts_service": options.tts_service,
        "tts_voice": options.tts_voice,
        "tts_rate": options.tts_rate,
        "voice_cloning_mode": options.voice_cloning_mode,
        "tts_ref_audio": options.tts_ref_audio,
        "tts_ref_text": options.tts_ref_text,
        "original_audio_volume": options.original_audio_volume,
        "tts_audio_volume": options.tts_audio_volume,
        "replace_original_audio": options.replace_original_audio,
        # Video quality
        "video_quality": options.video_quality,
        "format_id": options.format_id,
        "video_quality_label": options.video_quality_label,
        # Upload targets
        "upload_bilibili": options.upload_bilibili,
        "upload_douyin": options.upload_douyin,
        "upload_xiaohongshu": options.upload_xiaohongshu,
        "bilibili_account_uid": options.bilibili_account_uid,
        # Metadata
        "custom_title": options.custom_title,
        "custom_description": options.custom_description,
        "custom_tags": options.custom_tags,
        "metadata_preset_id": options.metadata_preset_id,
        "use_ai_preset_selection": options.use_ai_preset_selection,
        # Proofreading
        "enable_proofreading": options.enable_proofreading,
        "proofreading_auto_pause": options.proofreading_auto_pause,
        "proofreading_min_confidence": options.proofreading_min_confidence,
        "proofreading_auto_optimize": options.proofreading_auto_optimize,
        "proofreading_optimization_level": options.proofreading_optimization_level,
        # Directory
        "directory": options.directory,
    }
    await task_persistence.save_task_created(task_id, options_dict, "pending", task.message, directory=subscription.directory)

    # Queue for processing
    await task_executor.submit(task)

    logger.info(f"Auto-created task {task_id} from subscription {subscription.id}")
    return task


async def load_tasks_from_db():
    """Load tasks from database into pipeline memory for retry/resume capability"""
    try:
        # Load ALL recent tasks (including completed/failed) so they can be retried
        tasks = await task_persistence.load_all_tasks(limit=500)
        loaded_count = 0

        for task_data in tasks:
            task_id = task_data["task_id"]

            # Reconstruct ProcessingTask from database data
            options_data = task_data.get("options", {})
            if not options_data:
                logger.warning(f"Task {task_id} has no options, skipping")
                continue

            try:
                # Handle missing fields in old tasks
                options_data.setdefault("whisper_backend", "auto")
                options_data.setdefault("whisper_model", "auto")
                options_data.setdefault("whisper_device", "auto")

                options = ProcessingOptions(**options_data)
                task = ProcessingTask(task_id=task_id, options=options)

                # Restore state
                task.status = TaskStatus(task_data.get("status", "pending"))
                task.progress = task_data.get("progress", 0)
                task.message = task_data.get("message", "")
                task.video_info = task_data.get("video_info")
                task.error = task_data.get("error")

                # Restore timestamps from database
                if task_data.get("created_at"):
                    try:
                        task.created_at = datetime.fromisoformat(task_data["created_at"].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        pass  # Keep default if parse fails
                if task_data.get("updated_at"):
                    try:
                        task.updated_at = datetime.fromisoformat(task_data["updated_at"].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        pass

                # Restore file paths (verify they exist)
                files = task_data.get("files", {})
                if files.get("video"):
                    path = Path(files["video"])
                    if path.exists():
                        task.video_path = path
                if files.get("audio"):
                    path = Path(files["audio"])
                    if path.exists():
                        task.audio_path = path
                if files.get("original_subtitle"):
                    path = Path(files["original_subtitle"])
                    if path.exists():
                        task.subtitle_path = path
                if files.get("translated_subtitle"):
                    path = Path(files["translated_subtitle"])
                    if path.exists():
                        task.translated_subtitle_path = path
                if files.get("tts_audio"):
                    path = Path(files["tts_audio"])
                    if path.exists():
                        task.tts_audio_path = path
                if files.get("final_video"):
                    path = Path(files["final_video"])
                    if path.exists():
                        task.final_video_path = path
                if files.get("thumbnail"):
                    path = Path(files["thumbnail"])
                    if path.exists():
                        task.thumbnail_path = path

                # Restore step results from database
                steps_data = task_data.get("steps", {})
                for step_name, step_info in steps_data.items():
                    if step_name in task.steps:
                        task.steps[step_name].status = StepStatus(step_info.get("status", "pending"))
                        task.steps[step_name].error = step_info.get("error")
                        task.steps[step_name].output_files = step_info.get("output_files", {})
                        task.steps[step_name].metadata = step_info.get("metadata", {})
                        if step_info.get("duration_seconds"):
                            task.steps[step_name].duration_seconds = step_info["duration_seconds"]

                # Restore timing info
                task.total_processing_time = task_data.get("total_processing_time")

                # Restore task folder name
                task.task_folder = task_data.get("task_folder")

                # Restore proofreading and optimization results
                if task_data.get("proofreading_result"):
                    task.proofreading_result = task_data["proofreading_result"]
                if task_data.get("optimization_result"):
                    task.optimization_result = task_data["optimization_result"]

                pipeline.tasks[task_id] = task
                loaded_count += 1

            except Exception as e:
                logger.warning(f"Failed to restore task {task_id}: {e}")
                continue

        logger.info(f"Loaded {loaded_count} tasks from database (all statuses, for retry capability)")
    except Exception as e:
        logger.error(f"Failed to load tasks from database: {e}")


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="iDubb API",
    description="自动视频翻译和多平台发布系统",
    version="1.0.0",
    lifespan=lifespan
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS for frontend - configure allowed origins based on environment
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:3005,http://localhost:5173,http://localhost:5174,http://127.0.0.1:3000,http://127.0.0.1:3005,http://127.0.0.1:5173,http://127.0.0.1:5174").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Configured via CORS_ORIGINS env var
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Initialize pipeline
pipeline = VideoPipeline()

# Initialize video info parser
video_info_parser = VideoInfoParser()

# Background optimization jobs tracking
# Format: {job_id: {"status": "pending"|"running"|"completed"|"failed", "result": {...}, "error": str}}
optimization_jobs: Dict[str, Dict[str, Any]] = {}


# === Pydantic Models ===

class CreateTaskRequest(BaseModel):
    """Request to create a new processing task"""
    source_url: str = Field("", description="YouTube or TikTok video URL (empty for local uploads)")
    source_platform: str = Field("auto", description="Platform: auto, youtube, tiktok, local")
    local_file_path: Optional[str] = Field(None, description="Path to uploaded local video file")

    # Video quality
    video_quality: str = Field("1080p", description="Video quality: 2160p, 1080p, 720p, 480p")
    format_id: Optional[str] = Field(None, description="Specific format ID from video info")
    video_quality_label: Optional[str] = Field(None, description="Human-readable quality label")

    # Language settings
    source_language: str = Field("auto", description="Source language code or 'auto'")
    target_language: str = Field("zh-CN", description="Target language code")

    # Whisper transcription settings
    whisper_backend: str = Field("auto", description="Whisper backend: auto, faster, openai")
    whisper_model: str = Field("auto", description="Whisper model: auto, faster:tiny, faster:base, faster:small, etc.")
    whisper_device: str = Field("auto", description="Whisper device: auto, cpu, cuda, mps (mps only for openai backend)")

    # Subtitle options
    add_subtitles: bool = Field(True, description="Add subtitles to video")
    dual_subtitles: bool = Field(True, description="Show both original and translated")
    use_existing_subtitles: bool = Field(True, description="Use video's existing subtitles if available")
    subtitle_language: Optional[str] = Field(None, description="Preferred subtitle language to download")
    subtitle_preset: Optional[str] = Field(None, description="Subtitle preset ID for styling")

    # TTS options
    add_tts: bool = Field(True, description="Add TTS voiceover")
    tts_service: str = Field("edge", description="TTS engine: edge, index, cosyvoice")
    tts_voice: str = Field("zh-CN-XiaoxiaoNeural", description="TTS voice name")
    tts_ref_audio: Optional[str] = Field(None, description="Reference audio path for voice cloning")
    tts_ref_text: Optional[str] = Field(None, description="Transcript of reference audio")
    voice_cloning_mode: str = Field("disabled", description="Voice cloning mode: disabled, video_audio, custom")

    # Audio mixing options
    replace_original_audio: bool = Field(False, description="Replace original audio with TTS")
    original_audio_volume: float = Field(0.3, description="Original audio volume when mixing (0-1)")
    tts_audio_volume: float = Field(1.0, description="TTS audio volume (0-1)")

    # Translation options
    skip_translation: bool = Field(False, description="Skip translation - subtitles only mode")
    translation_engine: str = Field("google", description="Translation engine: google, deepl, claude, gpt")

    # Upload targets
    upload_bilibili: bool = Field(False, description="Upload to Bilibili")
    upload_douyin: bool = Field(False, description="Upload to Douyin")
    upload_xiaohongshu: bool = Field(False, description="Upload to Xiaohongshu")
    bilibili_account_uid: Optional[str] = Field(None, description="Specific Bilibili account UID (None = default)")

    # Custom metadata
    custom_title: Optional[str] = Field(None, description="Custom video title")
    custom_description: Optional[str] = Field(None, description="Custom description")
    custom_tags: List[str] = Field(default_factory=list, description="Custom tags")

    # Metadata preset
    metadata_preset_id: Optional[str] = Field(None, description="Metadata preset ID for title prefix and signature")
    use_ai_preset_selection: bool = Field(False, description="Use AI to automatically select best preset")

    # Use global settings as defaults
    use_global_settings: bool = Field(True, description="Use global settings as defaults")

    # Directory for organizing tasks (immutable after creation)
    directory: Optional[str] = Field(None, description="Directory name for organizing tasks")


class StepResponse(BaseModel):
    """Individual step status response"""
    step_name: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    duration_formatted: Optional[str] = None
    error: Optional[str] = None
    output_files: Dict[str, Optional[str]] = {}
    metadata: Dict[str, Any] = {}


class TaskResponse(BaseModel):
    """Task status response with step details"""
    task_id: str
    status: str
    progress: int
    message: str
    created_at: str
    updated_at: str
    current_step: Optional[str] = None
    task_folder: Optional[str] = None  # Task folder name (task_id + video title)
    steps: Dict[str, StepResponse] = {}
    video_info: Optional[Dict[str, Any]] = None
    files: Dict[str, Optional[str]] = {}
    upload_results: Dict[str, Any] = {}
    error: Optional[str] = None
    # Timing and thumbnail fields
    thumbnail_url: Optional[str] = None
    step_timings: Optional[Dict[str, Any]] = None
    total_processing_time: Optional[float] = None
    total_time_formatted: Optional[str] = None
    # Metadata approval fields
    generated_metadata: Optional[Dict[str, Any]] = None
    metadata_approved: bool = False
    metadata_approved_at: Optional[str] = None
    # Proofreading results
    proofreading_result: Optional[Dict[str, Any]] = None
    # Optimization results
    optimization_result: Optional[Dict[str, Any]] = None
    # Queue position (0 = processing, >0 = waiting in queue, -1 = not queued)
    queue_position: int = -1
    # Task settings/options (subtitle_preset, translation settings, etc.)
    settings: Optional[Dict[str, Any]] = None


class VideoInfoResponse(BaseModel):
    """Video info response"""
    title: str
    description: str
    duration: int
    thumbnail_url: str
    uploader: str
    platform: str
    tags: List[str]


class PlatformCredentials(BaseModel):
    """Platform credentials for authentication"""
    platform: str = Field(..., description="Platform name: bilibili, douyin, xiaohongshu")
    cookies: Optional[str] = Field(None, description="Full cookie string")
    sessdata: Optional[str] = Field(None, description="Bilibili SESSDATA")
    bili_jct: Optional[str] = Field(None, description="Bilibili bili_jct")
    buvid3: Optional[str] = Field(None, description="Bilibili buvid3")


class LanguageInfo(BaseModel):
    """Language information"""
    code: str
    name: str


class VoiceInfo(BaseModel):
    """TTS voice information"""
    name: str
    display_name: str
    gender: str
    locale: str


# === Helper Functions ===

def task_to_response(task: ProcessingTask) -> TaskResponse:
    """Convert ProcessingTask to TaskResponse"""
    steps = {}
    for name, step in task.steps.items():
        steps[name] = StepResponse(
            step_name=step.step_name,
            status=step.status.value,
            started_at=step.started_at.isoformat() if step.started_at else None,
            completed_at=step.completed_at.isoformat() if step.completed_at else None,
            duration_seconds=step.duration_seconds,
            duration_formatted=step._format_duration() if step.duration_seconds else None,
            error=step.error,
            output_files=step.output_files,
            metadata=step.metadata,
        )

    # Generate thumbnail URL if available
    thumbnail_url = None
    if task.thumbnail_path and task.thumbnail_path.exists():
        thumbnail_url = f"/api/tasks/{task.task_id}/files/thumbnail"

    # Get queue position from task executor
    queue_position = task_executor.get_queue_position(task.task_id)

    # Extract settings from task options
    settings = None
    if hasattr(task, 'options') and task.options:
        settings = {
            "subtitle_preset": getattr(task.options, 'subtitle_preset', None),
            "source_language": getattr(task.options, 'source_language', None),
            "target_language": getattr(task.options, 'target_language', None),
            "translation_engine": getattr(task.options, 'translation_engine', None),
            "tts_provider": getattr(task.options, 'tts_provider', None),
            "voice_cloning_mode": getattr(task.options, 'voice_cloning_mode', None),
            "add_subtitles": getattr(task.options, 'add_subtitles', None),
            "dual_subtitles": getattr(task.options, 'dual_subtitles', None),
            "add_tts": getattr(task.options, 'add_tts', None),
        }
        # Remove None values for cleaner output
        settings = {k: v for k, v in settings.items() if v is not None}

    return TaskResponse(
        task_id=task.task_id,
        status=task.status.value,
        progress=task.progress,
        message=task.message or "",
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
        current_step=task.current_step,
        task_folder=task.task_folder,
        steps=steps,
        video_info=task.video_info,
        files=task._get_files_dict(),
        upload_results=task.upload_results,
        error=task.error,
        thumbnail_url=thumbnail_url,
        step_timings=task.get_step_timings(),
        total_processing_time=task.total_processing_time,
        total_time_formatted=task.get_total_time_formatted(),
        proofreading_result=task.proofreading_result,
        optimization_result=task.optimization_result,
        queue_position=queue_position,
        settings=settings,
    )


# === API Endpoints ===

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "app": "iDubb",
        "version": "1.0.0"
    }


@app.post("/api/tasks", response_model=TaskResponse)
@limiter.limit("10/minute")
async def create_task(request: Request, task_request: CreateTaskRequest, background_tasks: BackgroundTasks):
    """Create a new video processing task (rate limited: 10/minute)"""
    task_id = str(uuid.uuid4())[:8]

    # Debug: Log received subtitle_preset value
    logger.debug(f"[DEBUG] Task creation received subtitle_preset: '{task_request.subtitle_preset}'")

    options = ProcessingOptions(
        source_url=task_request.source_url,
        source_platform=task_request.source_platform,
        local_file_path=task_request.local_file_path,
        source_language=task_request.source_language,
        target_language=task_request.target_language,
        skip_translation=task_request.skip_translation,
        whisper_backend=task_request.whisper_backend,
        whisper_model=task_request.whisper_model,
        whisper_device=task_request.whisper_device,
        translation_engine=task_request.translation_engine,
        video_quality=task_request.video_quality,
        format_id=task_request.format_id,
        video_quality_label=task_request.video_quality_label,
        add_subtitles=task_request.add_subtitles,
        dual_subtitles=task_request.dual_subtitles,
        use_existing_subtitles=task_request.use_existing_subtitles,
        subtitle_language=task_request.subtitle_language,
        subtitle_preset=task_request.subtitle_preset,
        add_tts=task_request.add_tts,
        tts_service=task_request.tts_service,
        tts_voice=task_request.tts_voice,
        tts_ref_audio=task_request.tts_ref_audio,
        tts_ref_text=task_request.tts_ref_text,
        voice_cloning_mode=task_request.voice_cloning_mode,
        original_audio_volume=task_request.original_audio_volume,
        tts_audio_volume=task_request.tts_audio_volume,
        replace_original_audio=task_request.replace_original_audio,
        upload_bilibili=task_request.upload_bilibili,
        upload_douyin=task_request.upload_douyin,
        upload_xiaohongshu=task_request.upload_xiaohongshu,
        bilibili_account_uid=task_request.bilibili_account_uid,
        custom_title=task_request.custom_title,
        custom_description=task_request.custom_description,
        custom_tags=task_request.custom_tags,
        metadata_preset_id=task_request.metadata_preset_id,
        use_ai_preset_selection=task_request.use_ai_preset_selection,
        directory=task_request.directory,
    )

    task = ProcessingTask(task_id=task_id, options=options)
    task.message = "任务已创建"

    # Store task before starting background processing
    pipeline.tasks[task_id] = task

    # Save to database for persistence
    options_dict = {
        "source_url": options.source_url,
        "source_platform": options.source_platform,
        "local_file_path": options.local_file_path,
        "source_language": options.source_language,
        "target_language": options.target_language,
        "skip_translation": options.skip_translation,
        "whisper_backend": options.whisper_backend,
        "whisper_model": options.whisper_model,
        "whisper_device": options.whisper_device,
        "translation_engine": options.translation_engine,
        "video_quality": options.video_quality,
        "format_id": options.format_id,
        "video_quality_label": options.video_quality_label,
        "add_subtitles": options.add_subtitles,
        "dual_subtitles": options.dual_subtitles,
        "use_existing_subtitles": options.use_existing_subtitles,
        "subtitle_language": options.subtitle_language,
        "subtitle_preset": options.subtitle_preset,
        "add_tts": options.add_tts,
        "tts_service": options.tts_service,
        "tts_voice": options.tts_voice,
        "tts_ref_audio": options.tts_ref_audio,
        "tts_ref_text": options.tts_ref_text,
        "voice_cloning_mode": options.voice_cloning_mode,
        "original_audio_volume": options.original_audio_volume,
        "tts_audio_volume": options.tts_audio_volume,
        "replace_original_audio": options.replace_original_audio,
        "upload_bilibili": options.upload_bilibili,
        "upload_douyin": options.upload_douyin,
        "upload_xiaohongshu": options.upload_xiaohongshu,
        "bilibili_account_uid": options.bilibili_account_uid,
        "custom_title": options.custom_title,
        "custom_description": options.custom_description,
        "custom_tags": options.custom_tags,
        "metadata_preset_id": options.metadata_preset_id,
        "use_ai_preset_selection": options.use_ai_preset_selection,
        "directory": options.directory,
    }
    await task_persistence.save_task_created(task_id, options_dict, "pending", "任务已创建", directory=task_request.directory)

    # Submit to task executor (queue-based processing)
    queue_position = await task_executor.submit(task)

    # Update task status based on queue position
    if queue_position > 0:
        task.status = TaskStatus.QUEUED
        task.message = f"排队中，前面有 {queue_position} 个任务"
        await task_persistence.save_task_status(task_id, "queued", 0, task.message, None)
    else:
        task.message = "开始处理..."

    logger.info(f"Created task {task_id} for {task_request.source_url}, queue position: {queue_position}")

    return task_to_response(task)


async def process_task_with_persistence(task: ProcessingTask):
    """Process task and save state changes to database"""
    try:
        # Process the task
        result = await pipeline.process(task)

        # Save final state
        await task_persistence.save_task_status(
            task.task_id,
            task.status.value,
            task.progress,
            task.message,
            task.error
        )

        # Save file paths
        await task_persistence.save_task_files(
            task.task_id,
            video_path=str(task.video_path) if task.video_path else None,
            audio_path=str(task.audio_path) if task.audio_path else None,
            subtitle_path=str(task.subtitle_path) if task.subtitle_path else None,
            translated_subtitle_path=str(task.translated_subtitle_path) if task.translated_subtitle_path else None,
            tts_audio_path=str(task.tts_audio_path) if task.tts_audio_path else None,
            final_video_path=str(task.final_video_path) if task.final_video_path else None,
            thumbnail_path=str(task.thumbnail_path) if task.thumbnail_path else None
        )

        # Save all step statuses
        for step_name, step in task.steps.items():
            await task_persistence.save_step_status(
                task.task_id,
                step_name,
                step.status.value,
                step.error,
                step.output_files,
                step.metadata
            )

        if task.video_info:
            await task_persistence.save_video_info(task.task_id, task.video_info)

        if task.upload_results:
            await task_persistence.save_upload_results(task.task_id, task.upload_results)

        # Save task folder name if set
        if task.task_folder:
            await task_persistence.update_task_folder(task.task_id, task.task_folder)

    except Exception as e:
        logger.error(f"Task processing error: {e}")
        await task_persistence.save_task_status(
            task.task_id,
            "failed",
            task.progress,
            f"处理失败: {e}",
            str(e)
        )
        # Also save step statuses when task fails
        for step_name, step in task.steps.items():
            await task_persistence.save_step_status(
                task.task_id,
                step_name,
                step.status.value,
                step.error,
                step.output_files,
                step.metadata
            )


@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get task status by ID with step details"""
    # First check in-memory tasks
    task = pipeline.tasks.get(task_id)

    if task:
        response = task_to_response(task)
        # Always fetch metadata approval status from database (it may have been updated separately)
        db_task = await task_persistence.load_task(task_id)
        if db_task:
            response.generated_metadata = db_task.get("generated_metadata")
            response.metadata_approved = db_task.get("metadata_approved", False)
            response.metadata_approved_at = db_task.get("metadata_approved_at")
        return response

    # If not in memory, try database
    task_data = await task_persistence.load_task(task_id)
    if task_data:
        # Extract settings from stored options
        db_options = task_data.get("options", {})
        db_settings = None
        if db_options:
            db_settings = {
                "subtitle_preset": db_options.get("subtitle_preset"),
                "source_language": db_options.get("source_language"),
                "target_language": db_options.get("target_language"),
                "translation_engine": db_options.get("translation_engine"),
                "tts_provider": db_options.get("tts_provider"),
                "voice_cloning_mode": db_options.get("voice_cloning_mode"),
                "add_subtitles": db_options.get("add_subtitles"),
                "dual_subtitles": db_options.get("dual_subtitles"),
                "add_tts": db_options.get("add_tts"),
            }
            # Remove None values for cleaner output
            db_settings = {k: v for k, v in db_settings.items() if v is not None}

        return TaskResponse(
            task_id=task_data["task_id"],
            status=task_data["status"],
            progress=task_data["progress"],
            message=task_data.get("message", ""),
            created_at=task_data["created_at"],
            updated_at=task_data["updated_at"],
            current_step=task_data.get("current_step"),
            task_folder=task_data.get("task_folder"),
            steps={name: StepResponse(**step) for name, step in task_data.get("steps", {}).items()},
            video_info=task_data.get("video_info"),
            files=task_data.get("files", {}),
            upload_results=task_data.get("upload_results", {}),
            error=task_data.get("error"),
            generated_metadata=task_data.get("generated_metadata"),
            metadata_approved=task_data.get("metadata_approved", False),
            metadata_approved_at=task_data.get("metadata_approved_at"),
            settings=db_settings,
        )

    raise HTTPException(status_code=404, detail="Task not found")


@app.get("/api/queue/status")
async def get_queue_status():
    """
    Get current queue status including active and pending tasks.

    Returns:
        - active_count: Number of tasks currently processing
        - pending_count: Number of tasks waiting in queue
        - max_concurrent: Maximum concurrent task limit
        - active_tasks: List of currently processing tasks
        - queue: List of pending tasks with their queue positions
    """
    return task_executor.get_status()


@app.get("/api/tasks", response_model=List[TaskResponse])
async def list_tasks(status: Optional[str] = None, directory: Optional[str] = None, limit: int = 50, include_history: bool = True):
    """
    List all tasks, optionally filtered by status and directory.
    Set include_history=True to include completed/failed tasks from database.
    """
    responses = []

    # Load all tasks from database for metadata info
    all_db_tasks = {}
    if include_history:
        db_tasks_list = await task_persistence.load_all_tasks(status=status, directory=directory, limit=limit * 2)
        for t in db_tasks_list:
            all_db_tasks[t["task_id"]] = t

    # Get in-memory tasks first (these are the most up-to-date)
    memory_task_ids = set()
    for task in pipeline.tasks.values():
        # Filter by status
        if status is not None and task.status.value != status:
            continue
        # Filter by directory (need to check db_task for directory info)
        db_task = all_db_tasks.get(task.task_id)
        if directory is not None:
            task_directory = db_task.get("directory") if db_task else None
            if task_directory != directory:
                continue
        response = task_to_response(task)
        # Add metadata approval info from database
        if db_task:
            response.generated_metadata = db_task.get("generated_metadata")
            response.metadata_approved = db_task.get("metadata_approved", False)
            response.metadata_approved_at = db_task.get("metadata_approved_at")
        responses.append(response)
        memory_task_ids.add(task.task_id)

    # Add tasks from database (for history)
    if include_history:
        for task_data in all_db_tasks.values():
            # Skip if already in memory
            if task_data["task_id"] in memory_task_ids:
                continue

            responses.append(TaskResponse(
                task_id=task_data["task_id"],
                status=task_data["status"],
                progress=task_data["progress"],
                message=task_data.get("message", ""),
                created_at=task_data["created_at"],
                updated_at=task_data["updated_at"],
                current_step=task_data.get("current_step"),
                task_folder=task_data.get("task_folder"),
                steps={name: StepResponse(**step) for name, step in task_data.get("steps", {}).items()},
                video_info=task_data.get("video_info"),
                files=task_data.get("files", {}),
                upload_results=task_data.get("upload_results", {}),
                error=task_data.get("error"),
                generated_metadata=task_data.get("generated_metadata"),
                metadata_approved=task_data.get("metadata_approved", False),
                metadata_approved_at=task_data.get("metadata_approved_at"),
            ))

    # Sort by updated_at descending (most recently modified first)
    def parse_updated_at(x):
        try:
            return datetime.fromisoformat(x.updated_at.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return datetime.min
    responses.sort(key=parse_updated_at, reverse=True)

    return responses[:limit]


async def _try_auto_refresh_youtube_cookies() -> bool:
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
            # Reload the pipeline's downloader
            global pipeline
            pipeline.youtube_downloader = pipeline.youtube_downloader.__class__(
                settings.DOWNLOADS_DIR
            )
            logger.info(f"Auto-refreshed YouTube cookies from {browser}: {result['cookie_count']} cookies")
            return True
        else:
            logger.warning(f"Failed to extract cookies from {browser}: {result.get('message', 'Unknown error')}")

    return False


@app.post("/api/video/info", response_model=VideoInfoResponse)
async def get_video_info(url: str):
    """Get video information without downloading. Auto-refreshes cookies on failure for YouTube."""
    cookies_refreshed = False

    async def _fetch_info():
        if pipeline.youtube_downloader.supports_url(url):
            return await pipeline.youtube_downloader.get_video_info(url), "youtube"
        elif pipeline.tiktok_downloader.supports_url(url):
            return await pipeline.tiktok_downloader.get_video_info(url), "tiktok"
        else:
            raise HTTPException(status_code=400, detail="Unsupported URL")

    try:
        info, platform = await _fetch_info()

        # If info is None and it's YouTube, try auto-refreshing cookies
        if info is None and platform == "youtube":
            logger.info("Video info fetch returned None, attempting auto-refresh cookies...")
            if await _try_auto_refresh_youtube_cookies():
                cookies_refreshed = True
                # Retry after cookie refresh
                info, _ = await _fetch_info()

        if not info:
            raise HTTPException(status_code=404, detail="Video not found")

        response = VideoInfoResponse(
            title=info.title,
            description=info.description,
            duration=info.duration,
            thumbnail_url=info.thumbnail_url,
            uploader=info.uploader,
            platform=info.platform,
            tags=info.tags or [],
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e).lower()
        # Check if error might be cookie-related for YouTube
        is_youtube = pipeline.youtube_downloader.supports_url(url)
        cookie_related_errors = ['sign in', 'login', 'cookie', 'authentication', 'bot', 'verify']

        if is_youtube and any(err in error_str for err in cookie_related_errors):
            logger.info(f"Possible cookie-related error: {e}, attempting auto-refresh...")
            if await _try_auto_refresh_youtube_cookies():
                cookies_refreshed = True
                try:
                    info, _ = await _fetch_info()
                    if info:
                        return VideoInfoResponse(
                            title=info.title,
                            description=info.description,
                            duration=info.duration,
                            thumbnail_url=info.thumbnail_url,
                            uploader=info.uploader,
                            platform=info.platform,
                            tags=info.tags or [],
                        )
                except Exception as retry_error:
                    logger.error(f"Retry after cookie refresh also failed: {retry_error}")

        logger.error(f"Get video info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/languages", response_model=List[LanguageInfo])
async def get_supported_languages():
    """Get list of supported languages for translation"""
    from translation import Translator

    languages = Translator.get_supported_languages()
    return [
        LanguageInfo(code=code, name=name)
        for code, name in languages.items()
    ]


@app.get("/api/voices", response_model=List[VoiceInfo])
async def get_available_voices(language: Optional[str] = None):
    """Get available TTS voices"""
    voices = await pipeline.tts_engine.get_available_voices(language)
    return [
        VoiceInfo(
            name=v["name"],
            display_name=v["display_name"],
            gender=v["gender"],
            locale=v["locale"],
        )
        for v in voices
    ]


def parse_cookie_string(cookie_string: str) -> dict:
    """Parse a cookie string into a dictionary"""
    cookies = {}
    if not cookie_string:
        return cookies
    for item in cookie_string.split(";"):
        if "=" in item:
            key, value = item.strip().split("=", 1)
            cookies[key.strip()] = value.strip()
    return cookies


@app.post("/api/platforms/authenticate")
@limiter.limit("5/minute")
async def authenticate_platform(request: Request, credentials: PlatformCredentials):
    """Authenticate with a platform and save credentials (rate limited: 5/minute)"""
    from platform_credentials import platform_credentials

    try:
        if credentials.platform == "bilibili":
            from uploaders import BilibiliUploader

            # Parse cookie string to extract required fields
            if credentials.cookies:
                parsed = parse_cookie_string(credentials.cookies)
                sessdata = parsed.get("SESSDATA", "")
                bili_jct = parsed.get("bili_jct", "")
                buvid3 = parsed.get("buvid3", "")
            else:
                # Fallback to individual fields (backward compatibility)
                sessdata = credentials.sessdata or ""
                bili_jct = credentials.bili_jct or ""
                buvid3 = credentials.buvid3 or ""

            if not sessdata or not bili_jct:
                return {
                    "success": False,
                    "platform": credentials.platform,
                    "error": "Cookie中缺少必需的SESSDATA或bili_jct字段"
                }

            uploader = BilibiliUploader()
            creds = {
                "SESSDATA": sessdata,
                "bili_jct": bili_jct,
                "buvid3": buvid3,
            }
            try:
                success = await uploader.authenticate(creds)
            finally:
                await uploader.close()

            # Save credentials if authentication succeeded
            if success:
                platform_credentials.set_bilibili(sessdata, bili_jct, buvid3)
                logger.info("Bilibili credentials saved")

        elif credentials.platform == "douyin":
            from uploaders import DouyinUploader
            uploader = DouyinUploader()
            try:
                success = await uploader.authenticate({"cookies": credentials.cookies})
            finally:
                await uploader.close()

            # Save credentials if authentication succeeded
            if success:
                platform_credentials.set_douyin(credentials.cookies)
                logger.info("Douyin credentials saved")

        elif credentials.platform == "xiaohongshu":
            from uploaders import XiaohongshuUploader
            uploader = XiaohongshuUploader()
            try:
                success = await uploader.authenticate({"cookies": credentials.cookies})
            finally:
                await uploader.close()

            # Save credentials if authentication succeeded
            if success:
                platform_credentials.set_xiaohongshu(credentials.cookies)
                logger.info("Xiaohongshu credentials saved")

        else:
            raise HTTPException(status_code=400, detail="Unknown platform")

        return {"success": success, "platform": credentials.platform}

    except Exception as e:
        logger.error(f"Auth error: {e}")
        return {"success": False, "platform": credentials.platform, "error": str(e)}


@app.get("/api/platforms/status")
async def get_platform_status():
    """Get authentication status for all platforms"""
    from platform_credentials import platform_credentials
    from douyin_accounts import douyin_account_manager
    from xiaohongshu_accounts import xiaohongshu_account_manager

    status = platform_credentials.get_status()

    # Check multi-account systems
    douyin_primary = await douyin_account_manager.get_primary()
    xiaohongshu_primary = await xiaohongshu_account_manager.get_primary()

    return {
        "bilibili": {
            "configured": status["bilibili"] or bool(settings.BILIBILI_SESSDATA),
            "authenticated": pipeline._bilibili.is_authenticated if pipeline._bilibili else False
        },
        "douyin": {
            "configured": bool(douyin_primary) or status["douyin"] or bool(settings.DOUYIN_COOKIES),
            "authenticated": pipeline._douyin.is_authenticated if pipeline._douyin else False
        },
        "xiaohongshu": {
            "configured": bool(xiaohongshu_primary) or status["xiaohongshu"] or bool(settings.XHS_COOKIES),
            "authenticated": pipeline._xiaohongshu.is_authenticated if pipeline._xiaohongshu else False
        }
    }


@app.get("/api/platforms/bilibili/partitions")
async def get_bilibili_partitions():
    """Get the list of Bilibili video partitions (分区)"""
    from uploaders.bilibili import BilibiliUploader
    return {
        "success": True,
        "partitions": BilibiliUploader.get_partition_list()
    }


@app.get("/api/platforms/cookie-status")
async def get_platform_cookie_status():
    """Check which browsers have cookies for each platform"""
    from utils.cookie_extractor import get_available_browsers, extract_chromium_cookies

    # Check if running in Docker
    in_docker = os.path.exists('/.dockerenv')
    if in_docker:
        return {
            "in_docker": True,
            "available_browsers": [],
            "platforms": {},
        }

    browsers = get_available_browsers()
    platforms_status = {}

    for platform in ["bilibili", "douyin", "xiaohongshu"]:
        browsers_with_cookies = []
        for browser in browsers:
            cookies, _ = extract_chromium_cookies(browser, platform)
            if cookies:
                browsers_with_cookies.append(browser)
        platforms_status[platform] = browsers_with_cookies

    return {
        "in_docker": False,
        "available_browsers": browsers,
        "platforms": platforms_status,
    }


@app.post("/api/platforms/{platform}/extract-cookies")
@limiter.limit("5/minute")
async def extract_platform_cookies(request: Request, platform: str, browser: str = "chrome"):
    """
    Extract cookies from browser for a specific platform (rate limited: 5/minute).
    Also authenticates the platform and saves credentials.
    Note: This only works when running locally, not in Docker.
    """
    from utils.cookie_extractor import extract_platform_credentials
    from platform_credentials import platform_credentials as creds_store

    if platform not in ["bilibili", "douyin", "xiaohongshu"]:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")

    # Check if running in Docker
    in_docker = os.path.exists('/.dockerenv')
    if in_docker:
        raise HTTPException(
            status_code=400,
            detail="Cannot extract cookies when running in Docker. Please input credentials manually."
        )

    result = extract_platform_credentials(browser, platform)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    credentials = result["credentials"]

    # Save credentials and authenticate
    try:
        if platform == "bilibili":
            from uploaders import BilibiliUploader

            # Parse cookie string to extract required fields
            cookie_string = credentials.get("cookies", "")
            parsed = parse_cookie_string(cookie_string)
            sessdata = parsed.get("SESSDATA", "")
            bili_jct = parsed.get("bili_jct", "")
            buvid3 = parsed.get("buvid3", "")

            # Save credentials
            creds_store.set_bilibili(sessdata, bili_jct, buvid3)

            # Authenticate
            uploader = BilibiliUploader()
            auth_success = await uploader.authenticate({
                "SESSDATA": sessdata,
                "bili_jct": bili_jct,
                "buvid3": buvid3,
            })

            if auth_success:
                pipeline._bilibili = uploader
                return {
                    "success": True,
                    "message": f"Successfully extracted and authenticated {platform} credentials",
                    "cookie_count": result.get("cookie_count", 0),
                }
            else:
                await uploader.close()  # Close on failure
                return {
                    "success": True,
                    "message": f"Credentials extracted and saved, but authentication failed. Cookies may be expired.",
                    "cookie_count": result.get("cookie_count", 0),
                    "authenticated": False,
                }

        elif platform == "douyin":
            from uploaders import DouyinUploader

            # Save credentials
            creds_store.set_douyin(credentials.get("cookies", ""))

            # Authenticate
            uploader = DouyinUploader()
            auth_success = await uploader.authenticate({"cookies": credentials.get("cookies", "")})

            if auth_success:
                pipeline._douyin = uploader
                return {
                    "success": True,
                    "message": f"Successfully extracted and authenticated {platform} credentials",
                    "cookie_count": result.get("cookie_count", 0),
                }
            else:
                await uploader.close()  # Close on failure
                return {
                    "success": True,
                    "message": f"Credentials extracted and saved, but authentication failed. Cookies may be expired.",
                    "cookie_count": result.get("cookie_count", 0),
                    "authenticated": False,
                }

        elif platform == "xiaohongshu":
            from uploaders import XiaohongshuUploader

            # Save credentials
            creds_store.set_xiaohongshu(credentials.get("cookies", ""))

            # Authenticate
            uploader = XiaohongshuUploader()
            auth_success = await uploader.authenticate({"cookies": credentials.get("cookies", "")})

            if auth_success:
                pipeline._xiaohongshu = uploader
                return {
                    "success": True,
                    "message": f"Successfully extracted and authenticated {platform} credentials",
                    "cookie_count": result.get("cookie_count", 0),
                }
            else:
                await uploader.close()  # Close on failure
                return {
                    "success": True,
                    "message": f"Credentials extracted and saved, but authentication failed. Cookies may be expired.",
                    "cookie_count": result.get("cookie_count", 0),
                    "authenticated": False,
                }

    except Exception as e:
        logger.error(f"Failed to authenticate {platform}: {e}")
        return {
            "success": True,
            "message": f"Credentials extracted but authentication error: {str(e)}",
            "cookie_count": result.get("cookie_count", 0),
            "authenticated": False,
        }

    return result


# === Bilibili Multi-Account Endpoints ===

@app.get("/api/bilibili/qrcode")
async def generate_bilibili_qrcode():
    """Generate QR code for Bilibili login"""
    from bilibili_auth import bilibili_qr_auth
    
    qrcode_key, qrcode_url = await bilibili_qr_auth.generate_qrcode()
    
    if not qrcode_key:
        raise HTTPException(status_code=500, detail="Failed to generate QR code")
    
    return {
        "qrcode_key": qrcode_key,
        "qrcode_url": qrcode_url,  # Use this to generate QR image on frontend
    }


@app.get("/api/bilibili/qrcode/poll")
async def poll_bilibili_qrcode(key: str, label: Optional[str] = None):
    """Poll QR code scan status"""
    from bilibili_auth import bilibili_qr_auth
    from bilibili_accounts import bilibili_account_manager
    
    result = await bilibili_qr_auth.poll_qrcode(key)
    
    # If login successful, save account
    if result.get("status") == "success" and result.get("account"):
        account = result["account"]
        is_new = await bilibili_account_manager.add_account(account, label=label)
        
        # Reload to get updated info (label, is_primary)
        saved_account = await bilibili_account_manager.get_account(account.uid)
        
        return {
            "status": "success",
            "message": result["message"],
            "is_new": is_new,
            "account": {
                "uid": saved_account.uid,
                "nickname": saved_account.nickname,
                "avatar": saved_account.avatar,
                "label": saved_account.label,
                "is_primary": saved_account.is_primary,
            }
        }
    
    return result


@app.get("/api/bilibili/accounts")
async def list_bilibili_accounts():
    """List all Bilibili accounts (without credentials)"""
    from bilibili_accounts import bilibili_account_manager
    
    accounts = await bilibili_account_manager.list_accounts()
    return {"accounts": accounts}


@app.delete("/api/bilibili/accounts/{uid}")
async def delete_bilibili_account(uid: str):
    """Delete a Bilibili account"""
    from bilibili_accounts import bilibili_account_manager
    
    removed = await bilibili_account_manager.remove_account(uid)
    
    if not removed:
        raise HTTPException(status_code=404, detail=f"Account {uid} not found")
    
    return {"success": True, "message": f"Account {uid} removed"}


class BilibiliCookieRequest(BaseModel):
    """Request body for manual cookie input"""
    sessdata: str
    bili_jct: str
    buvid3: str
    label: Optional[str] = None


@app.post("/api/bilibili/accounts/cookies")
async def add_bilibili_account_by_cookies(request: BilibiliCookieRequest):
    """Add Bilibili account by manually providing cookies"""
    from bilibili_accounts import bilibili_account_manager
    
    try:
        account = await bilibili_account_manager.add_from_cookies(
            sessdata=request.sessdata,
            bili_jct=request.bili_jct,
            buvid3=request.buvid3,
            label=request.label,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not account:
        raise HTTPException(
            status_code=400, 
            detail="Failed to verify cookies. Please check if they are valid."
        )
    
    return {
        "success": True,
        "account": {
            "uid": account.uid,
            "nickname": account.nickname,
            "avatar": account.avatar,
            "label": account.label,
            "is_primary": account.is_primary,
        }
    }


class BilibiliLabelRequest(BaseModel):
    """Request body for updating account label"""
    label: str


@app.put("/api/bilibili/accounts/{uid}/label")
async def update_bilibili_account_label(uid: str, request: BilibiliLabelRequest):
    """Update account label (must be unique)"""
    from bilibili_accounts import bilibili_account_manager
    
    try:
        success = await bilibili_account_manager.update_label(uid, request.label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Account {uid} not found")
    
    return {"success": True, "message": f"Label updated to '{request.label}'"}


@app.put("/api/bilibili/accounts/{uid}/primary")
async def set_bilibili_primary_account(uid: str):
    """Set account as primary"""
    from bilibili_accounts import bilibili_account_manager
    
    success = await bilibili_account_manager.set_primary(uid)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Account {uid} not found")
    
    return {"success": True, "message": "Primary account updated"}


# === Douyin Multi-Account Endpoints ===

@app.get("/api/douyin/qrcode")
async def generate_douyin_qrcode():
    """Generate QR code for Douyin login"""
    from douyin_auth import douyin_qr_auth
    
    token, qrcode_url = await douyin_qr_auth.generate_qrcode()
    
    if not token:
        raise HTTPException(status_code=500, detail="Failed to generate QR code")
    
    return {
        "token": token,
        "qrcode_url": qrcode_url,
    }


@app.get("/api/douyin/qrcode/poll")
async def poll_douyin_qrcode(token: str, label: Optional[str] = None):
    """Poll QR code scan status"""
    from douyin_auth import douyin_qr_auth
    from douyin_accounts import douyin_account_manager
    
    result = await douyin_qr_auth.poll_qrcode(token)
    
    if result.get("status") == "success" and result.get("account"):
        account = result["account"]
        is_new = await douyin_account_manager.add_account(account, label=label)
        
        saved_account = await douyin_account_manager.get_account(account.uid)
        
        return {
            "status": "success",
            "message": result["message"],
            "is_new": is_new,
            "account": {
                "uid": saved_account.uid,
                "nickname": saved_account.nickname,
                "avatar": saved_account.avatar,
                "label": saved_account.label,
                "is_primary": saved_account.is_primary,
            }
        }
    
    return result


@app.get("/api/douyin/accounts")
async def list_douyin_accounts():
    """List all Douyin accounts"""
    from douyin_accounts import douyin_account_manager
    
    accounts = await douyin_account_manager.list_accounts()
    return {"accounts": accounts}


@app.delete("/api/douyin/accounts/{uid}")
async def delete_douyin_account(uid: str):
    """Delete a Douyin account"""
    from douyin_accounts import douyin_account_manager
    
    removed = await douyin_account_manager.remove_account(uid)
    
    if not removed:
        raise HTTPException(status_code=404, detail=f"Account {uid} not found")
    
    return {"success": True, "message": f"Account {uid} removed"}


class DouyinCookieRequest(BaseModel):
    """Request body for manual cookie input"""
    cookies: str
    label: Optional[str] = None


@app.post("/api/douyin/accounts/cookies")
async def add_douyin_account_by_cookies(request: DouyinCookieRequest):
    """Add Douyin account by manually providing cookies"""
    from douyin_accounts import douyin_account_manager
    from douyin_auth import DouyinAccount
    from datetime import datetime
    import aiohttp
    
    # Parse cookies and verify
    try:
        # Create temporary account to verify
        cookies_dict = {}
        for item in request.cookies.split(";"):
            if "=" in item:
                key, value = item.strip().split("=", 1)
                cookies_dict[key] = value
        
        # Verify by getting user info
        import aiohttp
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Cookie": request.cookies,
            "Referer": "https://creator.douyin.com/",
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get("https://creator.douyin.com/web/api/media/user/info/") as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail="Cookie 验证失败")
                
                data = await resp.json()
                if data.get("status_code") != 0:
                    raise HTTPException(status_code=400, detail="Cookie 验证失败")
                
                user_data = data.get("user", {})
                
                account = DouyinAccount(
                    uid=str(user_data.get("uid", "")),
                    nickname=user_data.get("nickname", "抖音用户"),
                    avatar=user_data.get("avatar_url", ""),
                    cookies=request.cookies,
                    updated_at=datetime.now().isoformat(),
                    label=request.label or "",
                )
                
                await douyin_account_manager.add_account(account, label=request.label)
                
                return {
                    "success": True,
                    "account": {
                        "uid": account.uid,
                        "nickname": account.nickname,
                        "avatar": account.avatar,
                        "label": account.label,
                        "is_primary": account.is_primary,
                    }
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to verify cookies: {str(e)}")


class DouyinLabelRequest(BaseModel):
    """Request body for updating account label"""
    label: str


@app.put("/api/douyin/accounts/{uid}/label")
async def update_douyin_account_label(uid: str, request: DouyinLabelRequest):
    """Update account label"""
    from douyin_accounts import douyin_account_manager
    
    success = await douyin_account_manager.update_label(uid, request.label)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Account {uid} not found")
    
    return {"success": True, "message": f"Label updated to '{request.label}'"}


@app.put("/api/douyin/accounts/{uid}/primary")
async def set_douyin_primary_account(uid: str):
    """Set account as primary"""
    from douyin_accounts import douyin_account_manager
    
    success = await douyin_account_manager.set_primary(uid)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Account {uid} not found")
    
    return {"success": True, "message": "Primary account updated"}


# === Xiaohongshu Multi-Account Endpoints ===

@app.get("/api/xiaohongshu/qrcode")
async def generate_xiaohongshu_qrcode():
    """Generate QR code for Xiaohongshu login"""
    from xiaohongshu_auth import xiaohongshu_qr_auth
    
    qr_id, qrcode_url = await xiaohongshu_qr_auth.generate_qrcode()
    
    if not qr_id:
        raise HTTPException(status_code=500, detail="Failed to generate QR code")
    
    return {
        "qr_id": qr_id,
        "qrcode_url": qrcode_url,
    }


@app.get("/api/xiaohongshu/qrcode/poll")
async def poll_xiaohongshu_qrcode(qr_id: str, label: Optional[str] = None):
    """Poll QR code scan status"""
    from xiaohongshu_auth import xiaohongshu_qr_auth
    from xiaohongshu_accounts import xiaohongshu_account_manager
    
    result = await xiaohongshu_qr_auth.poll_qrcode(qr_id)
    
    if result.get("status") == "success" and result.get("account"):
        account = result["account"]
        is_new = await xiaohongshu_account_manager.add_account(account, label=label)
        
        saved_account = await xiaohongshu_account_manager.get_account(account.user_id)
        
        return {
            "status": "success",
            "message": result["message"],
            "is_new": is_new,
            "account": {
                "uid": saved_account.user_id,
                "user_id": saved_account.user_id,
                "nickname": saved_account.nickname,
                "avatar": saved_account.avatar,
                "label": saved_account.label,
                "is_primary": saved_account.is_primary,
            }
        }
    
    return result


@app.get("/api/xiaohongshu/accounts")
async def list_xiaohongshu_accounts():
    """List all Xiaohongshu accounts"""
    from xiaohongshu_accounts import xiaohongshu_account_manager
    
    accounts = await xiaohongshu_account_manager.list_accounts()
    return {"accounts": accounts}


@app.delete("/api/xiaohongshu/accounts/{user_id}")
async def delete_xiaohongshu_account(user_id: str):
    """Delete a Xiaohongshu account"""
    from xiaohongshu_accounts import xiaohongshu_account_manager
    
    removed = await xiaohongshu_account_manager.remove_account(user_id)
    
    if not removed:
        raise HTTPException(status_code=404, detail=f"Account {user_id} not found")
    
    return {"success": True, "message": f"Account {user_id} removed"}


class XiaohongshuCookieRequest(BaseModel):
    """Request body for manual cookie input"""
    cookies: str
    label: Optional[str] = None


@app.post("/api/xiaohongshu/accounts/cookies")
async def add_xiaohongshu_account_by_cookies(request: XiaohongshuCookieRequest):
    """Add Xiaohongshu account by manually providing cookies"""
    from xiaohongshu_accounts import xiaohongshu_account_manager
    from xiaohongshu_auth import XiaohongshuAccount
    from datetime import datetime
    import ssl
    import aiohttp
    
    try:
        # Verify by getting user info
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Cookie": request.cookies,
            "Origin": "https://www.xiaohongshu.com",
            "Referer": "https://www.xiaohongshu.com/",
        }
        
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            async with session.get("https://www.xiaohongshu.com/api/sns/web/v1/user/selfinfo") as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail="Cookie 验证失败")
                
                data = await resp.json()
                if not data.get("success"):
                    raise HTTPException(status_code=400, detail="Cookie 验证失败")
                
                user_data = data.get("data", {})
                
                account = XiaohongshuAccount(
                    user_id=user_data.get("user_id", ""),
                    nickname=user_data.get("nickname", "小红书用户"),
                    avatar=user_data.get("imageb", ""),
                    cookies=request.cookies,
                    updated_at=datetime.now().isoformat(),
                    label=request.label or "",
                )
                
                await xiaohongshu_account_manager.add_account(account, label=request.label)
                
                return {
                    "success": True,
                    "account": {
                        "uid": account.user_id,
                        "user_id": account.user_id,
                        "nickname": account.nickname,
                        "avatar": account.avatar,
                        "label": account.label,
                        "is_primary": account.is_primary,
                    }
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to verify cookies: {str(e)}")


class XiaohongshuLabelRequest(BaseModel):
    """Request body for updating account label"""
    label: str


@app.put("/api/xiaohongshu/accounts/{user_id}/label")
async def update_xiaohongshu_account_label(user_id: str, request: XiaohongshuLabelRequest):
    """Update account label"""
    from xiaohongshu_accounts import xiaohongshu_account_manager
    
    success = await xiaohongshu_account_manager.update_label(user_id, request.label)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Account {user_id} not found")
    
    return {"success": True, "message": f"Label updated to '{request.label}'"}


@app.put("/api/xiaohongshu/accounts/{user_id}/primary")
async def set_xiaohongshu_primary_account(user_id: str):
    """Set account as primary"""
    from xiaohongshu_accounts import xiaohongshu_account_manager
    
    success = await xiaohongshu_account_manager.set_primary(user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Account {user_id} not found")
    
    return {"success": True, "message": "Primary account updated"}


# === Step Control Endpoints ===

async def ensure_task_in_memory(task_id: str) -> Optional[ProcessingTask]:
    """
    Ensure a task is loaded into pipeline memory.
    If not in memory, load from database and reconstruct.
    """
    task = pipeline.tasks.get(task_id)
    if task:
        return task

    # Load from database
    task_data = await task_persistence.load_task(task_id)
    if not task_data:
        return None

    # Reconstruct ProcessingTask
    options_data = task_data.get("options", {})
    if not options_data:
        logger.warning(f"Task {task_id} has no options")
        return None

    try:
        # Handle missing fields
        options_data.setdefault("whisper_backend", "auto")
        options_data.setdefault("whisper_model", "auto")
        options_data.setdefault("whisper_device", "auto")

        options = ProcessingOptions(**options_data)
        task = ProcessingTask(task_id=task_id, options=options)

        # Restore state
        task.status = TaskStatus(task_data.get("status", "pending"))
        task.progress = task_data.get("progress", 0)
        task.message = task_data.get("message", "")
        task.video_info = task_data.get("video_info")
        task.error = task_data.get("error")
        task.task_folder = task_data.get("task_folder")

        # Restore timestamps from database
        if task_data.get("created_at"):
            try:
                task.created_at = datetime.fromisoformat(task_data["created_at"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass  # Keep default if parse fails
        if task_data.get("updated_at"):
            try:
                task.updated_at = datetime.fromisoformat(task_data["updated_at"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass

        # Restore file paths
        files = task_data.get("files", {})
        if files.get("video"):
            path = Path(files["video"])
            if path.exists():
                task.video_path = path
        if files.get("audio"):
            path = Path(files["audio"])
            if path.exists():
                task.audio_path = path
        if files.get("original_subtitle"):
            path = Path(files["original_subtitle"])
            if path.exists():
                task.subtitle_path = path
        if files.get("translated_subtitle"):
            path = Path(files["translated_subtitle"])
            if path.exists():
                task.translated_subtitle_path = path
        if files.get("tts_audio"):
            path = Path(files["tts_audio"])
            if path.exists():
                task.tts_audio_path = path
        if files.get("final_video"):
            path = Path(files["final_video"])
            if path.exists():
                task.final_video_path = path
        if files.get("thumbnail"):
            path = Path(files["thumbnail"])
            if path.exists():
                task.thumbnail_path = path

        # Restore step results
        steps_data = task_data.get("steps", {})
        for step_name, step_info in steps_data.items():
            if step_name in task.steps:
                task.steps[step_name].status = StepStatus(step_info.get("status", "pending"))
                task.steps[step_name].error = step_info.get("error")
                task.steps[step_name].output_files = step_info.get("output_files", {})
                task.steps[step_name].metadata = step_info.get("metadata", {})
                if step_info.get("duration_seconds"):
                    task.steps[step_name].duration_seconds = step_info["duration_seconds"]

        # Restore proofreading result
        if task_data.get("proofreading_result"):
            task.proofreading_result = task_data["proofreading_result"]

        # Add to pipeline memory
        pipeline.tasks[task_id] = task
        logger.info(f"Loaded task {task_id} from database into memory for retry")
        return task

    except Exception as e:
        logger.error(f"Failed to reconstruct task {task_id}: {e}")
        return None


async def retry_step_with_persistence(task_id: str, step_name: str):
    """Retry step and save state changes to database"""
    task = await ensure_task_in_memory(task_id)
    if not task:
        logger.error(f"Task {task_id} not found for retry")
        return

    try:
        await pipeline.retry_step(task_id, step_name)

        # Save final state
        await task_persistence.save_task_status(
            task.task_id,
            task.status.value,
            task.progress,
            task.message,
            task.error
        )

        # Save file paths
        await task_persistence.save_task_files(
            task.task_id,
            video_path=str(task.video_path) if task.video_path else None,
            audio_path=str(task.audio_path) if task.audio_path else None,
            subtitle_path=str(task.subtitle_path) if task.subtitle_path else None,
            translated_subtitle_path=str(task.translated_subtitle_path) if task.translated_subtitle_path else None,
            tts_audio_path=str(task.tts_audio_path) if task.tts_audio_path else None,
            final_video_path=str(task.final_video_path) if task.final_video_path else None,
            thumbnail_path=str(task.thumbnail_path) if task.thumbnail_path else None
        )

        # Save all step statuses
        for sname, step in task.steps.items():
            await task_persistence.save_step_status(
                task.task_id,
                sname,
                step.status.value,
                step.error,
                step.output_files,
                step.metadata
            )

        if task.video_info:
            await task_persistence.save_video_info(task.task_id, task.video_info)

    except Exception as e:
        logger.error(f"Retry step error: {e}")
        await task_persistence.save_task_status(
            task.task_id,
            task.status.value,
            task.progress,
            task.message,
            str(e)
        )
        # Save step statuses
        for sname, step in task.steps.items():
            await task_persistence.save_step_status(
                task.task_id,
                sname,
                step.status.value,
                step.error,
                step.output_files,
                step.metadata
            )


@app.post("/api/tasks/{task_id}/retry/{step_name}", response_model=TaskResponse)
async def retry_step(task_id: str, step_name: str, background_tasks: BackgroundTasks):
    """
    Retry a specific step of a task.

    Valid step names: download, transcribe, translate, proofread, tts, process_video, upload
    """
    # Ensure task is in memory (load from DB if needed)
    task = await ensure_task_in_memory(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    valid_steps = ["download", "transcribe", "translate", "proofread", "optimize", "tts", "process_video", "upload"]
    if step_name not in valid_steps:
        raise HTTPException(status_code=400, detail=f"Invalid step name. Valid steps: {valid_steps}")

    # Run retry in background with persistence
    background_tasks.add_task(retry_step_with_persistence, task_id, step_name)

    logger.info(f"Retrying step {step_name} for task {task_id}")
    return task_to_response(task)


@app.post("/api/tasks/{task_id}/continue/{step_name}", response_model=TaskResponse)
async def continue_from_step(task_id: str, step_name: str, background_tasks: BackgroundTasks):
    """
    Continue processing from a specific step (runs this step and all subsequent steps).

    Valid step names: download, transcribe, translate, proofread, tts, process_video, upload
    """
    # Ensure task is in memory (load from DB if needed)
    task = await ensure_task_in_memory(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    valid_steps = ["download", "transcribe", "translate", "proofread", "optimize", "tts", "process_video", "upload"]
    if step_name not in valid_steps:
        raise HTTPException(status_code=400, detail=f"Invalid step name. Valid steps: {valid_steps}")

    # Run continuation in background with persistence (reuse retry_step_with_persistence)
    background_tasks.add_task(retry_step_with_persistence, task_id, step_name)

    logger.info(f"Continuing from step {step_name} for task {task_id}")
    return task_to_response(task)


@app.post("/api/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    """Pause a running task"""
    task = pipeline.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task.status = TaskStatus.PAUSED
    task.message = "任务已暂停"
    logger.info(f"Paused task {task_id}")
    return {"success": True, "message": "Task paused"}


@app.get("/api/tasks/{task_id}/subtitles")
async def get_subtitles(task_id: str):
    """
    Get subtitle segments for editing.
    Returns both original and translated subtitle text with timestamps.
    """
    from subtitles.parser import SubtitleParser

    try:
        task = await ensure_task_in_memory(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        parser = SubtitleParser()
        segments = []

        # Get file paths from task properties
        original_file = task.subtitle_path
        translated_file = task.translated_subtitle_path

        logger.info(f"Getting subtitles for task {task_id}: translated={translated_file}, original={original_file}")

        if not translated_file or not translated_file.exists():
            raise HTTPException(status_code=404, detail=f"Translated subtitle file not found")

        # Parse translated subtitles
        translated_segments = parser.parse_file(translated_file)

        # Parse original subtitles if available
        original_segments = []
        if original_file and original_file.exists():
            original_segments = parser.parse_file(original_file)

        # Combine into response
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

        return {
            "success": True,
            "segments": segments,
            "original_file": str(original_file) if original_file else None,
            "translated_file": str(translated_file)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting subtitles for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading subtitles: {str(e)}")


@app.put("/api/tasks/{task_id}/subtitles")
async def update_subtitles(task_id: str, data: Dict[str, Any]):
    """
    Update subtitle segments after editing.
    Saves the modified subtitles to the translated subtitle file.
    """
    task = await ensure_task_in_memory(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    segments = data.get("segments", [])
    if not segments:
        raise HTTPException(status_code=400, detail="No segments provided")

    translated_file = task.translated_subtitle_path
    if not translated_file:
        raise HTTPException(status_code=404, detail="Translated subtitle file not found")

    # Generate SRT content
    def format_srt_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    srt_lines = []
    for seg in segments:
        idx = seg.get("index", 0) + 1
        start = format_srt_time(seg.get("start_time", 0))
        end = format_srt_time(seg.get("end_time", 0))
        text = seg.get("translated_text", "")

        srt_lines.append(str(idx))
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")

    srt_content = "\n".join(srt_lines)

    # Write to file
    try:
        translated_file.write_text(srt_content, encoding="utf-8")
        logger.info(f"Updated subtitles for task {task_id}: {len(segments)} segments")

        # Clear proofreading result since subtitles changed
        if hasattr(task, 'proofreading_result'):
            task.proofreading_result = None

        return {"success": True, "message": f"Updated {len(segments)} subtitle segments"}
    except Exception as e:
        logger.error(f"Failed to save subtitles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save subtitles: {e}")


async def _run_optimization_job(job_id: str, task_id: str, level: str):
    """Background task to run subtitle optimization"""
    from proofreading.optimizer import optimize_subtitles
    from subtitles.parser import SubtitleParser
    import asyncio

    try:
        optimization_jobs[job_id]["status"] = "running"

        task = await ensure_task_in_memory(task_id)
        if not task:
            optimization_jobs[job_id]["status"] = "failed"
            optimization_jobs[job_id]["error"] = "Task not found"
            return

        # Get subtitle files
        translated_file = task.translated_subtitle_path
        original_file = task.subtitle_path

        if not translated_file or not translated_file.exists():
            optimization_jobs[job_id]["status"] = "failed"
            optimization_jobs[job_id]["error"] = "Translated subtitle file not found"
            return

        # Parse subtitles
        parser = SubtitleParser()
        translated_segments = parser.parse_file(translated_file)
        original_segments = []
        if original_file and original_file.exists():
            original_segments = parser.parse_file(original_file)

        # Build segments list for optimizer
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

        # Get proofreading result if available
        proofreading_result = getattr(task, 'proofreading_result', None)

        # Get language settings
        source_lang = task.options.source_language if task.options else "en"
        target_lang = task.options.target_language if task.options else "zh-CN"

        logger.info(f"Running background optimization for task {task_id}, level={level}")

        # Run optimization
        result = await optimize_subtitles(
            segments=segments,
            proofreading_result=proofreading_result,
            source_lang=source_lang,
            target_lang=target_lang,
            level=level
        )

        if not result.success:
            optimization_jobs[job_id]["status"] = "failed"
            optimization_jobs[job_id]["error"] = result.error or "Optimization failed"
            return

        # Save optimized subtitles if changes were made
        if result.optimized_count > 0:
            # Generate SRT content
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
            translated_file.write_text(srt_content, encoding="utf-8")

            # Clear proofreading result since subtitles changed
            if hasattr(task, 'proofreading_result'):
                task.proofreading_result = None
                await task_persistence.save_proofreading_result(task_id, None)

            logger.info(f"Optimization complete for task {task_id}: {result.optimized_count} segments optimized")

        optimization_jobs[job_id]["status"] = "completed"
        optimization_jobs[job_id]["result"] = {
            "success": True,
            "optimized_count": result.optimized_count,
            "total_segments": result.total_segments,
            "changes": result.changes
        }

    except Exception as e:
        logger.error(f"Background optimization failed for task {task_id}: {e}")
        optimization_jobs[job_id]["status"] = "failed"
        optimization_jobs[job_id]["error"] = str(e)


@app.post("/api/tasks/{task_id}/optimize-subtitles")
async def optimize_subtitles_endpoint(task_id: str, background_tasks: BackgroundTasks, data: Optional[Dict[str, Any]] = None):
    """
    Start subtitle optimization as a background task.
    Returns immediately with a job_id that can be used to check status.

    Request body (optional):
    - level: Optimization level (minimal, moderate, aggressive). Default: moderate
    """
    try:
        task = await ensure_task_in_memory(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Get optimization level from request or settings
        level = "moderate"
        if data and data.get("level"):
            level = data["level"]
        else:
            from settings_store import settings_store
            settings = settings_store.load()
            level = settings.proofreading.optimization_level

        # Check if translated subtitle file exists
        translated_file = task.translated_subtitle_path
        if not translated_file or not translated_file.exists():
            raise HTTPException(status_code=404, detail="Translated subtitle file not found")

        # Create job ID and initialize job status
        job_id = f"opt_{task_id}_{uuid.uuid4().hex[:8]}"
        optimization_jobs[job_id] = {
            "status": "pending",
            "task_id": task_id,
            "level": level,
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat()
        }

        # Start background task
        import asyncio
        asyncio.create_task(_run_optimization_job(job_id, task_id, level))

        logger.info(f"Started background optimization job {job_id} for task {task_id}")

        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Optimization started in background"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start optimization for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")


@app.get("/api/optimization-jobs/{job_id}")
async def get_optimization_job_status(job_id: str):
    """
    Get the status and result of an optimization job.
    """
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")

    job = optimization_jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "task_id": job.get("task_id"),
    }

    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]

    return response


@app.post("/api/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    """
    Stop a running task immediately.
    The task will be paused and can be resumed with different settings.
    """
    task = pipeline.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        await pipeline.stop_task(task_id)

        # Persist the stopped state to database so it survives restart
        await task_persistence.save_task_status(
            task_id=task_id,
            status=task.status.value,
            progress=task.progress,
            message=task.message,
            error=task.error
        )

        # Persist the failed step status
        for step_name, step in task.steps.items():
            if step.status.value == "failed" and step.error == "用户手动停止":
                await task_persistence.save_step_status(
                    task_id=task_id,
                    step_name=step_name,
                    status=step.status.value,
                    error=step.error
                )

        logger.info(f"Stopped task {task_id} and persisted state")
        return {"success": True, "message": "Task stopped", "task": task_to_response(task)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task (soft delete).
    The task will be marked as deleted and hidden from the task list.
    All associated files will be deleted from disk.
    """
    from utils.storage import delete_task_directory

    task = pipeline.tasks.get(task_id)

    # Get task_folder and task_data from memory or database
    task_folder = None
    task_data = None

    if task:
        task_folder = task.task_folder
        # Build task_data dict with file paths
        # Note: ai_thumbnail_path is only in database model, not in memory ProcessingTask
        task_data = {
            'video_path': str(task.video_path) if task.video_path else None,
            'audio_path': str(task.audio_path) if task.audio_path else None,
            'subtitle_path': str(task.subtitle_path) if task.subtitle_path else None,
            'translated_subtitle_path': str(task.translated_subtitle_path) if task.translated_subtitle_path else None,
            'tts_audio_path': str(task.tts_audio_path) if task.tts_audio_path else None,
            'final_video_path': str(task.final_video_path) if task.final_video_path else None,
            'thumbnail_path': str(task.thumbnail_path) if task.thumbnail_path else None,
        }

    # Always try to get ai_thumbnail_path from database (not in memory ProcessingTask)
    # Also fill in missing data if not in memory
    try:
        db_task = await task_persistence.get_task(task_id)
        if db_task:
            task_folder = task_folder or db_task.get('task_folder')
            if task_data:
                # Add ai_thumbnail_path from database
                task_data['ai_thumbnail_path'] = db_task.get('ai_thumbnail_path')
            else:
                task_data = {
                    'video_path': db_task.get('video_path'),
                    'audio_path': db_task.get('audio_path'),
                    'subtitle_path': db_task.get('subtitle_path'),
                    'translated_subtitle_path': db_task.get('translated_subtitle_path'),
                    'tts_audio_path': db_task.get('tts_audio_path'),
                    'final_video_path': db_task.get('final_video_path'),
                    'thumbnail_path': db_task.get('thumbnail_path'),
                    'ai_thumbnail_path': db_task.get('ai_thumbnail_path'),
                }
    except Exception as e:
        logger.warning(f"Failed to get task data from database: {e}")

    try:
        # Delete task files from disk (pass task_data for accurate path detection)
        files_deleted = await delete_task_directory(task_id, task_folder, task_data)
        logger.info(f"Deleted task files for {task_id}: {files_deleted}")

        # Soft delete in database (mark as deleted)
        db_deleted = await task_persistence.soft_delete_task(task_id)

        # Remove from pipeline memory
        if task_id in pipeline.tasks:
            del pipeline.tasks[task_id]

        logger.info(f"Deleted task {task_id}: db={db_deleted}, files={files_deleted}")
        return {
            "success": True,
            "message": "Task deleted",
            "task_id": task_id,
            "files_deleted": files_deleted,
            "db_deleted": db_deleted
        }
    except Exception as e:
        logger.error(f"Failed to delete task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Video file upload settings
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10GB max file size


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a local video file for processing.
    Returns the stored file path that can be used with create_task.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {file_ext}. 支持的格式: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    # Create uploads directory if it doesn't exist
    uploads_dir = settings.DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{unique_id}_{Path(file.filename).stem}{file_ext}"
    file_path = uploads_dir / safe_filename

    try:
        # Stream file to disk to handle large files
        total_size = 0
        with open(file_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    # Clean up partial file
                    out_file.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE // (1024*1024*1024)}GB"
                    )
                out_file.write(chunk)

        logger.info(f"Uploaded video file: {file_path} ({total_size / (1024*1024):.1f}MB)")

        return {
            "success": True,
            "file_path": str(file_path),
            "original_filename": file.filename,
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


class LocalVideoInfoRequest(BaseModel):
    """Request to get local video file info"""
    file_path: str = Field(..., description="Path to the uploaded video file")


@app.post("/api/video/local-info")
async def get_local_video_info(request: LocalVideoInfoRequest):
    """
    Get video information (resolution, duration, etc.) from a local file.
    Used after uploading to display video details and limit quality options.
    """
    import subprocess
    import json as json_module

    file_path = Path(request.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    # Verify file is in uploads directory for security
    uploads_dir = settings.DATA_DIR / "uploads"
    try:
        file_path.resolve().relative_to(uploads_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="无效的文件路径")

    try:
        # Use ffprobe to get video info
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(file_path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise Exception(f"ffprobe failed: {result.stderr}")

        probe_data = json_module.loads(result.stdout)

        # Extract video stream info
        video_stream = None
        audio_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video" and not video_stream:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and not audio_stream:
                audio_stream = stream

        if not video_stream:
            raise HTTPException(status_code=400, detail="未检测到视频流")

        # Get video properties
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        duration = float(probe_data.get("format", {}).get("duration", 0))
        codec = video_stream.get("codec_name", "unknown")
        fps_str = video_stream.get("r_frame_rate", "0/1")
        try:
            fps_parts = fps_str.split("/")
            fps = round(float(fps_parts[0]) / float(fps_parts[1]), 2) if len(fps_parts) == 2 and float(fps_parts[1]) > 0 else 0
        except Exception:
            fps = 0

        # Determine resolution label
        if height >= 2160:
            resolution_label = "4K (2160p)"
            max_quality = "2160p"
        elif height >= 1440:
            resolution_label = "2K (1440p)"
            max_quality = "1440p"
        elif height >= 1080:
            resolution_label = "1080p"
            max_quality = "1080p"
        elif height >= 720:
            resolution_label = "720p"
            max_quality = "720p"
        elif height >= 480:
            resolution_label = "480p"
            max_quality = "480p"
        else:
            resolution_label = f"{height}p"
            max_quality = "480p"

        # Build available quality options (can't upscale)
        all_qualities = ["2160p", "1440p", "1080p", "720p", "480p"]
        quality_heights = {"2160p": 2160, "1440p": 1440, "1080p": 1080, "720p": 720, "480p": 480}
        available_qualities = [q for q in all_qualities if quality_heights[q] <= height]

        # If video is smaller than 480p, still allow 480p as minimum
        if not available_qualities:
            available_qualities = ["480p"]

        # Get file size
        file_size = file_path.stat().st_size

        return {
            "success": True,
            "title": file_path.stem,
            "duration": duration,
            "duration_formatted": f"{int(duration // 60)}分{int(duration % 60)}秒",
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "resolution_label": resolution_label,
            "max_quality": max_quality,
            "available_qualities": available_qualities,
            "codec": codec,
            "fps": fps,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "has_audio": audio_stream is not None,
            "platform": "local"
        }

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="视频分析超时")
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        raise HTTPException(status_code=500, detail=f"获取视频信息失败: {str(e)}")


class UpdateOptionsRequest(BaseModel):
    """Request to update task options"""
    translation_engine: Optional[str] = None
    whisper_backend: Optional[str] = None
    whisper_model: Optional[str] = None
    whisper_device: Optional[str] = None
    tts_service: Optional[str] = None
    tts_voice: Optional[str] = None
    target_language: Optional[str] = None
    add_subtitles: Optional[bool] = None
    dual_subtitles: Optional[bool] = None
    add_tts: Optional[bool] = None
    replace_original_audio: Optional[bool] = None
    subtitle_preset: Optional[str] = None  # Subtitle style preset ID
    # Upload platform options
    upload_bilibili: Optional[bool] = None
    upload_douyin: Optional[bool] = None
    upload_xiaohongshu: Optional[bool] = None
    bilibili_account_uid: Optional[str] = None  # Specific Bilibili account


@app.put("/api/tasks/{task_id}/options")
async def update_task_options(task_id: str, request: UpdateOptionsRequest):
    """
    Update task options (when completed, paused, or failed).
    Allows changing translation engine, TTS settings, subtitle preset, etc. before retrying.
    """
    # Ensure task is in memory (load from DB if needed)
    task = await ensure_task_in_memory(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    allowed_statuses = [
        TaskStatus.COMPLETED, TaskStatus.PAUSED, TaskStatus.FAILED,
        TaskStatus.PENDING_REVIEW, TaskStatus.PENDING_UPLOAD, TaskStatus.UPLOADED
    ]
    if task.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail="Can only update options when task is not actively processing"
        )

    # Build updates dict from non-None values
    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    if not updates:
        raise HTTPException(status_code=400, detail="No options to update")

    try:
        pipeline.update_task_options(task_id, updates)
        logger.info(f"Updated options for task {task_id}: {updates}")

        # Persist updated options to database
        # Get the full options dict from the task and save it
        opts = task.options
        options_dict = {
            "source_url": opts.source_url,
            "source_platform": opts.source_platform,
            "local_file_path": opts.local_file_path,  # Important for local uploads
            "source_language": opts.source_language,
            "target_language": opts.target_language,
            "video_quality": opts.video_quality,
            "format_id": opts.format_id,
            "video_quality_label": opts.video_quality_label,
            "whisper_backend": opts.whisper_backend,
            "whisper_model": opts.whisper_model,
            "whisper_device": opts.whisper_device,
            "translation_engine": opts.translation_engine,
            "add_subtitles": opts.add_subtitles,
            "dual_subtitles": opts.dual_subtitles,
            "use_existing_subtitles": opts.use_existing_subtitles,
            "subtitle_language": opts.subtitle_language,
            "subtitle_preset": opts.subtitle_preset,
            "add_tts": opts.add_tts,
            "tts_service": opts.tts_service,
            "tts_voice": opts.tts_voice,
            "tts_ref_audio": opts.tts_ref_audio,
            "tts_ref_text": opts.tts_ref_text,
            "replace_original_audio": opts.replace_original_audio,
            "upload_bilibili": opts.upload_bilibili,
            "upload_douyin": opts.upload_douyin,
            "upload_xiaohongshu": opts.upload_xiaohongshu,
            "bilibili_account_uid": opts.bilibili_account_uid,
            "custom_title": opts.custom_title,
            "custom_description": opts.custom_description,
            "custom_tags": opts.custom_tags,
        }
        # Update task's updated_at timestamp
        task.updated_at = datetime.now()

        # Persist to database (updated_at is handled automatically by repository.update)
        await task_persistence._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {"options": options_dict},
        })

        return {"success": True, "message": "Options updated", "task": task_to_response(task)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/tasks/{task_id}/options")
async def get_task_options(task_id: str):
    """Get current task options/configuration"""
    # Ensure task is in memory (load from DB if needed)
    task = await ensure_task_in_memory(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Return relevant options as dict
    opts = task.options
    return {
        "task_id": task_id,
        "status": task.status.value,
        "options": {
            "source_url": opts.source_url,
            "source_platform": opts.source_platform,
            "source_language": opts.source_language,
            "target_language": opts.target_language,
            "video_quality": opts.video_quality,
            "format_id": opts.format_id,
            "video_quality_label": opts.video_quality_label,
            "whisper_backend": opts.whisper_backend,
            "whisper_model": opts.whisper_model,
            "whisper_device": opts.whisper_device,
            "translation_engine": opts.translation_engine,
            "add_subtitles": opts.add_subtitles,
            "dual_subtitles": opts.dual_subtitles,
            "use_existing_subtitles": opts.use_existing_subtitles,
            "subtitle_language": opts.subtitle_language,
            "subtitle_preset": opts.subtitle_preset,
            "add_tts": opts.add_tts,
            "tts_service": opts.tts_service,
            "tts_voice": opts.tts_voice,
            "tts_ref_audio": opts.tts_ref_audio,
            "voice_cloning_mode": opts.voice_cloning_mode,
            "original_audio_volume": opts.original_audio_volume,
            "tts_audio_volume": opts.tts_audio_volume,
            "replace_original_audio": opts.replace_original_audio,
            "upload_bilibili": opts.upload_bilibili,
            "upload_douyin": opts.upload_douyin,
            "upload_xiaohongshu": opts.upload_xiaohongshu,
            "bilibili_account_uid": opts.bilibili_account_uid,
        }
    }


# === File Download Endpoints ===

async def get_task_file_paths(task_id: str) -> Dict[str, Optional[Path]]:
    """
    Get file paths for a task from either in-memory or database.
    Returns a dict mapping file type to Path object (or None if not available).
    """
    # First try in-memory task
    task = pipeline.tasks.get(task_id)
    if task:
        # Load AI thumbnail from database since it's not in memory
        ai_thumbnail_info = await task_persistence.load_ai_thumbnail_info(task_id)
        ai_thumbnail_path = None
        if ai_thumbnail_info and ai_thumbnail_info.get("ai_thumbnail_path"):
            ai_thumbnail_path = Path(ai_thumbnail_info["ai_thumbnail_path"])

        return {
            "video": task.video_path,
            "audio": task.audio_path,
            "original_subtitle": task.subtitle_path,
            "translated_subtitle": task.translated_subtitle_path,
            "tts_audio": task.tts_audio_path,
            "final_video": task.final_video_path,
            "thumbnail": task.thumbnail_path,
            "ai_thumbnail": ai_thumbnail_path,
        }

    # Fallback to database
    db_task = await task_persistence.load_task(task_id)
    if not db_task:
        return None

    # Convert string paths from database to Path objects
    files = db_task.get("files", {})
    return {
        "video": Path(files.get("video")) if files.get("video") else None,
        "audio": Path(files.get("audio")) if files.get("audio") else None,
        "original_subtitle": Path(files.get("original_subtitle")) if files.get("original_subtitle") else None,
        "translated_subtitle": Path(files.get("translated_subtitle")) if files.get("translated_subtitle") else None,
        "tts_audio": Path(files.get("tts_audio")) if files.get("tts_audio") else None,
        "final_video": Path(files.get("final_video")) if files.get("final_video") else None,
        "thumbnail": Path(files.get("thumbnail")) if files.get("thumbnail") else None,
        "ai_thumbnail": Path(files.get("ai_thumbnail")) if files.get("ai_thumbnail") else None,
    }


@app.get("/api/tasks/{task_id}/files/{file_type}")
async def download_file(task_id: str, file_type: str):
    """
    Download a file generated by a task.

    Valid file types:
    - video: Downloaded source video
    - audio: Extracted audio
    - original_subtitle: Original language subtitles (.srt)
    - translated_subtitle: Translated subtitles (.srt)
    - tts_audio: Generated TTS audio (.mp3)
    - final_video: Final processed video with subtitles/audio
    """
    file_map = await get_task_file_paths(task_id)
    if file_map is None:
        raise HTTPException(status_code=404, detail="Task not found")

    valid_types = ["video", "audio", "original_subtitle", "translated_subtitle", "tts_audio", "final_video", "thumbnail", "ai_thumbnail"]
    if file_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Valid types: {valid_types}"
        )

    file_path = file_map.get(file_type)

    if not file_path:
        raise HTTPException(status_code=404, detail=f"File not available yet: {file_type}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found on disk: {file_type}")

    # Determine content type
    content_type, _ = mimetypes.guess_type(str(file_path))
    if not content_type:
        content_type = "application/octet-stream"

    # Create filename for download
    filename = f"{task_id}_{file_type}{file_path.suffix}"

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=content_type,
    )


@app.get("/api/tasks/{task_id}/files")
async def list_task_files(task_id: str):
    """
    List all available files for a task with their status.
    """
    file_map = await get_task_file_paths(task_id)
    if file_map is None:
        raise HTTPException(status_code=404, detail="Task not found")

    files_info = {}
    for file_type, path in file_map.items():
        if path and path.exists():
            files_info[file_type] = {
                "available": True,
                "path": str(path),
                "size": path.stat().st_size,
                "download_url": f"/api/tasks/{task_id}/files/{file_type}",
            }
        else:
            files_info[file_type] = {
                "available": False,
                "path": str(path) if path else None,
                "size": 0,
                "download_url": None,
            }

    return files_info


# === Video Info Endpoints ===

@app.post("/api/video/detailed-info")
async def get_detailed_video_info(url: str):
    """
    Get detailed video information including all available formats,
    audio tracks, and subtitles. Auto-refreshes cookies on failure for YouTube.

    This should be called before creating a task to let the user
    choose video quality, subtitle source, etc.
    """
    global video_info_parser  # Declare at function level for potential reassignment

    cookies_refreshed = False
    is_youtube = pipeline.youtube_downloader.supports_url(url)

    async def _fetch_detailed():
        return await video_info_parser.get_detailed_info(url)

    try:
        logger.info(f"Fetching detailed video info for: {url}")
        info = await _fetch_detailed()

        # If info is None and it's YouTube, try auto-refreshing cookies
        if info is None and is_youtube:
            logger.info("Detailed info fetch returned None, attempting auto-refresh cookies...")
            if await _try_auto_refresh_youtube_cookies():
                cookies_refreshed = True
                # Reinitialize video_info_parser to use new cookies
                video_info_parser = VideoInfoParser()
                # Retry
                info = await _fetch_detailed()

        if not info:
            logger.warning(f"Could not get detailed info for: {url}")
            raise HTTPException(
                status_code=404,
                detail="无法获取视频详细信息，可能是视频不存在或需要登录才能访问"
            )

        logger.info(f"Got detailed info: {len(info.formats)} formats, {len(info.subtitles)} subtitles")
        result = info.to_dict()
        result["cookies_refreshed"] = cookies_refreshed
        return result

    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e).lower()
        cookie_related_errors = ['sign in', 'login', 'cookie', 'authentication', 'bot', 'verify']

        if is_youtube and any(err in error_str for err in cookie_related_errors):
            logger.info(f"Possible cookie-related error: {e}, attempting auto-refresh...")
            if await _try_auto_refresh_youtube_cookies():
                cookies_refreshed = True
                # Reinitialize video_info_parser to use new cookies
                video_info_parser = VideoInfoParser()
                try:
                    info = await _fetch_detailed()
                    if info:
                        result = info.to_dict()
                        result["cookies_refreshed"] = True
                        return result
                except Exception as retry_error:
                    logger.error(f"Retry after cookie refresh also failed: {retry_error}")

        logger.error(f"Get detailed video info error for {url}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取视频信息失败: {str(e)}")


# === Settings Endpoints ===

@app.get("/api/settings")
async def get_global_settings():
    """Get current global settings"""
    try:
        settings = settings_store.load()
        return settings.to_dict()
    except Exception as e:
        logger.error(f"Get settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/settings")
async def update_global_settings(updates: Dict[str, Any]):
    """
    Update global settings.

    Example request body:
    {
        "video": {"default_quality": "1080p"},
        "translation": {"engine": "deepl", "api_key": "your-api-key"},
        "tts": {"voice": "zh-CN-YunxiNeural"},
        "audio": {"replace_original": false, "original_volume": 0.3}
    }
    """
    try:
        settings = settings_store.update(updates)
        return settings.to_dict()
    except Exception as e:
        logger.error(f"Update settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/settings/reset")
async def reset_global_settings():
    """Reset settings to defaults"""
    try:
        settings = settings_store.reset()
        return settings.to_dict()
    except Exception as e:
        logger.error(f"Reset settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/settings/reload")
async def reload_global_settings():
    """Force reload settings from disk (useful after manual edits to settings.json)"""
    try:
        settings = settings_store.reload()
        logger.info("Settings reloaded from disk")
        return settings.to_dict()
    except Exception as e:
        logger.error(f"Reload settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === API Keys Management Endpoints ===

@app.get("/api/settings/api-keys")
async def get_api_keys():
    """
    Get all configured API keys (masked).
    Returns which services have keys configured without exposing actual values.
    """
    try:
        from database.connection import get_db
        from database.repository import ApiKeyRepository

        async with get_db() as session:
            repo = ApiKeyRepository(session)
            keys = await repo.get_all(decrypt=False)  # Masked values

            # Also return status info
            models = await repo.get_all_models()
            status = {}
            for model in models:
                status[model.service] = {
                    "has_key": bool(model.encrypted_key),
                    "is_valid": model.is_valid,
                    "last_verified": model.last_verified.isoformat() if model.last_verified else None,
                }

            return {
                "keys": keys,
                "status": status
            }
    except Exception as e:
        logger.error(f"Get API keys error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ApiKeyUpdate(BaseModel):
    """Request to update a single API key"""
    service: str = Field(..., description="Service name (openai, anthropic, deepseek, deepl, elevenlabs, youtube)")
    key: str = Field(..., description="API key value (empty string to delete)")


@app.put("/api/settings/api-keys")
async def update_api_key(update: ApiKeyUpdate):
    """
    Update a single API key (stored encrypted).
    Pass empty string to delete the key.
    """
    try:
        from database.connection import get_db
        from database.repository import ApiKeyRepository

        valid_services = ['openai', 'anthropic', 'deepseek', 'deepl', 'elevenlabs', 'youtube']
        if update.service not in valid_services:
            raise HTTPException(status_code=400, detail=f"Invalid service. Must be one of: {valid_services}")

        async with get_db() as session:
            repo = ApiKeyRepository(session)

            if update.key:
                await repo.set(update.service, update.key)
                logger.info(f"Updated API key for service: {update.service}")
            else:
                await repo.delete(update.service)
                logger.info(f"Deleted API key for service: {update.service}")

            # Reload settings to pick up the change
            settings_store.reload()

            return {"success": True, "service": update.service, "has_key": bool(update.key)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update API key error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ApiKeysBulkUpdate(BaseModel):
    """Request to update multiple API keys"""
    keys: Dict[str, str] = Field(..., description="Map of service name to API key value")


@app.put("/api/settings/api-keys/bulk")
async def update_api_keys_bulk(update: ApiKeysBulkUpdate):
    """
    Update multiple API keys at once (stored encrypted).
    Pass empty string for a key to delete it.
    """
    try:
        from database.connection import get_db
        from database.repository import ApiKeyRepository

        valid_services = ['openai', 'anthropic', 'deepseek', 'deepl', 'elevenlabs', 'youtube']

        async with get_db() as session:
            repo = ApiKeyRepository(session)

            for service, key in update.keys.items():
                if service not in valid_services:
                    logger.warning(f"Skipping invalid service: {service}")
                    continue

                if key and key != '***':
                    await repo.set(service, key)
                    logger.info(f"Updated API key for service: {service}")
                elif key == '':
                    await repo.delete(service)
                    logger.info(f"Deleted API key for service: {service}")
                # Skip '***' (masked) values

            # Reload settings to pick up changes
            settings_store.reload()

            return {"success": True, "updated": list(update.keys.keys())}
    except Exception as e:
        logger.error(f"Bulk update API keys error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Storage Settings Endpoints ===

class StorageSettingsRequest(BaseModel):
    """Request to update storage settings"""
    output_directory: str = Field(..., description="Output directory path for processed files")


@app.get("/api/settings/storage")
async def get_storage_settings():
    """Get current storage settings (output directory)"""
    try:
        from utils.storage import get_output_directory
        output_dir = await get_output_directory()

        # Get from database settings
        db_settings = await settings_persistence.get_global_settings()
        storage_config = db_settings.get("storage", {})
        configured_dir = storage_config.get("output_directory", "")

        return {
            "output_directory": configured_dir,
            "effective_directory": str(output_dir),
            "default_directory": str(settings.PROCESSED_DIR),
            "is_default": not configured_dir or configured_dir == str(settings.PROCESSED_DIR)
        }
    except Exception as e:
        logger.error(f"Get storage settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/settings/storage")
async def update_storage_settings(request: StorageSettingsRequest):
    """
    Update storage settings (output directory).

    The output directory must be a valid, writable path.
    All new tasks will store their files in subdirectories of this path.
    """
    try:
        from pathlib import Path

        output_dir = request.output_directory.strip()

        # Validate the directory path
        if output_dir:
            path = Path(output_dir)

            # Check if path is absolute
            if not path.is_absolute():
                raise HTTPException(
                    status_code=400,
                    detail="Output directory must be an absolute path"
                )

            # Try to create the directory if it doesn't exist
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Permission denied: Cannot create directory {output_dir}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot create directory: {str(e)}"
                )

            # Check if directory is writable
            test_file = path / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory is not writable: {output_dir}"
                )

        # Update in database
        await settings_persistence.update_global_settings({
            "storage": {
                "output_directory": output_dir
            }
        })

        logger.info(f"Updated output directory to: {output_dir or 'default'}")

        return {
            "success": True,
            "output_directory": output_dir,
            "effective_directory": output_dir if output_dir else str(settings.PROCESSED_DIR),
            "message": "Storage settings updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update storage settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Translation Engine Options ===

@app.get("/api/translation/engines")
async def get_translation_engines():
    """Get available translation engines"""
    return [
        {
            "id": "google",
            "name": "Google Translate",
            "description": "免费，质量一般，速度快",
            "requires_api_key": False,
            "free": True,
        },
        {
            "id": "deepl",
            "name": "DeepL",
            "description": "高质量翻译，需要API密钥",
            "requires_api_key": True,
            "free": False,
        },
        {
            "id": "claude",
            "name": "Claude (Anthropic)",
            "description": "AI翻译，支持上下文理解，需要API密钥",
            "requires_api_key": True,
            "free": False,
        },
        {
            "id": "gpt",
            "name": "GPT-4 (OpenAI)",
            "description": "AI翻译，高质量，需要API密钥",
            "requires_api_key": True,
            "free": False,
        },
        {
            "id": "deepseek",
            "name": "DeepSeek",
            "description": "AI翻译，性价比高，需要API密钥",
            "requires_api_key": True,
            "free": False,
        },
    ]


# === Transcription Models ===

@app.get("/api/transcription/models")
async def get_transcription_models(backend: Optional[str] = None):
    """
    Get available Whisper models for transcription.

    Args:
        backend: Filter by backend ("faster" or "openai"), None for all
    """
    models = get_whisper_models(backend)
    return {
        "models": models,
        "default": "faster:small",
        "recommended_cpu": "faster:small",
        "recommended_gpu": "faster:large-v3",
    }


@app.get("/api/transcription/estimate")
async def get_transcription_estimate(
    duration: float,
    model_id: str = "faster:small",
    device: str = "cpu"
):
    """
    Get estimated transcription time for a video.

    Args:
        duration: Video duration in seconds
        model_id: Model ID (e.g., "faster:base", "faster:small")
        device: "cpu" or "gpu"
    """
    return estimate_transcription_time(duration, model_id, device)


@app.get("/api/transcription/estimates")
async def get_all_transcription_estimates(duration: float, backend: Optional[str] = None):
    """
    Get transcription time estimates for all models.

    Args:
        duration: Video duration in seconds
        backend: Filter by backend ("faster" or "openai"), None for all
    """
    estimates = get_all_estimates(duration, backend)
    recommended = get_recommended_model(duration, "cpu")
    return {
        "duration_seconds": duration,
        "estimates": estimates,
        "recommended_model": recommended,
    }


# === TTS Voice Options ===

@app.get("/api/tts/engines")
async def get_tts_engines():
    """Get available TTS engines"""
    return [
        {
            "id": "edge",
            "name": "Microsoft Edge TTS",
            "description": "免费，高质量神经语音，无需本地部署",
            "requires_api_key": False,
            "requires_local_server": False,
            "supports_voice_cloning": False,
            "free": True,
        },
        {
            "id": "index",
            "name": "IndexTTS 2.0",
            "description": "开源语音克隆，支持零样本/少样本学习",
            "requires_api_key": False,
            "requires_local_server": True,
            "server_port": 9880,
            "supports_voice_cloning": True,
            "modes": ["preset", "global", "dynamic"],
            "free": True,
        },
        {
            "id": "cosyvoice",
            "name": "CosyVoice (阿里巴巴)",
            "description": "阿里开源语音克隆，支持跨语言/情感控制",
            "requires_api_key": False,
            "requires_local_server": True,
            "server_port": 50000,
            "supports_voice_cloning": True,
            "modes": ["preset", "zero_shot", "cross_lingual", "instruct"],
            "free": True,
        },
        {
            "id": "qwen3",
            "name": "Qwen3-TTS (阿里通义)",
            "description": "阿里最新开源TTS，支持10种语言和语音克隆",
            "requires_api_key": False,
            "requires_local_server": True,
            "server_port": 50001,
            "supports_voice_cloning": True,
            "modes": ["preset", "clone"],
            "free": True,
        },
    ]


@app.get("/api/tts/voices/{engine}")
async def get_tts_voices_by_engine(engine: str, language: Optional[str] = None):
    """Get available voices for a specific TTS engine"""
    from tts import EdgeTTSEngine, IndexTTSEngine, CosyVoiceEngine, Qwen3TTSEngine

    if engine == "edge":
        tts = EdgeTTSEngine()
    elif engine == "index":
        tts = IndexTTSEngine()
    elif engine == "cosyvoice":
        tts = CosyVoiceEngine()
    elif engine == "qwen3":
        tts = Qwen3TTSEngine()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown TTS engine: {engine}")

    voices = await tts.get_available_voices(language)
    return [
        VoiceInfo(
            name=v["name"],
            display_name=v["display_name"],
            gender=v["gender"],
            locale=v["locale"],
        )
        for v in voices
    ]


@app.get("/api/tts/health/{engine}")
async def check_tts_engine_health(engine: str):
    """Check if a TTS engine server is available"""
    from tts import IndexTTSEngine, CosyVoiceEngine, Qwen3TTSEngine

    if engine == "edge":
        # Edge TTS is always available (online service)
        return {"engine": engine, "available": True, "message": "Edge TTS is an online service"}

    elif engine == "index":
        tts = IndexTTSEngine(
            host=settings.INDEX_TTS_HOST,
            port=settings.INDEX_TTS_PORT,
        )
        available = await tts.check_health()
        return {
            "engine": engine,
            "available": available,
            "host": settings.INDEX_TTS_HOST,
            "port": settings.INDEX_TTS_PORT,
            "message": "IndexTTS server is running" if available else "IndexTTS server not available. Start with: python -m indextts.server"
        }

    elif engine == "cosyvoice":
        tts = CosyVoiceEngine(
            host=settings.COSYVOICE_HOST,
            port=settings.COSYVOICE_PORT,
        )
        available = await tts.check_health()
        return {
            "engine": engine,
            "available": available,
            "host": settings.COSYVOICE_HOST,
            "port": settings.COSYVOICE_PORT,
            "message": "CosyVoice server is running" if available else "CosyVoice server not available. Start with: python webui.py --api"
        }

    elif engine == "qwen3":
        tts = Qwen3TTSEngine()
        available = await tts.check_health()
        return {
            "engine": engine,
            "available": available,
            "host": "127.0.0.1",
            "port": 50001,
            "message": "Qwen3-TTS server is running" if available else "Qwen3-TTS server not available. Start with: ./scripts/start_qwen3tts.sh"
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unknown TTS engine: {engine}")


# === YouTube Cookie Management ===

@app.get("/api/youtube/cookies/status")
async def get_youtube_cookie_status():
    """Check YouTube cookie status"""
    from utils.cookie_extractor import validate_cookies, get_available_browsers

    cookie_path = settings.DATA_DIR / "youtube_cookies.txt"

    # Check if running in Docker (can't access host browser)
    in_docker = os.path.exists('/.dockerenv')

    result = {
        "cookie_file_exists": cookie_path.exists(),
        "in_docker": in_docker,
        "available_browsers": [] if in_docker else get_available_browsers(),
        "validation": validate_cookies(cookie_path) if cookie_path.exists() else None,
    }

    return result


@app.post("/api/youtube/cookies/extract")
@limiter.limit("5/minute")
async def extract_youtube_cookies(request: Request, browser: str = "chrome"):
    """
    Extract YouTube cookies from browser (rate limited: 5/minute).
    Note: This only works when running locally, not in Docker.
    """
    from utils.cookie_extractor import extract_youtube_cookies as extract_cookies

    # Check if running in Docker
    in_docker = os.path.exists('/.dockerenv')
    if in_docker:
        raise HTTPException(
            status_code=400,
            detail="Cannot extract cookies when running in Docker. Please upload cookies manually or use the CLI tool."
        )

    cookie_path = settings.DATA_DIR / "youtube_cookies.txt"
    result = extract_cookies(browser, cookie_path)

    if result["success"]:
        # Persist to database
        try:
            with open(cookie_path, 'r', encoding='utf-8') as f:
                cookies_content = f.read()
            from database.connection import get_db
            async with get_db() as session:
                cookie_repo = CookieRepository(session)
                await cookie_repo.save(platform="youtube", cookie_data=cookies_content)
            logger.info("YouTube cookies persisted to database")
        except Exception as db_error:
            logger.warning(f"Failed to persist cookies to database: {db_error}")

        # Reload the pipeline's downloader to use new cookies
        global pipeline
        pipeline.youtube_downloader = pipeline.youtube_downloader.__class__(
            settings.DOWNLOADS_DIR
        )
        logger.info("Reloaded YouTube downloader with new cookies")

    return result


@app.post("/api/youtube/cookies/upload")
@limiter.limit("5/minute")
async def upload_youtube_cookies(request: Request, cookies_content: str):
    """
    Upload YouTube cookies content directly (rate limited: 5/minute).
    Accepts Netscape format cookie content.
    Also persists to database for durability.
    """
    from utils.cookie_extractor import validate_cookies

    cookie_path = settings.DATA_DIR / "youtube_cookies.txt"

    try:
        # Validate the content looks like cookies
        lines = cookies_content.strip().split('\n')
        valid_lines = 0
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) >= 7:
                valid_lines += 1

        if valid_lines == 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid cookie format. Please provide cookies in Netscape format."
            )

        # Save the cookies to file
        cookie_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cookie_path, 'w', encoding='utf-8') as f:
            f.write(cookies_content)

        # Persist to database for durability
        try:
            from database.connection import get_db
            async with get_db() as session:
                cookie_repo = CookieRepository(session)
                await cookie_repo.save(
                    platform="youtube",
                    cookie_data=cookies_content,
                )
            logger.info("YouTube cookies persisted to database")
        except Exception as db_error:
            logger.warning(f"Failed to persist cookies to database: {db_error}")
            # Continue anyway, file-based cookies will still work

        # Validate saved cookies
        validation = validate_cookies(cookie_path)

        if validation["valid"]:
            # Reload the pipeline's downloader
            global pipeline
            pipeline.youtube_downloader = pipeline.youtube_downloader.__class__(
                settings.DOWNLOADS_DIR
            )
            logger.info("Reloaded YouTube downloader with uploaded cookies")

            return {
                "success": True,
                "message": f"Successfully saved cookies. Found {validation['youtube_cookies']} YouTube cookies.",
                "validation": validation,
            }
        else:
            return {
                "success": False,
                "message": "Cookies saved but no YouTube cookies found. Make sure you exported cookies from YouTube.",
                "validation": validation,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload cookies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/youtube/cookies")
async def delete_youtube_cookies():
    """Delete YouTube cookies file"""
    cookie_path = settings.DATA_DIR / "youtube_cookies.txt"

    if cookie_path.exists():
        cookie_path.unlink()
        return {"success": True, "message": "Cookies deleted"}
    else:
        return {"success": True, "message": "No cookies to delete"}


# === Cookie Persistence Endpoints ===

@app.get("/api/cookies")
async def list_all_cookies():
    """List all saved cookies from database"""
    try:
        from database.connection import get_db
        async with get_db() as session:
            cookie_repo = CookieRepository(session)
            cookies = await cookie_repo.get_all()
            return {
                "success": True,
                "cookies": [
                    {
                        "platform": c.platform,
                        "is_valid": c.is_valid,
                        "expires_at": c.expires_at.isoformat() if c.expires_at else None,
                        "last_verified": c.last_verified.isoformat() if c.last_verified else None,
                        "created_at": c.created_at.isoformat() if c.created_at else None,
                        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                    }
                    for c in cookies
                ]
            }
    except Exception as e:
        logger.error(f"Failed to list cookies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cookies/{platform}")
async def get_cookie_status(platform: str):
    """Get cookie status for a specific platform"""
    try:
        from database.connection import get_db
        async with get_db() as session:
            cookie_repo = CookieRepository(session)
            cookie = await cookie_repo.get(platform)

            if not cookie:
                return {
                    "success": True,
                    "platform": platform,
                    "exists": False,
                    "is_valid": False,
                }

            return {
                "success": True,
                "platform": platform,
                "exists": True,
                "is_valid": cookie.is_valid,
                "expires_at": cookie.expires_at.isoformat() if cookie.expires_at else None,
                "last_verified": cookie.last_verified.isoformat() if cookie.last_verified else None,
                "created_at": cookie.created_at.isoformat() if cookie.created_at else None,
                "updated_at": cookie.updated_at.isoformat() if cookie.updated_at else None,
            }
    except Exception as e:
        logger.error(f"Failed to get cookie status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SaveCookieRequest(BaseModel):
    """Request to save cookie data"""
    platform: str = Field(..., description="Platform name: youtube, bilibili, etc.")
    cookie_data: Optional[str] = Field(None, description="Cookie string in Netscape format")
    cookie_json: Optional[Dict[str, Any]] = Field(None, description="Cookie data as JSON")
    expires_at: Optional[str] = Field(None, description="Expiration datetime ISO format")


@app.post("/api/cookies")
async def save_cookie(request: SaveCookieRequest):
    """Save or update cookie for a platform"""
    try:
        from database.connection import get_db
        async with get_db() as session:
            cookie_repo = CookieRepository(session)

            expires_at = None
            if request.expires_at:
                expires_at = datetime.fromisoformat(request.expires_at)

            cookie = await cookie_repo.save(
                platform=request.platform,
                cookie_data=request.cookie_data,
                cookie_json=request.cookie_json,
                expires_at=expires_at,
            )

            # For YouTube, also write to file for yt-dlp compatibility
            if request.platform == "youtube" and request.cookie_data:
                cookie_path = settings.DATA_DIR / "youtube_cookies.txt"
                await cookie_repo.write_cookie_file(request.platform, cookie_path)

                # Reload YouTube downloader
                global pipeline
                pipeline.youtube_downloader = pipeline.youtube_downloader.__class__(
                    settings.DOWNLOADS_DIR
                )
                logger.info("Reloaded YouTube downloader with persisted cookies")

            return {
                "success": True,
                "message": f"Cookie saved for {request.platform}",
                "platform": request.platform,
            }
    except Exception as e:
        logger.error(f"Failed to save cookie: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cookies/{platform}/verify")
async def verify_cookie(platform: str):
    """Mark a cookie as verified (valid)"""
    try:
        from database.connection import get_db
        async with get_db() as session:
            cookie_repo = CookieRepository(session)
            success = await cookie_repo.mark_verified(platform)

            if not success:
                raise HTTPException(status_code=404, detail=f"Cookie not found for {platform}")

            return {
                "success": True,
                "message": f"Cookie verified for {platform}",
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify cookie: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cookies/{platform}/invalidate")
async def invalidate_cookie(platform: str):
    """Mark a cookie as invalid"""
    try:
        from database.connection import get_db
        async with get_db() as session:
            cookie_repo = CookieRepository(session)
            success = await cookie_repo.mark_invalid(platform)

            if not success:
                raise HTTPException(status_code=404, detail=f"Cookie not found for {platform}")

            return {
                "success": True,
                "message": f"Cookie marked invalid for {platform}",
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to invalidate cookie: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cookies/{platform}")
async def delete_cookie(platform: str):
    """Delete cookie for a platform"""
    try:
        from database.connection import get_db
        async with get_db() as session:
            cookie_repo = CookieRepository(session)
            success = await cookie_repo.delete(platform)

            # For YouTube, also delete the file
            if platform == "youtube":
                cookie_path = settings.DATA_DIR / "youtube_cookies.txt"
                if cookie_path.exists():
                    cookie_path.unlink()

            return {
                "success": True,
                "message": f"Cookie deleted for {platform}" if success else f"No cookie found for {platform}",
            }
    except Exception as e:
        logger.error(f"Failed to delete cookie: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cookies/{platform}/restore")
async def restore_cookie_file(platform: str):
    """Restore cookie file from database (for platforms that need file-based cookies)"""
    try:
        from database.connection import get_db
        async with get_db() as session:
            cookie_repo = CookieRepository(session)

            if platform == "youtube":
                cookie_path = settings.DATA_DIR / "youtube_cookies.txt"
                success = await cookie_repo.write_cookie_file(platform, cookie_path)

                if success:
                    # Reload YouTube downloader
                    global pipeline
                    pipeline.youtube_downloader = pipeline.youtube_downloader.__class__(
                        settings.DOWNLOADS_DIR
                    )
                    logger.info("Restored YouTube cookies from database")

                    return {
                        "success": True,
                        "message": f"Cookie file restored for {platform}",
                        "path": str(cookie_path),
                    }
                else:
                    return {
                        "success": False,
                        "message": f"No cookie data found for {platform}",
                    }
            else:
                return {
                "success": False,
                "message": f"File restoration not supported for {platform}",
            }
    except Exception as e:
        logger.error(f"Failed to restore cookie file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Font Management Endpoints ===

@app.get("/api/fonts")
async def get_available_fonts():
    """
    Get list of available fonts for subtitles.
    Includes system fonts and custom uploaded fonts.
    Detects script support (Latin, CJK, Arabic, Cyrillic, etc.) for each font.
    """
    import platform
    from pathlib import Path

    # Script detection patterns by font name
    SCRIPT_PATTERNS = {
        'cjk': {
            # Chinese - Apple/macOS fonts
            'pingfang', 'heiti', 'stheit', 'songti', 'kaiti', 'fangsong',
            'yuanti', 'lantinghei', 'baoli', 'hannotate', 'hanzipen',
            'xingkai', 'weibei', 'libian', 'hupo', 'yuppy', 'wawati',
            'stheiti', 'stkaiti', 'stsong', 'stfangsong', 'stxihei', 'stzhongs',
            'apple lisung', 'apple ligothic', 'gb18030', 'gbsn',
            # Chinese - Microsoft/Windows fonts
            'yahei', 'simhei', 'simsun', 'nsimsun', 'simkai', 'simfang', 'simli',
            'microsoft jhenghei', 'dengxian', 'fangzheng', 'fzshuti', 'fzyaoti', 'fzkai',
            # Chinese - Traditional
            'mingliu', 'pmingliu', 'dfkai', 'cwtext', 'cwyen',
            # Chinese - Common fonts
            'lisu', 'youyuan', 'huawenxihei', 'huawenzhongsong', 'huawenkaiti',
            # Chinese - Open source
            'wqy', 'wenquanyi', 'noto sans cjk', 'noto serif cjk', 'source han',
            'sarasa', 'iosevka', 'lxgw', 'oppo sans',
            # Chinese - Google Fonts open source
            'zcool', 'mashanzheng', 'zhimangxing', 'liujianmaocao', 'longcang',
            'mashan', 'zhanku', '站酷',
            # Japanese
            'hiragino', 'meiryo', 'msgothic', 'msmincho', 'yugothic', 'kozuka',
            'morisawa', 'iwata', 'hgp', 'hgs', 'ud digi',
            # Korean
            'malgun', 'batang', 'gulim', 'dotum', 'gungsuh', 'nanum', 'spoqa',
            # Generic CJK indicators
            'cjk', 'chinese', 'japanese', 'korean', 'sc', 'tc', 'hk', 'jp', 'kr', 'hans', 'hant',
        },
        'arabic': {
            'arabic', 'naskh', 'kufi', 'thuluth', 'nastaliq', 'ruqaa', 'diwani',
            'scheherazade', 'amiri', 'lateef', 'sakkal', 'traditional arabic',
            'simplified arabic', 'geeza', 'al bayan', 'baghdad',
            'decotype', 'kacstone', 'farsi', 'persian', 'urdu', 'noto sans arabic',
            'noto naskh', 'noto kufi',
        },
        'hebrew': {
            'hebrew', 'david', 'miriam', 'frank ruehl', 'aharoni', 'rod', 'gisha',
            'levenim', 'narkisim', 'raanana', 'noto sans hebrew', 'noto serif hebrew',
        },
        'thai': {
            'thai', 'sarabun', 'leelawadee', 'angsana', 'browallia',
            'cordia', 'dillen', 'eucrosia', 'freesia', 'iris', 'jasmine', 'kodchiang',
            'lily', 'norasi', 'noto sans thai', 'noto serif thai',
        },
        'devanagari': {
            'devanagari', 'mangal', 'aparajita', 'kokila', 'utsaah', 'sanskrit',
            'hindi', 'marathi', 'nepali', 'noto sans devanagari', 'noto serif devanagari',
            'lohit', 'gargi', 'kalimati',
        },
        'cyrillic': {
            'cyrillic', 'russian', 'bulgarian', 'serbian', 'ukrainian',
        },
        'greek': {
            'greek',
        },
    }

    # Common multilingual fonts
    MULTILINGUAL_FONTS = {'arial', 'times new roman', 'verdana', 'tahoma', 'calibri', 'segoe',
        'noto sans', 'noto serif', 'dejavu', 'liberation', 'freesans', 'freeserif',
        'ubuntu', 'droid', 'roboto', 'open sans', 'lato', 'source sans'}

    def detect_font_scripts(font_name: str) -> list:
        """Detect which scripts a font likely supports based on its name"""
        name_lower = font_name.lower().replace('-', ' ').replace('_', ' ')
        scripts = ['latin']  # Assume all fonts support basic Latin

        # Check for script-specific patterns
        for script, patterns in SCRIPT_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    if script not in scripts:
                        scripts.append(script)
                    break

        # Check for non-Latin characters in font name
        for char in font_name:
            if '\u4e00' <= char <= '\u9fff':  # Chinese
                if 'cjk' not in scripts:
                    scripts.append('cjk')
            elif '\u0600' <= char <= '\u06ff':  # Arabic
                if 'arabic' not in scripts:
                    scripts.append('arabic')
            elif '\u0590' <= char <= '\u05ff':  # Hebrew
                if 'hebrew' not in scripts:
                    scripts.append('hebrew')
            elif '\u0e00' <= char <= '\u0e7f':  # Thai
                if 'thai' not in scripts:
                    scripts.append('thai')
            elif '\u0900' <= char <= '\u097f':  # Devanagari
                if 'devanagari' not in scripts:
                    scripts.append('devanagari')

        # Multi-script fonts support Cyrillic and Greek
        for pattern in MULTILINGUAL_FONTS:
            if pattern in name_lower and 'noto' not in name_lower:
                if 'cyrillic' not in scripts:
                    scripts.append('cyrillic')
                if 'greek' not in scripts:
                    scripts.append('greek')
                break

        return scripts

    fonts = []
    system = platform.system()

    # System font directories by platform
    if system == "Darwin":  # macOS
        font_dirs = [
            Path("/System/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),  # Additional system fonts
            Path("/Library/Fonts"),
            Path.home() / "Library/Fonts",
        ]
    elif system == "Windows":
        font_dirs = [
            Path("C:/Windows/Fonts"),
        ]
    else:  # Linux
        font_dirs = [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".fonts",
            Path.home() / ".local/share/fonts",
        ]

    # Scan system fonts
    font_extensions = {".ttf", ".ttc", ".otf"}
    seen_names = set()

    for font_dir in font_dirs:
        if font_dir.exists():
            for font_file in font_dir.rglob("*"):
                if font_file.suffix.lower() in font_extensions:
                    # Get font name from filename
                    font_name = font_file.stem
                    if font_name not in seen_names:
                        seen_names.add(font_name)
                        font_scripts = detect_font_scripts(font_name)
                        fonts.append({
                            "name": font_name,
                            "path": str(font_file),
                            "is_custom": False,
                            "supported_scripts": font_scripts,
                        })

    # Add custom fonts from data/fonts directory
    custom_fonts_dir = settings.DATA_DIR / "fonts"
    if custom_fonts_dir.exists():
        for font_file in custom_fonts_dir.glob("*"):
            if font_file.suffix.lower() in font_extensions:
                font_name = font_file.stem
                if font_name not in seen_names:
                    seen_names.add(font_name)
                    font_scripts = detect_font_scripts(font_name)
                    fonts.append({
                        "name": font_name,
                        "path": str(font_file),
                        "is_custom": True,
                        "supported_scripts": font_scripts,
                    })

    # Sort fonts alphabetically
    fonts.sort(key=lambda x: x["name"].lower())

    # Add recommended font info
    from utils.subtitle_burner import find_cjk_font
    cjk_path, cjk_name = find_cjk_font()
    recommended = {
        "name": cjk_name,
        "path": cjk_path,
        "is_custom": False,
        "is_recommended": True,
        "supported_scripts": ['latin', 'cjk'],
    }

    return {
        "fonts": fonts,
        "recommended": recommended,
        "total": len(fonts),
    }


@app.post("/api/fonts/upload")
@limiter.limit("5/minute")
async def upload_font(request: Request):
    """
    Upload a custom font file (rate limited: 5/minute).
    Accepts .ttf, .ttc, or .otf files.
    """
    from fastapi import UploadFile, File
    import aiofiles

    # Get form data
    form = await request.form()
    file = form.get("file")

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file extension
    filename = file.filename
    valid_extensions = {".ttf", ".ttc", ".otf"}
    ext = Path(filename).suffix.lower()

    if ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(valid_extensions)}"
        )

    # Create fonts directory if needed
    fonts_dir = settings.DATA_DIR / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    # Save the font file
    dest_path = fonts_dir / filename
    content = await file.read()

    with open(dest_path, 'wb') as f:
        f.write(content)

    logger.info(f"Uploaded font: {filename}")

    return {
        "success": True,
        "message": f"Font '{filename}' uploaded successfully",
        "font": {
            "name": Path(filename).stem,
            "path": str(dest_path),
            "is_custom": True,
        }
    }


@app.delete("/api/fonts/{font_name}")
async def delete_custom_font(font_name: str):
    """Delete a custom uploaded font"""
    fonts_dir = settings.DATA_DIR / "fonts"

    # Find and delete the font file
    for ext in [".ttf", ".ttc", ".otf"]:
        font_path = fonts_dir / f"{font_name}{ext}"
        if font_path.exists():
            font_path.unlink()
            logger.info(f"Deleted font: {font_name}")
            return {"success": True, "message": f"Font '{font_name}' deleted"}

    raise HTTPException(status_code=404, detail=f"Font '{font_name}' not found")


# === Subtitle Preset Endpoints ===

@app.get("/api/subtitle-presets")
async def get_subtitle_presets():
    """
    Get all subtitle presets (built-in and custom).
    """
    from settings_store import BUILTIN_PRESETS

    # Get built-in presets
    presets = [p.to_dict() for p in BUILTIN_PRESETS]

    # Get custom presets from database
    try:
        custom_presets = await settings_persistence.get_custom_presets()
        # Add default values for missing fields in custom presets (migration support)
        for preset in custom_presets:
            if "is_vertical" not in preset:
                # Auto-detect vertical from ID or name
                preset_id = preset.get("id", "")
                preset_name = preset.get("name", "")
                preset["is_vertical"] = (
                    preset_id.startswith("vertical_") or
                    "竖屏" in preset_name or
                    "vertical" in preset_name.lower()
                )
            # Add max_width defaults to styles if missing
            for style_key in ["original_style", "translated_style"]:
                if preset.get(style_key) and "max_width" not in preset[style_key]:
                    # Use 75% for vertical, 90% for horizontal
                    preset[style_key]["max_width"] = 75 if preset.get("is_vertical") else 90
        presets.extend(custom_presets)
    except Exception as e:
        logger.warning(f"Failed to load custom presets: {e}")

    return {
        "presets": presets,
        "builtin_count": len(BUILTIN_PRESETS),
        "custom_count": len(presets) - len(BUILTIN_PRESETS),
    }


class CreatePresetRequest(BaseModel):
    """Request to create a custom subtitle preset"""
    name: str = Field(..., description="Preset name")
    description: str = Field("", description="Preset description")
    is_vertical: bool = Field(False, description="Whether this preset is for vertical videos (9:16)")
    subtitle_mode: str = Field("dual", description="Subtitle mode: 'dual', 'original_only', 'translated_only'")
    source_language: Optional[str] = Field("en", description="Source language code")
    target_language: Optional[str] = Field("zh-CN", description="Target language code")
    original_style: Optional[Dict[str, Any]] = Field(None, description="Original subtitle style config")
    translated_style: Optional[Dict[str, Any]] = Field(None, description="Translated subtitle style config")


@app.post("/api/subtitle-presets")
async def create_subtitle_preset(request: CreatePresetRequest):
    """
    Create a new custom subtitle preset.
    """
    import uuid

    # Generate unique ID for the preset
    preset_id = f"custom_{uuid.uuid4().hex[:8]}"

    preset_data = {
        "id": preset_id,
        "name": request.name,
        "description": request.description,
        "is_builtin": False,
        "is_vertical": request.is_vertical,
        "subtitle_mode": request.subtitle_mode,
        "source_language": request.source_language,
        "target_language": request.target_language,
    }

    # Only include styles that are needed based on subtitle_mode
    if request.subtitle_mode in ("dual", "original_only") and request.original_style:
        preset_data["original_style"] = request.original_style
    if request.subtitle_mode in ("dual", "translated_only") and request.translated_style:
        preset_data["translated_style"] = request.translated_style

    try:
        await settings_persistence.save_custom_preset(preset_data)
        logger.info(f"Created custom preset: {request.name}")

        return {
            "success": True,
            "message": f"Preset '{request.name}' created",
            "preset": preset_data,
        }
    except Exception as e:
        logger.error(f"Failed to save preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/subtitle-presets/{preset_id}")
async def update_subtitle_preset(preset_id: str, request: CreatePresetRequest):
    """
    Update an existing custom subtitle preset.
    Cannot update built-in presets.
    """
    from settings_store import get_preset_by_id

    # Check if it's a built-in preset
    if get_preset_by_id(preset_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot modify built-in presets"
        )

    preset_data = {
        "id": preset_id,
        "name": request.name,
        "description": request.description,
        "is_builtin": False,
        "is_vertical": request.is_vertical,
        "subtitle_mode": request.subtitle_mode,
        "source_language": request.source_language,
        "target_language": request.target_language,
    }

    # Only include styles that are needed based on subtitle_mode
    if request.subtitle_mode in ("dual", "original_only") and request.original_style:
        preset_data["original_style"] = request.original_style
    if request.subtitle_mode in ("dual", "translated_only") and request.translated_style:
        preset_data["translated_style"] = request.translated_style

    try:
        await settings_persistence.save_custom_preset(preset_data)
        logger.info(f"Updated custom preset: {preset_id}")

        return {
            "success": True,
            "message": f"Preset '{request.name}' updated",
            "preset": preset_data,
        }
    except Exception as e:
        logger.error(f"Failed to update preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/subtitle-presets/{preset_id}")
async def delete_subtitle_preset(preset_id: str):
    """
    Delete a custom subtitle preset.
    Cannot delete built-in presets.
    """
    from settings_store import get_preset_by_id

    # Check if it's a built-in preset
    if get_preset_by_id(preset_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete built-in presets"
        )

    try:
        success = await settings_persistence.delete_custom_preset(preset_id)
        if success:
            logger.info(f"Deleted custom preset: {preset_id}")
            return {"success": True, "message": f"Preset deleted"}
        else:
            raise HTTPException(status_code=404, detail="Preset not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Metadata Preset Endpoints ====================


class MetadataPresetRequest(BaseModel):
    """Request to create or update a metadata preset"""
    name: str = Field(..., description="Preset display name", min_length=1, max_length=100)
    description: str = Field("", description="Preset description")
    title_prefix: str = Field("", description="Title prefix (e.g., '[中字]')", max_length=50)
    custom_signature: str = Field("", description="Custom signature for description")
    tags: List[str] = Field(default_factory=list, description="Applicable scenario tags")


class MetadataPresetResponse(BaseModel):
    """Response model for metadata preset"""
    id: str
    name: str
    description: Optional[str]
    title_prefix: str
    custom_signature: str
    tags: List[str]
    is_default: bool
    is_builtin: bool
    sort_order: int
    created_at: Optional[str]
    updated_at: Optional[str]


class MetadataPresetsListResponse(BaseModel):
    """Response model for list of metadata presets"""
    presets: List[MetadataPresetResponse]
    builtin_count: int
    custom_count: int


@app.get("/api/metadata-presets", response_model=MetadataPresetsListResponse)
async def list_metadata_presets():
    """Get all metadata presets"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = MetadataPresetRepository(session)
        presets = await repo.get_all()

        preset_list = [MetadataPresetResponse(**p.to_dict()) for p in presets]
        builtin_count = sum(1 for p in presets if p.is_builtin)
        custom_count = sum(1 for p in presets if not p.is_builtin)

        return MetadataPresetsListResponse(
            presets=preset_list,
            builtin_count=builtin_count,
            custom_count=custom_count
        )


@app.get("/api/metadata-presets/{preset_id}", response_model=MetadataPresetResponse)
async def get_metadata_preset(preset_id: str):
    """Get a specific metadata preset by ID"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = MetadataPresetRepository(session)
        preset = await repo.get(preset_id)

        if not preset:
            raise HTTPException(status_code=404, detail="Preset not found")

        return MetadataPresetResponse(**preset.to_dict())


@app.post("/api/metadata-presets", response_model=dict)
async def create_metadata_preset(request: MetadataPresetRequest):
    """Create a new custom metadata preset"""
    import uuid

    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = MetadataPresetRepository(session)

        # Generate a unique ID
        preset_id = f"custom_{uuid.uuid4().hex[:8]}"

        # Get max sort_order for custom presets
        presets = await repo.get_all()
        max_order = max((p.sort_order for p in presets), default=0)

        preset = await repo.create(
            preset_id=preset_id,
            name=request.name,
            description=request.description,
            title_prefix=request.title_prefix,
            custom_signature=request.custom_signature,
            tags=request.tags,
            is_default=False,
            is_builtin=False,
            sort_order=max_order + 1,
        )

        await session.commit()

        return {
            "success": True,
            "message": "Preset created successfully",
            "preset": MetadataPresetResponse(**preset.to_dict())
        }


@app.put("/api/metadata-presets/{preset_id}", response_model=dict)
async def update_metadata_preset(preset_id: str, request: MetadataPresetRequest):
    """Update a metadata preset (only custom presets can be fully edited)"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = MetadataPresetRepository(session)

        preset = await repo.get(preset_id)
        if not preset:
            raise HTTPException(status_code=404, detail="Preset not found")

        if preset.is_builtin:
            raise HTTPException(
                status_code=400,
                detail="Cannot modify builtin presets. You can only change the default preset."
            )

        updated = await repo.update(
            preset_id,
            name=request.name,
            description=request.description,
            title_prefix=request.title_prefix,
            custom_signature=request.custom_signature,
            tags=request.tags,
        )

        await session.commit()

        return {
            "success": True,
            "message": "Preset updated successfully",
            "preset": MetadataPresetResponse(**updated.to_dict())
        }


@app.delete("/api/metadata-presets/{preset_id}", response_model=dict)
async def delete_metadata_preset(preset_id: str):
    """Delete a custom metadata preset (builtin presets cannot be deleted)"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = MetadataPresetRepository(session)

        preset = await repo.get(preset_id)
        if not preset:
            raise HTTPException(status_code=404, detail="Preset not found")

        if preset.is_builtin:
            raise HTTPException(status_code=400, detail="Cannot delete builtin presets")

        success = await repo.delete(preset_id)
        await session.commit()

        if success:
            return {"success": True, "message": "Preset deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete preset")


@app.post("/api/metadata-presets/{preset_id}/set-default", response_model=dict)
async def set_default_metadata_preset(preset_id: str):
    """Set a metadata preset as the default"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = MetadataPresetRepository(session)

        preset = await repo.set_default(preset_id)
        if not preset:
            raise HTTPException(status_code=404, detail="Preset not found")

        await session.commit()

        return {
            "success": True,
            "message": f"'{preset.name}' is now the default preset",
            "preset": MetadataPresetResponse(**preset.to_dict())
        }


@app.get("/api/metadata-presets/default", response_model=MetadataPresetResponse)
async def get_default_metadata_preset():
    """Get the default metadata preset"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = MetadataPresetRepository(session)
        preset = await repo.get_default()

        if not preset:
            # Return first preset if no default set
            presets = await repo.get_all()
            if presets:
                preset = presets[0]
            else:
                raise HTTPException(status_code=404, detail="No presets found")

        return MetadataPresetResponse(**preset.to_dict())


class AIPresetSelectRequest(BaseModel):
    """Request for AI preset selection"""
    task_id: Optional[str] = Field(None, description="Task ID to analyze")
    video_info: Optional[Dict[str, Any]] = Field(None, description="Video info for analysis")
    transcript_snippet: Optional[str] = Field(None, description="Transcript snippet for analysis")


class AIPresetSelectResponse(BaseModel):
    """Response for AI preset selection"""
    success: bool
    preset_id: str
    preset_name: str
    confidence: float
    reason: str
    all_matches: List[Dict[str, Any]]


@app.post("/api/metadata-presets/ai-select", response_model=AIPresetSelectResponse)
async def ai_select_metadata_preset(request: AIPresetSelectRequest):
    """
    Use AI to select the most appropriate metadata preset based on video content.

    Analyzes video title, description, tags, platform, and transcript to recommend
    the best preset for title prefix and signature.
    """
    from metadata import select_preset_for_task, preset_matcher

    video_info = request.video_info
    transcript = request.transcript_snippet

    # If task_id provided, get video info and transcript from task
    if request.task_id:
        task = pipeline.tasks.get(request.task_id)
        if not task:
            task = await ensure_task_in_memory(request.task_id)

        if task:
            if not video_info and task.video_info:
                video_info = task.video_info

            if not transcript and task.subtitle_path and task.subtitle_path.exists():
                # Read first 1000 chars of transcript
                try:
                    full_transcript = task.subtitle_path.read_text(encoding="utf-8")
                    transcript = full_transcript[:1000]
                except Exception as e:
                    logger.warning(f"Failed to read transcript: {e}")

    # Run the matcher
    result = preset_matcher.match(video_info=video_info, transcript=transcript)

    # Get preset name from database
    preset_name = result.recommended.preset_id
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = MetadataPresetRepository(session)
        preset = await repo.get(result.recommended.preset_id)
        if preset:
            preset_name = preset.name

    return AIPresetSelectResponse(
        success=True,
        preset_id=result.recommended.preset_id,
        preset_name=preset_name,
        confidence=result.recommended.confidence,
        reason=result.recommended.reason,
        all_matches=[
            {
                "preset_id": m.preset_id,
                "confidence": m.confidence,
                "reason": m.reason
            }
            for m in result.all_matches
        ]
    )


# ==================== Metadata Generation Endpoints ====================


class MetadataGenerateRequest(BaseModel):
    """Request model for metadata generation"""
    task_id: str = Field(..., description="Task ID to generate metadata for")
    source_language: str = Field("en", description="Source language code")
    target_language: str = Field("zh-CN", description="Target language code")
    include_source_url: bool = Field(True, description="Include source URL in description")
    title_prefix: Optional[str] = Field(None, description="Prefix to add before title (e.g., '[中字]')")
    custom_signature: Optional[str] = Field(None, description="Custom signature to append")
    max_keywords: int = Field(10, description="Maximum number of keywords", ge=1, le=20)
    preset_id: Optional[str] = Field(None, description="Metadata preset ID to use (overrides title_prefix and custom_signature)")
    use_ai_preset_selection: bool = Field(False, description="Use AI to automatically select best preset")
    platform: str = Field("generic", description="Target platform (douyin, bilibili, xiaohongshu, generic)")


class MetadataResponse(BaseModel):
    """Response model for metadata generation"""
    success: bool
    title: str = ""
    title_translated: str = ""
    description: str = ""
    keywords: List[str] = []
    error: Optional[str] = None


@app.post("/api/tasks/{task_id}/generate-metadata", response_model=MetadataResponse)
@limiter.limit("10/minute")
async def generate_task_metadata(task_id: str, request: Request, body: MetadataGenerateRequest):
    """
    Generate AI-powered metadata (title, description, keywords) for a task.

    Uses the translation engine configured in settings.
    Requires the transcription step to be completed.
    """
    # Get task
    task = pipeline.tasks.get(task_id)
    if not task:
        # Try to load from database
        task = await ensure_task_in_memory(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

    # Check if transcription is completed (optional for transfer-only tasks)
    has_transcript = task.subtitle_path and task.subtitle_path.exists()

    try:
        from metadata import MetadataGenerator

        # Get translation settings for API key
        global_settings = settings_store.load()
        engine = global_settings.translation.engine
        api_key = global_settings.translation.get_api_key()
        model = global_settings.translation.model
        metadata_settings = global_settings.metadata

        if not api_key and engine not in ["google"]:
            raise HTTPException(
                status_code=400,
                detail=f"API key not configured for {engine}. Please configure in settings."
            )

        # Read transcript from subtitle file (if available)
        transcript = ""
        if has_transcript:
            transcript = task.subtitle_path.read_text(encoding="utf-8")
        elif task.video_info:
            # For transfer-only tasks, use video description as fallback
            transcript = task.video_info.get("description", "")

        # Get original title
        original_title = ""
        if task.video_info:
            original_title = task.video_info.get("title", "")

        # Get source URL
        source_url = ""
        if body.include_source_url or metadata_settings.include_source_url:
            source_url = task.options.source_url or ""

        # Determine which preset to use:
        # 1. If preset_id is explicitly provided, use that
        # 2. If AI preset selection is enabled, let AI choose
        # 3. Otherwise, use the default preset
        preset_id = body.preset_id
        title_prefix = None
        signature = None

        # AI preset selection if enabled and no preset specified
        if body.use_ai_preset_selection and not preset_id:
            try:
                from metadata import select_preset_for_task
                logger.info(f"Using AI to select metadata preset for task {task_id}")
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
                if ai_result and ai_result.preset_id:
                    preset_id = ai_result.preset_id
                    logger.info(f"AI selected preset: {preset_id} (confidence: {ai_result.confidence:.2f})")
            except Exception as e:
                logger.warning(f"AI preset selection failed: {e}, using default")

        # Fetch preset (specified, AI-selected, or default)
        session_factory = get_session_factory()
        async with session_factory() as session:
            preset_repo = MetadataPresetRepository(session)
            
            if preset_id:
                preset = await preset_repo.get(preset_id)
                if preset:
                    title_prefix = preset.title_prefix
                    signature = preset.custom_signature
                    logger.info(f"Using metadata preset: {preset.name}")
            
            # Fall back to default preset if no preset specified or not found
            if title_prefix is None:
                default_preset = await preset_repo.get_default()
                if default_preset:
                    title_prefix = default_preset.title_prefix
                    signature = default_preset.custom_signature
                    logger.info(f"Using default metadata preset: {default_preset.name}")

        # Final fallback to empty if still not set
        title_prefix = title_prefix or ""
        signature = signature or ""

        # Create generator and generate metadata
        generator = MetadataGenerator(engine=engine, api_key=api_key, model=model)
        result = await generator.generate(
            original_title=original_title,
            transcript=transcript,
            source_url=source_url,
            source_language=body.source_language,
            target_language=body.target_language,
            title_prefix=title_prefix,
            custom_signature=signature,
            max_keywords=body.max_keywords or metadata_settings.max_keywords,
            platform=body.platform
        )

        return MetadataResponse(
            success=result.success,
            title=result.title,
            title_translated=result.title_translated,
            description=result.description,
            keywords=result.keywords,
            error=result.error
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metadata generation error: {e}")
        return MetadataResponse(success=False, error=str(e))


class SaveMetadataRequest(BaseModel):
    """Request to save generated metadata"""
    title: str = Field(..., description="Title (translated)")
    description: str = Field(..., description="Description")
    keywords: List[str] = Field(default_factory=list, description="Keywords list")


@app.post("/api/tasks/{task_id}/metadata")
async def save_task_metadata(task_id: str, body: SaveMetadataRequest):
    """
    Save generated/edited metadata for a task.
    This persists the metadata to database so it survives restarts.
    """
    try:
        metadata = {
            "title": body.title,
            "description": body.description,
            "keywords": body.keywords,
        }
        await task_persistence.save_generated_metadata(task_id, metadata)
        return {"success": True, "message": "Metadata saved successfully"}
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save metadata: {e}")


@app.get("/api/tasks/{task_id}/metadata")
async def get_task_metadata(task_id: str):
    """
    Get saved metadata for a task.
    Returns the previously saved/generated metadata if available.
    """
    try:
        metadata = await task_persistence.load_generated_metadata(task_id)
        is_approved = await task_persistence.is_metadata_approved(task_id)
        if metadata:
            return {
                "success": True,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "keywords": metadata.get("keywords", []),
                "generated_at": metadata.get("generated_at"),
                "approved": is_approved,
            }
        return {"success": False, "message": "No metadata found", "approved": False}
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {e}")


@app.post("/api/tasks/{task_id}/metadata/approve")
async def approve_task_metadata(task_id: str, background_tasks: BackgroundTasks):
    """
    Approve metadata for a task.
    Changes task status from pending_review to pending_upload.
    If auto_upload is enabled for any platform, triggers upload.
    """
    try:
        # Check if metadata exists
        metadata = await task_persistence.load_generated_metadata(task_id)
        if not metadata:
            raise HTTPException(status_code=400, detail="No metadata to approve. Generate metadata first.")

        # Approve metadata and set status to pending_upload (synchronous, not queued)
        success = await task_persistence.approve_metadata(task_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to approve metadata in database")

        # Get task (in-memory or from database) to check upload options
        task = pipeline.tasks.get(task_id)
        if not task:
            task = await ensure_task_in_memory(task_id)
        
        if task:
            task.status = TaskStatus.PENDING_UPLOAD
            task.message = "元数据已审核，等待上传"
            logger.info(f"Updated in-memory task {task_id} to pending_upload")

        # Check if any upload platforms are selected for THIS task (not global settings)
        should_upload = False
        if task and task.options:
            should_upload = (
                task.options.upload_bilibili or
                task.options.upload_douyin or
                task.options.upload_xiaohongshu
            )

        if should_upload:
            # Trigger upload in background
            background_tasks.add_task(run_upload_step, task_id)
            return {"success": True, "message": "Metadata approved, upload started", "uploading": True}

        return {"success": True, "message": "Metadata approved", "uploading": False}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to approve metadata: {e}")


@app.post("/api/tasks/{task_id}/redownload-thumbnail")
async def redownload_thumbnail(task_id: str):
    """
    Re-download the thumbnail for a task.
    Useful when the initial thumbnail download failed.
    """
    try:
        # Get task
        task = await ensure_task_in_memory(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Get video info for thumbnail URL
        thumbnail_url = task.video_info.get("thumbnail_url") if task.video_info else None
        if not thumbnail_url:
            raise HTTPException(status_code=400, detail="No thumbnail URL available for this task")

        # Get task directory
        from utils.storage import get_task_directory
        task_folder = await task_persistence.get_task_folder(task_id)
        task_dir = await get_task_directory(task_id, task_folder)
        thumbnail_path = task_dir / "thumbnail.jpg"

        # Download thumbnail
        from downloaders.youtube import YouTubeDownloader
        downloader = YouTubeDownloader(output_dir=task_dir)

        logger.info(f"Re-downloading thumbnail for task {task_id} from {thumbnail_url}")
        success = await downloader.download_thumbnail(thumbnail_url, thumbnail_path)

        if success and thumbnail_path.exists():
            # Update task with thumbnail path
            await task_persistence.save_task_files(task_id, thumbnail_path=str(thumbnail_path))
            logger.info(f"Thumbnail re-downloaded successfully: {thumbnail_path}")
            return {
                "success": True,
                "message": "封面下载成功",
                "thumbnail_path": str(thumbnail_path)
            }
        else:
            return {
                "success": False,
                "message": "封面下载失败，请检查网络连接"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to re-download thumbnail: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to re-download thumbnail: {e}")


@app.get("/api/tasks/{task_id}/thumbnails")
async def get_task_thumbnails(task_id: str):
    """
    Get thumbnail information for a task.
    Returns both original and AI-generated thumbnail info.
    """
    task = await ensure_task_in_memory(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Get AI thumbnail info from database
    ai_info = await task_persistence.load_ai_thumbnail_info(task_id)

    # Check if original thumbnail exists
    original_exists = task.thumbnail_path and Path(task.thumbnail_path).exists()

    # Check if AI thumbnail exists
    ai_exists = ai_info and ai_info.get("ai_thumbnail_path") and Path(ai_info["ai_thumbnail_path"]).exists()

    return {
        "original": {
            "url": f"/api/tasks/{task_id}/files/thumbnail" if original_exists else None,
            "exists": original_exists,
        },
        "ai_generated": {
            "url": f"/api/tasks/{task_id}/files/ai_thumbnail" if ai_exists else None,
            "exists": ai_exists,
            "title": ai_info.get("ai_thumbnail_title") if ai_info else None,
        },
        "selected": "ai_generated" if (ai_info and ai_info.get("use_ai_thumbnail")) else "original"
    }


class GenerateThumbnailRequest(BaseModel):
    """Request to generate AI thumbnail"""
    custom_title: Optional[str] = None  # Custom title to use (skips AI generation)
    style: Optional[str] = None  # Override style (gradient_bar, top_banner, corner_tag)


@app.post("/api/tasks/{task_id}/generate-ai-thumbnail")
async def generate_ai_thumbnail(task_id: str, request: GenerateThumbnailRequest = None):
    """
    Generate or regenerate AI thumbnail for a task.
    Uses DeepSeek AI to generate catchy Chinese title and overlays it on original thumbnail.
    """
    task = await ensure_task_in_memory(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if original thumbnail exists
    if not task.thumbnail_path or not Path(task.thumbnail_path).exists():
        raise HTTPException(status_code=400, detail="Original thumbnail not found. Please download it first.")

    try:
        from utils.thumbnail_generator import ThumbnailGenerator
        from utils.storage import get_task_directory

        # Get settings
        global_settings = settings_store.load()
        thumbnail_settings = global_settings.thumbnail

        # Get API key from translation settings (DeepSeek)
        api_key = global_settings.translation.api_keys.get_key_for_engine("deepseek")
        if not api_key:
            logger.warning("No DeepSeek API key configured, will use original title for thumbnail")

        # Determine style
        style = request.style if request and request.style else thumbnail_settings.style

        # Create generator
        generator = ThumbnailGenerator(
            api_key=api_key,
            style=style,
            font_name=thumbnail_settings.font_name if thumbnail_settings.font_name else None,
            font_size=thumbnail_settings.font_size,
            text_color=thumbnail_settings.text_color,
            gradient_color=thumbnail_settings.gradient_color,
            gradient_opacity=thumbnail_settings.gradient_opacity,
        )

        # Get output path
        task_folder = await task_persistence.get_task_folder(task_id)
        task_dir = await get_task_directory(task_id, task_folder)
        ai_thumbnail_path = task_dir / "ai_thumbnail.jpg"

        # Get video info for title generation
        video_title = task.video_info.get("title", "") if task.video_info else ""
        video_description = task.video_info.get("description", "") if task.video_info else ""
        keywords = task.video_info.get("tags", []) if task.video_info else []

        # Generate thumbnail
        result = await generator.generate(
            original_thumbnail_path=Path(task.thumbnail_path),
            output_path=ai_thumbnail_path,
            video_title=video_title,
            video_description=video_description,
            keywords=keywords,
            custom_title=request.custom_title if request else None,
        )

        if result.success:
            # Save to database
            await task_persistence.save_ai_thumbnail(
                task_id=task_id,
                ai_thumbnail_path=str(result.output_path),
                ai_thumbnail_title=result.title,
            )

            return {
                "success": True,
                "message": "AI封面生成成功",
                "title": result.title,
                "thumbnail_url": f"/api/tasks/{task_id}/files/ai_thumbnail",
            }
        else:
            return {
                "success": False,
                "message": f"AI封面生成失败: {result.error}",
            }

    except Exception as e:
        logger.error(f"Failed to generate AI thumbnail: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate AI thumbnail: {e}")


class SelectThumbnailRequest(BaseModel):
    """Request to select thumbnail"""
    selected: str = Field(..., description="'original' or 'ai_generated'")


@app.put("/api/tasks/{task_id}/select-thumbnail")
async def select_thumbnail(task_id: str, request: SelectThumbnailRequest):
    """
    Select which thumbnail to use for upload.
    """
    task = await ensure_task_in_memory(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if request.selected not in ["original", "ai_generated"]:
        raise HTTPException(status_code=400, detail="Invalid selection. Must be 'original' or 'ai_generated'")

    use_ai = request.selected == "ai_generated"

    # Update database
    await task_persistence.update_use_ai_thumbnail(task_id, use_ai)

    return {
        "success": True,
        "selected": request.selected,
        "message": f"已选择{'AI生成的' if use_ai else '原始'}封面"
    }


async def run_upload_step(task_id: str):
    """Run the upload step for a task and save state changes to database"""
    task = await ensure_task_in_memory(task_id)
    if not task:
        logger.error(f"Task {task_id} not found for upload")
        return

    # Check if any upload platforms are selected for this task
    opts = task.options
    has_upload_platforms = opts.upload_bilibili or opts.upload_douyin or opts.upload_xiaohongshu
    
    if not has_upload_platforms:
        # No upload platforms selected - stay at pending_upload, don't set to uploaded
        logger.info(f"Task {task_id} has no upload platforms selected, staying at pending_upload")
        task.message = "处理完成（无上传平台）"
        await task_persistence.save_task_status(task_id, "pending_upload", message=task.message)
        return

    try:
        # Update status to uploading
        task.status = TaskStatus.UPLOADING
        task.message = "正在上传..."
        await task_persistence.save_task_status(task_id, "uploading", message="正在上传...")

        # Run upload step
        await pipeline.step_upload(task)

        # Check if any uploads actually succeeded
        successful_uploads = [p for p, r in task.upload_results.items() if r.get("success")] if task.upload_results else []
        
        if successful_uploads:
            # At least one upload succeeded
            task.status = TaskStatus.UPLOADED
            task.message = f"上传完成 ({', '.join(successful_uploads)})"
        else:
            # No successful uploads (all failed or none attempted)
            task.status = TaskStatus.PENDING_UPLOAD
            task.message = "上传失败，请重试"
        
        task.current_step = None

        # Save final state
        await task_persistence.save_task_status(
            task.task_id,
            task.status.value,
            task.progress,
            task.message,
            task.error
        )

        # Save upload results
        if task.upload_results:
            await task_persistence.save_upload_results(task_id, task.upload_results)

        logger.info(f"Task {task_id} upload step completed: {task.status.value}")

    except Exception as e:
        logger.error(f"Upload failed for task {task_id}: {e}")
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.message = f"上传失败: {e}"
        await task_persistence.save_task_status(
            task.task_id,
            "failed",
            task.progress,
            task.message,
            task.error
        )


@app.post("/api/generate-metadata-preview")
@limiter.limit("10/minute")
async def generate_metadata_preview(request: Request, body: dict):
    """
    Preview metadata generation without a task.
    Useful for testing custom signature and settings.

    Request body:
    - title: Original video title
    - transcript: Sample transcript text (optional, can be short)
    - source_url: Source URL (optional)
    - target_language: Target language (default: zh-CN)
    - title_prefix: Prefix to add before title (e.g., '[中字]')
    - custom_signature: Custom signature to preview
    """
    try:
        from metadata import MetadataGenerator

        title = body.get("title", "Sample Video Title")
        transcript = body.get("transcript", "This is a sample video about technology and innovation.")
        source_url = body.get("source_url", "")
        target_language = body.get("target_language", "zh-CN")
        title_prefix = body.get("title_prefix", "")
        custom_signature = body.get("custom_signature", "")

        # Get translation settings
        global_settings = settings_store.load()
        engine = global_settings.translation.engine
        api_key = global_settings.translation.get_api_key()
        model = global_settings.translation.model

        if not api_key and engine not in ["google"]:
            raise HTTPException(
                status_code=400,
                detail=f"API key not configured for {engine}."
            )

        generator = MetadataGenerator(engine=engine, api_key=api_key, model=model)
        result = await generator.generate(
            original_title=title,
            transcript=transcript,
            source_url=source_url,
            source_language="en",
            target_language=target_language,
            title_prefix=title_prefix,
            custom_signature=custom_signature,
            max_keywords=5
        )

        return {
            "success": result.success,
            "title": result.title,
            "title_translated": result.title_translated,
            "description": result.description,
            "keywords": result.keywords,
            "error": result.error
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metadata preview error: {e}")
        return {"success": False, "error": str(e)}


# === Directory API Models ===

class CreateDirectoryRequest(BaseModel):
    """Request to create a new directory"""
    name: str = Field(..., description="Directory name (unique)", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Directory description")


class DirectoryResponse(BaseModel):
    """Directory response"""
    id: int
    name: str
    description: Optional[str]
    task_count: int
    created_at: Optional[str]
    updated_at: Optional[str]


class DirectoryListResponse(BaseModel):
    """Directory list response"""
    directories: List[DirectoryResponse]
    count: int


# === Subscription API Models ===

class CreateSubscriptionRequest(BaseModel):
    """Request to create a new subscription"""
    platform: str = Field(..., description="Platform: youtube, bilibili")
    channel_url: str = Field(..., description="Channel URL or identifier")
    directory: str = Field(..., description="Directory name for organizing tasks")
    check_interval: int = Field(60, description="Check interval in minutes", ge=5, le=1440)
    auto_process: bool = Field(True, description="Automatically create tasks for new videos")
    process_options: Optional[Dict[str, Any]] = Field(None, description="Task processing options template")


class UpdateSubscriptionRequest(BaseModel):
    """Request to update a subscription"""
    check_interval: Optional[int] = Field(None, ge=5, le=1440)
    auto_process: Optional[bool] = None
    process_options: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class SubscriptionResponse(BaseModel):
    """Subscription response"""
    id: str
    platform: str
    channel_id: str
    channel_name: str
    channel_url: Optional[str]
    channel_avatar: Optional[str]
    directory: Optional[str]
    last_video_id: Optional[str]
    last_video_title: Optional[str]
    last_video_published_at: Optional[str]
    check_interval: int
    next_check_at: Optional[str]
    last_checked_at: Optional[str]
    auto_process: bool
    process_options: Optional[Dict[str, Any]]
    enabled: bool
    error_count: int
    last_error: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class ChannelLookupResponse(BaseModel):
    """Channel lookup response"""
    success: bool
    platform: Optional[str] = None
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    channel_url: Optional[str] = None
    channel_avatar: Optional[str] = None
    error: Optional[str] = None


class NewVideosResponse(BaseModel):
    """New videos check response"""
    videos: List[Dict[str, Any]]
    count: int


class FetchHistoricalVideosRequest(BaseModel):
    """Request to fetch videos in a date range"""
    start_date: str = Field(..., description="Start date in ISO format (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date in ISO format (YYYY-MM-DD)")
    max_videos: int = Field(50, description="Maximum videos to fetch", ge=1, le=100)


class VideoItem(BaseModel):
    """Video item for batch operations"""
    video_id: str
    title: str
    url: str
    published_at: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None


class FetchHistoricalVideosResponse(BaseModel):
    """Response with videos in date range"""
    success: bool
    videos: List[VideoItem] = []
    count: int = 0
    start_date: str
    end_date: str
    error: Optional[str] = None


class BatchCreateTasksRequest(BaseModel):
    """Request to create tasks for multiple videos"""
    videos: List[VideoItem] = Field(..., description="Videos to create tasks for")
    process_options: Optional[Dict[str, Any]] = Field(None, description="Task processing options")


class BatchCreateTasksResponse(BaseModel):
    """Response for batch task creation"""
    success: bool
    created_count: int = 0
    failed_count: int = 0
    task_ids: List[str] = []
    errors: List[str] = []


# === Trending Video Models ===

class TrendingVideoResponse(BaseModel):
    """Trending video response"""
    id: int
    video_id: str
    title: str
    channel_name: str
    channel_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: int = 0
    view_count: int = 0
    category: str
    platform: str = "youtube"
    video_url: str
    fetched_at: Optional[str] = None


class TrendingVideosListResponse(BaseModel):
    """List of trending videos"""
    videos: List[TrendingVideoResponse]
    count: int
    category: Optional[str] = None


class TrendingCategoriesResponse(BaseModel):
    """Available trending categories"""
    categories: List[Dict[str, Any]]


class TrendingSettingsResponse(BaseModel):
    """Trending settings"""
    enabled: bool
    update_interval: int
    last_updated: Optional[str] = None
    enabled_categories: List[str]
    max_videos_per_category: int
    time_filter: str = "week"
    sort_by: str = "upload_date"
    min_view_count: int = 10000
    max_duration: int = 1800
    has_youtube_api_key: bool = False  # Whether API key is configured


class TrendingSettingsUpdateRequest(BaseModel):
    """Update trending settings"""
    enabled: Optional[bool] = None
    update_interval: Optional[int] = None
    enabled_categories: Optional[List[str]] = None
    max_videos_per_category: Optional[int] = None
    time_filter: Optional[str] = None
    sort_by: Optional[str] = None
    min_view_count: Optional[int] = None
    max_duration: Optional[int] = None
    youtube_api_key: Optional[str] = None  # YouTube Data API v3 key


# YouTube Search Models
class YouTubeSearchRequest(BaseModel):
    """YouTube search request"""
    query: str
    max_results: int = 20


class YouTubeSearchResult(BaseModel):
    """Single YouTube search result"""
    video_id: str
    title: str
    channel_name: str
    channel_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: int = 0
    view_count: int = 0
    published_at: Optional[str] = None
    video_url: str


class YouTubeSearchResponse(BaseModel):
    """YouTube search response"""
    success: bool
    results: List[YouTubeSearchResult] = []
    count: int = 0
    error: Optional[str] = None


class TrendingRefreshResponse(BaseModel):
    """Refresh trending videos response"""
    success: bool
    message: str
    categories_updated: List[str] = []
    total_videos: int = 0


class TrendingBatchCreateRequest(BaseModel):
    """Batch create tasks from trending videos request"""
    video_ids: List[str] = Field(..., description="List of video IDs to create tasks for")
    process_options: Optional[Dict[str, Any]] = Field(None, description="Task processing options")


class TrendingBatchCreateResponse(BaseModel):
    """Batch create tasks from trending videos"""
    success: bool
    created_count: int = 0
    failed_count: int = 0
    task_ids: List[str] = []
    errors: List[str] = []


# === Directory API Endpoints ===

@app.get("/api/directories", response_model=DirectoryListResponse)
async def list_directories():
    """Get all directories"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = DirectoryRepository(session)
        directories = await repo.get_all()

        return DirectoryListResponse(
            directories=[DirectoryResponse(**d.to_dict()) for d in directories],
            count=len(directories)
        )


@app.post("/api/directories", response_model=DirectoryResponse)
async def create_directory(request: CreateDirectoryRequest):
    """Create a new directory"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = DirectoryRepository(session)

        # Check if directory already exists
        if await repo.exists(request.name):
            raise HTTPException(
                status_code=409,
                detail=f"Directory '{request.name}' already exists"
            )

        directory = await repo.create(
            name=request.name,
            description=request.description
        )
        await session.commit()

        return DirectoryResponse(**directory.to_dict())


@app.get("/api/directories/{name}/exists")
async def check_directory_exists(name: str):
    """Check if a directory name already exists"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = DirectoryRepository(session)
        exists = await repo.exists(name)
        return {"exists": exists, "name": name}


@app.post("/api/directories/sync-counts")
async def sync_directory_task_counts():
    """Sync task_count for all directories based on actual tasks"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        dir_repo = DirectoryRepository(session)
        task_repo = TaskRepository(session)
        updated = await dir_repo.sync_task_counts(task_repo)
        await session.commit()
        return {
            "success": True,
            "updated": updated,
            "message": f"Synced {len(updated)} directories"
        }


@app.delete("/api/directories/{directory_id}")
async def delete_directory(directory_id: int):
    """Delete a directory (only if it has no tasks)"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = DirectoryRepository(session)
        directory = await repo.get(directory_id)

        if not directory:
            raise HTTPException(status_code=404, detail="Directory not found")

        if directory.task_count > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete directory with {directory.task_count} tasks"
            )

        await repo.delete(directory_id)
        await session.commit()

        return {"success": True, "message": "Directory deleted"}


# === Trending Videos API Endpoints ===

# Import categories from scheduler
from trending.scheduler import YOUTUBE_CATEGORIES as TRENDING_CATEGORIES


@app.get("/api/trending/categories", response_model=TrendingCategoriesResponse)
async def get_trending_categories():
    """Get available trending video categories"""
    return TrendingCategoriesResponse(
        categories=[
            {
                "id": k,  # Use string key (tech, gaming, etc.) for filtering
                "youtube_id": v.get("id"),  # Keep numeric YouTube category ID
                "name": v.get("name"),
                "display_name": v.get("display_name"),
                "search_queries": v.get("search_queries", []),
            }
            for k, v in TRENDING_CATEGORIES.items()
        ]
    )


@app.get("/api/trending/videos", response_model=TrendingVideosListResponse)
async def get_trending_videos(category: Optional[str] = None, limit: int = 50):
    """Get trending videos, optionally filtered by category"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = TrendingRepository(session)

        if category:
            videos = await repo.get_by_category(category, limit=limit)
        else:
            videos = await repo.get_all(limit=limit)

        return TrendingVideosListResponse(
            videos=[TrendingVideoResponse(**v.to_dict()) for v in videos],
            count=len(videos),
            category=category,
        )


@app.get("/api/trending/settings", response_model=TrendingSettingsResponse)
async def get_trending_settings():
    """Get trending video settings"""
    settings = settings_store.load()
    return TrendingSettingsResponse(
        enabled=settings.trending.enabled,
        update_interval=settings.trending.update_interval,
        last_updated=settings.trending.last_updated,
        enabled_categories=settings.trending.enabled_categories,
        max_videos_per_category=settings.trending.max_videos_per_category,
        time_filter=settings.trending.time_filter,
        sort_by=settings.trending.sort_by,
        min_view_count=settings.trending.min_view_count,
        max_duration=settings.trending.max_duration,
        has_youtube_api_key=bool(settings.trending.youtube_api_key),
    )


@app.put("/api/trending/settings", response_model=TrendingSettingsResponse)
async def update_trending_settings(request: TrendingSettingsUpdateRequest):
    """Update trending video settings"""
    updates = {}
    if request.enabled is not None:
        updates["enabled"] = request.enabled
    if request.update_interval is not None:
        updates["update_interval"] = request.update_interval
    if request.enabled_categories is not None:
        updates["enabled_categories"] = request.enabled_categories
    if request.max_videos_per_category is not None:
        updates["max_videos_per_category"] = request.max_videos_per_category
    if request.time_filter is not None:
        updates["time_filter"] = request.time_filter
    if request.sort_by is not None:
        updates["sort_by"] = request.sort_by
    if request.min_view_count is not None:
        updates["min_view_count"] = request.min_view_count
    if request.max_duration is not None:
        updates["max_duration"] = request.max_duration
    if request.youtube_api_key is not None and request.youtube_api_key != "***":
        # Don't overwrite with masked value
        updates["youtube_api_key"] = request.youtube_api_key

    if updates:
        settings_store.update({"trending": updates})

    settings = settings_store.load()
    return TrendingSettingsResponse(
        enabled=settings.trending.enabled,
        update_interval=settings.trending.update_interval,
        last_updated=settings.trending.last_updated,
        enabled_categories=settings.trending.enabled_categories,
        max_videos_per_category=settings.trending.max_videos_per_category,
        time_filter=settings.trending.time_filter,
        sort_by=settings.trending.sort_by,
        min_view_count=settings.trending.min_view_count,
        max_duration=settings.trending.max_duration,
        has_youtube_api_key=bool(settings.trending.youtube_api_key),
    )


@app.post("/api/trending/refresh", response_model=TrendingRefreshResponse)
async def refresh_trending_videos():
    """Manually refresh trending videos"""
    from trending import trending_scheduler

    try:
        # Clear all old videos first
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = TrendingRepository(session)
            await repo.clear_all()
            await session.commit()

        # Fetch new trending videos
        results = await trending_scheduler.fetch_trending_videos()
        total = sum(len(videos) for videos in results.values())

        # Save to database
        async with session_factory() as session:
            repo = TrendingRepository(session)
            for category, videos in results.items():
                for video_data in videos:
                    await repo.upsert_video(category=category, **video_data)
            await session.commit()

        return TrendingRefreshResponse(
            success=True,
            message=f"Refreshed {total} trending videos",
            categories_updated=list(results.keys()),
            total_videos=total,
        )
    except Exception as e:
        logger.error(f"Failed to refresh trending videos: {e}")
        return TrendingRefreshResponse(
            success=False,
            message=str(e),
        )


@app.post("/api/trending/refresh/{category}", response_model=TrendingRefreshResponse)
async def refresh_trending_category(category: str):
    """Manually refresh trending videos for a specific category only"""
    from trending import trending_scheduler

    try:
        # Validate category
        from trending.scheduler import YOUTUBE_CATEGORIES
        if category not in YOUTUBE_CATEGORIES:
            return TrendingRefreshResponse(
                success=False,
                message=f"Unknown category: {category}. Valid categories: {list(YOUTUBE_CATEGORIES.keys())}",
            )

        # Clear only this category's videos first
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = TrendingRepository(session)
            await repo.clear_category(category)
            await session.commit()

        # Fetch new trending videos for this category only
        videos = await trending_scheduler.fetch_single_category(category)

        # Save to database
        async with session_factory() as session:
            repo = TrendingRepository(session)
            for video_data in videos:
                await repo.upsert_video(category=category, **video_data)
            await session.commit()

        return TrendingRefreshResponse(
            success=True,
            message=f"Refreshed {len(videos)} trending videos for {category}",
            categories_updated=[category],
            total_videos=len(videos),
        )
    except Exception as e:
        logger.error(f"Failed to refresh trending videos for {category}: {e}")
        return TrendingRefreshResponse(
            success=False,
            message=str(e),
        )


@app.post("/api/youtube/search", response_model=YouTubeSearchResponse)
async def youtube_search(request: YouTubeSearchRequest):
    """Search YouTube videos using YouTube Data API v3"""
    import aiohttp
    import re

    settings = settings_store.load()
    api_key = settings.trending.youtube_api_key

    if not api_key:
        return YouTubeSearchResponse(
            success=False,
            error="YouTube API key not configured. Please add your API key in Settings > YouTube.",
        )

    try:
        # Search for videos
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part": "snippet",
            "q": request.query,
            "type": "video",
            "maxResults": min(request.max_results, 50),
            "key": api_key,
        }

        async with aiohttp.ClientSession() as session:
            # Step 1: Search for videos
            async with session.get(search_url, params=search_params) as response:
                if response.status != 200:
                    error_data = await response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    logger.error(f"YouTube search API error: {response.status} - {error_msg}")
                    return YouTubeSearchResponse(
                        success=False,
                        error=f"YouTube API error: {error_msg}",
                    )

                search_data = await response.json()

            # Get video IDs for fetching details (duration, view count)
            video_ids = [item["id"]["videoId"] for item in search_data.get("items", [])]

            if not video_ids:
                return YouTubeSearchResponse(success=True, results=[], count=0)

            # Step 2: Get video details (duration, statistics)
            videos_url = "https://www.googleapis.com/youtube/v3/videos"
            videos_params = {
                "part": "contentDetails,statistics",
                "id": ",".join(video_ids),
                "key": api_key,
            }

            async with session.get(videos_url, params=videos_params) as response:
                if response.status != 200:
                    # If details fail, return basic info without duration/views
                    video_details = {}
                else:
                    details_data = await response.json()
                    video_details = {
                        item["id"]: item
                        for item in details_data.get("items", [])
                    }

        # Parse ISO 8601 duration to seconds
        def parse_duration(duration_str: str) -> int:
            if not duration_str or not duration_str.startswith("PT"):
                return 0
            hours = minutes = seconds = 0
            h_match = re.search(r'(\d+)H', duration_str)
            m_match = re.search(r'(\d+)M', duration_str)
            s_match = re.search(r'(\d+)S', duration_str)
            if h_match:
                hours = int(h_match.group(1))
            if m_match:
                minutes = int(m_match.group(1))
            if s_match:
                seconds = int(s_match.group(1))
            return hours * 3600 + minutes * 60 + seconds

        # Build results
        results = []
        for item in search_data.get("items", []):
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            details = video_details.get(video_id, {})

            # Get duration and view count from details
            duration = 0
            view_count = 0
            if details:
                content_details = details.get("contentDetails", {})
                statistics = details.get("statistics", {})
                duration = parse_duration(content_details.get("duration", ""))
                view_count = int(statistics.get("viewCount", 0))

            results.append(YouTubeSearchResult(
                video_id=video_id,
                title=snippet.get("title", ""),
                channel_name=snippet.get("channelTitle", ""),
                channel_url=f"https://www.youtube.com/channel/{snippet.get('channelId', '')}",
                thumbnail_url=snippet.get("thumbnails", {}).get("high", {}).get("url")
                    or snippet.get("thumbnails", {}).get("medium", {}).get("url")
                    or f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                duration=duration,
                view_count=view_count,
                published_at=snippet.get("publishedAt"),
                video_url=f"https://www.youtube.com/watch?v={video_id}",
            ))

        return YouTubeSearchResponse(
            success=True,
            results=results,
            count=len(results),
        )

    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return YouTubeSearchResponse(
            success=False,
            error=str(e),
        )


@app.post("/api/trending/batch-create", response_model=TrendingBatchCreateResponse)
async def batch_create_trending_tasks(request: TrendingBatchCreateRequest):
    """Create tasks for multiple trending videos"""
    session_factory = get_session_factory()
    created_ids = []
    errors = []

    video_ids = request.video_ids
    process_options = request.process_options

    # Load global settings as defaults
    global_settings = settings_store.load()

    async with session_factory() as session:
        repo = TrendingRepository(session)

        for video_id in video_ids:
            try:
                video = await repo.get_by_video_id(video_id)
                if not video:
                    errors.append(f"Video not found: {video_id}")
                    continue

                # Create task
                task_id = str(uuid.uuid4())[:8]
                opts = process_options or {}

                # Use global settings as defaults, override with opts if provided
                options = ProcessingOptions(
                    source_url=video.video_url,
                    source_platform="youtube",
                    # Transcription
                    source_language=opts.get("source_language", global_settings.subtitle.source_language),
                    target_language=opts.get("target_language", global_settings.subtitle.target_language),
                    whisper_backend=opts.get("whisper_backend", global_settings.video.whisper_backend),
                    whisper_model=opts.get("whisper_model", global_settings.video.whisper_model),
                    whisper_device=opts.get("whisper_device", global_settings.video.whisper_device),
                    # Translation
                    skip_translation=opts.get("skip_translation", False),
                    translation_engine=opts.get("translation_engine", global_settings.translation.engine),
                    # Subtitles
                    add_subtitles=opts.get("add_subtitles", global_settings.subtitle.enabled),
                    subtitle_style=opts.get("subtitle_style", global_settings.subtitle.style),
                    dual_subtitles=opts.get("dual_subtitles", global_settings.subtitle.dual_subtitles),
                    use_existing_subtitles=opts.get("use_existing_subtitles", global_settings.video.prefer_existing_subtitles),
                    subtitle_language=opts.get("subtitle_language"),
                    subtitle_preset=opts.get("subtitle_preset", global_settings.subtitle.default_preset),
                    # TTS
                    add_tts=opts.get("add_tts", global_settings.audio.generate_tts),
                    tts_service=opts.get("tts_service", global_settings.tts.engine),
                    tts_voice=opts.get("tts_voice", global_settings.tts.voice),
                    tts_rate=opts.get("tts_rate", global_settings.tts.rate),
                    voice_cloning_mode=opts.get("voice_cloning_mode", global_settings.tts.voice_cloning_mode),
                    tts_ref_audio=opts.get("tts_ref_audio", global_settings.tts.ref_audio_path),
                    tts_ref_text=opts.get("tts_ref_text", global_settings.tts.ref_audio_text),
                    original_audio_volume=opts.get("original_audio_volume", global_settings.audio.original_volume),
                    tts_audio_volume=opts.get("tts_audio_volume", global_settings.audio.tts_volume),
                    replace_original_audio=opts.get("replace_original_audio", global_settings.audio.replace_original),
                    # Video quality - default to highest available for trending
                    video_quality=opts.get("video_quality", "2160p"),  # Will fallback to best available
                    format_id=opts.get("format_id"),
                    video_quality_label=opts.get("video_quality_label"),
                    # Metadata
                    custom_title=opts.get("custom_title"),
                    custom_description=opts.get("custom_description"),
                    custom_tags=opts.get("custom_tags", []),
                    metadata_preset_id=opts.get("metadata_preset_id"),
                    use_ai_preset_selection=opts.get("use_ai_preset_selection", global_settings.metadata.default_use_ai_preset_selection),
                    # Upload targets
                    upload_bilibili=opts.get("upload_bilibili", global_settings.auto_upload_bilibili),
                    upload_douyin=opts.get("upload_douyin", global_settings.auto_upload_douyin),
                    upload_xiaohongshu=opts.get("upload_xiaohongshu", global_settings.auto_upload_xiaohongshu),
                    bilibili_account_uid=opts.get("bilibili_account_uid"),
                    # Proofreading
                    enable_proofreading=opts.get("enable_proofreading", global_settings.proofreading.enabled),
                    proofreading_auto_pause=opts.get("proofreading_auto_pause", global_settings.proofreading.auto_pause),
                    proofreading_min_confidence=opts.get("proofreading_min_confidence", global_settings.proofreading.min_confidence),
                    proofreading_auto_optimize=opts.get("proofreading_auto_optimize", global_settings.proofreading.auto_optimize),
                    proofreading_optimization_level=opts.get("proofreading_optimization_level", global_settings.proofreading.optimization_level),
                    # Directory
                    directory=opts.get("directory"),
                )

                task = ProcessingTask(task_id=task_id, options=options)
                task.message = f"从热门视频创建: {video.title[:50]}"

                # Store task
                pipeline.tasks[task_id] = task

                # Persist to database
                options_dict = {
                    "source_url": options.source_url,
                    "source_platform": options.source_platform,
                    # Transcription
                    "source_language": options.source_language,
                    "target_language": options.target_language,
                    "whisper_backend": options.whisper_backend,
                    "whisper_model": options.whisper_model,
                    "whisper_device": options.whisper_device,
                    # Translation
                    "skip_translation": options.skip_translation,
                    "translation_engine": options.translation_engine,
                    # Subtitles
                    "add_subtitles": options.add_subtitles,
                    "subtitle_style": options.subtitle_style,
                    "dual_subtitles": options.dual_subtitles,
                    "use_existing_subtitles": options.use_existing_subtitles,
                    "subtitle_language": options.subtitle_language,
                    "subtitle_preset": options.subtitle_preset,
                    # TTS
                    "add_tts": options.add_tts,
                    "tts_service": options.tts_service,
                    "tts_voice": options.tts_voice,
                    "tts_rate": options.tts_rate,
                    "voice_cloning_mode": options.voice_cloning_mode,
                    "tts_ref_audio": options.tts_ref_audio,
                    "tts_ref_text": options.tts_ref_text,
                    "original_audio_volume": options.original_audio_volume,
                    "tts_audio_volume": options.tts_audio_volume,
                    "replace_original_audio": options.replace_original_audio,
                    # Video quality
                    "video_quality": options.video_quality,
                    "format_id": options.format_id,
                    "video_quality_label": options.video_quality_label,
                    # Metadata
                    "custom_title": options.custom_title,
                    "custom_description": options.custom_description,
                    "custom_tags": options.custom_tags,
                    "metadata_preset_id": options.metadata_preset_id,
                    "use_ai_preset_selection": options.use_ai_preset_selection,
                    # Upload targets
                    "upload_bilibili": options.upload_bilibili,
                    "upload_douyin": options.upload_douyin,
                    "upload_xiaohongshu": options.upload_xiaohongshu,
                    # Proofreading
                    "enable_proofreading": options.enable_proofreading,
                    "proofreading_auto_pause": options.proofreading_auto_pause,
                    "proofreading_min_confidence": options.proofreading_min_confidence,
                    "proofreading_auto_optimize": options.proofreading_auto_optimize,
                    "proofreading_optimization_level": options.proofreading_optimization_level,
                    # Directory
                    "directory": options.directory,
                }
                await task_persistence.save_task_created(
                    task_id, options_dict, "pending", task.message,
                    directory=options.directory
                )

                # Queue for processing
                await task_executor.submit(task)
                created_ids.append(task_id)

            except Exception as e:
                logger.error(f"Failed to create task for video {video_id}: {e}")
                errors.append(f"Failed for {video_id}: {str(e)}")

    return TrendingBatchCreateResponse(
        success=len(errors) == 0,
        created_count=len(created_ids),
        failed_count=len(errors),
        task_ids=created_ids,
        errors=errors,
    )


# === TikTok Discovery API Endpoints ===

@app.get("/api/tiktok/tags")
async def get_tiktok_tags():
    """Get available TikTok tags"""
    from tiktok.scheduler import DEFAULT_TIKTOK_TAGS, TIKTOK_REGIONS
    
    settings = settings_store.load()
    enabled_tags = settings.tiktok.enabled_tags
    
    return {
        "tags": [
            {
                "id": tag,
                "name": f"#{tag}",
                "enabled": tag in enabled_tags,
            }
            for tag in DEFAULT_TIKTOK_TAGS
        ],
        "regions": [
            {"code": code, "name": name}
            for code, name in TIKTOK_REGIONS.items()
        ],
    }


@app.get("/api/tiktok/videos")
async def get_tiktok_videos(tag: Optional[str] = None, limit: int = 50):
    """Get TikTok videos from database"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = TrendingRepository(session)
        
        if tag:
            # Get videos for specific tag
            videos = await repo.get_by_category(f"tiktok_{tag}", limit=limit)
        else:
            # Get all TikTok videos
            from sqlalchemy import select
            from database.models import TrendingVideoModel
            stmt = select(TrendingVideoModel).where(
                TrendingVideoModel.category.like("tiktok_%")
            ).order_by(TrendingVideoModel.fetched_at.desc()).limit(limit)
            result = await session.execute(stmt)
            videos = result.scalars().all()
        
        return {
            "videos": [v.to_dict() for v in videos],
            "total": len(videos),
        }


@app.get("/api/tiktok/settings")
async def get_tiktok_settings():
    """Get TikTok discovery settings"""
    settings = settings_store.load()
    return settings.tiktok.to_dict()


@app.put("/api/tiktok/settings")
async def update_tiktok_settings(updates: Dict[str, Any]):
    """Update TikTok discovery settings"""
    settings = settings_store.update({"tiktok": updates})
    return settings.tiktok.to_dict()


@app.post("/api/tiktok/refresh")
async def refresh_tiktok_videos():
    """Manually refresh all TikTok videos"""
    from tiktok import tiktok_scheduler
    
    try:
        # Clear all TikTok videos first
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = TrendingRepository(session)
            # Delete videos with category starting with "tiktok_"
            from sqlalchemy import delete
            from database.models import TrendingVideoModel
            stmt = delete(TrendingVideoModel).where(
                TrendingVideoModel.category.like("tiktok_%")
            )
            await session.execute(stmt)
            await session.commit()
        
        # Fetch new TikTok videos
        results = await tiktok_scheduler.fetch_tiktok_videos()
        total = sum(len(videos) for videos in results.values())
        
        return {
            "success": True,
            "message": f"Refreshed {total} TikTok videos",
            "tags_updated": list(results.keys()),
            "total_videos": total,
        }
    except Exception as e:
        logger.error(f"Failed to refresh TikTok videos: {e}")
        return {
            "success": False,
            "message": str(e),
        }


@app.post("/api/tiktok/refresh/{tag}")
async def refresh_tiktok_tag(tag: str):
    """Manually refresh TikTok videos for a specific tag"""
    from tiktok import tiktok_scheduler
    
    try:
        # Clear videos for this tag
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = TrendingRepository(session)
            await repo.clear_category(f"tiktok_{tag}")
            await session.commit()
        
        # Fetch new videos for this tag
        videos = await tiktok_scheduler.fetch_single_tag(tag)
        
        return {
            "success": True,
            "message": f"Refreshed {len(videos)} TikTok videos for #{tag}",
            "tags_updated": [tag],
            "total_videos": len(videos),
        }
    except Exception as e:
        logger.error(f"Failed to refresh TikTok videos for #{tag}: {e}")
        return {
            "success": False,
            "message": str(e),
        }


# === Subscription API Endpoints ===

@app.get("/api/subscriptions/platforms")
async def get_supported_platforms():
    """Get list of supported platforms for subscriptions"""
    return {
        "platforms": [
            {"id": k, **v} for k, v in SUPPORTED_PLATFORMS.items()
        ]
    }


@app.post("/api/subscriptions/lookup", response_model=ChannelLookupResponse)
async def lookup_channel(platform: str, url: str):
    """Look up channel information from URL"""
    fetcher = get_fetcher(platform)
    if not fetcher:
        return ChannelLookupResponse(
            success=False,
            error=f"Unsupported platform: {platform}"
        )

    try:
        channel_info = await fetcher.lookup_channel(url)
        if not channel_info:
            return ChannelLookupResponse(
                success=False,
                error="Channel not found"
            )

        return ChannelLookupResponse(
            success=True,
            platform=channel_info.platform,
            channel_id=channel_info.channel_id,
            channel_name=channel_info.channel_name,
            channel_url=channel_info.channel_url,
            channel_avatar=channel_info.avatar_url,
        )
    except Exception as e:
        logger.error(f"Channel lookup error: {e}")
        return ChannelLookupResponse(
            success=False,
            error=str(e)
        )
    finally:
        await fetcher.close()


@app.get("/api/subscriptions", response_model=List[SubscriptionResponse])
async def list_subscriptions(
    platform: Optional[str] = None,
    enabled_only: bool = False,
    limit: int = 100
):
    """Get all subscriptions"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)
        subscriptions = await repo.get_all(
            enabled_only=enabled_only,
            platform=platform,
            limit=limit
        )

        return [
            SubscriptionResponse(**sub.to_dict())
            for sub in subscriptions
        ]


@app.post("/api/subscriptions", response_model=SubscriptionResponse)
async def create_subscription(request: CreateSubscriptionRequest):
    """Create a new subscription"""
    # Validate platform
    if request.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported platform: {request.platform}. Supported: {list(SUPPORTED_PLATFORMS.keys())}"
        )

    # Look up channel info
    fetcher = get_fetcher(request.platform)
    if not fetcher:
        raise HTTPException(status_code=400, detail="Failed to create fetcher")

    try:
        channel_info = await fetcher.lookup_channel(request.channel_url)
        if not channel_info:
            raise HTTPException(status_code=404, detail="Channel not found")

        # Get latest video to set as baseline (so only truly new videos trigger tasks)
        latest_videos = await fetcher.get_latest_videos(channel_info.channel_id, limit=1)
        latest_video = latest_videos[0] if latest_videos else None
    finally:
        await fetcher.close()

    # Check if already subscribed
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)
        dir_repo = DirectoryRepository(session)

        existing = await repo.get_by_channel(request.platform, channel_info.channel_id)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Already subscribed to this channel: {channel_info.channel_name}"
            )

        # Ensure directory exists (create if not)
        await dir_repo.get_or_create(request.directory)

        # Create subscription with latest video as baseline
        subscription_id = str(uuid.uuid4())[:8]
        subscription = await repo.create(
            subscription_id=subscription_id,
            platform=channel_info.platform,
            channel_id=channel_info.channel_id,
            channel_name=channel_info.channel_name,
            channel_url=channel_info.channel_url,
            channel_avatar=channel_info.avatar_url,
            directory=request.directory,
            check_interval=request.check_interval,
            auto_process=request.auto_process,
            process_options=request.process_options,
            # Set latest video as baseline so only truly new videos trigger tasks
            last_video_id=latest_video.video_id if latest_video else None,
            last_video_title=latest_video.title if latest_video else None,
            last_video_published_at=latest_video.published_at if latest_video else None,
        )

        await session.commit()

        return SubscriptionResponse(**subscription.to_dict())


@app.get("/api/subscriptions/{subscription_id}", response_model=SubscriptionResponse)
async def get_subscription(subscription_id: str):
    """Get a single subscription"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)
        subscription = await repo.get(subscription_id)

        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")

        return SubscriptionResponse(**subscription.to_dict())


@app.put("/api/subscriptions/{subscription_id}", response_model=SubscriptionResponse)
async def update_subscription(subscription_id: str, request: UpdateSubscriptionRequest):
    """Update a subscription"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)
        subscription = await repo.get(subscription_id)

        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")

        # Build updates
        updates = {}
        if request.check_interval is not None:
            updates["check_interval"] = request.check_interval
        if request.auto_process is not None:
            updates["auto_process"] = request.auto_process
        if request.process_options is not None:
            updates["process_options"] = request.process_options
        if request.enabled is not None:
            updates["enabled"] = request.enabled

        if updates:
            subscription = await repo.update(subscription_id, **updates)
            await session.commit()

        return SubscriptionResponse(**subscription.to_dict())


@app.delete("/api/subscriptions/{subscription_id}")
async def delete_subscription(subscription_id: str):
    """Delete a subscription"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)

        deleted = await repo.delete(subscription_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Subscription not found")

        await session.commit()

        return {"success": True, "message": "Subscription deleted"}


@app.post("/api/subscriptions/{subscription_id}/check", response_model=NewVideosResponse)
async def check_subscription_now(subscription_id: str):
    """Manually trigger a check for new videos"""
    from datetime import timedelta

    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)
        subscription = await repo.get(subscription_id)

        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")

        try:
            videos = await subscription_scheduler.check_now(subscription)

            # Update last checked time
            next_check = datetime.now() + timedelta(minutes=subscription.check_interval)
            await repo.update_check_status(
                subscription_id,
                next_check_at=next_check,
                error=None
            )
            await session.commit()

            # Convert videos to dict format
            video_list = [
                {
                    "video_id": v.video_id,
                    "title": v.title,
                    "url": v.url,
                    "published_at": v.published_at.isoformat() if v.published_at else None,
                    "thumbnail_url": v.thumbnail_url,
                }
                for v in videos
            ]

            return NewVideosResponse(videos=video_list, count=len(video_list))

        except Exception as e:
            logger.error(f"Check subscription error: {e}")
            await repo.update_check_status(
                subscription_id,
                next_check_at=datetime.now() + timedelta(minutes=5),
                error=str(e)
            )
            await session.commit()
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/subscriptions/{subscription_id}/enable")
async def enable_subscription(subscription_id: str):
    """Enable a subscription"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)
        subscription = await repo.set_enabled(subscription_id, True)

        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")

        await session.commit()

        return {"success": True, "enabled": True}


@app.post("/api/subscriptions/{subscription_id}/disable")
async def disable_subscription(subscription_id: str):
    """Disable a subscription"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)
        subscription = await repo.set_enabled(subscription_id, False)

        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")

        await session.commit()

        return {"success": True, "enabled": False}


@app.post("/api/subscriptions/{subscription_id}/fetch-videos", response_model=FetchHistoricalVideosResponse)
async def fetch_historical_videos(subscription_id: str, request: FetchHistoricalVideosRequest):
    """
    Fetch videos from a subscribed channel within a date range.
    The date range is limited to the last 3 months from today.
    """
    from datetime import timedelta

    # Parse dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError as e:
        return FetchHistoricalVideosResponse(
            success=False,
            start_date=request.start_date,
            end_date=request.end_date,
            error=f"Invalid date format: {e}. Use YYYY-MM-DD."
        )

    # Validate date range (max 3 months back)
    today = datetime.now().replace(hour=23, minute=59, second=59)
    three_months_ago = today - timedelta(days=90)

    if start_date < three_months_ago:
        start_date = three_months_ago
        logger.info(f"Start date adjusted to 3 months ago: {start_date.date()}")

    if end_date > today:
        end_date = today
        logger.info(f"End date adjusted to today: {end_date.date()}")

    if start_date > end_date:
        return FetchHistoricalVideosResponse(
            success=False,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            error="Start date must be before or equal to end date"
        )

    # Get subscription
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)
        subscription = await repo.get(subscription_id)

        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")

        # Check platform support for historical video fetching
        supported_platforms = ["youtube", "tiktok"]
        if subscription.platform not in supported_platforms:
            return FetchHistoricalVideosResponse(
                success=False,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                error=f"Historical video fetching is only supported for: {', '.join(supported_platforms)}. {subscription.platform} is not supported yet."
            )

        # Get fetcher
        fetcher = get_fetcher(subscription.platform)
        if not fetcher:
            return FetchHistoricalVideosResponse(
                success=False,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                error=f"Failed to create fetcher for {subscription.platform}"
            )

        try:
            # Fetch videos in date range
            videos = await fetcher.get_videos_in_date_range(
                channel_id=subscription.channel_id,
                start_date=start_date,
                end_date=end_date,
                max_videos=request.max_videos
            )

            # Convert to response format
            video_items = [
                VideoItem(
                    video_id=v.video_id,
                    title=v.title,
                    url=v.url,
                    published_at=v.published_at.isoformat() if v.published_at else None,
                    thumbnail_url=v.thumbnail_url,
                    duration=v.duration,
                )
                for v in videos
            ]

            return FetchHistoricalVideosResponse(
                success=True,
                videos=video_items,
                count=len(video_items),
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

        except Exception as e:
            logger.error(f"Error fetching historical videos: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return FetchHistoricalVideosResponse(
                success=False,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                error=str(e)
            )
        finally:
            await fetcher.close()


@app.post("/api/subscriptions/{subscription_id}/batch-create-tasks", response_model=BatchCreateTasksResponse)
async def batch_create_tasks(subscription_id: str, request: BatchCreateTasksRequest):
    """
    Create processing tasks for multiple videos from a subscription.
    Uses the subscription's process_options as defaults, which can be overridden.
    """
    # Get subscription
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = SubscriptionRepository(session)
        subscription = await repo.get(subscription_id)

        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")

        # Merge process options (request options override subscription defaults)
        base_options = subscription.process_options or {}
        override_options = request.process_options or {}
        process_options = {**base_options, **override_options}

        # Interpret processing_mode to set proper flags
        processing_mode = process_options.get("processing_mode", "full")
        logger.info(f"Batch create processing_mode: {processing_mode}")
        
        if processing_mode == "direct":
            process_options["skip_translation"] = True
            process_options["add_subtitles"] = False
            process_options["add_tts"] = False
            process_options["dual_subtitles"] = False
        elif processing_mode == "subtitle":
            process_options["skip_translation"] = False
            process_options["add_subtitles"] = True
            process_options["add_tts"] = False
        elif processing_mode == "auto":
            process_options.setdefault("skip_translation", False)
            process_options.setdefault("add_subtitles", True)
            process_options.setdefault("add_tts", False)
        else:  # "full" or default
            process_options.setdefault("skip_translation", False)
            process_options.setdefault("add_subtitles", True)
            process_options.setdefault("add_tts", True)
        
        logger.info(f"Batch task flags: skip_translation={process_options.get('skip_translation')}, add_subtitles={process_options.get('add_subtitles')}, add_tts={process_options.get('add_tts')}")

        # Default options if not specified
        defaults = {
            "source_language": "auto",
            "target_language": "zh-CN",
            "whisper_backend": "auto",
            "whisper_model": "auto",
            "whisper_device": "auto",
            "translation_engine": "google",
            "video_quality": "best",
            "use_existing_subtitles": True,
            "tts_service": "edge",
            "tts_voice": "zh-CN-XiaoxiaoNeural",
            "replace_original_audio": False,
            "upload_bilibili": False,
            "upload_douyin": False,
            "upload_xiaohongshu": False,
        }

        for key, value in defaults.items():
            if key not in process_options:
                process_options[key] = value

        # Create tasks for each video
        task_ids = []
        errors = []

        for video in request.videos:
            try:
                # Create unique task ID
                task_id = str(uuid.uuid4())[:8]

                # Build ProcessingOptions
                options = ProcessingOptions(
                    source_url=video.url,
                    source_platform=subscription.platform,
                    source_language=process_options.get("source_language", "auto"),
                    target_language=process_options.get("target_language", "zh-CN"),
                    skip_translation=process_options.get("skip_translation", False),
                    whisper_backend=process_options.get("whisper_backend", "auto"),
                    whisper_model=process_options.get("whisper_model", "auto"),
                    whisper_device=process_options.get("whisper_device", "auto"),
                    translation_engine=process_options.get("translation_engine", "google"),
                    video_quality=process_options.get("video_quality", "best"),
                    add_subtitles=process_options.get("add_subtitles", True),
                    dual_subtitles=process_options.get("dual_subtitles", True),
                    use_existing_subtitles=process_options.get("use_existing_subtitles", True),
                    subtitle_preset=process_options.get("subtitle_preset"),
                    add_tts=process_options.get("add_tts", True),
                    tts_service=process_options.get("tts_service", "edge"),
                    tts_voice=process_options.get("tts_voice", "zh-CN-XiaoxiaoNeural"),
                    replace_original_audio=process_options.get("replace_original_audio", False),
                    metadata_preset_id=process_options.get("metadata_preset_id"),
                    use_ai_preset_selection=process_options.get("use_ai_preset_selection", False),
                    upload_bilibili=process_options.get("upload_bilibili", False),
                    upload_douyin=process_options.get("upload_douyin", False),
                    upload_xiaohongshu=process_options.get("upload_xiaohongshu", False),
                    bilibili_account_uid=process_options.get("bilibili_account_uid"),
                )

                # Create ProcessingTask and store in pipeline
                task = ProcessingTask(task_id=task_id, options=options)
                task.message = "任务已创建（批量导入）"
                pipeline.tasks[task_id] = task

                # Build options dict for database persistence
                options_dict = {
                    "source_url": options.source_url,
                    "source_platform": options.source_platform,
                    "source_language": options.source_language,
                    "target_language": options.target_language,
                    "whisper_backend": options.whisper_backend,
                    "whisper_model": options.whisper_model,
                    "whisper_device": options.whisper_device,
                    "translation_engine": options.translation_engine,
                    "video_quality": options.video_quality,
                    "add_subtitles": options.add_subtitles,
                    "dual_subtitles": options.dual_subtitles,
                    "use_existing_subtitles": options.use_existing_subtitles,
                    "subtitle_preset": options.subtitle_preset,
                    "add_tts": options.add_tts,
                    "tts_service": options.tts_service,
                    "tts_voice": options.tts_voice,
                    "replace_original_audio": options.replace_original_audio,
                    "metadata_preset_id": options.metadata_preset_id,
                    "use_ai_preset_selection": options.use_ai_preset_selection,
                    "upload_bilibili": options.upload_bilibili,
                    "upload_douyin": options.upload_douyin,
                    "upload_xiaohongshu": options.upload_xiaohongshu,
                    "directory": subscription.directory,
                }

                # Save to database for persistence
                await task_persistence.save_task_created(
                    task_id, options_dict, "pending", "任务已创建（批量导入）", directory=subscription.directory
                )

                # Submit to task executor for processing
                await task_executor.submit(task)

                task_ids.append(task_id)
                logger.info(f"Created and queued task {task_id} for video: {video.title}")

            except Exception as e:
                error_msg = f"Failed to create task for '{video.title}': {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        return BatchCreateTasksResponse(
            success=len(task_ids) > 0,
            created_count=len(task_ids),
            failed_count=len(errors),
            task_ids=task_ids,
            errors=errors,
        )


# === User Glossary API ===
from translation.terminology.user_glossary import user_glossary_manager, GlossaryEntry


class GlossaryEntryRequest(BaseModel):
    """Request to add/update a glossary entry"""
    source: str
    target: str
    note: str = ""
    category: str = "general"


class GlossaryEntryResponse(BaseModel):
    """Response for a glossary entry"""
    source: str
    target: str
    note: str
    category: str
    created_at: str
    updated_at: str


class GlossaryListResponse(BaseModel):
    """Response for glossary list"""
    entries: List[GlossaryEntryResponse]
    total: int
    categories: List[str]


class GlossaryImportRequest(BaseModel):
    """Request to import glossary entries"""
    format: str = "json"  # json or csv
    data: str  # JSON string or CSV content


class GlossaryImportResponse(BaseModel):
    """Response for glossary import"""
    success: bool
    imported_count: int
    message: str


@app.get("/api/glossary", response_model=GlossaryListResponse)
async def get_glossary(
    category: Optional[str] = None,
    search: Optional[str] = None
):
    """Get user glossary entries"""
    entries = user_glossary_manager.get_entries(category=category, search=search)
    categories = user_glossary_manager.get_categories()
    
    return GlossaryListResponse(
        entries=[
            GlossaryEntryResponse(
                source=e.source,
                target=e.target,
                note=e.note,
                category=e.category,
                created_at=e.created_at,
                updated_at=e.updated_at,
            )
            for e in entries
        ],
        total=len(entries),
        categories=categories,
    )


@app.post("/api/glossary", response_model=GlossaryEntryResponse)
async def add_glossary_entry(request: GlossaryEntryRequest):
    """Add or update a glossary entry"""
    entry = user_glossary_manager.add_entry(
        source=request.source,
        target=request.target,
        note=request.note,
        category=request.category,
    )
    
    return GlossaryEntryResponse(
        source=entry.source,
        target=entry.target,
        note=entry.note,
        category=entry.category,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@app.put("/api/glossary/{source}")
async def update_glossary_entry(
    source: str,
    request: GlossaryEntryRequest
):
    """Update a glossary entry"""
    entry = user_glossary_manager.update_entry(
        source=source,
        target=request.target,
        note=request.note,
        category=request.category,
    )
    
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    return GlossaryEntryResponse(
        source=entry.source,
        target=entry.target,
        note=entry.note,
        category=entry.category,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@app.delete("/api/glossary/{source}")
async def delete_glossary_entry(source: str):
    """Delete a glossary entry"""
    success = user_glossary_manager.delete_entry(source)
    
    if not success:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    return {"success": True, "message": f"Deleted entry: {source}"}


@app.post("/api/glossary/import", response_model=GlossaryImportResponse)
async def import_glossary(request: GlossaryImportRequest):
    """Import glossary entries from JSON or CSV"""
    try:
        if request.format == "json":
            import json
            data = json.loads(request.data)
            count = user_glossary_manager.import_json(data)
        elif request.format == "csv":
            count = user_glossary_manager.import_csv(request.data)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
        
        return GlossaryImportResponse(
            success=True,
            imported_count=count,
            message=f"Successfully imported {count} entries",
        )
    except Exception as e:
        return GlossaryImportResponse(
            success=False,
            imported_count=0,
            message=f"Import failed: {str(e)}",
        )


@app.get("/api/glossary/export")
async def export_glossary(format: str = "json"):
    """Export glossary as JSON or CSV"""
    if format == "json":
        data = user_glossary_manager.export_json()
        return data
    elif format == "csv":
        csv_content = user_glossary_manager.export_csv()
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=glossary.csv"}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@app.delete("/api/glossary")
async def clear_glossary():
    """Clear all glossary entries"""
    user_glossary_manager.clear()
    return {"success": True, "message": "Glossary cleared"}


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    await pipeline.close()


# Run with: uvicorn api.main:app --reload --loop asyncio
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        loop="asyncio"  # 禁用 uvloop，使用标准 asyncio (uvloop 在 macOS ARM64 不稳定)
    )

"""
Database Models for iDubb
SQLAlchemy models for persistent storage
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, JSON, ForeignKey, Enum as SQLEnum, LargeBinary
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


class TaskStatusEnum(str, enum.Enum):
    """Task status enum for database"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    GENERATING_TTS = "generating_tts"
    PROCESSING_VIDEO = "processing_video"
    PENDING_REVIEW = "pending_review"  # Video ready, waiting for metadata review
    PENDING_UPLOAD = "pending_upload"  # Metadata approved, ready for upload
    UPLOADING = "uploading"
    UPLOADED = "uploaded"  # Upload completed (final state)
    FAILED = "failed"
    PAUSED = "paused"


class StepStatusEnum(str, enum.Enum):
    """Step status enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DirectoryModel(Base):
    """Directory model for organizing tasks"""
    __tablename__ = "directories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)  # Unique directory name
    description = Column(Text, nullable=True)
    task_count = Column(Integer, default=0)  # Cached count of tasks
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_count": self.task_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TaskModel(Base):
    """Task persistence model"""
    __tablename__ = "tasks"

    task_id = Column(String(36), primary_key=True)
    status = Column(String(50), default=TaskStatusEnum.PENDING.value)
    progress = Column(Integer, default=0)
    message = Column(Text, default="")
    current_step = Column(String(50), nullable=True)
    error = Column(Text, nullable=True)

    # Directory for organizing tasks (cannot be changed after creation)
    directory = Column(String(100), nullable=True, index=True)

    # Soft delete flag
    deleted = Column(Boolean, default=False, nullable=False)

    # Task folder name (task_id + sanitized video name)
    task_folder = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Processing options (stored as JSON)
    options = Column(JSON, nullable=False)

    # Video info
    video_info = Column(JSON, nullable=True)

    # File paths
    video_path = Column(Text, nullable=True)
    audio_path = Column(Text, nullable=True)
    subtitle_path = Column(Text, nullable=True)
    translated_subtitle_path = Column(Text, nullable=True)
    tts_audio_path = Column(Text, nullable=True)
    final_video_path = Column(Text, nullable=True)
    thumbnail_path = Column(Text, nullable=True)  # Local thumbnail path

    # AI thumbnail fields
    ai_thumbnail_path = Column(Text, nullable=True)  # AI-generated thumbnail path
    ai_thumbnail_title = Column(Text, nullable=True)  # AI-generated title text on thumbnail
    use_ai_thumbnail = Column(Boolean, default=False)  # Whether to use AI thumbnail for upload

    # Upload results
    upload_results = Column(JSON, default=dict)

    # Step timing statistics (JSON: {step_name: {duration_seconds, started_at, completed_at}})
    step_timings = Column(JSON, default=dict)

    # Total processing time in seconds
    total_processing_time = Column(Float, nullable=True)

    # AI-generated metadata (JSON: {title, title_translated, description, keywords, generated_at})
    generated_metadata = Column(JSON, nullable=True)

    # Metadata approval status
    metadata_approved = Column(Boolean, default=False, nullable=False)
    metadata_approved_at = Column(DateTime, nullable=True)

    # Proofreading results (JSON: {overall_confidence, segments, issues_summary, ...})
    proofreading_result = Column(JSON, nullable=True)

    # Optimization results (JSON: {success, optimized_count, total_segments, changes, ...})
    optimization_result = Column(JSON, nullable=True)

    # Relationships
    steps = relationship("StepResultModel", back_populates="task", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "current_step": self.current_step,
            "error": self.error,
            "directory": self.directory,
            "deleted": self.deleted,
            "task_folder": self.task_folder,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "options": self.options or {},  # Include processing options for task reconstruction
            "video_info": self.video_info,
            "files": {
                "video": self.video_path,
                "audio": self.audio_path,
                "original_subtitle": self.subtitle_path,
                "translated_subtitle": self.translated_subtitle_path,
                "tts_audio": self.tts_audio_path,
                "final_video": self.final_video_path,
                "thumbnail": self.thumbnail_path,
                "ai_thumbnail": self.ai_thumbnail_path,
            },
            "ai_thumbnail_title": self.ai_thumbnail_title,
            "use_ai_thumbnail": self.use_ai_thumbnail,
            "upload_results": self.upload_results or {},
            "steps": {step.step_name: step.to_dict() for step in self.steps},
            "step_timings": self.step_timings or {},
            "total_processing_time": self.total_processing_time,
            "generated_metadata": self.generated_metadata,
            "metadata_approved": self.metadata_approved,
            "metadata_approved_at": self.metadata_approved_at.isoformat() if self.metadata_approved_at else None,
            "proofreading_result": self.proofreading_result,
            "optimization_result": self.optimization_result,
        }


class StepResultModel(Base):
    """Step result persistence model with timing"""
    __tablename__ = "step_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False)
    step_name = Column(String(50), nullable=False)
    status = Column(String(20), default=StepStatusEnum.PENDING.value)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)  # Step duration in seconds
    error = Column(Text, nullable=True)
    output_files = Column(JSON, default=dict)
    step_metadata = Column(JSON, default=dict)

    # Relationship
    task = relationship("TaskModel", back_populates="steps")

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "step_name": self.step_name,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "output_files": self.output_files or {},
            "metadata": self.step_metadata or {},
        }


class SettingsModel(Base):
    """Global settings persistence model"""
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    @classmethod
    def get_default_settings(cls) -> dict:
        """Return default global settings"""
        return {
            "storage": {
                "output_directory": "",  # Empty means use default (data/processed)
            },
            "video": {
                "default_quality": "1080p",
                "preferred_format": "mp4",
                "max_duration": 3600,
                "download_subtitles": True,
                "prefer_existing_subtitles": True,
            },
            "translation": {
                "engine": "google",
                "api_key": None,
                "model": "gpt-4",
                "preserve_formatting": True,
                "batch_size": 50,
            },
            "tts": {
                "engine": "edge",
                "voice": "zh-CN-XiaoxiaoNeural",
                "rate": "+0%",
                "volume": "100%",
                "pitch": "+0Hz",
                "api_key": None,
            },
            "subtitle": {
                "enabled": True,
                "dual_subtitles": True,
                "font_name": "Arial",
                "font_size": 24,
                "position": "bottom",
                "style": "default",
            },
            "audio": {
                "generate_tts": True,
                "replace_original": False,
                "original_volume": 0.3,
                "tts_volume": 1.0,
            },
            "auto_upload_bilibili": False,
            "auto_upload_douyin": False,
            "auto_upload_xiaohongshu": False,
        }


class ApiKeyModel(Base):
    """API key persistence model with encryption"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    service = Column(String(50), unique=True, nullable=False)  # openai, anthropic, deepseek, deepl, elevenlabs, etc.
    encrypted_key = Column(Text, nullable=True)  # Encrypted API key
    is_valid = Column(Boolean, default=True)  # Whether key is still valid
    last_verified = Column(DateTime, nullable=True)  # Last verification time
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self, include_key: bool = False) -> dict:
        """Convert to dictionary (key masked by default)"""
        result = {
            "service": self.service,
            "has_key": bool(self.encrypted_key),
            "is_valid": self.is_valid,
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_key:
            # Caller is responsible for decryption
            result["encrypted_key"] = self.encrypted_key
        return result


class CookieModel(Base):
    """Cookie persistence model for platform authentication"""
    __tablename__ = "cookies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String(50), unique=True, nullable=False)  # youtube, bilibili, etc.
    cookie_data = Column(Text, nullable=True)  # Cookie content (Netscape format)
    cookie_json = Column(JSON, nullable=True)  # Cookie as JSON for browsers
    expires_at = Column(DateTime, nullable=True)  # Cookie expiration time
    is_valid = Column(Boolean, default=True)  # Whether cookie is still valid
    last_verified = Column(DateTime, nullable=True)  # Last verification time
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary (without sensitive data)"""
        return {
            "platform": self.platform,
            "is_valid": self.is_valid,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MetadataPresetModel(Base):
    """Metadata preset model for title prefix and signature combinations"""
    __tablename__ = "metadata_presets"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)  # Display name
    description = Column(Text, nullable=True)  # Description/notes
    title_prefix = Column(String(50), default="")  # Title prefix (e.g., "[中字]")
    custom_signature = Column(Text, default="")  # Custom signature for description
    tags = Column(JSON, default=list)  # Applicable scenario tags
    is_default = Column(Boolean, default=False)  # Whether this is the default preset
    is_builtin = Column(Boolean, default=False)  # Whether this is a builtin preset
    sort_order = Column(Integer, default=0)  # Sort order
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "title_prefix": self.title_prefix,
            "custom_signature": self.custom_signature,
            "tags": self.tags or [],
            "is_default": self.is_default,
            "is_builtin": self.is_builtin,
            "sort_order": self.sort_order,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TrendingVideoModel(Base):
    """Trending video model for YouTube trending videos cache"""
    __tablename__ = "trending_videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    channel_name = Column(String(255), nullable=False)
    channel_url = Column(Text, nullable=True)
    thumbnail_url = Column(Text, nullable=True)
    duration = Column(Integer, default=0)  # Duration in seconds
    view_count = Column(Integer, default=0)
    category = Column(String(50), nullable=False, index=True)  # tech, gaming, lifestyle
    platform = Column(String(20), default="youtube")
    video_url = Column(Text, nullable=False)
    published_at = Column(DateTime, nullable=True)  # Video upload/publish date
    fetched_at = Column(DateTime, default=datetime.now)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "video_id": self.video_id,
            "title": self.title,
            "channel_name": self.channel_name,
            "channel_url": self.channel_url,
            "thumbnail_url": self.thumbnail_url,
            "duration": self.duration,
            "view_count": self.view_count,
            "category": self.category,
            "platform": self.platform,
            "video_url": self.video_url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SubscriptionModel(Base):
    """Subscription model for tracking YouTube/Bilibili channels"""
    __tablename__ = "subscriptions"

    id = Column(String(36), primary_key=True)
    platform = Column(String(20), nullable=False)  # youtube, bilibili
    channel_id = Column(String(100), nullable=False)
    channel_name = Column(String(255), nullable=False)
    channel_url = Column(Text, nullable=True)
    channel_avatar = Column(Text, nullable=True)

    # Directory for organizing tasks from this subscription
    directory = Column(String(100), nullable=True, index=True)

    # Video tracking
    last_video_id = Column(String(100), nullable=True)
    last_video_title = Column(Text, nullable=True)
    last_video_published_at = Column(DateTime, nullable=True)

    # Scheduling configuration
    check_interval = Column(Integer, default=60)  # minutes
    next_check_at = Column(DateTime, nullable=True)
    last_checked_at = Column(DateTime, nullable=True)

    # Auto-processing
    auto_process = Column(Boolean, default=True)
    process_options = Column(JSON, nullable=True)  # Task template options

    # Status
    enabled = Column(Boolean, default=True)
    error_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "platform": self.platform,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "channel_url": self.channel_url,
            "channel_avatar": self.channel_avatar,
            "directory": self.directory,
            "last_video_id": self.last_video_id,
            "last_video_title": self.last_video_title,
            "last_video_published_at": self.last_video_published_at.isoformat() if self.last_video_published_at else None,
            "check_interval": self.check_interval,
            "next_check_at": self.next_check_at.isoformat() if self.next_check_at else None,
            "last_checked_at": self.last_checked_at.isoformat() if self.last_checked_at else None,
            "auto_process": self.auto_process,
            "process_options": self.process_options,
            "enabled": self.enabled,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

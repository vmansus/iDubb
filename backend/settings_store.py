"""
Global Settings Store
管理用户全局设置：视频质量、翻译引擎、TTS配置等
使用数据库存储，每个配置类别独立存储
"""
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from loguru import logger


# ==================== Data Classes ====================

@dataclass
class TranslationApiKeys:
    """API keys for different translation engines"""
    openai: str = ""
    anthropic: str = ""
    deepseek: str = ""
    deepl: str = ""

    def to_dict(self, mask: bool = True) -> Dict[str, Any]:
        if mask:
            return {
                "openai": "***" if self.openai else "",
                "anthropic": "***" if self.anthropic else "",
                "deepseek": "***" if self.deepseek else "",
                "deepl": "***" if self.deepl else "",
            }
        return {
            "openai": self.openai,
            "anthropic": self.anthropic,
            "deepseek": self.deepseek,
            "deepl": self.deepl,
        }

    def get_key_for_engine(self, engine: str) -> Optional[str]:
        """Get the API key for a specific engine"""
        key_map = {
            "gpt": self.openai,
            "openai": self.openai,
            "claude": self.anthropic,
            "anthropic": self.anthropic,
            "deepseek": self.deepseek,
            "deepl": self.deepl,
        }
        key = key_map.get(engine.lower(), "")
        return key if key else None


@dataclass
class TranslationSettings:
    """Translation engine settings"""
    engine: str = "google"
    api_key: Optional[str] = None  # Legacy single key (deprecated)
    api_keys: TranslationApiKeys = field(default_factory=TranslationApiKeys)
    model: str = "gpt-4"
    preserve_formatting: bool = True
    batch_size: int = 10
    use_optimized: bool = True
    fast_mode: bool = False
    use_two_step_mode: bool = True
    enable_alignment: bool = False
    enable_length_control: bool = True
    max_chars_per_line: int = 42
    max_lines: int = 2
    enable_localization: bool = True
    use_custom_glossary: bool = True

    def get_api_key(self) -> Optional[str]:
        """Get API key for current engine"""
        key = self.api_keys.get_key_for_engine(self.engine)
        if key:
            return key
        return self.api_key

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "model": self.model,
            "preserve_formatting": self.preserve_formatting,
            "batch_size": self.batch_size,
            "api_keys": self.api_keys.to_dict(mask=True),
            "use_optimized": self.use_optimized,
            "fast_mode": self.fast_mode,
            "use_two_step_mode": self.use_two_step_mode,
            "enable_alignment": self.enable_alignment,
            "enable_length_control": self.enable_length_control,
            "max_chars_per_line": self.max_chars_per_line,
            "max_lines": self.max_lines,
            "enable_localization": self.enable_localization,
            "use_custom_glossary": self.use_custom_glossary,
        }


@dataclass
class TTSSettings:
    """TTS (Text-to-Speech) settings"""
    engine: str = "edge"
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+0%"
    volume: str = "+0%"
    pitch: str = "+0Hz"
    api_key: Optional[str] = None
    qwen3_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    qwen3_language: str = "Chinese"
    voice_cloning_mode: str = "disabled"
    ref_audio_path: Optional[str] = None
    ref_audio_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "voice": self.voice,
            "rate": self.rate,
            "volume": self.volume,
            "pitch": self.pitch,
            "api_key": "***" if self.api_key else None,
            "qwen3_model": self.qwen3_model,
            "qwen3_language": self.qwen3_language,
            "voice_cloning_mode": self.voice_cloning_mode,
            "ref_audio_path": self.ref_audio_path,
            "ref_audio_text": self.ref_audio_text,
        }


@dataclass
class ProcessingSettings:
    """Task processing and concurrency settings"""
    max_concurrent_tasks: int = 2
    use_gpu_lock: bool = True
    translation_timeout: int = 300
    translation_retry_count: int = 3
    timezone: str = "Asia/Shanghai"  # User's preferred timezone

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StorageSettings:
    """Storage settings for output directory"""
    output_directory: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoSettings:
    """Video processing settings"""
    default_quality: str = "1080p"
    preferred_format: str = "mp4"
    max_duration: int = 3600
    download_subtitles: bool = True
    prefer_existing_subtitles: bool = True
    whisper_backend: str = "faster"
    whisper_model: str = "faster:small"
    whisper_device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubtitleStyleConfig:
    """Style configuration for a single language subtitle"""
    font_name: str = ""
    font_size: int = 24
    color: str = "#FFFFFF"
    bold: bool = True
    italic: bool = False
    outline_color: str = "#000000"
    outline_width: int = 2
    shadow: int = 1
    shadow_color: str = "#000000"
    alignment: str = "bottom"
    margin_h: int = 20
    margin_v: int = 30
    back_color: str = "#000000"
    back_opacity: int = 0
    spacing: int = 0
    scale_x: int = 100
    scale_y: int = 100
    max_width: int = 90

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_ass_color(self, hex_color: str) -> str:
        """Convert hex color (#RRGGBB) to ASS format (&HBBGGRR)"""
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        if len(hex_color) == 6:
            r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
            return f"&H{b}{g}{r}"
        return "&HFFFFFF"

    def to_ass_back_color(self) -> str:
        """Convert back_color and back_opacity to ASS format"""
        hex_color = self.back_color
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        if len(hex_color) == 6:
            r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
            alpha = int((100 - self.back_opacity) * 255 / 100)
            return f"&H{alpha:02X}{b}{g}{r}"
        return "&H80000000"


@dataclass
class SubtitlePreset:
    """Subtitle style preset"""
    id: str
    name: str
    description: str
    is_builtin: bool = True
    is_vertical: bool = False
    subtitle_mode: str = "dual"
    source_language: str = "en"
    target_language: str = "zh-CN"
    original_style: Optional[SubtitleStyleConfig] = field(default_factory=SubtitleStyleConfig)
    translated_style: Optional[SubtitleStyleConfig] = field(default_factory=SubtitleStyleConfig)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "is_builtin": self.is_builtin,
            "is_vertical": self.is_vertical,
            "subtitle_mode": self.subtitle_mode,
            "source_language": self.source_language,
            "target_language": self.target_language,
        }
        if self.original_style:
            result["original_style"] = self.original_style.to_dict()
        if self.translated_style:
            result["translated_style"] = self.translated_style.to_dict()
        return result


# Built-in subtitle presets
BUILTIN_PRESETS: List[SubtitlePreset] = [
    SubtitlePreset(
        id="classic", name="经典", description="白色中文 + 灰色原文，黑色描边",
        is_builtin=True, subtitle_mode="dual", source_language="en", target_language="zh-CN",
        original_style=SubtitleStyleConfig(font_size=20, color="#CCCCCC", bold=False, outline_width=2, shadow=1, margin_v=60),
        translated_style=SubtitleStyleConfig(font_size=26, color="#FFFFFF", bold=True, outline_width=2, shadow=1, margin_v=60),
    ),
    SubtitlePreset(
        id="modern", name="现代", description="黄色中文 + 白色原文，无描边半透明背景",
        is_builtin=True, subtitle_mode="dual", source_language="en", target_language="zh-CN",
        original_style=SubtitleStyleConfig(font_size=18, color="#FFFFFF", bold=False, outline_width=0, shadow=0, back_opacity=60, margin_v=50),
        translated_style=SubtitleStyleConfig(font_size=24, color="#FFEB3B", bold=True, outline_width=0, shadow=0, back_opacity=60, margin_v=50),
    ),
    SubtitlePreset(
        id="minimal", name="极简", description="小号白色字幕，细描边",
        is_builtin=True, subtitle_mode="dual", source_language="en", target_language="zh-CN",
        original_style=SubtitleStyleConfig(font_size=16, color="#E0E0E0", bold=False, outline_width=1, shadow=0, margin_v=40),
        translated_style=SubtitleStyleConfig(font_size=20, color="#FFFFFF", bold=False, outline_width=1, shadow=0, margin_v=40),
    ),
    SubtitlePreset(
        id="cinematic", name="电影", description="大号中文，强阴影效果",
        is_builtin=True, subtitle_mode="dual", source_language="en", target_language="zh-CN",
        original_style=SubtitleStyleConfig(font_size=18, color="#B0B0B0", bold=False, outline_width=2, shadow=3, shadow_color="#000000", margin_v=70),
        translated_style=SubtitleStyleConfig(font_size=28, color="#FFFFFF", bold=True, outline_width=3, shadow=4, shadow_color="#000000", margin_v=70),
    ),
    SubtitlePreset(
        id="chinese_only", name="仅中文", description="仅显示中文翻译字幕",
        is_builtin=True, subtitle_mode="translated_only", source_language="en", target_language="zh-CN",
        original_style=None,
        translated_style=SubtitleStyleConfig(font_size=26, color="#FFFFFF", bold=True, outline_width=2, shadow=1, margin_v=50),
    ),
    SubtitlePreset(
        id="english_only", name="仅原文", description="仅显示原文字幕",
        is_builtin=True, subtitle_mode="original_only", source_language="en", target_language="zh-CN",
        original_style=SubtitleStyleConfig(font_size=24, color="#FFFFFF", bold=False, outline_width=2, shadow=1, margin_v=50),
        translated_style=None,
    ),
    # Vertical video presets
    SubtitlePreset(
        id="vertical_classic", name="竖屏经典", description="竖屏视频专用，字幕居中靠下",
        is_builtin=True, is_vertical=True, subtitle_mode="dual", source_language="en", target_language="zh-CN",
        original_style=SubtitleStyleConfig(font_size=18, color="#CCCCCC", bold=False, outline_width=2, shadow=1, margin_v=120, margin_h=30, max_width=75),
        translated_style=SubtitleStyleConfig(font_size=22, color="#FFFFFF", bold=True, outline_width=2, shadow=1, margin_v=120, margin_h=30, max_width=75),
    ),
    SubtitlePreset(
        id="vertical_tiktok", name="TikTok风格", description="模仿TikTok字幕风格",
        is_builtin=True, is_vertical=True, subtitle_mode="translated_only", source_language="en", target_language="zh-CN",
        original_style=None,
        translated_style=SubtitleStyleConfig(font_size=28, color="#FFFFFF", bold=True, outline_width=3, shadow=0, back_color="#000000", back_opacity=50, margin_v=180, margin_h=60, max_width=80),
    ),
]


def get_preset_by_id(preset_id: str) -> Optional[SubtitlePreset]:
    """Get a preset by its ID"""
    for preset in BUILTIN_PRESETS:
        if preset.id == preset_id:
            return preset
    return None


@dataclass
class SubtitleSettings:
    """Subtitle settings with per-language configuration"""
    enabled: bool = True
    dual_subtitles: bool = True
    default_preset: Optional[str] = "classic"
    font_name: str = ""
    font_size: int = 24
    position: str = "bottom"
    style: str = "default"
    chinese_on_top: bool = True
    source_language: str = "en"
    target_language: str = "zh-CN"
    original_style: SubtitleStyleConfig = field(default_factory=lambda: SubtitleStyleConfig(font_size=20, color="#CCCCCC", bold=False))
    translated_style: SubtitleStyleConfig = field(default_factory=lambda: SubtitleStyleConfig(font_size=26, color="#FFFFFF", bold=True))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "dual_subtitles": self.dual_subtitles,
            "default_preset": self.default_preset,
            "font_name": self.font_name,
            "font_size": self.font_size,
            "position": self.position,
            "style": self.style,
            "chinese_on_top": self.chinese_on_top,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "original_style": self.original_style.to_dict(),
            "translated_style": self.translated_style.to_dict(),
        }


@dataclass
class AudioSettings:
    """Audio processing settings"""
    generate_tts: bool = True
    replace_original: bool = False
    original_volume: float = 0.3
    tts_volume: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetadataSettings:
    """AI metadata generation settings"""
    enabled: bool = True
    auto_generate: bool = True
    require_review: bool = True
    include_source_url: bool = True
    max_keywords: int = 10
    default_use_ai_preset_selection: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BilibiliSettings:
    """Bilibili upload settings"""
    is_original: bool = False
    default_tid: int = 0
    auto_match_partition: bool = True
    source_url_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThumbnailSettings:
    """AI thumbnail generation settings"""
    enabled: bool = False
    auto_generate: bool = True
    default_use_ai: bool = False
    style: str = "gradient_bar"
    font_name: str = ""
    font_size: int = 72
    text_color: str = "#FFD700"
    gradient_color: str = "#000000"
    gradient_opacity: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendingSettings:
    """YouTube trending video settings"""
    enabled: bool = True
    youtube_api_key: Optional[str] = None
    use_official_api: bool = True
    update_interval: int = 60
    last_updated: Optional[str] = None
    enabled_categories: List[str] = field(default_factory=lambda: ["tech", "gaming", "lifestyle"])
    max_videos_per_category: int = 20
    time_filter: str = "week"
    sort_by: str = "upload_date"
    min_view_count: int = 10000
    max_duration: int = 1800
    region_code: str = "US"
    exclude_shorts: bool = True  # Exclude YouTube Shorts (videos <= 60 seconds)

    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "youtube_api_key": ("***" if self.youtube_api_key else None) if mask_secrets else self.youtube_api_key,
            "use_official_api": self.use_official_api,
            "update_interval": self.update_interval,
            "last_updated": self.last_updated,
            "enabled_categories": self.enabled_categories,
            "max_videos_per_category": self.max_videos_per_category,
            "time_filter": self.time_filter,
            "sort_by": self.sort_by,
            "min_view_count": self.min_view_count,
            "max_duration": self.max_duration,
            "region_code": self.region_code,
            "exclude_shorts": self.exclude_shorts,
        }


@dataclass
class TikTokSettings:
    """TikTok trending video settings"""
    enabled: bool = False
    update_interval: int = 60  # minutes
    last_updated: Optional[str] = None
    region_code: str = "US"  # TikTok region
    enabled_tags: List[str] = field(default_factory=lambda: ["trending", "fyp", "viral"])
    max_videos_per_tag: int = 20
    min_view_count: int = 10000
    min_like_count: int = 1000
    max_duration: int = 180  # 3 minutes max for TikTok
    max_publish_age: int = 7  # days - only include videos published within this many days

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "update_interval": self.update_interval,
            "last_updated": self.last_updated,
            "region_code": self.region_code,
            "enabled_tags": self.enabled_tags,
            "max_videos_per_tag": self.max_videos_per_tag,
            "min_view_count": self.min_view_count,
            "min_like_count": self.min_like_count,
            "max_duration": self.max_duration,
            "max_publish_age": self.max_publish_age,
        }


@dataclass
class ProofreadingSettings:
    """AI proofreading settings"""
    enabled: bool = True
    auto_pause: bool = True
    min_confidence: float = 0.6
    check_grammar: bool = True
    check_terminology: bool = True
    check_timing: bool = True
    check_formatting: bool = True
    use_ai_validation: bool = True
    auto_optimize: bool = False
    optimization_level: str = "moderate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UploadSettings:
    """Upload platform settings"""
    auto_upload_bilibili: bool = False
    auto_upload_douyin: bool = False
    auto_upload_xiaohongshu: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalSettings:
    """Global application settings - container for all settings"""
    storage: StorageSettings = field(default_factory=StorageSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    subtitle: SubtitleSettings = field(default_factory=SubtitleSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    metadata: MetadataSettings = field(default_factory=MetadataSettings)
    bilibili: BilibiliSettings = field(default_factory=BilibiliSettings)
    thumbnail: ThumbnailSettings = field(default_factory=ThumbnailSettings)
    proofreading: ProofreadingSettings = field(default_factory=ProofreadingSettings)
    trending: TrendingSettings = field(default_factory=TrendingSettings)
    tiktok: TikTokSettings = field(default_factory=TikTokSettings)
    upload: UploadSettings = field(default_factory=UploadSettings)

    # Legacy compatibility - map to upload settings
    @property
    def auto_upload_bilibili(self) -> bool:
        return self.upload.auto_upload_bilibili

    @auto_upload_bilibili.setter
    def auto_upload_bilibili(self, value: bool):
        self.upload.auto_upload_bilibili = value

    @property
    def auto_upload_douyin(self) -> bool:
        return self.upload.auto_upload_douyin

    @auto_upload_douyin.setter
    def auto_upload_douyin(self, value: bool):
        self.upload.auto_upload_douyin = value

    @property
    def auto_upload_xiaohongshu(self) -> bool:
        return self.upload.auto_upload_xiaohongshu

    @auto_upload_xiaohongshu.setter
    def auto_upload_xiaohongshu(self, value: bool):
        self.upload.auto_upload_xiaohongshu = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage": self.storage.to_dict(),
            "processing": self.processing.to_dict(),
            "video": self.video.to_dict(),
            "translation": self.translation.to_dict(),
            "tts": self.tts.to_dict(),
            "subtitle": self.subtitle.to_dict(),
            "audio": self.audio.to_dict(),
            "metadata": self.metadata.to_dict(),
            "bilibili": self.bilibili.to_dict(),
            "thumbnail": self.thumbnail.to_dict(),
            "proofreading": self.proofreading.to_dict(),
            "trending": self.trending.to_dict(),
            "tiktok": self.tiktok.to_dict(),
            "upload": self.upload.to_dict(),
            # Legacy compatibility
            "auto_upload_bilibili": self.upload.auto_upload_bilibili,
            "auto_upload_douyin": self.upload.auto_upload_douyin,
            "auto_upload_xiaohongshu": self.upload.auto_upload_xiaohongshu,
        }


# ==================== Settings Store ====================

# Setting keys in database (one key per category)
SETTING_KEYS = [
    "storage", "processing", "video", "translation", "tts", "subtitle",
    "audio", "metadata", "bilibili", "thumbnail", "proofreading", "trending", "tiktok", "upload"
]

# Separate key for custom presets
CUSTOM_PRESETS_KEY = "custom_presets"

# Legacy key (for migration)
LEGACY_GLOBAL_KEY = "global_settings"


class SettingsStore:
    """
    Persistent settings storage using database.
    Each setting category is stored as a separate key for better performance.
    """

    def __init__(self):
        self._settings: Optional[GlobalSettings] = None
        self._initialized = False
        # Legacy JSON path for migration
        project_root = Path(__file__).parent.parent
        self._legacy_json_path = project_root / "data" / "settings.json"

    def _run_async(self, coro):
        """Run async coroutine from sync context"""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(coro)

    async def _get_setting(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a single setting by key"""
        from database.connection import get_db
        from database.repository import SettingsRepository
        try:
            async with get_db() as session:
                repo = SettingsRepository(session)
                return await repo.get(key)
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
            return None

    async def _set_setting(self, key: str, value: Dict[str, Any]) -> bool:
        """Set a single setting"""
        from database.connection import get_db
        from database.repository import SettingsRepository
        try:
            async with get_db() as session:
                repo = SettingsRepository(session)
                await repo.set(key, value)
                return True
        except Exception as e:
            logger.error(f"Failed to set setting {key}: {e}")
            return False

    async def _migrate_from_legacy(self) -> bool:
        """Migrate from legacy global_settings blob to separate keys"""
        from database.connection import get_db
        from database.repository import SettingsRepository

        try:
            async with get_db() as session:
                repo = SettingsRepository(session)

                # Check for legacy global_settings
                legacy_data = await repo.get(LEGACY_GLOBAL_KEY)
                if not legacy_data:
                    return False

                logger.info("Migrating from legacy global_settings to separate keys...")

                # Extract and save each category
                for key in SETTING_KEYS:
                    if key in legacy_data:
                        await repo.set(key, legacy_data[key])
                        logger.debug(f"Migrated setting: {key}")

                # Handle legacy upload settings (top-level auto_upload_*)
                upload_data = legacy_data.get("upload", {})
                if not upload_data:
                    upload_data = {
                        "auto_upload_bilibili": legacy_data.get("auto_upload_bilibili", False),
                        "auto_upload_douyin": legacy_data.get("auto_upload_douyin", False),
                        "auto_upload_xiaohongshu": legacy_data.get("auto_upload_xiaohongshu", False),
                    }
                await repo.set("upload", upload_data)

                # Migrate custom_presets separately
                if "custom_presets" in legacy_data:
                    await repo.set(CUSTOM_PRESETS_KEY, legacy_data["custom_presets"])
                    logger.debug("Migrated custom_presets")

                # Delete legacy key
                await repo.delete(LEGACY_GLOBAL_KEY)
                logger.info("Migration complete, deleted legacy global_settings")

                return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    async def _load_from_db(self) -> GlobalSettings:
        """Load all settings from database"""
        settings = GlobalSettings()

        # Try migration first
        await self._migrate_from_legacy()

        # Load each category
        for key in SETTING_KEYS:
            data = await self._get_setting(key)
            if data:
                self._apply_category_data(settings, key, data)

        # Load API keys from encrypted table
        await self._load_api_keys(settings)

        return settings

    async def _load_api_keys(self, settings: GlobalSettings) -> None:
        """Load all API keys from encrypted api_keys table"""
        from database.connection import get_db
        from database.repository import ApiKeyRepository

        try:
            async with get_db() as session:
                repo = ApiKeyRepository(session)

                # Translation API keys
                openai_key = await repo.get("openai")
                anthropic_key = await repo.get("anthropic")
                deepseek_key = await repo.get("deepseek")
                deepl_key = await repo.get("deepl")

                settings.translation.api_keys = TranslationApiKeys(
                    openai=openai_key or "",
                    anthropic=anthropic_key or "",
                    deepseek=deepseek_key or "",
                    deepl=deepl_key or "",
                )

                # TTS API key (ElevenLabs)
                elevenlabs_key = await repo.get("elevenlabs")
                if elevenlabs_key:
                    settings.tts.api_key = elevenlabs_key

                # YouTube API key (for trending)
                youtube_key = await repo.get("youtube")
                if youtube_key:
                    settings.trending.youtube_api_key = youtube_key

        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")

    async def _save_to_db(self, settings: GlobalSettings) -> bool:
        """Save all settings to database (each category separately)"""
        try:
            # Save each category
            await self._set_setting("storage", asdict(settings.storage))
            await self._set_setting("processing", asdict(settings.processing))
            await self._set_setting("video", asdict(settings.video))
            await self._set_setting("translation", self._translation_to_dict(settings.translation))
            await self._set_setting("tts", self._tts_to_dict(settings.tts))
            await self._set_setting("subtitle", self._subtitle_to_dict(settings.subtitle))
            await self._set_setting("audio", asdict(settings.audio))
            await self._set_setting("metadata", asdict(settings.metadata))
            await self._set_setting("bilibili", asdict(settings.bilibili))
            await self._set_setting("thumbnail", asdict(settings.thumbnail))
            await self._set_setting("proofreading", asdict(settings.proofreading))
            await self._set_setting("trending", settings.trending.to_dict(mask_secrets=False))
            await self._set_setting("tiktok", settings.tiktok.to_dict())
            await self._set_setting("upload", asdict(settings.upload))

            # Save API keys to separate encrypted table
            await self._save_api_keys(settings)

            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False

    async def _save_api_keys(self, settings: GlobalSettings) -> None:
        """Save all API keys to encrypted api_keys table"""
        from database.connection import get_db
        from database.repository import ApiKeyRepository

        async with get_db() as session:
            repo = ApiKeyRepository(session)

            # Translation API keys
            api_keys = settings.translation.api_keys
            if api_keys.openai:
                await repo.set("openai", api_keys.openai)
            if api_keys.anthropic:
                await repo.set("anthropic", api_keys.anthropic)
            if api_keys.deepseek:
                await repo.set("deepseek", api_keys.deepseek)
            if api_keys.deepl:
                await repo.set("deepl", api_keys.deepl)

            # TTS API key (ElevenLabs)
            if settings.tts.api_key:
                await repo.set("elevenlabs", settings.tts.api_key)

            # YouTube API key (for trending)
            if settings.trending.youtube_api_key:
                await repo.set("youtube", settings.trending.youtube_api_key)

    def _translation_to_dict(self, settings: TranslationSettings) -> Dict[str, Any]:
        """Convert translation settings to dict (API keys stored separately)"""
        return {
            "engine": settings.engine,
            "model": settings.model,
            "preserve_formatting": settings.preserve_formatting,
            "batch_size": settings.batch_size,
            # API keys are stored in api_keys table, not here
            "use_optimized": settings.use_optimized,
            "fast_mode": settings.fast_mode,
            "use_two_step_mode": settings.use_two_step_mode,
            "enable_alignment": settings.enable_alignment,
            "enable_length_control": settings.enable_length_control,
            "max_chars_per_line": settings.max_chars_per_line,
            "max_lines": settings.max_lines,
            "enable_localization": settings.enable_localization,
            "use_custom_glossary": settings.use_custom_glossary,
        }

    def _tts_to_dict(self, settings: TTSSettings) -> Dict[str, Any]:
        """Convert TTS settings to dict (API key stored separately)"""
        return {
            "engine": settings.engine,
            "voice": settings.voice,
            "rate": settings.rate,
            "volume": settings.volume,
            "pitch": settings.pitch,
            # API key is stored in api_keys table, not here
            "qwen3_model": settings.qwen3_model,
            "qwen3_language": settings.qwen3_language,
            "voice_cloning_mode": settings.voice_cloning_mode,
            "ref_audio_path": settings.ref_audio_path,
            "ref_audio_text": settings.ref_audio_text,
        }

    def _subtitle_to_dict(self, settings: SubtitleSettings) -> Dict[str, Any]:
        """Convert subtitle settings to dict"""
        return {
            "enabled": settings.enabled,
            "dual_subtitles": settings.dual_subtitles,
            "default_preset": settings.default_preset,
            "font_name": settings.font_name,
            "font_size": settings.font_size,
            "position": settings.position,
            "style": settings.style,
            "chinese_on_top": settings.chinese_on_top,
            "source_language": settings.source_language,
            "target_language": settings.target_language,
            "original_style": asdict(settings.original_style),
            "translated_style": asdict(settings.translated_style),
        }

    def _apply_category_data(self, settings: GlobalSettings, key: str, data: Dict[str, Any]):
        """Apply data to a specific settings category"""
        if key == "storage":
            for k, v in data.items():
                if hasattr(settings.storage, k):
                    setattr(settings.storage, k, v)
        elif key == "processing":
            for k, v in data.items():
                if hasattr(settings.processing, k):
                    setattr(settings.processing, k, v)
        elif key == "video":
            for k, v in data.items():
                if hasattr(settings.video, k):
                    setattr(settings.video, k, v)
        elif key == "translation":
            self._apply_translation_data(settings.translation, data)
        elif key == "tts":
            self._apply_tts_data(settings.tts, data)
        elif key == "subtitle":
            self._apply_subtitle_data(settings.subtitle, data)
        elif key == "audio":
            for k, v in data.items():
                if hasattr(settings.audio, k):
                    setattr(settings.audio, k, v)
        elif key == "metadata":
            for k, v in data.items():
                if hasattr(settings.metadata, k):
                    setattr(settings.metadata, k, v)
        elif key == "bilibili":
            for k, v in data.items():
                if hasattr(settings.bilibili, k):
                    setattr(settings.bilibili, k, v)
        elif key == "thumbnail":
            for k, v in data.items():
                if hasattr(settings.thumbnail, k):
                    setattr(settings.thumbnail, k, v)
        elif key == "proofreading":
            for k, v in data.items():
                if hasattr(settings.proofreading, k):
                    setattr(settings.proofreading, k, v)
        elif key == "trending":
            self._apply_trending_data(settings.trending, data)
        elif key == "tiktok":
            for k, v in data.items():
                if hasattr(settings.tiktok, k):
                    setattr(settings.tiktok, k, v)
        elif key == "upload":
            for k, v in data.items():
                if hasattr(settings.upload, k):
                    setattr(settings.upload, k, v)

    def _apply_translation_data(self, settings: TranslationSettings, data: Dict[str, Any]):
        """Apply translation settings data (API keys stored in encrypted api_keys table)"""
        for key, value in data.items():
            if key == 'api_keys':
                # Update API keys in encrypted storage
                if isinstance(value, dict):
                    self._run_async(self._update_translation_api_keys(value, settings))
                continue
            if hasattr(settings, key) and value != '***':
                setattr(settings, key, value)

    async def _update_translation_api_keys(self, api_keys_data: Dict[str, str], settings: TranslationSettings) -> None:
        """Update translation API keys in encrypted api_keys table"""
        from database.connection import get_db
        from database.repository import ApiKeyRepository

        try:
            async with get_db() as session:
                repo = ApiKeyRepository(session)

                # Update each key (skip masked values)
                for service in ['openai', 'anthropic', 'deepseek', 'deepl']:
                    key = api_keys_data.get(service)
                    if key and key != '***':
                        await repo.set(service, key)
                        # Also update in-memory settings
                        setattr(settings.api_keys, service, key)
                        logger.debug(f"Updated {service} API key in encrypted storage")

        except Exception as e:
            logger.error(f"Failed to update translation API keys: {e}")

    def _apply_tts_data(self, settings: TTSSettings, data: Dict[str, Any]):
        """Apply TTS settings data (API key stored in encrypted api_keys table)"""
        for key, value in data.items():
            if key == 'api_key':
                # Update API key in encrypted storage
                if value and value != '***':
                    self._run_async(self._update_tts_api_key(value, settings))
                continue
            if hasattr(settings, key) and value != '***':
                setattr(settings, key, value)

    async def _update_tts_api_key(self, api_key: str, settings: TTSSettings) -> None:
        """Update TTS API key in encrypted api_keys table"""
        from database.connection import get_db
        from database.repository import ApiKeyRepository

        try:
            async with get_db() as session:
                repo = ApiKeyRepository(session)
                await repo.set("elevenlabs", api_key)
                # Also update in-memory settings
                settings.api_key = api_key
                logger.debug("Updated ElevenLabs API key in encrypted storage")

        except Exception as e:
            logger.error(f"Failed to update TTS API key: {e}")

    def _apply_subtitle_data(self, settings: SubtitleSettings, data: Dict[str, Any]):
        """Apply subtitle settings data"""
        if 'original_style' in data and isinstance(data['original_style'], dict):
            for key, value in data['original_style'].items():
                if hasattr(settings.original_style, key):
                    setattr(settings.original_style, key, value)
        if 'translated_style' in data and isinstance(data['translated_style'], dict):
            for key, value in data['translated_style'].items():
                if hasattr(settings.translated_style, key):
                    setattr(settings.translated_style, key, value)
        for key, value in data.items():
            if key in ('original_style', 'translated_style'):
                continue
            if hasattr(settings, key):
                setattr(settings, key, value)

    def _apply_trending_data(self, settings: TrendingSettings, data: Dict[str, Any]):
        """Apply trending settings data (API key stored in encrypted api_keys table)"""
        for key, value in data.items():
            if key == 'youtube_api_key':
                # Update API key in encrypted storage
                if value and value != '***':
                    self._run_async(self._update_youtube_api_key(value, settings))
                continue
            if hasattr(settings, key):
                setattr(settings, key, value)

    async def _update_youtube_api_key(self, api_key: str, settings: TrendingSettings) -> None:
        """Update YouTube API key in encrypted api_keys table"""
        from database.connection import get_db
        from database.repository import ApiKeyRepository

        try:
            async with get_db() as session:
                repo = ApiKeyRepository(session)
                await repo.set("youtube", api_key)
                # Also update in-memory settings
                settings.youtube_api_key = api_key
                logger.debug("Updated YouTube API key in encrypted storage")

        except Exception as e:
            logger.error(f"Failed to update YouTube API key: {e}")

    # ==================== Public Interface ====================

    def load(self) -> GlobalSettings:
        """Load settings (sync interface)"""
        if self._settings is not None:
            return self._settings
        self._settings = self._run_async(self._load_from_db())
        self._initialized = True
        return self._settings

    async def load_async(self) -> GlobalSettings:
        """Load settings (async interface)"""
        if self._settings is not None:
            return self._settings
        self._settings = await self._load_from_db()
        self._initialized = True
        return self._settings

    def save(self, settings: GlobalSettings) -> bool:
        """Save settings (sync interface)"""
        success = self._run_async(self._save_to_db(settings))
        if success:
            self._settings = settings
        return success

    async def save_async(self, settings: GlobalSettings) -> bool:
        """Save settings (async interface)"""
        success = await self._save_to_db(settings)
        if success:
            self._settings = settings
        return success

    def update(self, updates: Dict[str, Any]) -> GlobalSettings:
        """Update specific settings (sync interface)"""
        settings = self.load()
        self._apply_updates(settings, updates)
        self.save(settings)
        return settings

    async def update_async(self, updates: Dict[str, Any]) -> GlobalSettings:
        """Update specific settings (async interface)"""
        settings = await self.load_async()
        self._apply_updates(settings, updates)
        await self.save_async(settings)
        return settings

    def _apply_updates(self, settings: GlobalSettings, updates: Dict[str, Any]):
        """Apply updates to settings object"""
        for key in SETTING_KEYS:
            if key in updates:
                self._apply_category_data(settings, key, updates[key])

        # Legacy compatibility
        for key in ['auto_upload_bilibili', 'auto_upload_douyin', 'auto_upload_xiaohongshu']:
            if key in updates:
                setattr(settings, key, updates[key])

    def reset(self) -> GlobalSettings:
        """Reset to default settings (sync interface)"""
        self._settings = GlobalSettings()
        self.save(self._settings)
        return self._settings

    async def reset_async(self) -> GlobalSettings:
        """Reset to default settings (async interface)"""
        self._settings = GlobalSettings()
        await self.save_async(self._settings)
        return self._settings

    def reload(self) -> GlobalSettings:
        """Force reload settings from database (sync interface)"""
        self._settings = None
        self._initialized = False
        return self.load()

    async def reload_async(self) -> GlobalSettings:
        """Force reload settings from database (async interface)"""
        self._settings = None
        self._initialized = False
        return await self.load_async()


# Global instance
settings_store = SettingsStore()

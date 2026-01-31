"""
Repository classes for database operations
Provides CRUD operations for tasks and settings
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from loguru import logger

from .models import TaskModel, StepResultModel, SettingsModel, StepStatusEnum, CookieModel, MetadataPresetModel, SubscriptionModel, DirectoryModel, TrendingVideoModel, ApiKeyModel
from utils.encryption import encrypt_string, decrypt_string, decrypt_if_needed, is_encrypted


class TaskRepository:
    """Repository for task CRUD operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        task_id: str,
        options: Dict[str, Any],
        status: str = "pending",
        message: str = "",
        directory: str = None
    ) -> TaskModel:
        """Create a new task"""
        task = TaskModel(
            task_id=task_id,
            status=status,
            message=message,
            options=options,
            directory=directory,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Create step records
        step_names = ["download", "transcribe", "translate", "tts", "process_video", "upload"]
        for step_name in step_names:
            step = StepResultModel(
                task_id=task_id,
                step_name=step_name,
                status=StepStatusEnum.PENDING.value,
            )
            task.steps.append(step)

        self.session.add(task)
        await self.session.flush()
        logger.info(f"Created task: {task_id}")
        return task

    async def get(self, task_id: str) -> Optional[TaskModel]:
        """Get task by ID with steps"""
        stmt = select(TaskModel).options(
            selectinload(TaskModel.steps)
        ).where(TaskModel.task_id == task_id)

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        status: Optional[str] = None,
        directory: Optional[str] = None,
        limit: int = 100,
        include_deleted: bool = False
    ) -> List[TaskModel]:
        """Get all tasks, optionally filtered by status and directory. Excludes deleted tasks by default."""
        stmt = select(TaskModel).options(
            selectinload(TaskModel.steps)
        ).order_by(TaskModel.created_at.desc()).limit(limit)

        # Filter out deleted tasks by default
        if not include_deleted:
            stmt = stmt.where(TaskModel.deleted == False)

        if status:
            stmt = stmt.where(TaskModel.status == status)

        if directory:
            stmt = stmt.where(TaskModel.directory == directory)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(
        self,
        task_id: str,
        **kwargs
    ) -> Optional[TaskModel]:
        """Update task fields"""
        task = await self.get(task_id)
        if not task:
            return None

        kwargs["updated_at"] = datetime.now()

        # Pre-process: Convert ISO format strings to datetime objects for datetime columns
        for key in list(kwargs.keys()):
            value = kwargs[key]
            if key.endswith("_at") and isinstance(value, str):
                try:
                    kwargs[key] = datetime.fromisoformat(value)
                    logger.debug(f"Converted {key} from string to datetime: {kwargs[key]}")
                except ValueError as e:
                    logger.warning(f"Failed to convert {key} to datetime: {e}")

        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)

        await self.session.flush()
        logger.debug(f"Updated task {task_id}: {list(kwargs.keys())}")
        return task

    async def update_status(
        self,
        task_id: str,
        status: str,
        progress: int = None,
        message: str = None,
        error: str = None
    ) -> Optional[TaskModel]:
        """Update task status and progress"""
        updates = {"status": status, "updated_at": datetime.now()}
        if progress is not None:
            updates["progress"] = progress
        if message is not None:
            updates["message"] = message
        if error is not None:
            updates["error"] = error
        # Auto-clear error when task completes successfully
        elif status in ("completed", "uploaded", "pending_upload", "pending_review"):
            updates["error"] = None

        return await self.update(task_id, **updates)

    async def update_files(
        self,
        task_id: str,
        video_path: str = None,
        audio_path: str = None,
        subtitle_path: str = None,
        translated_subtitle_path: str = None,
        tts_audio_path: str = None,
        final_video_path: str = None,
        thumbnail_path: str = None,
        ai_thumbnail_path: str = None
    ) -> Optional[TaskModel]:
        """Update task file paths"""
        updates = {}
        if video_path:
            updates["video_path"] = video_path
        if audio_path:
            updates["audio_path"] = audio_path
        if subtitle_path:
            updates["subtitle_path"] = subtitle_path
        if translated_subtitle_path:
            updates["translated_subtitle_path"] = translated_subtitle_path
        if tts_audio_path:
            updates["tts_audio_path"] = tts_audio_path
        if final_video_path:
            updates["final_video_path"] = final_video_path
        if thumbnail_path:
            updates["thumbnail_path"] = thumbnail_path
        if ai_thumbnail_path:
            updates["ai_thumbnail_path"] = ai_thumbnail_path

        if updates:
            return await self.update(task_id, **updates)
        return await self.get(task_id)

    async def update_step(
        self,
        task_id: str,
        step_name: str,
        status: str,
        error: str = None,
        output_files: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[StepResultModel]:
        """Update a step's status, creating it if it doesn't exist"""
        task = await self.get(task_id)
        if not task:
            return None

        # Try to find existing step
        for step in task.steps:
            if step.step_name == step_name:
                step.status = status
                if status == StepStatusEnum.RUNNING.value:
                    step.started_at = datetime.now()
                elif status in [StepStatusEnum.COMPLETED.value, StepStatusEnum.FAILED.value, StepStatusEnum.SKIPPED.value]:
                    step.completed_at = datetime.now()
                # Always update error (will be None for success, clearing previous errors)
                step.error = error
                if output_files:
                    step.output_files = output_files
                if metadata:
                    step.step_metadata = metadata

                await self.session.flush()
                return step

        # Step doesn't exist, create it
        new_step = StepResultModel(
            task_id=task_id,
            step_name=step_name,
            status=status,
            error=error,
            output_files=output_files or {},
            step_metadata=metadata or {}
        )
        if status == StepStatusEnum.RUNNING.value:
            new_step.started_at = datetime.now()
        elif status in [StepStatusEnum.COMPLETED.value, StepStatusEnum.FAILED.value, StepStatusEnum.SKIPPED.value]:
            new_step.completed_at = datetime.now()

        self.session.add(new_step)
        await self.session.flush()
        logger.debug(f"Created new step record: task={task_id}, step={step_name}, status={status}")
        return new_step

    async def delete(self, task_id: str) -> bool:
        """Hard delete a task (permanently removes from database)"""
        task = await self.get(task_id)
        if task:
            await self.session.delete(task)
            await self.session.flush()
            logger.info(f"Hard deleted task: {task_id}")
            return True
        return False

    async def soft_delete(self, task_id: str) -> bool:
        """Soft delete a task (marks as deleted but keeps in database)"""
        task = await self.get(task_id)
        if task:
            task.deleted = True
            task.updated_at = datetime.now()
            await self.session.flush()
            logger.info(f"Soft deleted task: {task_id}")
            return True
        return False

    async def update_task_folder(self, task_id: str, task_folder: str) -> Optional[TaskModel]:
        """Update task folder name"""
        return await self.update(task_id, task_folder=task_folder)

    async def get_task_folder(self, task_id: str) -> Optional[str]:
        """Get task folder name"""
        task = await self.get(task_id)
        if task:
            return task.task_folder
        return None

    async def count(self, status: Optional[str] = None) -> int:
        """Count tasks"""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(TaskModel)
        if status:
            stmt = stmt.where(TaskModel.status == status)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def exists_pending_by_url(self, source_url: str) -> bool:
        """Check if a pending/processing task exists for the given source URL.
        Used to prevent duplicate tasks for the same video.
        """
        from sqlalchemy import func

        # Check for pending or processing tasks with the same source_url
        # options is a JSON column - use json_extract for SQLite compatibility
        stmt = select(func.count()).select_from(TaskModel).where(
            TaskModel.status.in_(["pending", "processing", "queued"]),
            TaskModel.deleted == False,
            func.json_extract(TaskModel.options, '$.source_url') == source_url
        )
        result = await self.session.execute(stmt)
        count = result.scalar() or 0
        return count > 0


class SettingsRepository:
    """Repository for settings CRUD operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, key: str) -> Optional[Any]:
        """Get setting value by key"""
        stmt = select(SettingsModel).where(SettingsModel.key == key)
        result = await self.session.execute(stmt)
        setting = result.scalar_one_or_none()
        return setting.value if setting else None

    async def get_all(self) -> Dict[str, Any]:
        """Get all settings as dictionary"""
        stmt = select(SettingsModel)
        result = await self.session.execute(stmt)
        settings = {}
        for row in result.scalars().all():
            settings[row.key] = row.value
        return settings

    async def set(self, key: str, value: Any) -> SettingsModel:
        """Set a setting value (create or update)"""
        stmt = select(SettingsModel).where(SettingsModel.key == key)
        result = await self.session.execute(stmt)
        setting = result.scalar_one_or_none()

        if setting:
            setting.value = value
            setting.updated_at = datetime.now()
        else:
            setting = SettingsModel(key=key, value=value)
            self.session.add(setting)

        await self.session.flush()
        logger.debug(f"Set setting: {key}")
        return setting

    async def set_many(self, settings_dict: Dict[str, Any]) -> None:
        """Set multiple settings at once"""
        for key, value in settings_dict.items():
            await self.set(key, value)

    async def delete(self, key: str) -> bool:
        """Delete a setting"""
        stmt = delete(SettingsModel).where(SettingsModel.key == key)
        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def get_global_settings(self) -> Dict[str, Any]:
        """Get global settings with defaults"""
        stored = await self.get("global_settings")
        defaults = SettingsModel.get_default_settings()

        if stored:
            # Merge stored with defaults (stored takes priority)
            return self._deep_merge(defaults, stored)
        return defaults

    async def update_global_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update global settings (partial update)"""
        current = await self.get_global_settings()
        merged = self._deep_merge(current, updates)
        await self.set("global_settings", merged)
        return merged

    async def reset_global_settings(self) -> Dict[str, Any]:
        """Reset to default settings"""
        defaults = SettingsModel.get_default_settings()
        await self.set("global_settings", defaults)
        return defaults

    def _deep_merge(self, base: dict, updates: dict) -> dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


class CookieRepository:
    """Repository for cookie CRUD operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, platform: str) -> Optional[CookieModel]:
        """Get cookie by platform"""
        stmt = select(CookieModel).where(CookieModel.platform == platform)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(self) -> List[CookieModel]:
        """Get all cookies"""
        stmt = select(CookieModel).order_by(CookieModel.platform)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def save(
        self,
        platform: str,
        cookie_data: str = None,
        cookie_json: Dict[str, Any] = None,
        expires_at: datetime = None,
    ) -> CookieModel:
        """Save or update cookie for a platform (encrypts sensitive data)"""
        cookie = await self.get(platform)

        # Encrypt cookie data before storage
        encrypted_cookie_data = encrypt_string(cookie_data) if cookie_data else None

        if cookie:
            if cookie_data is not None:
                cookie.cookie_data = encrypted_cookie_data
            if cookie_json is not None:
                cookie.cookie_json = cookie_json
            if expires_at is not None:
                cookie.expires_at = expires_at
            cookie.is_valid = True
            cookie.updated_at = datetime.now()
        else:
            cookie = CookieModel(
                platform=platform,
                cookie_data=encrypted_cookie_data,
                cookie_json=cookie_json,
                expires_at=expires_at,
                is_valid=True,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self.session.add(cookie)

        await self.session.flush()
        logger.info(f"Saved encrypted cookie for platform: {platform}")
        return cookie

    async def mark_invalid(self, platform: str) -> bool:
        """Mark a cookie as invalid"""
        cookie = await self.get(platform)
        if cookie:
            cookie.is_valid = False
            cookie.updated_at = datetime.now()
            await self.session.flush()
            logger.info(f"Marked cookie invalid for platform: {platform}")
            return True
        return False

    async def mark_verified(self, platform: str) -> bool:
        """Mark a cookie as verified"""
        cookie = await self.get(platform)
        if cookie:
            cookie.is_valid = True
            cookie.last_verified = datetime.now()
            cookie.updated_at = datetime.now()
            await self.session.flush()
            return True
        return False

    async def delete(self, platform: str) -> bool:
        """Delete a cookie"""
        stmt = delete(CookieModel).where(CookieModel.platform == platform)
        result = await self.session.execute(stmt)
        if result.rowcount > 0:
            logger.info(f"Deleted cookie for platform: {platform}")
            return True
        return False

    async def get_cookie_data(self, platform: str) -> Optional[str]:
        """Get decrypted cookie data string for a platform (Netscape format)"""
        cookie = await self.get(platform)
        if cookie and cookie.is_valid and cookie.cookie_data:
            # Decrypt cookie data (handles both encrypted and legacy plaintext)
            return decrypt_if_needed(cookie.cookie_data)
        return None

    async def write_cookie_file(self, platform: str, file_path: Path) -> bool:
        """Write cookie to file in Netscape format"""
        cookie_data = await self.get_cookie_data(platform)
        if cookie_data:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cookie_data)
            logger.info(f"Wrote cookie file for {platform}: {file_path}")
            return True
        return False


class MetadataPresetRepository:
    """Repository for metadata preset CRUD operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, preset_id: str) -> Optional[MetadataPresetModel]:
        """Get preset by ID"""
        stmt = select(MetadataPresetModel).where(MetadataPresetModel.id == preset_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(self) -> List[MetadataPresetModel]:
        """Get all presets ordered by sort_order and name"""
        stmt = select(MetadataPresetModel).order_by(
            MetadataPresetModel.sort_order,
            MetadataPresetModel.name
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_default(self) -> Optional[MetadataPresetModel]:
        """Get the default preset"""
        stmt = select(MetadataPresetModel).where(MetadataPresetModel.is_default == True)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        preset_id: str,
        name: str,
        description: str = "",
        title_prefix: str = "",
        custom_signature: str = "",
        tags: List[str] = None,
        is_default: bool = False,
        is_builtin: bool = False,
        sort_order: int = 0,
    ) -> MetadataPresetModel:
        """Create a new preset"""
        # If this is set as default, unset other defaults
        if is_default:
            await self._clear_default()

        preset = MetadataPresetModel(
            id=preset_id,
            name=name,
            description=description,
            title_prefix=title_prefix,
            custom_signature=custom_signature,
            tags=tags or [],
            is_default=is_default,
            is_builtin=is_builtin,
            sort_order=sort_order,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.session.add(preset)
        await self.session.flush()
        logger.info(f"Created metadata preset: {preset_id}")
        return preset

    async def update(
        self,
        preset_id: str,
        **kwargs
    ) -> Optional[MetadataPresetModel]:
        """Update a preset"""
        preset = await self.get(preset_id)
        if not preset:
            return None

        # Builtin presets can only have is_default updated
        if preset.is_builtin:
            allowed_keys = {"is_default"}
            kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

        # If setting as default, clear other defaults first
        if kwargs.get("is_default"):
            await self._clear_default()

        kwargs["updated_at"] = datetime.now()

        for key, value in kwargs.items():
            if hasattr(preset, key):
                setattr(preset, key, value)

        await self.session.flush()
        logger.debug(f"Updated preset {preset_id}: {list(kwargs.keys())}")
        return preset

    async def delete(self, preset_id: str) -> bool:
        """Delete a preset (only custom presets can be deleted)"""
        preset = await self.get(preset_id)
        if not preset:
            return False

        # Cannot delete builtin presets
        if preset.is_builtin:
            logger.warning(f"Cannot delete builtin preset: {preset_id}")
            return False

        stmt = delete(MetadataPresetModel).where(MetadataPresetModel.id == preset_id)
        result = await self.session.execute(stmt)
        if result.rowcount > 0:
            logger.info(f"Deleted metadata preset: {preset_id}")
            return True
        return False

    async def set_default(self, preset_id: str) -> Optional[MetadataPresetModel]:
        """Set a preset as the default"""
        preset = await self.get(preset_id)
        if not preset:
            return None

        # Clear existing default
        await self._clear_default()

        # Set this as default
        preset.is_default = True
        preset.updated_at = datetime.now()
        await self.session.flush()
        logger.info(f"Set default metadata preset: {preset_id}")
        return preset

    async def _clear_default(self):
        """Clear the is_default flag from all presets"""
        from sqlalchemy import update as sql_update
        stmt = sql_update(MetadataPresetModel).where(
            MetadataPresetModel.is_default == True
        ).values(is_default=False, updated_at=datetime.now())
        await self.session.execute(stmt)

    async def count(self) -> int:
        """Count total presets"""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(MetadataPresetModel)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def exists(self, preset_id: str) -> bool:
        """Check if a preset exists"""
        preset = await self.get(preset_id)
        return preset is not None


class SubscriptionRepository:
    """Repository for subscription CRUD operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        subscription_id: str,
        platform: str,
        channel_id: str,
        channel_name: str,
        channel_url: str = None,
        channel_avatar: str = None,
        check_interval: int = 60,
        auto_process: bool = True,
        process_options: Dict[str, Any] = None,
        directory: str = None,
        last_video_id: str = None,
        last_video_title: str = None,
        last_video_published_at: datetime = None,
    ) -> SubscriptionModel:
        """Create a new subscription"""
        subscription = SubscriptionModel(
            id=subscription_id,
            platform=platform,
            channel_id=channel_id,
            channel_name=channel_name,
            channel_url=channel_url,
            channel_avatar=channel_avatar,
            directory=directory,
            check_interval=check_interval,
            auto_process=auto_process,
            process_options=process_options,
            enabled=True,
            error_count=0,
            last_video_id=last_video_id,
            last_video_title=last_video_title,
            last_video_published_at=last_video_published_at,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.session.add(subscription)
        await self.session.flush()
        logger.info(f"Created subscription: {subscription_id} for {platform}/{channel_id}")
        return subscription

    async def get(self, subscription_id: str) -> Optional[SubscriptionModel]:
        """Get subscription by ID"""
        stmt = select(SubscriptionModel).where(SubscriptionModel.id == subscription_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_channel(self, platform: str, channel_id: str) -> Optional[SubscriptionModel]:
        """Get subscription by platform and channel ID"""
        stmt = select(SubscriptionModel).where(
            SubscriptionModel.platform == platform,
            SubscriptionModel.channel_id == channel_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        enabled_only: bool = False,
        platform: str = None,
        limit: int = 100
    ) -> List[SubscriptionModel]:
        """Get all subscriptions with optional filters"""
        stmt = select(SubscriptionModel).order_by(
            SubscriptionModel.created_at.desc()
        ).limit(limit)

        if enabled_only:
            stmt = stmt.where(SubscriptionModel.enabled == True)

        if platform:
            stmt = stmt.where(SubscriptionModel.platform == platform)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_due_subscriptions(self) -> List[SubscriptionModel]:
        """Get subscriptions that are due for checking"""
        now = datetime.now()
        stmt = select(SubscriptionModel).where(
            SubscriptionModel.enabled == True,
            (SubscriptionModel.next_check_at == None) | (SubscriptionModel.next_check_at <= now)
        ).order_by(SubscriptionModel.next_check_at.asc())

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(
        self,
        subscription_id: str,
        **kwargs
    ) -> Optional[SubscriptionModel]:
        """Update subscription fields"""
        subscription = await self.get(subscription_id)
        if not subscription:
            return None

        kwargs["updated_at"] = datetime.now()

        for key, value in kwargs.items():
            if hasattr(subscription, key):
                setattr(subscription, key, value)

        await self.session.flush()
        logger.debug(f"Updated subscription {subscription_id}: {list(kwargs.keys())}")
        return subscription

    async def update_last_video(
        self,
        subscription_id: str,
        video_id: str,
        video_title: str,
        published_at: datetime
    ) -> Optional[SubscriptionModel]:
        """Update the last video information"""
        return await self.update(
            subscription_id,
            last_video_id=video_id,
            last_video_title=video_title,
            last_video_published_at=published_at
        )

    async def update_check_status(
        self,
        subscription_id: str,
        next_check_at: datetime,
        error: str = None
    ) -> Optional[SubscriptionModel]:
        """Update check status after a check attempt"""
        updates = {
            "last_checked_at": datetime.now(),
            "next_check_at": next_check_at
        }

        subscription = await self.get(subscription_id)
        if not subscription:
            return None

        if error:
            updates["error_count"] = subscription.error_count + 1
            updates["last_error"] = error
        else:
            updates["error_count"] = 0
            updates["last_error"] = None

        return await self.update(subscription_id, **updates)

    async def delete(self, subscription_id: str) -> bool:
        """Delete a subscription"""
        subscription = await self.get(subscription_id)
        if subscription:
            await self.session.delete(subscription)
            await self.session.flush()
            logger.info(f"Deleted subscription: {subscription_id}")
            return True
        return False

    async def set_enabled(self, subscription_id: str, enabled: bool) -> Optional[SubscriptionModel]:
        """Enable or disable a subscription"""
        return await self.update(subscription_id, enabled=enabled)

    async def count(self, enabled_only: bool = False) -> int:
        """Count subscriptions"""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(SubscriptionModel)
        if enabled_only:
            stmt = stmt.where(SubscriptionModel.enabled == True)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def exists_by_channel(self, platform: str, channel_id: str) -> bool:
        """Check if a subscription exists for this channel"""
        subscription = await self.get_by_channel(platform, channel_id)
        return subscription is not None


class DirectoryRepository:
    """Repository for directory CRUD operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, description: str = None) -> DirectoryModel:
        """Create a new directory"""
        directory = DirectoryModel(
            name=name,
            description=description,
            task_count=0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.session.add(directory)
        await self.session.flush()
        logger.info(f"Created directory: {name}")
        return directory

    async def get(self, directory_id: int) -> Optional[DirectoryModel]:
        """Get directory by ID"""
        stmt = select(DirectoryModel).where(DirectoryModel.id == directory_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[DirectoryModel]:
        """Get directory by name"""
        stmt = select(DirectoryModel).where(DirectoryModel.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def exists(self, name: str) -> bool:
        """Check if directory exists"""
        directory = await self.get_by_name(name)
        return directory is not None

    async def get_all(self, limit: int = 100) -> List[DirectoryModel]:
        """Get all directories"""
        stmt = select(DirectoryModel).order_by(
            DirectoryModel.name.asc()
        ).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, directory_id: int, **kwargs) -> Optional[DirectoryModel]:
        """Update directory fields"""
        kwargs['updated_at'] = datetime.now()
        stmt = update(DirectoryModel).where(
            DirectoryModel.id == directory_id
        ).values(**kwargs)
        await self.session.execute(stmt)
        await self.session.flush()
        return await self.get(directory_id)

    async def delete(self, directory_id: int) -> bool:
        """Delete a directory"""
        directory = await self.get(directory_id)
        if directory:
            await self.session.delete(directory)
            await self.session.flush()
            logger.info(f"Deleted directory: {directory.name}")
            return True
        return False

    async def increment_task_count(self, name: str) -> None:
        """Increment task count for a directory"""
        directory = await self.get_by_name(name)
        if directory:
            await self.update(directory.id, task_count=directory.task_count + 1)

    async def decrement_task_count(self, name: str) -> None:
        """Decrement task count for a directory"""
        directory = await self.get_by_name(name)
        if directory and directory.task_count > 0:
            await self.update(directory.id, task_count=directory.task_count - 1)

    async def get_or_create(self, name: str, description: str = None) -> DirectoryModel:
        """Get existing directory or create new one"""
        directory = await self.get_by_name(name)
        if directory:
            return directory
        return await self.create(name, description)

    async def sync_task_counts(self, task_repo: 'TaskRepository') -> Dict[str, int]:
        """
        Sync task_count for all directories based on actual tasks in database.
        Returns dict of {directory_name: actual_count}
        """
        from collections import defaultdict

        # Get all tasks and count by directory
        all_tasks = await task_repo.get_all(limit=10000)
        counts = defaultdict(int)
        for task in all_tasks:
            if task.directory:
                counts[task.directory] += 1

        # Update each directory's task_count
        directories = await self.get_all()
        updated = {}
        for directory in directories:
            actual_count = counts.get(directory.name, 0)
            if directory.task_count != actual_count:
                await self.update(directory.id, task_count=actual_count)
                updated[directory.name] = actual_count

        return updated


class TrendingRepository:
    """Repository for trending video CRUD operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_category(self, category: str, limit: int = 50, sort_by: str = "view_count") -> List[TrendingVideoModel]:
        """Get trending videos by category, ordered by view_count (most popular) or published_at (newest)"""
        stmt = select(TrendingVideoModel).where(
            TrendingVideoModel.category == category
        )
        
        if sort_by == "published_at":
            stmt = stmt.order_by(TrendingVideoModel.published_at.desc().nullslast())
        else:  # Default: view_count
            stmt = stmt.order_by(TrendingVideoModel.view_count.desc().nullslast())
        
        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_all(self, limit: int = 100, sort_by: str = "view_count") -> List[TrendingVideoModel]:
        """Get all trending videos ordered by view_count (most popular) or published_at (newest)"""
        stmt = select(TrendingVideoModel)
        
        if sort_by == "published_at":
            stmt = stmt.order_by(TrendingVideoModel.published_at.desc().nullslast())
        else:  # Default: view_count
            stmt = stmt.order_by(TrendingVideoModel.view_count.desc().nullslast())
        
        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_video_id(self, video_id: str) -> Optional[TrendingVideoModel]:
        """Get trending video by video ID"""
        stmt = select(TrendingVideoModel).where(TrendingVideoModel.video_id == video_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_video(
        self,
        video_id: str,
        title: str,
        channel_name: str,
        category: str,
        video_url: str,
        channel_url: str = None,
        thumbnail_url: str = None,
        duration: int = 0,
        view_count: int = 0,
        platform: str = "youtube",
        published_at: datetime = None,
    ) -> TrendingVideoModel:
        """Insert or update a trending video"""
        # Convert ISO string to datetime if needed
        if isinstance(published_at, str):
            try:
                published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                published_at = None

        existing = await self.get_by_video_id(video_id)

        if existing:
            existing.title = title
            existing.channel_name = channel_name
            existing.channel_url = channel_url
            existing.thumbnail_url = thumbnail_url
            existing.duration = duration
            existing.view_count = view_count
            existing.category = category
            existing.video_url = video_url
            existing.published_at = published_at
            existing.fetched_at = datetime.now()
            existing.updated_at = datetime.now()
            await self.session.flush()
            return existing

        video = TrendingVideoModel(
            video_id=video_id,
            title=title,
            channel_name=channel_name,
            channel_url=channel_url,
            thumbnail_url=thumbnail_url,
            duration=duration,
            view_count=view_count,
            category=category,
            platform=platform,
            video_url=video_url,
            published_at=published_at,
            fetched_at=datetime.now(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.session.add(video)
        await self.session.flush()
        return video

    async def upsert_videos(self, videos: List[Dict[str, Any]]) -> int:
        """Bulk upsert trending videos. Returns count of upserted videos."""
        count = 0
        for video_data in videos:
            await self.upsert_video(**video_data)
            count += 1
        return count

    async def clear_old_videos(self, before: datetime) -> int:
        """Delete videos fetched before the given time. Returns deleted count."""
        stmt = delete(TrendingVideoModel).where(
            TrendingVideoModel.fetched_at < before
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        deleted = result.rowcount
        if deleted > 0:
            logger.info(f"Cleared {deleted} old trending videos")
        return deleted

    async def clear_category(self, category: str) -> int:
        """Clear all videos in a category. Returns deleted count."""
        stmt = delete(TrendingVideoModel).where(
            TrendingVideoModel.category == category
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount

    async def clear_all(self) -> int:
        """Clear all trending videos. Returns deleted count."""
        stmt = delete(TrendingVideoModel)
        result = await self.session.execute(stmt)
        await self.session.flush()
        deleted = result.rowcount
        if deleted > 0:
            logger.info(f"Cleared all {deleted} trending videos")
        return deleted

    async def get_categories(self) -> List[str]:
        """Get list of unique categories"""
        from sqlalchemy import distinct
        stmt = select(distinct(TrendingVideoModel.category))
        result = await self.session.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def count(self, category: Optional[str] = None) -> int:
        """Count trending videos"""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(TrendingVideoModel)
        if category:
            stmt = stmt.where(TrendingVideoModel.category == category)
        result = await self.session.execute(stmt)
        return result.scalar() or 0


class ApiKeyRepository:
    """Repository for API key CRUD operations with encryption"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, service: str) -> Optional[str]:
        """Get decrypted API key for a service"""
        stmt = select(ApiKeyModel).where(ApiKeyModel.service == service)
        result = await self.session.execute(stmt)
        api_key = result.scalar_one_or_none()

        if not api_key or not api_key.encrypted_key:
            return None

        # Decrypt the key
        try:
            return decrypt_string(api_key.encrypted_key)
        except Exception as e:
            logger.error(f"Failed to decrypt API key for {service}: {e}")
            return None

    async def get_model(self, service: str) -> Optional[ApiKeyModel]:
        """Get the ApiKeyModel for a service"""
        stmt = select(ApiKeyModel).where(ApiKeyModel.service == service)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def set(self, service: str, key: str) -> bool:
        """Save or update API key (encrypts automatically)"""
        if not key:
            return await self.delete(service)

        try:
            encrypted_key = encrypt_string(key)
            existing = await self.get_model(service)

            if existing:
                existing.encrypted_key = encrypted_key
                existing.updated_at = datetime.now()
            else:
                api_key = ApiKeyModel(
                    service=service,
                    encrypted_key=encrypted_key,
                    is_valid=True,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                self.session.add(api_key)

            await self.session.flush()
            logger.info(f"Saved encrypted API key for service: {service}")
            return True
        except Exception as e:
            logger.error(f"Failed to save API key for {service}: {e}")
            return False

    async def delete(self, service: str) -> bool:
        """Delete API key for a service"""
        stmt = delete(ApiKeyModel).where(ApiKeyModel.service == service)
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount > 0

    async def get_all(self, decrypt: bool = False) -> Dict[str, Any]:
        """Get all API keys (masked or decrypted)"""
        stmt = select(ApiKeyModel)
        result = await self.session.execute(stmt)
        api_keys = result.scalars().all()

        keys_dict = {}
        for api_key in api_keys:
            if decrypt and api_key.encrypted_key:
                try:
                    keys_dict[api_key.service] = decrypt_string(api_key.encrypted_key)
                except Exception:
                    keys_dict[api_key.service] = None
            else:
                keys_dict[api_key.service] = "***" if api_key.encrypted_key else ""
        return keys_dict

    async def get_all_models(self) -> List[ApiKeyModel]:
        """Get all API key models"""
        stmt = select(ApiKeyModel)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def set_many(self, keys: Dict[str, str]) -> bool:
        """Set multiple API keys at once"""
        try:
            for service, key in keys.items():
                await self.set(service, key)
            return True
        except Exception as e:
            logger.error(f"Failed to set multiple API keys: {e}")
            return False

    async def mark_invalid(self, service: str) -> bool:
        """Mark an API key as invalid"""
        existing = await self.get_model(service)
        if existing:
            existing.is_valid = False
            existing.updated_at = datetime.now()
            await self.session.flush()
            return True
        return False

    async def mark_verified(self, service: str) -> bool:
        """Mark an API key as verified"""
        existing = await self.get_model(service)
        if existing:
            existing.is_valid = True
            existing.last_verified = datetime.now()
            existing.updated_at = datetime.now()
            await self.session.flush()
            return True
        return False

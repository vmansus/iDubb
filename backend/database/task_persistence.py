"""
Task Persistence Layer
Bridges the in-memory pipeline with database persistence
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import asdict
from loguru import logger

from .connection import get_db
from .repository import TaskRepository, SettingsRepository
from .models import TaskModel, StepStatusEnum


class TaskPersistenceManager:
    """
    Manages task persistence to database.
    Works alongside VideoPipeline to save state changes.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._save_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        logger.info("TaskPersistenceManager initialized")

    async def start(self):
        """Start the background save worker"""
        if self._running:
            return
        self._running = True
        asyncio.create_task(self._save_worker())
        logger.info("Task persistence worker started")

    async def stop(self):
        """Stop the background save worker"""
        self._running = False
        # Process remaining items
        while not self._save_queue.empty():
            try:
                item = self._save_queue.get_nowait()
                await self._process_save(item)
            except asyncio.QueueEmpty:
                break
        logger.info("Task persistence worker stopped")

    async def _save_worker(self):
        """Background worker to process save queue"""
        while self._running:
            try:
                item = await asyncio.wait_for(self._save_queue.get(), timeout=1.0)
                await self._process_save(item)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Save worker error: {e}")

    async def _process_save(self, item: Dict[str, Any]):
        """Process a save item"""
        try:
            action = item.get("action")
            async with get_db() as session:
                repo = TaskRepository(session)

                if action == "create":
                    await repo.create(
                        task_id=item["task_id"],
                        options=item["options"],
                        status=item.get("status", "pending"),
                        message=item.get("message", ""),
                        directory=item.get("directory")
                    )

                elif action == "update_status":
                    await repo.update_status(
                        task_id=item["task_id"],
                        status=item["status"],
                        progress=item.get("progress"),
                        message=item.get("message"),
                        error=item.get("error")
                    )

                elif action == "update_files":
                    await repo.update_files(
                        task_id=item["task_id"],
                        **item.get("files", {})
                    )

                elif action == "update_step":
                    await repo.update_step(
                        task_id=item["task_id"],
                        step_name=item["step_name"],
                        status=item["status"],
                        error=item.get("error"),
                        output_files=item.get("output_files"),
                        metadata=item.get("metadata")
                    )

                elif action == "update_full":
                    await repo.update(item["task_id"], **item.get("data", {}))

        except Exception as e:
            logger.error(f"Failed to save task state: {e}")

    # Public API methods

    async def save_task_created(
        self,
        task_id: str,
        options: Dict[str, Any],
        status: str = "pending",
        message: str = "",
        directory: str = None
    ):
        """Save a newly created task - uses synchronous save to ensure persistence"""
        # Use synchronous save for task creation to ensure it's persisted immediately
        try:
            async with get_db() as session:
                repo = TaskRepository(session)
                await repo.create(
                    task_id=task_id,
                    options=options,
                    status=status,
                    message=message,
                    directory=directory
                )
            logger.info(f"Task {task_id} saved to database")
        except Exception as e:
            logger.error(f"Failed to save task {task_id}: {e}")
            raise

    async def save_task_status(
        self,
        task_id: str,
        status: str,
        progress: int = None,
        message: str = None,
        error: str = None
    ):
        """Save task status update"""
        await self._save_queue.put({
            "action": "update_status",
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
        })

    async def save_task_files(
        self,
        task_id: str,
        video_path: str = None,
        audio_path: str = None,
        subtitle_path: str = None,
        translated_subtitle_path: str = None,
        tts_audio_path: str = None,
        final_video_path: str = None,
        thumbnail_path: str = None
    ):
        """Save task file paths"""
        files = {}
        if video_path:
            files["video_path"] = video_path
        if audio_path:
            files["audio_path"] = audio_path
        if subtitle_path:
            files["subtitle_path"] = subtitle_path
        if translated_subtitle_path:
            files["translated_subtitle_path"] = translated_subtitle_path
        if tts_audio_path:
            files["tts_audio_path"] = tts_audio_path
        if final_video_path:
            files["final_video_path"] = final_video_path
        if thumbnail_path:
            files["thumbnail_path"] = thumbnail_path

        if files:
            await self._save_queue.put({
                "action": "update_files",
                "task_id": task_id,
                "files": files,
            })

    async def save_step_status(
        self,
        task_id: str,
        step_name: str,
        status: str,
        error: str = None,
        output_files: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Save step status update"""
        await self._save_queue.put({
            "action": "update_step",
            "task_id": task_id,
            "step_name": step_name,
            "status": status,
            "error": error,
            "output_files": output_files,
            "metadata": metadata,
        })

    async def save_video_info(self, task_id: str, video_info: Dict[str, Any]):
        """Save video info"""
        await self._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {"video_info": video_info},
        })

    async def save_upload_results(self, task_id: str, upload_results: Dict[str, Any]):
        """Save upload results"""
        await self._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {"upload_results": upload_results},
        })

    async def save_options(self, task_id: str, options: Dict[str, Any]):
        """Save updated task options"""
        await self._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {"options": options},
        })

    async def save_generated_metadata(self, task_id: str, metadata: Dict[str, Any]):
        """Save AI-generated metadata for a task"""
        # Add timestamp if not present
        if "generated_at" not in metadata:
            metadata["generated_at"] = datetime.now().isoformat()
        await self._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {"generated_metadata": metadata},
        })

    async def load_generated_metadata(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load generated metadata for a task"""
        async with get_db() as session:
            repo = TaskRepository(session)
            task = await repo.get(task_id)
            if task:
                return task.generated_metadata
        return None

    async def approve_metadata(self, task_id: str) -> bool:
        """Approve metadata for a task and update status to pending_upload.
        This is done synchronously (not queued) to avoid race conditions.
        """
        approval_time = datetime.now()
        logger.info(f"Approving metadata for task {task_id}, approval_time: {approval_time}")
        try:
            async with get_db() as session:
                repo = TaskRepository(session)
                await repo.update(
                    task_id,
                    metadata_approved=True,
                    metadata_approved_at=approval_time,
                    status="pending_upload"
                )
            logger.info(f"Metadata approved for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to approve metadata for task {task_id}: {e}")
            return False

    async def is_metadata_approved(self, task_id: str) -> bool:
        """Check if metadata is approved for a task"""
        async with get_db() as session:
            repo = TaskRepository(session)
            task = await repo.get(task_id)
            if task:
                return task.metadata_approved
        return False

    async def save_ai_thumbnail(
        self,
        task_id: str,
        ai_thumbnail_path: str,
        ai_thumbnail_title: str
    ):
        """Save AI-generated thumbnail info"""
        await self._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {
                "ai_thumbnail_path": ai_thumbnail_path,
                "ai_thumbnail_title": ai_thumbnail_title,
            },
        })

    async def update_use_ai_thumbnail(self, task_id: str, use_ai: bool):
        """Update whether to use AI thumbnail for upload"""
        await self._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {"use_ai_thumbnail": use_ai},
        })

    async def save_proofreading_result(self, task_id: str, result: Dict[str, Any]):
        """Save proofreading result for a task"""
        await self._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {"proofreading_result": result},
        })

    async def load_proofreading_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load proofreading result for a task"""
        async with get_db() as session:
            repo = TaskRepository(session)
            task = await repo.get(task_id)
            if task:
                return task.proofreading_result
        return None

    async def save_optimization_result(self, task_id: str, result: Dict[str, Any]):
        """Save optimization result for a task"""
        await self._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {"optimization_result": result},
        })

    async def load_optimization_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load optimization result for a task"""
        async with get_db() as session:
            repo = TaskRepository(session)
            task = await repo.get(task_id)
            if task:
                return task.optimization_result
        return None

    async def load_ai_thumbnail_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load AI thumbnail info for a task"""
        async with get_db() as session:
            repo = TaskRepository(session)
            task = await repo.get(task_id)
            if task:
                return {
                    "ai_thumbnail_path": task.ai_thumbnail_path,
                    "ai_thumbnail_title": task.ai_thumbnail_title,
                    "use_ai_thumbnail": task.use_ai_thumbnail,
                }
        return None

    # Load methods (synchronous database access)

    async def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load a task from database"""
        async with get_db() as session:
            repo = TaskRepository(session)
            task = await repo.get(task_id)
            if task:
                return task.to_dict()
        return None

    async def load_all_tasks(
        self,
        status: Optional[str] = None,
        directory: Optional[str] = None,
        limit: int = 100,
        include_deleted: bool = False
    ) -> List[Dict[str, Any]]:
        """Load all tasks from database. Excludes deleted tasks by default."""
        async with get_db() as session:
            repo = TaskRepository(session)
            tasks = await repo.get_all(status=status, directory=directory, limit=limit, include_deleted=include_deleted)
            return [task.to_dict() for task in tasks]

    async def delete_task(self, task_id: str) -> bool:
        """Hard delete a task from database"""
        async with get_db() as session:
            repo = TaskRepository(session)
            return await repo.delete(task_id)

    async def soft_delete_task(self, task_id: str) -> bool:
        """Soft delete a task (marks as deleted)"""
        async with get_db() as session:
            repo = TaskRepository(session)
            return await repo.soft_delete(task_id)

    async def update_task_folder(self, task_id: str, task_folder: str):
        """Update task folder name"""
        await self._save_queue.put({
            "action": "update_full",
            "task_id": task_id,
            "data": {"task_folder": task_folder},
        })

    async def get_task_folder(self, task_id: str) -> Optional[str]:
        """Get task folder name"""
        async with get_db() as session:
            repo = TaskRepository(session)
            return await repo.get_task_folder(task_id)


class SettingsPersistenceManager:
    """
    Manages settings persistence to database.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_global_settings(self) -> Dict[str, Any]:
        """Get global settings (using separate keys per category)"""
        async with get_db() as session:
            repo = SettingsRepository(session)

            # Load each category separately
            settings = {}
            categories = ["storage", "processing", "video", "translation", "tts",
                         "subtitle", "audio", "metadata", "bilibili", "thumbnail",
                         "proofreading", "trending", "upload"]

            for category in categories:
                data = await repo.get(category)
                if data:
                    settings[category] = data

            return settings

    async def update_global_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update global settings (using separate keys per category)"""
        async with get_db() as session:
            repo = SettingsRepository(session)

            # Update each category separately
            for category, data in updates.items():
                if isinstance(data, dict):
                    # Get existing data and merge
                    existing = await repo.get(category) or {}
                    merged = {**existing, **data}
                    await repo.set(category, merged)

            # Return updated settings
            return await self.get_global_settings()

    async def reset_global_settings(self) -> Dict[str, Any]:
        """Reset to default settings"""
        async with get_db() as session:
            repo = SettingsRepository(session)

            # Delete all category keys
            categories = ["storage", "processing", "video", "translation", "tts",
                         "subtitle", "audio", "metadata", "bilibili", "thumbnail",
                         "proofreading", "trending", "upload"]

            for category in categories:
                await repo.delete(category)

            return {}

    # Custom Preset Methods

    async def get_custom_presets(self) -> List[Dict[str, Any]]:
        """Get all custom subtitle presets"""
        async with get_db() as session:
            repo = SettingsRepository(session)
            # Use dedicated custom_presets key (new storage)
            presets = await repo.get("custom_presets")
            if presets is not None:
                return presets

            # Fallback to legacy global_settings (for migration)
            settings = await repo.get_global_settings()
            return settings.get("custom_presets", [])

    async def save_custom_preset(self, preset: Dict[str, Any]) -> None:
        """Save a custom subtitle preset"""
        async with get_db() as session:
            repo = SettingsRepository(session)
            # Use dedicated custom_presets key
            presets = await repo.get("custom_presets") or []

            # Check if preset with this ID already exists
            existing_idx = None
            for i, p in enumerate(presets):
                if p.get("id") == preset.get("id"):
                    existing_idx = i
                    break

            if existing_idx is not None:
                presets[existing_idx] = preset
            else:
                presets.append(preset)

            await repo.set("custom_presets", presets)

    async def delete_custom_preset(self, preset_id: str) -> bool:
        """Delete a custom subtitle preset"""
        async with get_db() as session:
            repo = SettingsRepository(session)
            # Use dedicated custom_presets key
            presets = await repo.get("custom_presets") or []

            new_presets = [p for p in presets if p.get("id") != preset_id]

            if len(new_presets) == len(presets):
                return False  # Preset not found

            await repo.set("custom_presets", new_presets)
            return True

    async def get_preset_by_id(self, preset_id: str) -> Optional[Dict[str, Any]]:
        """Get a preset by ID (builtin or custom)"""
        from settings_store import BUILTIN_PRESETS

        # First check builtin presets
        for preset in BUILTIN_PRESETS:
            if preset.id == preset_id:
                return preset.to_dict()

        # Check custom presets
        custom_presets = await self.get_custom_presets()
        for preset in custom_presets:
            if preset.get("id") == preset_id:
                return preset

        return None


# Global instances
task_persistence = TaskPersistenceManager()
settings_persistence = SettingsPersistenceManager()

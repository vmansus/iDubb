#!/usr/bin/env python3
"""
Migration script to add video dimensions (width, height, is_vertical) to historical tasks.
Run this once to update all existing tasks that are missing video dimension info.

Usage:
    cd backend
    python scripts/migrate_video_dimensions.py
"""
import asyncio
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import sys
from sqlalchemy.orm.attributes import flag_modified

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db
from database.repository import TaskRepository
from database.models import TaskModel


async def get_video_dimensions(video_path: Path) -> Optional[Dict[str, Any]]:
    """Get video dimensions using ffprobe"""
    if not video_path or not video_path.exists():
        return None

    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Find video stream
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                width = stream.get("width", 0)
                height = stream.get("height", 0)

                if width > 0 and height > 0:
                    aspect_ratio = width / height
                    is_vertical = aspect_ratio < 0.9

                    return {
                        "width": width,
                        "height": height,
                        "is_vertical": is_vertical
                    }

        return None
    except Exception as e:
        logger.error(f"Failed to get video dimensions for {video_path}: {e}")
        return None


async def get_image_dimensions(image_path: Path) -> Optional[Dict[str, Any]]:
    """Get image dimensions using ffprobe (for thumbnails)"""
    if not image_path or not image_path.exists():
        return None

    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(image_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Find video stream (images are treated as single-frame video)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                width = stream.get("width", 0)
                height = stream.get("height", 0)

                if width > 0 and height > 0:
                    aspect_ratio = width / height
                    is_vertical = aspect_ratio < 0.9

                    return {
                        "width": width,
                        "height": height,
                        "is_vertical": is_vertical
                    }

        return None
    except Exception as e:
        logger.error(f"Failed to get image dimensions for {image_path}: {e}")
        return None


async def find_video_file(task_data: Dict[str, Any]) -> Optional[Path]:
    """Find the video file for a task"""
    task_id = task_data.get("task_id", "")
    task_folder = task_data.get("task_folder")
    video_info = task_data.get("video_info") or {}

    # Get the data directory (relative to backend)
    data_dir = Path(__file__).parent.parent.parent / "data"

    # Try video_path first (stored in database)
    video_path = task_data.get("video_path")
    if video_path:
        path = Path(video_path)
        if path.exists():
            return path

    # Extract video ID from thumbnail_path (e.g., "/data/downloads/ikDNH4aG-lI_title_thumb.jpg")
    thumbnail_path = video_info.get("thumbnail_path", "")
    video_id = None
    if thumbnail_path:
        filename = Path(thumbnail_path).stem  # e.g., "ikDNH4aG-lI_title_thumb"
        if "_" in filename:
            video_id = filename.split("_")[0]  # e.g., "ikDNH4aG-lI"

    # Search in downloads directory for video files
    downloads_dir = data_dir / "downloads"
    if downloads_dir.exists():
        for ext in [".webm", ".mp4", ".mkv", ".mov", ".avi"]:
            # Search by video_id prefix
            if video_id:
                for video_file in downloads_dir.glob(f"{video_id}_*{ext}"):
                    if video_file.is_file() and "_thumb" not in video_file.name:
                        return video_file

    # Try to find video in task folder within processed directory
    if task_folder:
        for base_dir in [data_dir / "processed" / task_folder, data_dir / "downloads" / task_folder]:
            if base_dir.exists():
                for ext in [".webm", ".mp4", ".mkv", ".mov", ".avi"]:
                    for pattern in ["video*", "final*", "original*", "*"]:
                        for video_file in base_dir.glob(f"{pattern}{ext}"):
                            if video_file.is_file() and "_thumb" not in video_file.name:
                                return video_file

    # Check uploads directory by task_id prefix
    uploads_dir = data_dir / "uploads"
    if uploads_dir.exists():
        for ext in [".webm", ".mp4", ".mkv", ".mov", ".avi"]:
            for video_file in uploads_dir.glob(f"{task_id[:8]}*{ext}"):
                if video_file.is_file():
                    return video_file

    return None


async def migrate_task(session, task: TaskModel) -> bool:
    """Migrate a single task to add video dimensions"""
    task_id = task.task_id
    task_data = task.to_dict()
    video_info = task_data.get("video_info") or {}

    # Skip if already has dimensions
    if video_info.get("is_vertical") is not None:
        logger.debug(f"Task {task_id[:8]} already has video dimensions, skipping")
        return False

    dimensions = None

    # Try 1: Get dimensions from video file
    video_path = await find_video_file(task_data)
    if video_path:
        dimensions = await get_video_dimensions(video_path)
        if dimensions:
            logger.debug(f"Task {task_id[:8]}: Got dimensions from video file")

    # Try 2: Get dimensions from thumbnail image
    if not dimensions:
        thumbnail_path = video_info.get("thumbnail_path")
        if thumbnail_path:
            thumb_path = Path(thumbnail_path)
            if thumb_path.exists():
                dimensions = await get_image_dimensions(thumb_path)
                if dimensions:
                    logger.debug(f"Task {task_id[:8]}: Got dimensions from thumbnail")

    # Try 3: Infer from actual_height if available (assume standard 16:9)
    if not dimensions and video_info.get("actual_height"):
        actual_height = video_info["actual_height"]
        # Standard 16:9 ratios
        standard_16_9 = {
            720: 1280, 1080: 1920, 1440: 2560, 2160: 3840, 3840: 7680
        }
        if actual_height in standard_16_9:
            width = standard_16_9[actual_height]
            dimensions = {
                "width": width,
                "height": actual_height,
                "is_vertical": False  # Standard resolutions are horizontal
            }
            logger.debug(f"Task {task_id[:8]}: Inferred 16:9 dimensions from actual_height")

    if not dimensions:
        logger.warning(f"Task {task_id[:8]}: No dimensions could be determined")
        return False

    # Create a NEW dict with the updated video_info (important for SQLAlchemy change detection)
    new_video_info = dict(video_info)
    new_video_info["width"] = dimensions["width"]
    new_video_info["height"] = dimensions["height"]
    new_video_info["is_vertical"] = dimensions["is_vertical"]

    # Set the new dict and explicitly mark as modified
    task.video_info = new_video_info
    flag_modified(task, "video_info")

    logger.info(
        f"Task {task_id[:8]}: Updated dimensions "
        f"{dimensions['width']}x{dimensions['height']} "
        f"(vertical={dimensions['is_vertical']})"
    )
    return True


async def main():
    """Main migration function"""
    logger.info("Starting video dimensions migration...")

    updated_count = 0
    skipped_count = 0
    error_count = 0

    async with get_db() as session:
        repo = TaskRepository(session)

        # Get all tasks (including deleted ones for completeness)
        tasks = await repo.get_all(limit=1000, include_deleted=True)
        total_tasks = len(tasks)

        logger.info(f"Found {total_tasks} tasks to process")

        for i, task in enumerate(tasks):
            try:
                if await migrate_task(session, task):
                    updated_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                logger.error(f"Error migrating task {task.task_id[:8]}: {e}")
                error_count += 1

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{total_tasks}")

        # Commit all changes
        await session.commit()

    logger.info(f"""
Migration completed:
  - Total tasks: {total_tasks}
  - Updated: {updated_count}
  - Skipped (already had dimensions or no video): {skipped_count}
  - Errors: {error_count}
""")


if __name__ == "__main__":
    asyncio.run(main())

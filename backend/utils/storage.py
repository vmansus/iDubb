"""
Storage utilities for task folder management
"""
import re
import shutil
from pathlib import Path
from typing import Optional
from loguru import logger

from config import settings


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    Sanitize a string for use as a filename.
    Removes/replaces invalid characters and limits length.

    Args:
        name: Original name to sanitize
        max_length: Maximum length of the result

    Returns:
        Sanitized filename-safe string
    """
    if not name:
        return ""

    # Replace common invalid characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, '_', name)

    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')

    return sanitized


def generate_task_folder_name(task_id: str, video_title: Optional[str] = None) -> str:
    """
    Generate a task folder name from task ID and video title.

    Format: {task_id}_{sanitized_video_title}
    Example: abc123_How_to_Cook_Pasta

    Args:
        task_id: The task ID (typically 8 chars)
        video_title: Optional video title to append

    Returns:
        Folder name string
    """
    if not video_title:
        return task_id

    sanitized_title = sanitize_filename(video_title, max_length=50)

    if sanitized_title:
        return f"{task_id}_{sanitized_title}"

    return task_id


async def get_output_directory() -> Path:
    """
    Get the configured output directory from settings.
    Falls back to default PROCESSED_DIR if not configured.

    Returns:
        Path to the output directory
    """
    from database.task_persistence import settings_persistence

    try:
        global_settings = await settings_persistence.get_global_settings()
        storage_settings = global_settings.get("storage", {})
        output_dir = storage_settings.get("output_directory", "")

        if output_dir and output_dir.strip():
            output_path = Path(output_dir)
            # Create directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path
    except Exception as e:
        logger.warning(f"Failed to get output directory from settings: {e}")

    # Fall back to default
    return settings.PROCESSED_DIR


def get_output_directory_sync() -> Path:
    """
    Synchronous version of get_output_directory.
    Used during initialization or non-async contexts.

    Returns:
        Path to the output directory
    """
    return settings.PROCESSED_DIR


async def get_task_directory(
    task_id: str,
    task_folder: Optional[str] = None,
    directory: Optional[str] = None
) -> Path:
    """
    Get the directory for a specific task's files.

    Args:
        task_id: The task ID
        task_folder: Optional pre-computed task folder name
        directory: Optional directory name for grouping tasks

    Returns:
        Path to the task's directory
    """
    output_dir = await get_output_directory()

    # Add directory layer if specified
    if directory:
        output_dir = output_dir / sanitize_filename(directory, max_length=100)

    folder_name = task_folder if task_folder else task_id
    task_dir = output_dir / folder_name

    # Create directory if it doesn't exist
    task_dir.mkdir(parents=True, exist_ok=True)

    return task_dir


async def delete_task_directory(
    task_id: str,
    task_folder: Optional[str] = None,
    task_data: Optional[dict] = None
) -> bool:
    """
    Delete a task's directory and all its contents.

    Args:
        task_id: The task ID
        task_folder: Optional task folder name (for backwards compatibility)
        task_data: Optional task data dict containing file paths

    Returns:
        True if deleted successfully, False otherwise
    """
    output_dir = await get_output_directory()
    deleted = False
    deleted_paths = set()  # Track deleted paths to avoid duplicates

    # Helper to delete a directory
    def delete_dir(dir_path: Path) -> bool:
        if dir_path and dir_path.exists() and str(dir_path) not in deleted_paths:
            try:
                shutil.rmtree(dir_path)
                deleted_paths.add(str(dir_path))
                logger.info(f"Deleted task directory: {dir_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete directory {dir_path}: {e}")
        return False

    # 1. Try to find and delete the task directory from file paths in task_data
    if task_data:
        # Extract directory from any file path in task data
        path_keys = ['video_path', 'audio_path', 'subtitle_path', 'translated_subtitle_path',
                     'tts_audio_path', 'final_video_path', 'thumbnail_path', 'ai_thumbnail_path']

        for key in path_keys:
            file_path = task_data.get(key)
            if file_path:
                path = Path(file_path)
                if path.exists():
                    # Delete the parent directory (task folder)
                    task_dir = path.parent
                    if task_dir.name.startswith(task_id) or (task_folder and task_dir.name == task_folder):
                        if delete_dir(task_dir):
                            deleted = True
                            break  # Directory deleted, no need to continue

    # 2. Try configured output directory
    if task_folder:
        task_dir = output_dir / task_folder
        if delete_dir(task_dir):
            deleted = True

    # 3. Try task_id only
    task_dir_by_id = output_dir / task_id
    if delete_dir(task_dir_by_id):
        deleted = True

    # 4. Check common external data locations
    external_locations = [
        Path("/Users/vmansus/external/data"),
        Path.home() / "external" / "data",
        settings.DATA_DIR.parent / "external" / "data",
    ]

    for ext_dir in external_locations:
        if ext_dir.exists():
            # Try with task_folder
            if task_folder:
                task_dir = ext_dir / task_folder
                if delete_dir(task_dir):
                    deleted = True

            # Try to find any folder starting with task_id
            try:
                for folder in ext_dir.iterdir():
                    if folder.is_dir() and folder.name.startswith(task_id):
                        if delete_dir(folder):
                            deleted = True
            except Exception as e:
                logger.warning(f"Failed to scan {ext_dir}: {e}")

    # 5. Check for legacy files with task_id prefix in output directory
    try:
        for file_path in output_dir.glob(f"{task_id}*"):
            if file_path.is_file():
                file_path.unlink()
                logger.info(f"Deleted legacy task file: {file_path}")
                deleted = True
    except Exception as e:
        logger.error(f"Failed to delete legacy task files: {e}")

    # 6. Check in downloads directory
    downloads_dir = settings.DOWNLOADS_DIR
    try:
        for file_path in downloads_dir.glob(f"*{task_id}*"):
            if file_path.is_file():
                file_path.unlink()
                logger.info(f"Deleted download file: {file_path}")
                deleted = True
    except Exception as e:
        logger.error(f"Failed to delete download files: {e}")

    return deleted

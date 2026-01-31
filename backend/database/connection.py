"""
Database Connection Management
Async SQLite connection using aiosqlite
"""
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from loguru import logger

from .models import Base
from config import settings

# Database path
DB_PATH = settings.DATA_DIR / "idubb.db"

# Async engine
_engine = None
_session_factory = None


def get_engine():
    """Get or create async engine"""
    global _engine
    if _engine is None:
        # Ensure data directory exists
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        db_url = f"sqlite+aiosqlite:///{DB_PATH}"
        logger.info(f"Creating database engine: {db_url}")

        _engine = create_async_engine(
            db_url,
            echo=False,  # Disable SQL logging (too noisy even in DEBUG mode)
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return _engine


def get_session_factory():
    """Get or create session factory"""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def init_db():
    """Initialize database - create tables if not exist"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully")

    # Run migrations for existing tables
    await run_migrations()

    # Initialize builtin metadata presets
    await init_builtin_metadata_presets()


async def run_migrations():
    """Run database migrations for existing tables"""
    from sqlalchemy import text

    engine = get_engine()

    migrations = [
        # Add deleted column to tasks table (soft delete support)
        {
            "check": "SELECT deleted FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN deleted BOOLEAN DEFAULT 0 NOT NULL",
            "description": "Add deleted column to tasks"
        },
        # Add task_folder column to tasks table
        {
            "check": "SELECT task_folder FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN task_folder VARCHAR(255)",
            "description": "Add task_folder column to tasks"
        },
        # Add generated_metadata column to tasks table (AI metadata persistence)
        {
            "check": "SELECT generated_metadata FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN generated_metadata JSON",
            "description": "Add generated_metadata column to tasks"
        },
        # Add metadata_approved column to tasks table
        {
            "check": "SELECT metadata_approved FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN metadata_approved BOOLEAN DEFAULT 0 NOT NULL",
            "description": "Add metadata_approved column to tasks"
        },
        # Add metadata_approved_at column to tasks table
        {
            "check": "SELECT metadata_approved_at FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN metadata_approved_at DATETIME",
            "description": "Add metadata_approved_at column to tasks"
        },
        # Add AI thumbnail columns
        {
            "check": "SELECT ai_thumbnail_path FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN ai_thumbnail_path TEXT",
            "description": "Add ai_thumbnail_path column to tasks"
        },
        {
            "check": "SELECT ai_thumbnail_title FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN ai_thumbnail_title TEXT",
            "description": "Add ai_thumbnail_title column to tasks"
        },
        {
            "check": "SELECT use_ai_thumbnail FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN use_ai_thumbnail BOOLEAN DEFAULT 0",
            "description": "Add use_ai_thumbnail column to tasks"
        },
        # Add proofreading_result column for storing proofreading results
        {
            "check": "SELECT proofreading_result FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN proofreading_result JSON",
            "description": "Add proofreading_result column to tasks"
        },
        # Add optimization_result column for storing optimization results
        {
            "check": "SELECT optimization_result FROM tasks LIMIT 1",
            "migrate": "ALTER TABLE tasks ADD COLUMN optimization_result JSON",
            "description": "Add optimization_result column to tasks"
        },
    ]

    async with engine.connect() as conn:
        for migration in migrations:
            try:
                # Check if column exists
                await conn.execute(text(migration["check"]))
                logger.debug(f"Migration already applied: {migration['description']}")
            except Exception:
                # Column doesn't exist, run migration
                try:
                    await conn.execute(text(migration["migrate"]))
                    await conn.commit()
                    logger.info(f"Migration applied: {migration['description']}")
                except Exception as e:
                    logger.warning(f"Migration failed: {migration['description']} - {e}")

        # Data migration: Convert 'completed' status to new statuses
        # - If has upload_results with success -> 'uploaded'
        # - If metadata_approved -> 'pending_upload'
        # - Otherwise -> 'pending_review'
        try:
            import json

            # First, update tasks with successful uploads to 'uploaded'
            result = await conn.execute(text(
                "SELECT task_id, upload_results FROM tasks WHERE status = 'completed'"
            ))
            completed_tasks = result.fetchall()

            for row in completed_tasks:
                task_id = row[0]
                upload_results_str = row[1]

                # Determine new status based on upload_results
                new_status = 'pending_review'  # Default

                if upload_results_str:
                    try:
                        upload_results = json.loads(upload_results_str) if isinstance(upload_results_str, str) else upload_results_str
                        # Check if any platform upload was successful
                        if upload_results and any(
                            r.get('success', False) for r in upload_results.values()
                        ):
                            new_status = 'uploaded'
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass

                # Update the task status
                await conn.execute(text(
                    "UPDATE tasks SET status = :new_status WHERE task_id = :task_id"
                ), {"new_status": new_status, "task_id": task_id})

            if completed_tasks:
                await conn.commit()
                logger.info(f"Migrated {len(completed_tasks)} tasks from 'completed' to new statuses")

        except Exception as e:
            logger.warning(f"Data migration for completed tasks failed: {e}")


async def close_db():
    """Close database connection"""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
    logger.info("Database connection closed")


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session as async context manager"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            await session.close()


async def init_builtin_metadata_presets():
    """Initialize builtin metadata presets if they don't exist"""
    from .repository import MetadataPresetRepository

    # Builtin presets definition
    BUILTIN_PRESETS = [
        {
            "id": "chinese_subtitles",
            "name": "中文字幕",
            "description": "适合翻译类视频，添加[中字]前缀",
            "title_prefix": "[中字]",
            "custom_signature": "",
            "tags": ["通用", "字幕", "翻译"],
            "is_default": True,
            "sort_order": 1,
        },
        {
            "id": "bilingual_subtitles",
            "name": "双语字幕",
            "description": "适合双语字幕视频，添加[双语]前缀",
            "title_prefix": "[双语]",
            "custom_signature": "",
            "tags": ["通用", "双语", "字幕"],
            "is_default": False,
            "sort_order": 2,
        },
        {
            "id": "tech_tutorial",
            "name": "科技教程",
            "description": "适合科技、AI、教程类视频",
            "title_prefix": "[翻译]",
            "custom_signature": "",
            "tags": ["科技", "教程", "AI", "技术"],
            "is_default": False,
            "sort_order": 3,
        },
        {
            "id": "entertainment",
            "name": "娱乐休闲",
            "description": "适合娱乐、休闲类视频",
            "title_prefix": "[中字]",
            "custom_signature": "",
            "tags": ["娱乐", "休闲", "综艺"],
            "is_default": False,
            "sort_order": 4,
        },
        {
            "id": "no_prefix",
            "name": "无前缀",
            "description": "不添加任何前缀，适合原创或自制内容",
            "title_prefix": "",
            "custom_signature": "",
            "tags": ["自制", "原创", "无前缀"],
            "is_default": False,
            "sort_order": 5,
        },
    ]

    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            repo = MetadataPresetRepository(session)

            for preset_data in BUILTIN_PRESETS:
                # Check if preset already exists
                existing = await repo.get(preset_data["id"])
                if not existing:
                    await repo.create(
                        preset_id=preset_data["id"],
                        name=preset_data["name"],
                        description=preset_data["description"],
                        title_prefix=preset_data["title_prefix"],
                        custom_signature=preset_data["custom_signature"],
                        tags=preset_data["tags"],
                        is_default=preset_data["is_default"],
                        is_builtin=True,
                        sort_order=preset_data["sort_order"],
                    )
                    logger.debug(f"Created builtin metadata preset: {preset_data['id']}")

            await session.commit()
            logger.info("Builtin metadata presets initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize builtin metadata presets: {e}")
            await session.rollback()


async def get_db_session() -> AsyncSession:
    """Get a new database session (caller must manage lifecycle)"""
    session_factory = get_session_factory()
    return session_factory()

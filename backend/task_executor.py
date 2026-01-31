"""
Task Executor - Manages concurrent task execution with queue

Provides:
- Configurable concurrent task limit
- Task queue for pending tasks
- Queue position tracking
- Automatic recovery on restart
"""
import asyncio
from typing import Dict, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from config import settings


@dataclass
class QueuedTask:
    """Represents a task in the queue"""
    task_id: str
    task: any  # ProcessingTask
    queued_at: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Higher = more urgent


class TaskExecutor:
    """
    Manages concurrent task execution with a queue system.

    Features:
    - Limits concurrent task execution
    - Queues tasks when limit is reached
    - Tracks queue position
    - Supports priority (future)
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

        # Get max concurrent from global settings if available, fallback to config
        try:
            from settings_store import settings_store
            global_settings = settings_store.load()
            self.max_concurrent = global_settings.processing.max_concurrent_tasks
        except Exception:
            self.max_concurrent = getattr(settings, 'MAX_CONCURRENT_TASKS', 2)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.queue: asyncio.Queue[QueuedTask] = asyncio.Queue()

        # Track tasks by ID
        self.active_tasks: Dict[str, QueuedTask] = {}
        self.pending_tasks: Dict[str, QueuedTask] = {}

        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._process_func: Optional[Callable] = None

        logger.info(f"TaskExecutor initialized with max_concurrent={self.max_concurrent}")

    def set_process_function(self, func: Callable[[any], Awaitable[None]]):
        """Set the function used to process tasks"""
        self._process_func = func

    async def start(self):
        """Start the worker loop"""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("TaskExecutor worker started")

    async def stop(self):
        """Stop the worker loop"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("TaskExecutor worker stopped")

    async def submit(self, task: any) -> int:
        """
        Submit a task for processing.

        Args:
            task: ProcessingTask to process

        Returns:
            Queue position (0 = processing immediately, 1+ = waiting in queue)
        """
        queued_task = QueuedTask(
            task_id=task.task_id,
            task=task,
            queued_at=datetime.now()
        )

        # Add to pending tracking
        self.pending_tasks[task.task_id] = queued_task

        # Add to queue
        await self.queue.put(queued_task)

        position = self.get_queue_position(task.task_id)
        logger.info(f"Task {task.task_id} submitted, queue position: {position}")

        return position

    def get_queue_position(self, task_id: str) -> int:
        """
        Get the queue position of a task.

        Returns:
            0 = currently processing
            1+ = position in queue
            -1 = not found
        """
        if task_id in self.active_tasks:
            return 0

        if task_id not in self.pending_tasks:
            return -1

        # Count position in pending tasks
        position = 1
        for tid in self.pending_tasks:
            if tid == task_id:
                return position
            position += 1

        return -1

    def get_status(self) -> Dict:
        """Get current executor status"""
        return {
            "active_count": len(self.active_tasks),
            "pending_count": len(self.pending_tasks),
            "max_concurrent": self.max_concurrent,
            "active_tasks": [
                {
                    "task_id": qt.task_id,
                    "title": qt.task.video_info.get("title", "Unknown") if qt.task.video_info else "Unknown",
                    "started_at": qt.queued_at.isoformat(),
                }
                for qt in self.active_tasks.values()
            ],
            "queue": [
                {
                    "task_id": qt.task_id,
                    "position": i + 1,
                    "title": qt.task.video_info.get("title", "Unknown") if qt.task.video_info else "Unknown",
                    "queued_at": qt.queued_at.isoformat(),
                }
                for i, qt in enumerate(self.pending_tasks.values())
            ]
        }

    def is_task_queued(self, task_id: str) -> bool:
        """Check if a task is in the pending queue"""
        return task_id in self.pending_tasks

    def is_task_active(self, task_id: str) -> bool:
        """Check if a task is currently being processed"""
        return task_id in self.active_tasks

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued task (removes from queue if pending).
        Cannot cancel active tasks.

        Returns:
            True if task was cancelled
        """
        if task_id in self.pending_tasks:
            del self.pending_tasks[task_id]
            logger.info(f"Task {task_id} cancelled from queue")
            return True
        return False

    async def _worker_loop(self):
        """Background worker that processes tasks from the queue"""
        logger.info("TaskExecutor worker loop started")

        while self._running:
            try:
                # Get next task from queue (with timeout to allow checking _running)
                try:
                    queued_task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Check if task was cancelled while in queue
                if queued_task.task_id not in self.pending_tasks:
                    logger.info(f"Task {queued_task.task_id} was cancelled, skipping")
                    continue

                # Process task with semaphore
                asyncio.create_task(self._process_with_semaphore(queued_task))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1)

        logger.info("TaskExecutor worker loop ended")

    async def _process_with_semaphore(self, queued_task: QueuedTask):
        """Process a task with semaphore-based concurrency control"""
        task_id = queued_task.task_id

        async with self.semaphore:
            # Move from pending to active
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]
            self.active_tasks[task_id] = queued_task

            logger.info(f"Task {task_id} started processing (active: {len(self.active_tasks)}, pending: {len(self.pending_tasks)})")

            try:
                if self._process_func:
                    await self._process_func(queued_task.task)
                else:
                    logger.error("No process function set for TaskExecutor")
            except Exception as e:
                logger.error(f"Task {task_id} processing error: {e}")
            finally:
                # Remove from active
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]

                logger.info(f"Task {task_id} completed (active: {len(self.active_tasks)}, pending: {len(self.pending_tasks)})")


# Global instance
task_executor = TaskExecutor()

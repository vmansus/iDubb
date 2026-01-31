"""
Subscription scheduler for automatic video checking.
Uses asyncio for lightweight scheduling, similar to the TaskExecutor pattern.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Callable, Optional, Awaitable, Dict, Any, List
import uuid

from loguru import logger

from .base import VideoInfo
from .fetcher_factory import get_fetcher


class SubscriptionScheduler:
    """
    Async scheduler for checking subscriptions.
    Runs in the background and checks subscriptions at their configured intervals.
    """

    def __init__(
        self,
        check_interval: int = 60,  # Check for due subscriptions every 60 seconds
        max_concurrent_checks: int = 3,
    ):
        """
        Initialize the scheduler.

        Args:
            check_interval: How often to check for due subscriptions (seconds)
            max_concurrent_checks: Maximum concurrent subscription checks
        """
        self._check_interval = check_interval
        self._max_concurrent = max_concurrent_checks
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Callbacks
        self._get_due_subscriptions: Optional[Callable[[], Awaitable[List[Any]]]] = None
        self._update_subscription: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
        self._on_new_videos: Optional[Callable[[str, List[VideoInfo]], Awaitable[None]]] = None

    def set_callbacks(
        self,
        get_due_subscriptions: Callable[[], Awaitable[List[Any]]],
        update_subscription: Callable[[str, Dict[str, Any]], Awaitable[None]],
        on_new_videos: Callable[[str, List[VideoInfo]], Awaitable[None]],
    ):
        """
        Set the callback functions for database and task operations.

        Args:
            get_due_subscriptions: Async function to get subscriptions due for checking
            update_subscription: Async function to update subscription data
            on_new_videos: Async function called when new videos are found
        """
        self._get_due_subscriptions = get_due_subscriptions
        self._update_subscription = update_subscription
        self._on_new_videos = on_new_videos

    async def start(self):
        """Start the scheduler"""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Subscription scheduler started")

    async def stop(self):
        """Stop the scheduler"""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Subscription scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                await self._check_due_subscriptions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")

            # Wait before next check
            await asyncio.sleep(self._check_interval)

    async def _check_due_subscriptions(self):
        """Check all due subscriptions"""
        if not self._get_due_subscriptions:
            return

        try:
            subscriptions = await self._get_due_subscriptions()

            if not subscriptions:
                return

            logger.debug(f"Found {len(subscriptions)} subscriptions due for checking")

            # Check subscriptions with concurrency limit
            tasks = []
            for sub in subscriptions:
                task = asyncio.create_task(self._check_subscription_with_semaphore(sub))
                tasks.append(task)

            # Wait for all checks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error checking due subscriptions: {e}")

    async def _check_subscription_with_semaphore(self, subscription):
        """Check a subscription with semaphore for concurrency control"""
        async with self._semaphore:
            await self._check_subscription(subscription)

    async def _check_subscription(self, subscription):
        """Check a single subscription for new videos"""
        sub_id = subscription.id if hasattr(subscription, 'id') else subscription.get('id')
        platform = subscription.platform if hasattr(subscription, 'platform') else subscription.get('platform')
        channel_id = subscription.channel_id if hasattr(subscription, 'channel_id') else subscription.get('channel_id')
        check_interval = subscription.check_interval if hasattr(subscription, 'check_interval') else subscription.get('check_interval', 60)
        last_video_id = subscription.last_video_id if hasattr(subscription, 'last_video_id') else subscription.get('last_video_id')
        last_video_published_at = subscription.last_video_published_at if hasattr(subscription, 'last_video_published_at') else subscription.get('last_video_published_at')

        logger.debug(f"Checking subscription {sub_id} ({platform}/{channel_id})")

        fetcher = get_fetcher(platform)
        if not fetcher:
            logger.warning(f"No fetcher for platform: {platform}")
            await self._update_check_status(sub_id, check_interval, f"Unsupported platform: {platform}")
            return

        try:
            # Check if this is first check (no baseline set)
            is_first_check = not last_video_id and not last_video_published_at

            if is_first_check:
                # First check - only fetch latest video to set baseline, don't create tasks
                # This avoids duplicate API call (get_new_videos would also call get_latest_videos)
                logger.info(f"First check for {platform}/{channel_id}: setting baseline without creating tasks")
                latest_videos = await fetcher.get_latest_videos(channel_id, limit=1)
                if latest_videos and self._update_subscription:
                    latest = latest_videos[0]
                    await self._update_subscription(sub_id, {
                        "last_video_id": latest.video_id,
                        "last_video_title": latest.title,
                        "last_video_published_at": latest.published_at,
                    })
                    logger.info(f"Baseline set for {platform}/{channel_id}: {latest.title}")
            else:
                # Normal check - get new videos since last check
                new_videos = await fetcher.get_new_videos(
                    channel_id=channel_id,
                    after_video_id=last_video_id,
                    after_date=last_video_published_at,
                    limit=10
                )

                if new_videos:
                    logger.info(f"Found {len(new_videos)} new videos for {platform}/{channel_id}")

                    # IMPORTANT: Process videos FIRST, then update baseline
                    # This ensures videos aren't lost if processing fails
                    process_success = True
                    if self._on_new_videos:
                        try:
                            await self._on_new_videos(sub_id, new_videos)
                        except Exception as e:
                            logger.error(f"Failed to process new videos for {platform}/{channel_id}: {e}")
                            process_success = False

                    # Only update last video info if processing succeeded
                    if process_success:
                        latest = new_videos[0]
                        if self._update_subscription:
                            await self._update_subscription(sub_id, {
                                "last_video_id": latest.video_id,
                                "last_video_title": latest.title,
                                "last_video_published_at": latest.published_at,
                            })
                    else:
                        logger.warning(f"Skipping baseline update for {platform}/{channel_id} due to processing failure")
                else:
                    logger.debug(f"No new videos for {platform}/{channel_id}")

            # Update check status (success)
            await self._update_check_status(sub_id, check_interval, None)

        except Exception as e:
            logger.error(f"Error checking {platform}/{channel_id}: {e}")
            await self._update_check_status(sub_id, check_interval, str(e))
        finally:
            await fetcher.close()

    async def _update_check_status(
        self,
        subscription_id: str,
        check_interval: int,
        error: Optional[str]
    ):
        """Update the subscription's check status"""
        if not self._update_subscription:
            return

        next_check = datetime.now() + timedelta(minutes=check_interval)

        updates = {
            "last_checked_at": datetime.now(),
            "next_check_at": next_check,
        }

        if error:
            # We'll increment error_count in the update_subscription callback
            updates["last_error"] = error
        else:
            updates["error_count"] = 0
            updates["last_error"] = None

        await self._update_subscription(subscription_id, updates)

    async def check_now(self, subscription) -> List[VideoInfo]:
        """
        Manually trigger a check for a subscription.
        Returns list of new videos found.

        Args:
            subscription: Subscription object or dict

        Returns:
            List of new VideoInfo
        """
        platform = subscription.platform if hasattr(subscription, 'platform') else subscription.get('platform')
        channel_id = subscription.channel_id if hasattr(subscription, 'channel_id') else subscription.get('channel_id')
        last_video_id = subscription.last_video_id if hasattr(subscription, 'last_video_id') else subscription.get('last_video_id')
        last_video_published_at = subscription.last_video_published_at if hasattr(subscription, 'last_video_published_at') else subscription.get('last_video_published_at')

        fetcher = get_fetcher(platform)
        if not fetcher:
            raise ValueError(f"Unsupported platform: {platform}")

        try:
            return await fetcher.get_new_videos(
                channel_id=channel_id,
                after_video_id=last_video_id,
                after_date=last_video_published_at,
                limit=10
            )
        finally:
            await fetcher.close()


# Global scheduler instance
subscription_scheduler = SubscriptionScheduler()

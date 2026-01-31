"""
Douyin (抖音) Video Uploader using Playwright

使用 Playwright 模拟浏览器操作上传视频到抖音创作者中心
比 API 方式更稳定，因为使用的是真实的前端接口
"""
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from .base import BaseUploader, UploadResult, VideoMetadata


class DouyinPlaywrightUploader(BaseUploader):
    """
    Douyin video uploader using Playwright browser automation

    More stable than API-based approach as it uses the actual web interface
    """

    CREATOR_URL = "https://creator.douyin.com"
    UPLOAD_URL = "https://creator.douyin.com/creator-micro/content/upload"

    def __init__(self):
        super().__init__()
        self._cookies: List[Dict[str, Any]] = []
        self._browser = None
        self._context = None
        self._page = None

    def get_required_credentials(self) -> List[str]:
        return ["cookies"]

    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """
        Authenticate by parsing and storing cookies
        Actual browser login happens during upload
        """
        try:
            cookie_string = credentials.get("cookies", "")
            if not cookie_string:
                logger.error("No Douyin cookies provided")
                return False

            # Parse cookie string into Playwright format
            self._cookies = []
            for item in cookie_string.split(";"):
                item = item.strip()
                if "=" in item:
                    key, value = item.split("=", 1)
                    key = key.strip()
                    if key:
                        self._cookies.append({
                            "name": key,
                            "value": value,
                            "domain": ".douyin.com",
                            "path": "/",
                        })

            logger.info(f"Parsed {len(self._cookies)} cookies for Playwright")
            self._authenticated = True
            return True

        except Exception as e:
            logger.error(f"Failed to parse cookies: {e}")
            return False

    async def _init_browser(self):
        """Initialize Playwright browser"""
        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()

            # Launch browser (headless by default for server, set PLAYWRIGHT_HEADLESS=false for debugging)
            headless = os.environ.get("PLAYWRIGHT_HEADLESS", "true").lower() != "false"

            self._browser = await self._playwright.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ]
            )

            # Create context with cookies
            self._context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )

            # Add cookies
            await self._context.add_cookies(self._cookies)

            # Create page
            self._page = await self._context.new_page()

            logger.info("Playwright browser initialized")
            return True

        except ImportError:
            logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
            return False
        except Exception as e:
            logger.error(f"Failed to init browser: {e}")
            return False

    async def _close_browser(self):
        """Close browser resources"""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if hasattr(self, '_playwright') and self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")

    async def _dismiss_popups(self):
        """Dismiss common popup dialogs on Douyin creator center"""
        popup_buttons = [
            'button:has-text("我知道了")',
            'button:has-text("知道了")',
            'button:has-text("完成")',
            'button:has-text("关闭")',
            'button:has-text("取消")',
            '[class*="modal"] button[class*="close"]',
            '[class*="dialog"] button[class*="close"]',
            '[class*="popup"] button[class*="close"]',
            '[class*="toast"] button',
            '.semi-modal-close',
            '[aria-label="关闭"]',
        ]

        dismissed_count = 0
        for selector in popup_buttons:
            try:
                buttons = await self._page.query_selector_all(selector)
                for btn in buttons:
                    if await btn.is_visible():
                        await btn.click()
                        dismissed_count += 1
                        logger.debug(f"Dismissed popup with selector: {selector}")
                        await asyncio.sleep(0.5)
            except Exception:
                continue

        if dismissed_count > 0:
            logger.info(f"Dismissed {dismissed_count} popup(s)")
            await asyncio.sleep(1)

        return dismissed_count

    async def _handle_cover_prompt(self):
        """
        Handle the "请设置封面后再发布" prompt by selecting a recommended cover.
        Based on social-auto-upload library approach.
        """
        return await self._handle_auto_video_cover()
    
    async def _handle_auto_video_cover(self):
        """
        处理必须设置封面的情况，点击推荐封面的第一个
        Directly ported from social-auto-upload library
        """
        try:
            # 1. 判断是否出现 "请设置封面后再发布" 的提示
            cover_prompt = self._page.get_by_text("请设置封面后再发布")
            is_visible = await cover_prompt.first.is_visible()
            logger.debug(f"封面提示 '请设置封面后再发布' 可见: {is_visible}")
            if is_visible:
                logger.info("检测到需要设置封面提示: '请设置封面后再发布'")

                # 2. 定位"智能推荐封面"区域下的第一个封面
                # 使用 class^= 前缀匹配，避免 hash 变化导致失效
                recommend_cover = self._page.locator('[class^="recommendCover-"]').first

                if await recommend_cover.count():
                    logger.info("正在选择第一个推荐封面...")
                    try:
                        await recommend_cover.click()
                        await asyncio.sleep(1)

                        # 3. 处理可能的确认弹窗 "是否确认应用此封面？"
                        confirm_text = self._page.get_by_text("是否确认应用此封面？")
                        if await confirm_text.first.is_visible():
                            logger.info("检测到确认弹窗: 是否确认应用此封面？")
                            confirm_btn = self._page.get_by_role("button", name="确定")
                            if await confirm_btn.count():
                                await confirm_btn.click()
                                logger.info("已点击确认应用封面")
                                await asyncio.sleep(1)

                        logger.info("已完成封面选择流程")
                        return True
                    except Exception as e:
                        logger.warning(f"选择封面失败: {e}")
                else:
                    logger.warning("提示可见但未找到推荐封面")
            
            return False
        except Exception as e:
            logger.debug(f"封面处理错误: {e}")
            return False

    async def upload(
        self,
        video_path: Path,
        metadata: VideoMetadata
    ) -> UploadResult:
        """Upload video to Douyin using Playwright"""
        if not self._authenticated:
            return UploadResult(
                success=False,
                platform="douyin",
                error="未认证，请先在设置中配置抖音Cookies"
            )

        try:
            # Check file exists and size
            if not video_path.exists():
                return UploadResult(
                    success=False,
                    platform="douyin",
                    error=f"视频文件不存在: {video_path}"
                )

            file_size = video_path.stat().st_size
            logger.info(f"Starting Douyin Playwright upload: {video_path.name} ({file_size / 1024 / 1024:.2f} MB)")

            # Check file size limit
            max_size = 4 * 1024 * 1024 * 1024  # 4GB
            if file_size > max_size:
                return UploadResult(
                    success=False,
                    platform="douyin",
                    error=f"视频文件过大 ({file_size / 1024 / 1024:.0f}MB)，抖音限制4GB"
                )

            # Initialize browser
            if not await self._init_browser():
                return UploadResult(
                    success=False,
                    platform="douyin",
                    error="无法启动浏览器，请确保已安装 Playwright"
                )

            try:
                # Navigate to upload page
                logger.info("Navigating to Douyin creator center...")
                await self._page.goto(self.UPLOAD_URL, wait_until="networkidle", timeout=30000)

                # Wait a bit for page to fully load
                await asyncio.sleep(2)

                # Check if we need to login
                current_url = self._page.url
                if "login" in current_url.lower() or "passport" in current_url.lower():
                    logger.error("Cookies expired, redirected to login page")
                    return UploadResult(
                        success=False,
                        platform="douyin",
                        error="Cookies已过期，请重新登录抖音创作者中心并更新Cookies"
                    )

                # Dismiss any popup dialogs first
                await self._dismiss_popups()

                # Find and click upload area or input
                logger.info("Looking for upload input...")

                # Try to find the file input
                file_input = await self._page.query_selector('input[type="file"]')
                if not file_input:
                    # Try to click upload button first
                    upload_btn = await self._page.query_selector('[class*="upload"]')
                    if upload_btn:
                        await upload_btn.click()
                        await asyncio.sleep(1)
                        file_input = await self._page.query_selector('input[type="file"]')

                if not file_input:
                    # Take screenshot for debugging
                    screenshot_path = video_path.parent / "douyin_upload_debug.png"
                    await self._page.screenshot(path=str(screenshot_path))
                    logger.error(f"Cannot find upload input. Screenshot saved to {screenshot_path}")
                    return UploadResult(
                        success=False,
                        platform="douyin",
                        error="无法找到上传按钮，页面结构可能已变化"
                    )

                # Upload file
                logger.info(f"Uploading video file: {video_path}")
                await file_input.set_input_files(str(video_path))

                # Wait for upload to start
                await asyncio.sleep(3)

                # Wait for upload to complete using multiple detection methods
                logger.info("Waiting for upload to complete...")
                upload_timeout = 600  # 10 minutes max
                start_time = asyncio.get_event_loop().time()
                last_progress = ""

                while True:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    
                    if elapsed > upload_timeout:
                        # Take debug screenshot before timeout
                        timeout_screenshot = video_path.parent / "douyin_upload_timeout.png"
                        await self._page.screenshot(path=str(timeout_screenshot), full_page=True)
                        logger.error(f"Upload timeout after {elapsed:.0f}s. Screenshot: {timeout_screenshot}")
                        return UploadResult(
                            success=False,
                            platform="douyin",
                            error="上传超时"
                        )

                    # Dismiss any popups that might appear during upload
                    await self._dismiss_popups()

                    # Get full page text for checking
                    try:
                        page_text = await self._page.inner_text("body")
                    except Exception:
                        page_text = ""

                    # Method 1: Check for completion text indicators
                    completion_indicators = ["上传完成", "重新上传", "上传成功", "发布设置", "作品设置"]
                    for indicator in completion_indicators:
                        if indicator in page_text:
                            logger.info(f"Upload complete - found '{indicator}'")
                            break
                    else:
                        # Method 2: Check for editable description field (form is ready)
                        # Douyin uses a contenteditable div, not textarea
                        desc_selectors = [
                            '[contenteditable="true"]',
                            '.editor-kit-outer-container',
                            '[data-placeholder*="描述"]',
                            '[data-placeholder*="作品"]',
                            'textarea',
                            '[class*="editor"]',
                        ]
                        form_ready = False
                        for sel in desc_selectors:
                            try:
                                el = await self._page.query_selector(sel)
                                if el and await el.is_visible():
                                    # Check if this is the main editor, not some other contenteditable
                                    box = await el.bounding_box()
                                    if box and box['width'] > 200 and box['height'] > 50:
                                        logger.info(f"Upload complete - editor ready with selector: {sel}")
                                        form_ready = True
                                        break
                            except Exception:
                                continue
                        
                        if form_ready:
                            break

                        # Method 3: Check progress text and log it
                        progress_selectors = [
                            '[class*="progress"]',
                            '[class*="percent"]',
                            '[class*="upload"] span',
                            '[class*="status"]',
                        ]
                        for sel in progress_selectors:
                            try:
                                els = await self._page.query_selector_all(sel)
                                for el in els:
                                    if await el.is_visible():
                                        text = await el.inner_text()
                                        text = text.strip()
                                        if text and text != last_progress and len(text) < 50:
                                            # Check if it's a percentage
                                            if "%" in text:
                                                logger.info(f"Upload progress: {text}")
                                                last_progress = text
                                                if "100%" in text:
                                                    logger.info("Upload complete - 100% reached")
                                                    form_ready = True
                                                    break
                            except Exception:
                                continue
                        
                        if form_ready:
                            break

                        # Method 4: Check for video thumbnail/preview (video is ready)
                        thumb_selectors = [
                            'video',  # Video preview
                            '[class*="cover"] img',
                            '[class*="thumbnail"] img',
                            '[class*="preview"] img',
                        ]
                        for sel in thumb_selectors:
                            try:
                                el = await self._page.query_selector(sel)
                                if el and await el.is_visible():
                                    # Make sure it's not a tiny icon
                                    box = await el.bounding_box()
                                    if box and box['width'] > 100 and box['height'] > 100:
                                        logger.info(f"Upload complete - video preview visible: {sel}")
                                        form_ready = True
                                        break
                            except Exception:
                                continue
                        
                        if form_ready:
                            break

                        # Still uploading, wait and retry
                        if int(elapsed) % 10 == 0:  # Log every 10 seconds
                            logger.info(f"Still waiting for upload... ({elapsed:.0f}s elapsed)")
                        await asyncio.sleep(2)
                        continue
                    break  # If we hit the completion_indicators break

                # Wait a bit more for the form to fully load
                await asyncio.sleep(3)

                # Dismiss any popups before filling metadata
                await self._dismiss_popups()

                # Fill in metadata
                logger.info("Filling in video metadata...")

                # On Douyin creator center, the main text area is "作品描述" (Work description)
                # It combines title info with hashtags in a single text field
                # Build the description text with title and hashtags
                desc_parts = []

                # Add a short title/summary (Douyin description field handles this)
                if metadata.title:
                    # Truncate title if too long (Douyin has char limits)
                    short_title = metadata.title[:30] if len(metadata.title) > 30 else metadata.title
                    desc_parts.append(short_title)

                # Add hashtags
                if metadata.tags:
                    hashtags = " ".join([f"#{tag}" for tag in metadata.tags[:5]])
                    desc_parts.append(hashtags)

                # Add description if available
                if metadata.description:
                    desc_parts.append(metadata.description[:200])

                full_description = " ".join(desc_parts)[:1000]  # Limit total length
                logger.info(f"Prepared description: {full_description[:100]}...")

                # Find and fill the description field
                # Douyin uses contenteditable div, not textarea
                desc_selectors = [
                    '.editor-kit-outer-container [contenteditable="true"]',
                    '[contenteditable="true"]',
                    '[data-placeholder*="描述"]',
                    '[data-placeholder*="作品"]',
                    'textarea[placeholder*="描述"]',
                    'textarea[placeholder*="简介"]',
                    'textarea',
                ]

                desc_filled = False
                for selector in desc_selectors:
                    try:
                        elements = await self._page.query_selector_all(selector)
                        for desc_input in elements:
                            if not await desc_input.is_visible():
                                continue
                            # Check if it's a reasonable size for a description field
                            box = await desc_input.bounding_box()
                            if not box or box['width'] < 200:
                                continue
                            
                            # Click to focus
                            await desc_input.click()
                            await asyncio.sleep(0.3)
                            
                            # Clear existing content
                            await self._page.keyboard.press("Control+a")
                            await asyncio.sleep(0.1)
                            
                            # Type the description (works for both textarea and contenteditable)
                            await self._page.keyboard.type(full_description, delay=10)
                            desc_filled = True
                            logger.info(f"Filled description with selector: {selector}")
                            break
                        if desc_filled:
                            break
                    except Exception as e:
                        logger.debug(f"Failed to fill with selector {selector}: {e}")
                        continue

                if not desc_filled:
                    logger.warning("Could not find description input")

                await asyncio.sleep(1)

                # Wait for video encoding and content detection to complete
                logger.info("Waiting for video encoding and content detection...")
                encoding_timeout = 120  # 2 minutes max for encoding
                encoding_start = asyncio.get_event_loop().time()
                
                while True:
                    elapsed = asyncio.get_event_loop().time() - encoding_start
                    if elapsed > encoding_timeout:
                        logger.warning("Encoding timeout, proceeding anyway...")
                        break
                    
                    try:
                        page_text = await self._page.inner_text("body")
                        
                        # Check if still encoding
                        if "转码中" in page_text or "检测中" in page_text:
                            # Extract percentage if available
                            match = re.search(r'检测中(\d+)%', page_text)
                            if match:
                                pct = match.group(1)
                                logger.info(f"Content detection progress: {pct}%")
                            else:
                                logger.info("Still encoding/detecting...")
                            await asyncio.sleep(3)
                            continue
                        else:
                            logger.info("Encoding/detection complete!")
                            break
                    except Exception as e:
                        logger.debug(f"Error checking encoding status: {e}")
                        await asyncio.sleep(2)
                        continue

                await asyncio.sleep(2)

                # Try to set cover by clicking AI recommended thumbnails BEFORE publishing
                # This is more reliable than waiting for the "请设置封面后再发布" prompt
                logger.info("Attempting to set cover from AI recommendations...")
                
                try:
                    # Click on one of the AI recommended cover images
                    # These are the small thumbnails next to "AI智能推荐封面"
                    ai_covers = self._page.locator('[class^="recommendCover-"]')
                    if await ai_covers.count() > 0:
                        await ai_covers.first.click()
                        logger.info("Clicked first AI recommended cover")
                        await asyncio.sleep(1)
                        
                        # Handle any confirmation dialog
                        confirm_btn = self._page.get_by_role("button", name="确定")
                        if await confirm_btn.count() > 0 and await confirm_btn.first.is_visible():
                            await confirm_btn.first.click()
                            logger.info("Confirmed cover selection")
                            await asyncio.sleep(1)
                    else:
                        logger.info("No AI recommended covers found, trying alternative selectors...")
                        # Try alternative selectors for cover images
                        cover_imgs = await self._page.query_selector_all('img[src*="douyinpic"], img[src*="bytedance"]')
                        for img in cover_imgs[:3]:  # Try first 3 images
                            try:
                                if await img.is_visible():
                                    box = await img.bounding_box()
                                    # Look for small thumbnail-sized images (60-120px)
                                    if box and 50 < box['width'] < 150 and 50 < box['height'] < 150:
                                        await img.click()
                                        logger.info("Clicked cover thumbnail image")
                                        await asyncio.sleep(1)
                                        break
                            except Exception:
                                continue
                except Exception as e:
                    logger.debug(f"Cover pre-selection failed: {e}")

                # Dismiss any popups that might be on the page
                await self._dismiss_popups()

                # ============================================================
                # Publish loop - keep trying until success or max attempts
                max_attempts = 10
                for attempt in range(max_attempts):
                    try:
                        # Find publish button
                        publish_button = self._page.get_by_role('button', name="发布", exact=True)
                        if await publish_button.count() == 0:
                            publish_button = self._page.get_by_role('button', name="发布")
                        if await publish_button.count() == 0:
                            publish_button = self._page.locator('button:has-text("发布")').first
                        
                        if await publish_button.count() > 0:
                            await publish_button.click()
                            logger.info(f"Clicked publish button (attempt {attempt + 1})")
                        
                        # Wait for navigation to manage page (success indicator)
                        await self._page.wait_for_url("**/manage**", timeout=5000)
                        logger.info("Navigated to manage page - upload successful!")
                        
                        return UploadResult(
                            success=True,
                            platform="douyin",
                            video_url=self._page.url,
                        )
                    except Exception as e:
                        logger.debug(f"Attempt {attempt + 1}/{max_attempts} - waiting: {e}")
                        
                        # Check for SMS verification dialog (anti-automation measure)
                        sms_dialog = self._page.get_by_text("接收短信验证码")
                        if await sms_dialog.first.is_visible():
                            logger.error("抖音要求短信验证码验证身份，无法自动发布")
                            # Take screenshot for reference
                            await self._page.screenshot(path=str(video_path.parent / "douyin_sms_verification.png"))
                            return UploadResult(
                                success=False,
                                platform="douyin",
                                error="抖音要求短信验证码，请手动登录验证后重试"
                            )
                        
                        # Try to handle cover selection if prompted
                        cover_handled = await self._handle_auto_video_cover()
                        if cover_handled:
                            logger.info("Cover prompt handled successfully")
                        await asyncio.sleep(1)
                
                # All attempts failed
                current_url = self._page.url
                logger.warning(f"All {max_attempts} publish attempts failed. Current URL: {current_url}")
                
                # Final screenshot
                await self._page.screenshot(path=str(video_path.parent / "douyin_final_failed.png"), full_page=True)
                
                # Check if somehow we ended up on success page
                if "manage" in current_url.lower():
                    return UploadResult(
                        success=True,
                        platform="douyin", 
                        video_url=current_url,
                    )
                
                return UploadResult(
                    success=False,
                    platform="douyin",
                    error="发布失败，请检查截图确认问题"
                )
                
            except Exception as e:
                logger.error(f"Upload error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return UploadResult(
                    success=False,
                    platform="douyin",
                    error=f"上传出错: {str(e)}"
                )

            finally:
                await self._close_browser()

        except asyncio.TimeoutError:
            logger.error("Douyin upload timeout")
            return UploadResult(
                success=False,
                platform="douyin",
                error="操作超时"
            )
        except Exception as e:
            logger.error(f"Douyin Playwright upload error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return UploadResult(
                success=False,
                platform="douyin",
                error=f"上传出错: {str(e)}"
            )

    async def close(self):
        """Close uploader resources"""
        await self._close_browser()

    async def check_upload_status(self, video_id: str) -> Dict[str, Any]:
        """Check upload/processing status - not implemented for Playwright uploader"""
        return {"status": "unknown", "message": "Status check not available for Playwright uploader"}

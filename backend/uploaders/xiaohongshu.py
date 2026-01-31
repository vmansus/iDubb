"""
Xiaohongshu (小红书) Video Uploader

注意：小红书没有官方开放API
此实现基于Web端逆向，需要处理各种加密签名
"""
import asyncio
import json
import time
import hashlib
import random
import string
from pathlib import Path
from typing import Dict, Any, List, Optional
import aiohttp
import aiofiles
from loguru import logger

from .base import BaseUploader, UploadResult, VideoMetadata


class XiaohongshuUploader(BaseUploader):
    """
    Xiaohongshu video uploader

    Warning: Xiaohongshu doesn't have official API.
    Uses web interface reverse engineering.
    May break due to anti-bot measures.
    """

    BASE_URL = "https://www.xiaohongshu.com"
    CREATOR_URL = "https://creator.xiaohongshu.com"

    def __init__(self):
        super().__init__()
        self._session: Optional[aiohttp.ClientSession] = None
        self._cookies = {}
        self._user_id = None

    def get_required_credentials(self) -> List[str]:
        """
        Required cookies:
        - a1 (device fingerprint)
        - webId
        - web_session (login session)
        - xsecappid
        """
        return ["cookies"]

    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Authenticate using cookies"""
        try:
            cookie_string = credentials.get("cookies", "")
            if not cookie_string:
                logger.error("No Xiaohongshu cookies provided")
                return False

            # Parse cookies
            self._cookies = {}
            for item in cookie_string.split(";"):
                if "=" in item:
                    key, value = item.strip().split("=", 1)
                    self._cookies[key] = value

            # Create SSL context that doesn't verify certificates (for macOS compatibility)
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            # Create session
            self._session = aiohttp.ClientSession(
                connector=connector,
                cookies=self._cookies,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Origin": self.BASE_URL,
                    "Referer": f"{self.BASE_URL}/",
                }
            )

            # Verify login
            async with self._session.get(
                f"{self.BASE_URL}/api/sns/web/v1/user/selfinfo"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success"):
                        self._user_id = data.get("data", {}).get("user_id")
                        self._authenticated = True
                        logger.info(f"Xiaohongshu authenticated, user_id: {self._user_id}")
                        return True
                    else:
                        logger.error(f"Xiaohongshu auth failed: {data}")
                        return False

            return False

        except Exception as e:
            logger.error(f"Xiaohongshu authentication error: {e}")
            return False

    def _generate_trace_id(self) -> str:
        """Generate trace ID for API calls"""
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(32))

    async def _get_upload_token(self) -> Optional[Dict[str, Any]]:
        """Get upload token for video upload"""
        try:
            async with self._session.get(
                f"{self.CREATOR_URL}/api/media/v1/upload/web/permit",
                params={
                    "biz": "video",
                    "scene": "publish_video",
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success"):
                        return data.get("data")
                return None

        except Exception as e:
            logger.error(f"Get upload token error: {e}")
            return None

    async def upload(
        self,
        video_path: Path,
        metadata: VideoMetadata
    ) -> UploadResult:
        """Upload video to Xiaohongshu"""
        if not self._authenticated:
            return UploadResult(
                success=False,
                platform="xiaohongshu",
                error="未认证，请先在设置中配置小红书Cookies"
            )

        try:
            logger.info(f"Starting Xiaohongshu upload: {video_path.name}")
            file_size = video_path.stat().st_size
            logger.info(f"Video size: {file_size / 1024 / 1024:.2f} MB")

            # Check file size limit (小红书限制)
            max_size = 5 * 1024 * 1024 * 1024  # 5GB
            if file_size > max_size:
                return UploadResult(
                    success=False,
                    platform="xiaohongshu",
                    error=f"视频文件过大 ({file_size / 1024 / 1024:.0f}MB)，小红书限制5GB"
                )

            # Get upload token with retry
            upload_token = None
            for attempt in range(3):
                upload_token = await self._get_upload_token()
                if upload_token:
                    break
                logger.warning(f"获取上传Token失败，重试 {attempt + 1}/3")
                await asyncio.sleep(1)

            if not upload_token:
                return UploadResult(
                    success=False,
                    platform="xiaohongshu",
                    error="无法获取上传Token，请检查登录状态"
                )

            # Read video file
            async with aiofiles.open(video_path, "rb") as f:
                video_data = await f.read()

            # Upload to Xiaohongshu storage
            upload_url = upload_token.get("upload_addr")
            if not upload_url:
                upload_url = f"{self.CREATOR_URL}/api/media/v1/upload/web/video"

            form_data = aiohttp.FormData()
            form_data.add_field(
                "file",
                video_data,
                filename=video_path.name,
                content_type="video/mp4"
            )
            form_data.add_field("token", upload_token.get("token", ""))

            logger.info(f"Uploading to: {upload_url}")

            # Upload with timeout
            timeout = aiohttp.ClientTimeout(total=600)  # 10 minutes timeout
            async with self._session.post(
                upload_url,
                data=form_data,
                headers={
                    "X-Xhs-Trace-Id": self._generate_trace_id(),
                },
                timeout=timeout
            ) as resp:
                logger.info(f"Upload response status: {resp.status}")
                response_text = await resp.text()

                if resp.status == 200:
                    try:
                        upload_result = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON response: {response_text[:500]}")
                        return UploadResult(
                            success=False,
                            platform="xiaohongshu",
                            error="服务器返回无效响应"
                        )

                    if upload_result.get("success"):
                        file_id = upload_result.get("data", {}).get("file_id")
                        logger.info(f"Video uploaded, file_id: {file_id}")

                        # Wait for video processing
                        logger.info("等待视频处理...")
                        await asyncio.sleep(3)

                        # Publish note with video
                        publish_result = await self._publish_note(file_id, metadata)

                        if publish_result:
                            note_id = publish_result.get("note_id")
                            logger.info(f"Xiaohongshu upload success! note_id: {note_id}")
                            return UploadResult(
                                success=True,
                                video_id=note_id,
                                video_url=f"https://www.xiaohongshu.com/explore/{note_id}",
                                platform="xiaohongshu",
                            )
                        else:
                            return UploadResult(
                                success=False,
                                platform="xiaohongshu",
                                error="视频上传成功但发布失败"
                            )
                    else:
                        error_msg = upload_result.get("msg", "上传失败")
                        logger.error(f"Xiaohongshu upload failed: {upload_result}")
                        return UploadResult(
                            success=False,
                            platform="xiaohongshu",
                            error=f"上传失败: {error_msg}"
                        )
                else:
                    logger.error(f"Xiaohongshu upload HTTP error: {resp.status}")
                    return UploadResult(
                        success=False,
                        platform="xiaohongshu",
                        error=f"HTTP错误: {resp.status}"
                    )

        except asyncio.TimeoutError:
            logger.error("Xiaohongshu upload timeout")
            return UploadResult(
                success=False,
                platform="xiaohongshu",
                error="上传超时，请检查网络连接"
            )
        except Exception as e:
            logger.error(f"Xiaohongshu upload error: {e}")
            return UploadResult(
                success=False,
                platform="xiaohongshu",
                error=f"上传出错: {str(e)}"
            )

    async def _publish_note(
        self,
        file_id: str,
        metadata: VideoMetadata
    ) -> Optional[Dict[str, Any]]:
        """Publish note with uploaded video"""
        try:
            # Build hashtags
            hashtags = []
            for tag in metadata.tags[:5]:
                hashtags.append({
                    "name": tag,
                    "type": "topic"
                })

            # Build description with hashtags
            desc = metadata.description
            for tag in metadata.tags[:5]:
                desc += f" #{tag}"

            publish_data = {
                "common": {
                    "type": "video",  # or "normal" for image posts
                    "note_id": "",
                    "post_time": "",
                },
                "video_info": {
                    "file_id": file_id,
                },
                "title": metadata.title[:20],  # Max 20 chars for title
                "desc": desc[:1000],  # Max 1000 chars
                "hash_tag": hashtags,
                "at_user": [],
                "privacy": {
                    "type": 0,  # 0=public
                },
            }

            headers = {
                "X-Xhs-Trace-Id": self._generate_trace_id(),
            }

            async with self._session.post(
                f"{self.CREATOR_URL}/api/galaxy/creator/note/publish",
                json=publish_data,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result.get("success"):
                        return result.get("data")
                    else:
                        logger.error(f"Publish note failed: {result}")
                return None

        except Exception as e:
            logger.error(f"Publish note error: {e}")
            return None

    async def check_upload_status(self, video_id: str) -> Dict[str, Any]:
        """Check note status"""
        try:
            async with self._session.get(
                f"{self.BASE_URL}/api/sns/web/v1/feed",
                params={"source_note_id": video_id}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success"):
                        return {
                            "status": "published",
                            "data": data.get("data")
                        }
                return {"status": "unknown"}

        except Exception as e:
            logger.error(f"Status check error: {e}")
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close session"""
        if self._session:
            await self._session.close()


class XiaohongshuQRLogin:
    """
    Xiaohongshu QR code login helper

    小红书扫码登录流程:
    1. 获取二维码
    2. 用户使用小红书APP扫码
    3. 轮询登录状态
    4. 成功后获取cookies
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def get_qr_code(self) -> Optional[Dict[str, Any]]:
        """Get QR code for login"""
        try:
            # Create SSL context that doesn't verify certificates (for macOS compatibility)
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            self._session = aiohttp.ClientSession(
                connector=connector,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                }
            )

            async with self._session.get(
                "https://www.xiaohongshu.com/api/sns/web/v1/login/qrcode/create"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success"):
                        qr_data = data.get("data", {})
                        return {
                            "qrcode_url": qr_data.get("url"),
                            "code": qr_data.get("code"),
                        }
                return None

        except Exception as e:
            logger.error(f"Get QR code error: {e}")
            return None

    async def poll_qr_status(self, code: str) -> Optional[Dict[str, str]]:
        """Poll QR code scan status"""
        try:
            max_attempts = 120  # 2 minutes timeout
            for _ in range(max_attempts):
                async with self._session.get(
                    "https://www.xiaohongshu.com/api/sns/web/v1/login/qrcode/status",
                    params={"code": code}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("success"):
                            status = data.get("data", {}).get("status")

                            if status == 0:  # Waiting
                                await asyncio.sleep(1)
                                continue
                            elif status == 1:  # Scanned
                                logger.info("QR scanned, waiting confirmation...")
                                await asyncio.sleep(1)
                                continue
                            elif status == 2:  # Confirmed
                                # Extract cookies
                                cookies = {}
                                for cookie in resp.cookies.values():
                                    cookies[cookie.key] = cookie.value
                                return cookies
                            elif status == 3:  # Expired
                                logger.error("QR code expired")
                                return None
                            elif status == 4:  # Cancelled
                                logger.error("Login cancelled")
                                return None

                await asyncio.sleep(1)

            return None

        except Exception as e:
            logger.error(f"Poll QR status error: {e}")
            return None

    async def close(self):
        if self._session:
            await self._session.close()

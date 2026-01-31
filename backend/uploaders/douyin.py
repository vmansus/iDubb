"""
Douyin (抖音) Video Uploader

注意：抖音没有官方开放API，此实现基于逆向工程
需要通过扫码或cookies登录
实际使用时需要处理各种反爬措施
"""
import asyncio
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import aiohttp
import aiofiles
from loguru import logger

from .base import BaseUploader, UploadResult, VideoMetadata


class DouyinUploader(BaseUploader):
    """
    Douyin video uploader

    Warning: Douyin doesn't have official API.
    This is based on reverse engineering and may break.
    Use at your own risk.
    """

    # Creator center URL
    CREATOR_URL = "https://creator.douyin.com"

    def __init__(self):
        super().__init__()
        self._session: Optional[aiohttp.ClientSession] = None
        self._cookies = {}

    def get_required_credentials(self) -> List[str]:
        """
        Required cookies from Douyin login:
        - sessionid
        - sessionid_ss
        - ttwid
        - passport_csrf_token
        """
        return ["cookies"]  # Pass full cookie string

    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """
        Authenticate using cookies

        Args:
            credentials: Dict with 'cookies' key containing cookie string
        """
        try:
            cookie_string = credentials.get("cookies", "")
            logger.debug(f"Douyin authenticate: credentials keys={list(credentials.keys())}, cookie_string length={len(cookie_string) if cookie_string else 0}")
            if cookie_string:
                logger.debug(f"Douyin cookie preview: {cookie_string[:100]}...")
            if not cookie_string:
                logger.error("No Douyin cookies provided")
                return False

            # Parse cookie string
            self._cookies = {}
            for item in cookie_string.split(";"):
                item = item.strip()
                if "=" in item:
                    key, value = item.split("=", 1)
                    key = key.strip()
                    if key:  # Skip empty keys
                        self._cookies[key] = value

            logger.debug(f"Parsed {len(self._cookies)} cookies: {list(self._cookies.keys())[:10]}...")

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
                    "Referer": "https://creator.douyin.com/",
                    "Origin": "https://creator.douyin.com",
                }
            )

            # Verify login
            async with self._session.get(
                f"{self.CREATOR_URL}/creator-micro/home"
            ) as resp:
                if resp.status == 200:
                    self._authenticated = True
                    logger.info("Douyin authenticated successfully")
                    return True
                else:
                    logger.error(f"Douyin auth failed: {resp.status}")
                    return False

        except Exception as e:
            import traceback
            logger.error(f"Douyin authentication error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def _get_upload_params(self) -> Optional[Dict[str, Any]]:
        """Get upload parameters from Douyin"""
        try:
            # Try multiple API endpoints as Douyin frequently changes them
            endpoints = [
                "/creator-micro/upload/video/auth",
                "/web/api/media/aweme/web/upload/auth/",
                "/web/api/media/aweme/web/upload/signature/",
            ]

            # Get CSRF token from cookies
            csrf_token = self._cookies.get("passport_csrf_token", "")
            x_secsdk_csrf_token = self._cookies.get("__ac_nonce", "") or self._cookies.get("x-secsdk-csrf-token", "")

            headers = {
                "X-CSRFToken": csrf_token,
                "X-Secsdk-Csrf-Token": x_secsdk_csrf_token,
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/json",
            }

            for endpoint in endpoints:
                url = f"{self.CREATOR_URL}{endpoint}"
                logger.debug(f"Trying upload params endpoint: {url}")

                try:
                    async with self._session.get(url, headers=headers) as resp:
                        logger.debug(f"Upload params response status: {resp.status}")
                        if resp.status == 200:
                            data = await resp.json()
                            logger.debug(f"Upload params response: {data}")
                            if data.get("status_code") == 0:
                                return data.get("data")
                            elif data.get("data"):
                                # Some endpoints return data even with non-zero status_code
                                return data.get("data")
                        elif resp.status == 302:
                            # Redirect might indicate auth issue
                            logger.warning(f"Redirect response from {endpoint}, may need re-authentication")
                except Exception as e:
                    logger.debug(f"Endpoint {endpoint} failed: {e}")
                    continue

            # If all endpoints fail, try to get upload URL directly
            logger.warning("All upload param endpoints failed, trying direct upload approach")
            return {"upload_host": f"{self.CREATOR_URL}/web/api/media/aweme/web/upload/video/"}

        except Exception as e:
            logger.error(f"Get upload params error: {e}")
            return None

    async def upload(
        self,
        video_path: Path,
        metadata: VideoMetadata
    ) -> UploadResult:
        """
        Upload video to Douyin

        Note: This is based on reverse engineering.
        Douyin's API changes frequently and may require updates.
        """
        if not self._authenticated:
            return UploadResult(
                success=False,
                platform="douyin",
                error="未认证，请先在设置中配置抖音Cookies"
            )

        try:
            logger.info(f"Starting Douyin upload: {video_path.name}")
            file_size = video_path.stat().st_size
            logger.info(f"Video size: {file_size / 1024 / 1024:.2f} MB")

            # Check file size limit (抖音限制)
            max_size = 4 * 1024 * 1024 * 1024  # 4GB
            if file_size > max_size:
                return UploadResult(
                    success=False,
                    platform="douyin",
                    error=f"视频文件过大 ({file_size / 1024 / 1024:.0f}MB)，抖音限制4GB"
                )

            # Get upload signature with retry
            upload_params = None
            for attempt in range(3):
                upload_params = await self._get_upload_params()
                if upload_params:
                    break
                logger.warning(f"获取上传参数失败，重试 {attempt + 1}/3")
                await asyncio.sleep(1)

            if not upload_params:
                return UploadResult(
                    success=False,
                    platform="douyin",
                    error="无法获取上传参数，请检查登录状态"
                )

            # Upload video using chunked reading to avoid memory issues
            upload_url = upload_params.get("upload_host", "")
            if not upload_url:
                upload_url = f"{self.CREATOR_URL}/web/api/media/aweme/web/upload/video/"

            # Read file in chunks and create form data
            chunk_size = 10 * 1024 * 1024  # 10MB chunks for form data
            async with aiofiles.open(video_path, "rb") as f:
                video_data = await f.read()  # For small files, read all at once

            form_data = aiohttp.FormData()
            form_data.add_field(
                "file",
                video_data,
                filename=video_path.name,
                content_type="video/mp4"
            )

            logger.info(f"Uploading to: {upload_url}")

            # Upload with timeout
            timeout = aiohttp.ClientTimeout(total=600)  # 10 minutes timeout
            async with self._session.post(
                upload_url,
                data=form_data,
                headers={
                    "X-CSRFToken": self._cookies.get("passport_csrf_token", ""),
                },
                timeout=timeout
            ) as resp:
                logger.info(f"Upload response status: {resp.status}")
                response_text = await resp.text()

                if resp.status == 200:
                    try:
                        result = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON response: {response_text[:500]}")
                        return UploadResult(
                            success=False,
                            platform="douyin",
                            error="服务器返回无效响应"
                        )

                    if result.get("status_code") == 0:
                        video_id = result.get("data", {}).get("video_id", "")
                        logger.info(f"Video uploaded, video_id: {video_id}")

                        # Wait for processing
                        await asyncio.sleep(2)

                        # Publish video with metadata
                        publish_result = await self._publish_video(video_id, metadata)

                        if publish_result:
                            aweme_id = publish_result.get("aweme_id")
                            logger.info(f"Douyin upload success! aweme_id: {aweme_id}")
                            return UploadResult(
                                success=True,
                                video_id=aweme_id,
                                video_url=f"https://www.douyin.com/video/{aweme_id}",
                                platform="douyin",
                            )
                        else:
                            return UploadResult(
                                success=False,
                                platform="douyin",
                                error="视频上传成功但发布失败"
                            )
                    else:
                        error_msg = result.get("status_msg", "上传失败")
                        logger.error(f"Douyin upload failed: {result}")
                        return UploadResult(
                            success=False,
                            platform="douyin",
                            error=f"上传失败: {error_msg}"
                        )
                else:
                    logger.error(f"Douyin upload HTTP error: {resp.status}")
                    return UploadResult(
                        success=False,
                        platform="douyin",
                        error=f"HTTP错误: {resp.status}"
                    )

        except asyncio.TimeoutError:
            logger.error("Douyin upload timeout")
            return UploadResult(
                success=False,
                platform="douyin",
                error="上传超时，请检查网络连接"
            )
        except Exception as e:
            logger.error(f"Douyin upload error: {e}")
            return UploadResult(
                success=False,
                platform="douyin",
                error=f"上传出错: {str(e)}"
            )

    async def _publish_video(
        self,
        video_id: str,
        metadata: VideoMetadata
    ) -> Optional[Dict[str, Any]]:
        """Publish uploaded video"""
        try:
            publish_data = {
                "video_id": video_id,
                "text": metadata.title,
                "desc": metadata.description,
                "hashtag_list": [{"hashtag_name": tag} for tag in metadata.tags[:5]],
                "visibility_type": 0,  # 0=public, 1=friends, 2=private
                "sync_to_toutiao": False,
            }

            async with self._session.post(
                f"{self.CREATOR_URL}/web/api/media/aweme/web/create/",
                json=publish_data,
                headers={
                    "X-CSRFToken": self._cookies.get("passport_csrf_token", ""),
                }
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result.get("status_code") == 0:
                        return result.get("data")
                return None

        except Exception as e:
            logger.error(f"Publish video error: {e}")
            return None

    async def check_upload_status(self, video_id: str) -> Dict[str, Any]:
        """Check video status"""
        try:
            async with self._session.get(
                f"{self.CREATOR_URL}/web/api/media/aweme/web/detail/",
                params={"aweme_id": video_id}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("status_code") == 0:
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


class DouyinQRLogin:
    """
    Douyin QR code login helper

    Usage:
    1. Get QR code image
    2. User scans with Douyin app
    3. Poll for login result
    4. Get cookies on success
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
                "https://sso.douyin.com/get_qrcode/",
                params={
                    "service": "https://creator.douyin.com",
                    "need_logo": "false",
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "qrcode_url": data.get("data", {}).get("qrcode"),
                        "token": data.get("data", {}).get("token"),
                    }
                return None

        except Exception as e:
            logger.error(f"Get QR code error: {e}")
            return None

    async def poll_qr_status(self, token: str) -> Optional[Dict[str, str]]:
        """
        Poll QR code scan status

        Returns cookies dict on success
        """
        try:
            max_attempts = 60  # 60 seconds timeout
            for _ in range(max_attempts):
                async with self._session.get(
                    "https://sso.douyin.com/check_qrconnect/",
                    params={"token": token}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        status = data.get("data", {}).get("status")

                        if status == "1":  # Waiting
                            await asyncio.sleep(1)
                            continue
                        elif status == "2":  # Scanned
                            logger.info("QR code scanned, waiting confirmation...")
                            await asyncio.sleep(1)
                            continue
                        elif status == "3":  # Confirmed
                            # Extract cookies from response
                            cookies = {}
                            for cookie in resp.cookies.values():
                                cookies[cookie.key] = cookie.value
                            return cookies
                        elif status == "4":  # Expired
                            logger.error("QR code expired")
                            return None
                        elif status == "5":  # Cancelled
                            logger.error("QR code cancelled")
                            return None

                await asyncio.sleep(1)

            return None

        except Exception as e:
            logger.error(f"Poll QR status error: {e}")
            return None

    async def close(self):
        if self._session:
            await self._session.close()

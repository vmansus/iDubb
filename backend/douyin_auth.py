"""
Douyin QR Code Authentication

抖音扫码登录流程:
1. 申请二维码 -> 获取 token 和 url
2. 前端展示二维码
3. 用户用抖音 App 扫码
4. 轮询扫码状态
5. 扫码成功后获取 cookies
6. 获取用户信息
7. 保存到数据库 (按 UID 去重)
"""
import asyncio
import time
import uuid
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
from loguru import logger


@dataclass
class DouyinAccount:
    """Douyin account info"""
    uid: str
    nickname: str
    avatar: str
    cookies: str  # Full cookie string
    updated_at: str
    label: str = ""
    is_primary: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DouyinAccount":
        data.setdefault("label", "")
        data.setdefault("is_primary", False)
        return cls(**data)
    
    def to_auth_dict(self) -> Dict[str, str]:
        """Return dict for uploader authentication"""
        return {
            "cookies": self.cookies,
        }
    
    @property
    def display_name(self) -> str:
        return self.label if self.label else self.nickname


class DouyinQRAuth:
    """Douyin QR code authentication handler"""
    
    # API endpoints (creator.douyin.com)
    QR_GENERATE_URL = "https://sso.douyin.com/get_qrcode/"
    QR_POLL_URL = "https://sso.douyin.com/check_qrconnect/"
    USER_INFO_URL = "https://creator.douyin.com/web/api/media/user/info/"
    
    # Poll status
    STATUS_NOT_SCANNED = 1  # 未扫码
    STATUS_SCANNED = 2  # 已扫码待确认
    STATUS_SUCCESS = 3  # 已确认
    STATUS_EXPIRED = 4  # 已过期
    STATUS_CANCELED = 5  # 已取消
    
    def __init__(self):
        self._pending_qrcodes: Dict[str, Dict[str, Any]] = {}
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://creator.douyin.com/",
            "Origin": "https://creator.douyin.com",
        }
    
    async def generate_qrcode(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate QR code for login
        
        Returns:
            Tuple of (token, qrcode_url) or (None, None) on failure
        """
        try:
            # Generate a unique token
            token = str(uuid.uuid4())
            
            async with aiohttp.ClientSession(headers=self._get_headers()) as session:
                params = {
                    "service": "https://creator.douyin.com",
                    "need_logo": "false",
                    "aid": "2906",
                    "account_sdk_source": "sso",
                    "sdk_version": "2.2.5",
                    "language": "zh",
                }
                
                async with session.get(self.QR_GENERATE_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Douyin QR generate failed: HTTP {resp.status}")
                        return None, None
                    
                    data = await resp.json()
                    
                    if data.get("error_code") != 0:
                        logger.error(f"Douyin QR generate failed: {data}")
                        return None, None
                    
                    qr_data = data.get("data", {})
                    token = qr_data.get("token")
                    qrcode_url = qr_data.get("qrcode")
                    
                    if not token or not qrcode_url:
                        logger.error(f"Douyin QR generate missing data: {data}")
                        return None, None
                    
                    self._pending_qrcodes[token] = {
                        "url": qrcode_url,
                        "created_at": time.time()
                    }
                    
                    self._cleanup_pending()
                    
                    logger.info(f"Generated Douyin QR code: {token[:8]}...")
                    return token, qrcode_url
                    
        except Exception as e:
            logger.error(f"Douyin QR generate error: {e}")
            return None, None
    
    def _cleanup_pending(self):
        """Remove expired pending QR codes"""
        now = time.time()
        expired = [k for k, v in self._pending_qrcodes.items() 
                   if now - v["created_at"] > 300]
        for k in expired:
            del self._pending_qrcodes[k]
    
    async def poll_qrcode(self, token: str) -> Dict[str, Any]:
        """
        Poll QR code scan status
        
        Returns:
            {
                "status": "waiting" | "scanned" | "expired" | "success" | "error",
                "message": str,
                "account": DouyinAccount | None
            }
        """
        try:
            async with aiohttp.ClientSession(headers=self._get_headers()) as session:
                params = {
                    "token": token,
                    "service": "https://creator.douyin.com",
                    "aid": "2906",
                    "account_sdk_source": "sso",
                    "sdk_version": "2.2.5",
                }
                
                async with session.get(self.QR_POLL_URL, params=params) as resp:
                    if resp.status != 200:
                        return {
                            "status": "error",
                            "message": f"HTTP {resp.status}",
                            "account": None
                        }
                    
                    data = await resp.json()
                    error_code = data.get("error_code", -1)
                    
                    # Status mapping
                    if error_code == 2046:  # 未扫码
                        return {
                            "status": "waiting",
                            "message": "等待扫码",
                            "account": None
                        }
                    elif error_code == 2038:  # 已扫码待确认
                        return {
                            "status": "scanned",
                            "message": "已扫码，请在手机上确认",
                            "account": None
                        }
                    elif error_code == 2039:  # 二维码过期
                        return {
                            "status": "expired",
                            "message": "二维码已过期",
                            "account": None
                        }
                    elif error_code == 0:  # 登录成功
                        # Get cookies from response
                        redirect_url = data.get("data", {}).get("redirect_url", "")
                        
                        # Follow redirect to get cookies
                        cookies_dict = {}
                        async with session.get(redirect_url, allow_redirects=False) as redirect_resp:
                            # Collect cookies from redirect
                            for cookie in session.cookie_jar:
                                cookies_dict[cookie.key] = cookie.value
                        
                        # Get user info
                        account = await self._get_user_info(cookies_dict)
                        
                        if account:
                            return {
                                "status": "success",
                                "message": "登录成功",
                                "account": account
                            }
                        else:
                            return {
                                "status": "error",
                                "message": "获取用户信息失败",
                                "account": None
                            }
                    else:
                        return {
                            "status": "error",
                            "message": data.get("description", "未知错误"),
                            "account": None
                        }
                        
        except Exception as e:
            logger.error(f"Douyin QR poll error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "account": None
            }
    
    async def _get_user_info(self, cookies_dict: Dict[str, str]) -> Optional[DouyinAccount]:
        """Get user info after successful login"""
        try:
            cookie_string = "; ".join([f"{k}={v}" for k, v in cookies_dict.items()])
            
            headers = self._get_headers()
            headers["Cookie"] = cookie_string
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(self.USER_INFO_URL) as resp:
                    if resp.status != 200:
                        logger.error(f"Douyin user info failed: HTTP {resp.status}")
                        return None
                    
                    data = await resp.json()
                    
                    if data.get("status_code") != 0:
                        logger.error(f"Douyin user info failed: {data}")
                        return None
                    
                    user_data = data.get("user", {})
                    
                    return DouyinAccount(
                        uid=str(user_data.get("uid", "")),
                        nickname=user_data.get("nickname", "抖音用户"),
                        avatar=user_data.get("avatar_url", ""),
                        cookies=cookie_string,
                        updated_at=datetime.now().isoformat(),
                    )
                    
        except Exception as e:
            logger.error(f"Douyin get user info error: {e}")
            return None


# Global instance
douyin_qr_auth = DouyinQRAuth()

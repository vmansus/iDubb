"""
Bilibili QR Code Authentication

B站扫码登录流程:
1. 申请二维码 -> 获取 qrcode_key 和 url
2. 前端展示二维码
3. 用户用B站 App 扫码
4. 轮询扫码状态
5. 扫码成功后获取 cookies
6. 获取用户信息 (UID, 昵称, 头像)
7. 保存到数据库 (按 UID 去重)
"""
import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
from loguru import logger


@dataclass
class BilibiliAccount:
    """Bilibili account info"""
    uid: str
    nickname: str
    avatar: str
    sessdata: str
    bili_jct: str
    buvid3: str
    updated_at: str
    label: str = ""  # 唯一标签，用于显示和选择
    is_primary: bool = False  # 是否为主账号
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BilibiliAccount":
        # Handle old data without label/is_primary
        data.setdefault("label", "")
        data.setdefault("is_primary", False)
        return cls(**data)
    
    def to_auth_dict(self) -> Dict[str, str]:
        """Return dict for uploader authentication"""
        return {
            "SESSDATA": self.sessdata,
            "bili_jct": self.bili_jct,
            "buvid3": self.buvid3,
        }
    
    @property
    def display_name(self) -> str:
        """显示名称：优先显示标签，否则显示昵称"""
        return self.label if self.label else self.nickname


class BilibiliQRAuth:
    """Bilibili QR code authentication handler"""
    
    # API endpoints
    QR_GENERATE_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
    QR_POLL_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"
    USER_INFO_URL = "https://api.bilibili.com/x/web-interface/nav"
    
    # Poll status codes
    STATUS_NOT_SCANNED = 86101  # 未扫码
    STATUS_SCANNED_NOT_CONFIRMED = 86090  # 已扫码未确认
    STATUS_EXPIRED = 86038  # 二维码已过期
    STATUS_SUCCESS = 0  # 登录成功
    
    def __init__(self):
        self._pending_qrcodes: Dict[str, Dict[str, Any]] = {}  # key -> {url, created_at}
    
    async def generate_qrcode(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate QR code for login
        
        Returns:
            Tuple of (qrcode_key, qrcode_url) or (None, None) on failure
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://www.bilibili.com/",
                "Origin": "https://www.bilibili.com",
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(self.QR_GENERATE_URL) as resp:
                    if resp.status != 200:
                        logger.error(f"QR generate failed: HTTP {resp.status}")
                        return None, None
                    
                    data = await resp.json()
                    
                    if data.get("code") != 0:
                        logger.error(f"QR generate failed: {data}")
                        return None, None
                    
                    qrcode_key = data["data"]["qrcode_key"]
                    qrcode_url = data["data"]["url"]
                    
                    # Store pending QR code
                    self._pending_qrcodes[qrcode_key] = {
                        "url": qrcode_url,
                        "created_at": time.time()
                    }
                    
                    # Clean old pending codes (older than 5 minutes)
                    self._cleanup_pending()
                    
                    logger.info(f"Generated QR code: {qrcode_key[:8]}...")
                    return qrcode_key, qrcode_url
                    
        except Exception as e:
            logger.error(f"QR generate error: {e}")
            return None, None
    
    def _cleanup_pending(self):
        """Remove expired pending QR codes"""
        now = time.time()
        expired = [k for k, v in self._pending_qrcodes.items() 
                   if now - v["created_at"] > 300]  # 5 minutes
        for k in expired:
            del self._pending_qrcodes[k]
    
    def _generate_buvid3(self) -> str:
        """Generate a buvid3 device fingerprint"""
        import uuid
        import hashlib
        
        # buvid3 format: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXXinfoc
        # It's basically a UUID with "infoc" suffix
        raw_uuid = str(uuid.uuid4()).upper()
        return f"{raw_uuid}infoc"
    
    async def poll_qrcode(self, qrcode_key: str) -> Dict[str, Any]:
        """
        Poll QR code scan status
        
        Returns:
            {
                "status": "waiting" | "scanned" | "expired" | "success" | "error",
                "message": str,
                "account": BilibiliAccount | None  # Only on success
            }
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://www.bilibili.com/",
                "Origin": "https://www.bilibili.com",
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                    self.QR_POLL_URL,
                    params={"qrcode_key": qrcode_key}
                ) as resp:
                    if resp.status != 200:
                        return {"status": "error", "message": f"HTTP {resp.status}"}
                    
                    data = await resp.json()
                    code = data.get("data", {}).get("code")
                    
                    if code == self.STATUS_NOT_SCANNED:
                        return {"status": "waiting", "message": "等待扫码"}
                    
                    elif code == self.STATUS_SCANNED_NOT_CONFIRMED:
                        return {"status": "scanned", "message": "已扫码，请在手机上确认"}
                    
                    elif code == self.STATUS_EXPIRED:
                        # Clean up
                        self._pending_qrcodes.pop(qrcode_key, None)
                        return {"status": "expired", "message": "二维码已过期，请重新生成"}
                    
                    elif code == self.STATUS_SUCCESS:
                        # Extract cookies from response
                        logger.info(f"QR login success, extracting cookies...")
                        logger.debug(f"Response data: {data}")
                        
                        url = data["data"].get("url", "")
                        logger.debug(f"Redirect URL: {url}")
                        
                        cookies = self._parse_cookies_from_url(url)
                        logger.debug(f"Cookies from URL: {cookies}")
                        
                        # Also check Set-Cookie headers
                        set_cookies = resp.headers.getall("Set-Cookie", [])
                        logger.debug(f"Set-Cookie headers: {set_cookies}")
                        for cookie_str in set_cookies:
                            self._parse_set_cookie(cookie_str, cookies)
                        
                        # Also try refresh_token approach if no SESSDATA
                        if not cookies.get("SESSDATA"):
                            refresh_token = data["data"].get("refresh_token", "")
                            if refresh_token:
                                logger.info("No SESSDATA in URL, trying refresh_token...")
                                # The cookies might be in the response cookies
                                for cookie in resp.cookies.values():
                                    if cookie.key in ["SESSDATA", "bili_jct", "buvid3", "DedeUserID"]:
                                        cookies[cookie.key] = cookie.value
                                logger.debug(f"Cookies from resp.cookies: {cookies}")
                        
                        if not cookies.get("SESSDATA"):
                            logger.error(f"No SESSDATA in response: {data}")
                            return {"status": "error", "message": "登录成功但未获取到凭证"}
                        
                        # Get user info
                        account = await self._get_user_info(cookies)
                        
                        if account:
                            # Clean up pending
                            self._pending_qrcodes.pop(qrcode_key, None)
                            return {
                                "status": "success",
                                "message": f"登录成功: {account.nickname}",
                                "account": account
                            }
                        else:
                            return {"status": "error", "message": "获取用户信息失败"}
                    
                    else:
                        return {"status": "error", "message": f"未知状态: {code}"}
                        
        except Exception as e:
            logger.error(f"QR poll error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _parse_cookies_from_url(self, url: str) -> Dict[str, str]:
        """Parse cookies from redirect URL"""
        cookies = {}
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            for key in ["SESSDATA", "bili_jct", "buvid3", "DedeUserID"]:
                if key in params:
                    cookies[key] = params[key][0]
        except Exception as e:
            logger.warning(f"Failed to parse URL cookies: {e}")
        
        return cookies
    
    def _parse_set_cookie(self, cookie_str: str, cookies: Dict[str, str]):
        """Parse Set-Cookie header"""
        try:
            # Format: "name=value; Path=/; ..."
            parts = cookie_str.split(";")
            if parts:
                name_value = parts[0].strip()
                if "=" in name_value:
                    name, value = name_value.split("=", 1)
                    if name in ["SESSDATA", "bili_jct", "buvid3", "DedeUserID"]:
                        cookies[name] = value
        except Exception as e:
            logger.warning(f"Failed to parse Set-Cookie: {e}")
    
    async def _get_user_info(self, cookies: Dict[str, str]) -> Optional[BilibiliAccount]:
        """Get user info using cookies"""
        try:
            cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://www.bilibili.com/",
                "Cookie": cookie_header,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.USER_INFO_URL,
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Get user info failed: HTTP {resp.status}")
                        return None
                    
                    data = await resp.json()
                    
                    if data.get("code") != 0:
                        logger.error(f"Get user info failed: {data}")
                        return None
                    
                    user_data = data.get("data", {})
                    
                    # Generate buvid3 if not present (device fingerprint)
                    buvid3 = cookies.get("buvid3", "")
                    if not buvid3:
                        buvid3 = self._generate_buvid3()
                        logger.info(f"Generated buvid3: {buvid3[:20]}...")
                    
                    return BilibiliAccount(
                        uid=str(user_data.get("mid", "")),
                        nickname=user_data.get("uname", "Unknown"),
                        avatar=user_data.get("face", ""),
                        sessdata=cookies.get("SESSDATA", ""),
                        bili_jct=cookies.get("bili_jct", ""),
                        buvid3=buvid3,
                        updated_at=datetime.utcnow().isoformat() + "Z"
                    )
                    
        except Exception as e:
            logger.error(f"Get user info error: {e}")
            return None


# Global instance
bilibili_qr_auth = BilibiliQRAuth()

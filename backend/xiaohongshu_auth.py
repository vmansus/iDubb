"""
Xiaohongshu QR Code Authentication

小红书扫码登录流程 (使用 xhs 库):
1. 使用 xhs 库申请二维码
2. 前端展示二维码
3. 用户用小红书 App 扫码
4. 轮询扫码状态
5. 扫码成功后获取 cookies
6. 获取用户信息
7. 保存到数据库 (按 user_id 去重)

依赖: pip install xhs playwright
     playwright install chromium
"""
import asyncio
import time
import pathlib
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class XiaohongshuAccount:
    """Xiaohongshu account info"""
    user_id: str
    nickname: str
    avatar: str
    cookies: str  # Full cookie string
    updated_at: str
    label: str = ""
    is_primary: bool = False
    
    # Alias for compatibility
    @property
    def uid(self) -> str:
        return self.user_id
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "XiaohongshuAccount":
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


def _get_sign_function():
    """Create sign function using playwright"""
    from playwright.sync_api import sync_playwright
    from time import sleep
    
    def sign(uri, data=None, a1="", web_session=""):
        for _ in range(3):
            try:
                with sync_playwright() as playwright:
                    # stealth.min.js path
                    stealth_js_path = pathlib.Path(__file__).parent / "stealth.min.js"
                    
                    browser = playwright.chromium.launch(headless=True)
                    browser_context = browser.new_context()
                    
                    if stealth_js_path.exists():
                        browser_context.add_init_script(path=str(stealth_js_path))
                    
                    context_page = browser_context.new_page()
                    context_page.goto("https://www.xiaohongshu.com")
                    
                    if a1:
                        browser_context.add_cookies([
                            {'name': 'a1', 'value': a1, 'domain': ".xiaohongshu.com", 'path': "/"}
                        ])
                        context_page.reload()
                    
                    sleep(1)
                    encrypt_params = context_page.evaluate(
                        "([url, data]) => window._webmsxyw(url, data)", 
                        [uri, data]
                    )
                    browser.close()
                    return {
                        "x-s": encrypt_params["X-s"],
                        "x-t": str(encrypt_params["X-t"])
                    }
            except Exception as e:
                logger.warning(f"Sign attempt failed: {e}")
                continue
        raise Exception("签名失败")
    
    return sign


class XiaohongshuQRAuth:
    """Xiaohongshu QR code authentication handler using xhs library"""
    
    def __init__(self):
        self._pending_qrcodes: Dict[str, Dict[str, Any]] = {}
        self._xhs_client = None
    
    def _get_client(self):
        """Get or create XhsClient"""
        if self._xhs_client is None:
            try:
                from xhs import XhsClient
                sign = _get_sign_function()
                self._xhs_client = XhsClient(sign=sign, timeout=60)
            except ImportError:
                logger.error("xhs library not installed. Run: pip install xhs")
                raise
        return self._xhs_client
    
    async def generate_qrcode(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate QR code for login using xhs library
        
        Returns:
            Tuple of (qr_id, qrcode_url) or (None, None) on failure
        """
        try:
            # Run sync xhs client in thread pool
            loop = asyncio.get_event_loop()
            qr_res = await loop.run_in_executor(None, self._generate_qrcode_sync)
            
            if qr_res:
                qr_id = qr_res.get("qr_id")
                qrcode_url = qr_res.get("url")
                qr_code = qr_res.get("code")
                
                self._pending_qrcodes[qr_id] = {
                    "url": qrcode_url,
                    "code": qr_code,
                    "created_at": time.time()
                }
                
                self._cleanup_pending()
                
                logger.info(f"Generated Xiaohongshu QR code: {qr_id[:8]}...")
                return qr_id, qrcode_url
            
            return None, None
                    
        except Exception as e:
            logger.error(f"Xiaohongshu QR generate error: {e}")
            return None, None
    
    def _generate_qrcode_sync(self) -> Optional[Dict]:
        """Sync method to generate QR code"""
        try:
            client = self._get_client()
            return client.get_qrcode()
        except Exception as e:
            logger.error(f"XHS get_qrcode failed: {e}")
            return None
    
    def _cleanup_pending(self):
        """Remove expired pending QR codes"""
        now = time.time()
        expired = [k for k, v in self._pending_qrcodes.items() 
                   if now - v["created_at"] > 300]
        for k in expired:
            del self._pending_qrcodes[k]
    
    async def poll_qrcode(self, qr_id: str) -> Dict[str, Any]:
        """
        Poll QR code scan status
        
        Returns:
            {
                "status": "waiting" | "scanned" | "expired" | "success" | "error",
                "message": str,
                "account": XiaohongshuAccount | None
            }
        """
        try:
            pending = self._pending_qrcodes.get(qr_id)
            if not pending:
                return {
                    "status": "expired",
                    "message": "二维码已过期",
                    "account": None
                }
            
            qr_code = pending.get("code", "")
            
            # Run sync check in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._check_qrcode_sync, 
                qr_id, 
                qr_code
            )
            
            return result
                        
        except Exception as e:
            logger.error(f"Xiaohongshu QR poll error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "account": None
            }
    
    def _check_qrcode_sync(self, qr_id: str, qr_code: str) -> Dict[str, Any]:
        """Sync method to check QR code status"""
        try:
            client = self._get_client()
            result = client.check_qrcode(qr_id, qr_code)
            
            code_status = result.get("code_status", 0)
            
            # Status mapping:
            # 0 = waiting, 1 = scanned, 2 = confirmed, 3 = expired
            if code_status == 0:
                return {
                    "status": "waiting",
                    "message": "等待扫码",
                    "account": None
                }
            elif code_status == 1:
                return {
                    "status": "scanned",
                    "message": "已扫码，请在手机上确认",
                    "account": None
                }
            elif code_status == 2:
                # Login success!
                login_info = result.get("login_info", {})
                cookie = client.cookie
                
                # Get user info
                try:
                    user_info = client.get_self_info()
                    user_data = user_info.get("basic_info", {})
                    
                    account = XiaohongshuAccount(
                        user_id=user_data.get("red_id", "") or login_info.get("user_id", ""),
                        nickname=user_data.get("nickname", "小红书用户"),
                        avatar=user_data.get("imageb", "") or user_data.get("image", ""),
                        cookies=cookie,
                        updated_at=datetime.now().isoformat(),
                    )
                    
                    return {
                        "status": "success",
                        "message": "登录成功",
                        "account": account
                    }
                except Exception as e:
                    logger.error(f"Get user info failed: {e}")
                    # Still return success with basic info
                    account = XiaohongshuAccount(
                        user_id=login_info.get("user_id", "unknown"),
                        nickname="小红书用户",
                        avatar="",
                        cookies=cookie,
                        updated_at=datetime.now().isoformat(),
                    )
                    return {
                        "status": "success",
                        "message": "登录成功",
                        "account": account
                    }
            else:
                return {
                    "status": "expired",
                    "message": "二维码已过期",
                    "account": None
                }
                
        except Exception as e:
            logger.error(f"XHS check_qrcode failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "account": None
            }


# Global instance
xiaohongshu_qr_auth = XiaohongshuQRAuth()

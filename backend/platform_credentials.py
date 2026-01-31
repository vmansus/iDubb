"""
Platform Credentials Storage

Securely stores and manages platform authentication credentials.
Credentials are stored in database with Fernet encryption.
"""
import json
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class BilibiliCredentials:
    sessdata: str = ""
    bili_jct: str = ""
    buvid3: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.sessdata and self.bili_jct and self.buvid3)

    def to_auth_dict(self) -> Dict[str, str]:
        return {
            "SESSDATA": self.sessdata,
            "bili_jct": self.bili_jct,
            "buvid3": self.buvid3,
        }

    def to_json(self) -> str:
        return json.dumps({
            "sessdata": self.sessdata,
            "bili_jct": self.bili_jct,
            "buvid3": self.buvid3,
        })

    @classmethod
    def from_json(cls, data: str) -> "BilibiliCredentials":
        try:
            d = json.loads(data)
            return cls(
                sessdata=d.get("sessdata", ""),
                bili_jct=d.get("bili_jct", ""),
                buvid3=d.get("buvid3", ""),
            )
        except Exception:
            return cls()


@dataclass
class DouyinCredentials:
    cookies: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.cookies)

    def to_auth_dict(self) -> Dict[str, str]:
        return {"cookies": self.cookies}

    def to_json(self) -> str:
        return json.dumps({"cookies": self.cookies})

    @classmethod
    def from_json(cls, data: str) -> "DouyinCredentials":
        try:
            logger.debug(f"DouyinCredentials.from_json: data_len={len(data) if data else 0}, preview={data[:100] if data else 'None'}...")
            d = json.loads(data)
            cookies = d.get("cookies", "")
            logger.debug(f"DouyinCredentials.from_json: parsed cookies_len={len(cookies) if cookies else 0}")
            return cls(cookies=cookies)
        except Exception as e:
            logger.error(f"DouyinCredentials.from_json failed: {e}, data={data[:100] if data else 'None'}...")
            return cls()


@dataclass
class XiaohongshuCredentials:
    cookies: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.cookies)

    def to_auth_dict(self) -> Dict[str, str]:
        return {"cookies": self.cookies}

    def to_json(self) -> str:
        return json.dumps({"cookies": self.cookies})

    @classmethod
    def from_json(cls, data: str) -> "XiaohongshuCredentials":
        try:
            d = json.loads(data)
            return cls(cookies=d.get("cookies", ""))
        except Exception:
            return cls()


class PlatformCredentialsStore:
    """Manages storage and retrieval of platform credentials via database"""

    def __init__(self):
        self._bilibili: Optional[BilibiliCredentials] = None
        self._douyin: Optional[DouyinCredentials] = None
        self._xiaohongshu: Optional[XiaohongshuCredentials] = None
        self._loaded = False

    def _run_async(self, coro):
        """Run async coroutine from sync context"""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(coro)

    async def _load_async(self):
        """Load credentials from database"""
        from database.connection import get_db
        from database.repository import CookieRepository

        try:
            async with get_db() as session:
                repo = CookieRepository(session)

                # Load Bilibili
                bilibili_data = await repo.get_cookie_data("bilibili")
                if bilibili_data:
                    self._bilibili = BilibiliCredentials.from_json(bilibili_data)
                else:
                    self._bilibili = BilibiliCredentials()

                # Load Douyin
                douyin_data = await repo.get_cookie_data("douyin")
                if douyin_data:
                    self._douyin = DouyinCredentials.from_json(douyin_data)
                else:
                    self._douyin = DouyinCredentials()

                # Load Xiaohongshu
                xhs_data = await repo.get_cookie_data("xiaohongshu")
                if xhs_data:
                    self._xiaohongshu = XiaohongshuCredentials.from_json(xhs_data)
                else:
                    self._xiaohongshu = XiaohongshuCredentials()

                self._loaded = True
                logger.debug("Platform credentials loaded from database")

        except Exception as e:
            logger.error(f"Failed to load platform credentials: {e}")
            self._bilibili = BilibiliCredentials()
            self._douyin = DouyinCredentials()
            self._xiaohongshu = XiaohongshuCredentials()

    def _ensure_loaded(self):
        """Ensure credentials are loaded"""
        if not self._loaded:
            self._run_async(self._load_async())

    async def _save_credential_async(self, platform: str, data: str):
        """Save credential to database"""
        from database.connection import get_db
        from database.repository import CookieRepository

        try:
            async with get_db() as session:
                repo = CookieRepository(session)
                await repo.save(platform=platform, cookie_data=data)
                logger.debug(f"Saved {platform} credentials to database")
        except Exception as e:
            logger.error(f"Failed to save {platform} credentials: {e}")

    async def _clear_credential_async(self, platform: str):
        """Clear credential from database"""
        from database.connection import get_db
        from database.repository import CookieRepository

        try:
            async with get_db() as session:
                repo = CookieRepository(session)
                await repo.delete(platform)
                logger.debug(f"Cleared {platform} credentials from database")
        except Exception as e:
            logger.error(f"Failed to clear {platform} credentials: {e}")

    # Bilibili
    def set_bilibili(self, sessdata: str, bili_jct: str, buvid3: str):
        """Set Bilibili credentials"""
        self._bilibili = BilibiliCredentials(
            sessdata=sessdata,
            bili_jct=bili_jct,
            buvid3=buvid3,
        )
        self._run_async(self._save_credential_async("bilibili", self._bilibili.to_json()))

    def get_bilibili(self) -> BilibiliCredentials:
        """Get Bilibili credentials"""
        self._ensure_loaded()
        return self._bilibili or BilibiliCredentials()

    def clear_bilibili(self):
        """Clear Bilibili credentials"""
        self._bilibili = BilibiliCredentials()
        self._run_async(self._clear_credential_async("bilibili"))

    # Douyin
    def set_douyin(self, cookies: str):
        """Set Douyin credentials"""
        self._douyin = DouyinCredentials(cookies=cookies)
        self._run_async(self._save_credential_async("douyin", self._douyin.to_json()))

    def get_douyin(self) -> DouyinCredentials:
        """Get Douyin credentials"""
        self._ensure_loaded()
        return self._douyin or DouyinCredentials()

    def clear_douyin(self):
        """Clear Douyin credentials"""
        self._douyin = DouyinCredentials()
        self._run_async(self._clear_credential_async("douyin"))

    # Xiaohongshu
    def set_xiaohongshu(self, cookies: str):
        """Set Xiaohongshu credentials"""
        self._xiaohongshu = XiaohongshuCredentials(cookies=cookies)
        self._run_async(self._save_credential_async("xiaohongshu", self._xiaohongshu.to_json()))

    def get_xiaohongshu(self) -> XiaohongshuCredentials:
        """Get Xiaohongshu credentials"""
        self._ensure_loaded()
        return self._xiaohongshu or XiaohongshuCredentials()

    def clear_xiaohongshu(self):
        """Clear Xiaohongshu credentials"""
        self._xiaohongshu = XiaohongshuCredentials()
        self._run_async(self._clear_credential_async("xiaohongshu"))

    def get_status(self) -> Dict[str, bool]:
        """Get configuration status for all platforms"""
        self._ensure_loaded()
        return {
            "bilibili": (self._bilibili or BilibiliCredentials()).is_configured,
            "douyin": (self._douyin or DouyinCredentials()).is_configured,
            "xiaohongshu": (self._xiaohongshu or XiaohongshuCredentials()).is_configured,
        }

    def reload(self):
        """Force reload from database"""
        self._loaded = False
        self._ensure_loaded()


# Global instance
platform_credentials = PlatformCredentialsStore()

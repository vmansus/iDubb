"""
Bilibili Multi-Account Manager

支持多账号管理，按 UID 去重
支持标签（唯一）和主账号功能
"""
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from bilibili_auth import BilibiliAccount


class BilibiliAccountManager:
    """
    Manages multiple Bilibili accounts
    
    Storage key: "bilibili_accounts"
    Format: {uid: account_data, ...}
    
    Features:
    - Dedupe by UID
    - Unique labels for each account
    - One primary account (first account is primary by default)
    """
    
    STORAGE_KEY = "bilibili_accounts"
    
    def __init__(self):
        self._accounts: Dict[str, BilibiliAccount] = {}
        self._loaded = False
    
    async def load(self):
        """Load accounts from database"""
        if self._loaded:
            return
        
        try:
            from database.connection import get_session_factory
            from database.repository import CookieRepository
            
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = CookieRepository(session)
                data = await repo.get_cookie_data(self.STORAGE_KEY)
                
                if data:
                    accounts_dict = json.loads(data)
                    needs_save = False
                    for uid, account_data in accounts_dict.items():
                        try:
                            account = BilibiliAccount.from_dict(account_data)
                            # Fix accounts without buvid3
                            if not account.buvid3:
                                account.buvid3 = self._generate_buvid3()
                                logger.info(f"Generated buvid3 for account {account.nickname}")
                                needs_save = True
                            self._accounts[uid] = account
                        except Exception as e:
                            logger.warning(f"Failed to load account {uid}: {e}")
                    
                    # Ensure there's a primary account
                    self._ensure_primary()
                    
                    # Save if we fixed any accounts
                    if needs_save:
                        # Note: save() will be called after load() completes
                        pass
            
            self._loaded = True
            logger.info(f"Loaded {len(self._accounts)} Bilibili accounts")
            
            # Save fixed accounts
            if self._accounts:
                for acc in self._accounts.values():
                    if not acc.buvid3:
                        await self.save()
                        break
            
        except Exception as e:
            logger.error(f"Failed to load Bilibili accounts: {e}")
            self._loaded = True  # Mark as loaded to avoid infinite retries
    
    def _generate_buvid3(self) -> str:
        """Generate a buvid3 device fingerprint"""
        import uuid
        raw_uuid = str(uuid.uuid4()).upper()
        return f"{raw_uuid}infoc"
    
    def _ensure_primary(self):
        """Ensure exactly one primary account exists"""
        if not self._accounts:
            return
        
        # Check if any account is primary
        primary_accounts = [a for a in self._accounts.values() if a.is_primary]
        
        if len(primary_accounts) == 0:
            # No primary - set first account as primary
            first_account = next(iter(self._accounts.values()))
            first_account.is_primary = True
            logger.info(f"Set {first_account.nickname} as primary account (first account)")
        elif len(primary_accounts) > 1:
            # Multiple primaries - keep only the first one
            for i, acc in enumerate(primary_accounts):
                if i > 0:
                    acc.is_primary = False
            logger.warning("Fixed multiple primary accounts")
    
    async def save(self):
        """Save accounts to database"""
        try:
            from database.connection import get_session_factory
            from database.repository import CookieRepository
            
            accounts_dict = {
                uid: account.to_dict() 
                for uid, account in self._accounts.items()
            }
            
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = CookieRepository(session)
                await repo.save(
                    platform=self.STORAGE_KEY, 
                    cookie_data=json.dumps(accounts_dict, ensure_ascii=False)
                )
                await session.commit()
            
            logger.debug(f"Saved {len(self._accounts)} Bilibili accounts")
            
        except Exception as e:
            logger.error(f"Failed to save Bilibili accounts: {e}")
    
    def _is_label_unique(self, label: str, exclude_uid: Optional[str] = None) -> bool:
        """Check if label is unique"""
        if not label:
            return True
        for uid, acc in self._accounts.items():
            if uid != exclude_uid and acc.label == label:
                return False
        return True
    
    async def add_account(
        self, 
        account: BilibiliAccount, 
        label: Optional[str] = None,
        set_as_primary: bool = False
    ) -> bool:
        """
        Add or update account (deduped by UID)
        
        Args:
            account: Account to add
            label: Optional unique label
            set_as_primary: Whether to set as primary account
        
        Returns True if new account, False if updated existing
        """
        await self.load()
        
        is_new = account.uid not in self._accounts
        
        # Update timestamp
        account.updated_at = datetime.utcnow().isoformat() + "Z"
        
        # Handle label
        if label:
            if not self._is_label_unique(label, account.uid):
                raise ValueError(f"Label '{label}' already exists")
            account.label = label
        elif is_new and not account.label:
            # Auto-generate label for new account if not provided
            account.label = account.nickname
            # Make unique if needed
            counter = 1
            base_label = account.label
            while not self._is_label_unique(account.label, account.uid):
                account.label = f"{base_label}_{counter}"
                counter += 1
        
        # Handle primary account
        if is_new and len(self._accounts) == 0:
            # First account is always primary
            account.is_primary = True
        elif set_as_primary:
            # Clear other primaries
            for acc in self._accounts.values():
                acc.is_primary = False
            account.is_primary = True
        
        self._accounts[account.uid] = account
        await self.save()
        
        if is_new:
            logger.info(f"Added new Bilibili account: {account.display_name} ({account.uid})")
        else:
            logger.info(f"Updated Bilibili account: {account.display_name} ({account.uid})")
        
        return is_new
    
    async def update_label(self, uid: str, label: str) -> bool:
        """Update account label (must be unique)"""
        await self.load()
        
        if uid not in self._accounts:
            return False
        
        if not self._is_label_unique(label, uid):
            raise ValueError(f"Label '{label}' already exists")
        
        self._accounts[uid].label = label
        await self.save()
        return True
    
    async def set_primary(self, uid: str) -> bool:
        """Set account as primary"""
        await self.load()
        
        if uid not in self._accounts:
            return False
        
        # Clear all primaries
        for acc in self._accounts.values():
            acc.is_primary = False
        
        # Set new primary
        self._accounts[uid].is_primary = True
        await self.save()
        
        logger.info(f"Set {self._accounts[uid].display_name} as primary account")
        return True
    
    async def add_from_cookies(
        self, 
        sessdata: str, 
        bili_jct: str, 
        buvid3: str,
        label: Optional[str] = None,
        uid: Optional[str] = None,
        nickname: Optional[str] = None
    ) -> Optional[BilibiliAccount]:
        """
        Add account from raw cookies (for manual paste)
        
        If UID not provided, will try to fetch from API
        """
        from bilibili_auth import BilibiliQRAuth
        
        cookies = {
            "SESSDATA": sessdata,
            "bili_jct": bili_jct,
            "buvid3": buvid3,
        }
        
        # Try to get user info if UID not provided
        if not uid:
            auth = BilibiliQRAuth()
            account = await auth._get_user_info(cookies)
            if account:
                await self.add_account(account, label=label)
                return account
            else:
                logger.error("Failed to get user info from cookies")
                return None
        
        # Create account with provided info
        account = BilibiliAccount(
            uid=uid,
            nickname=nickname or f"User {uid}",
            avatar="",
            sessdata=sessdata,
            bili_jct=bili_jct,
            buvid3=buvid3,
            updated_at=datetime.utcnow().isoformat() + "Z"
        )
        
        await self.add_account(account, label=label)
        return account
    
    async def remove_account(self, uid: str) -> bool:
        """Remove account by UID"""
        await self.load()
        
        if uid not in self._accounts:
            return False
        
        account = self._accounts.pop(uid)
        was_primary = account.is_primary
        
        # If removed primary, set new primary
        if was_primary and self._accounts:
            first_account = next(iter(self._accounts.values()))
            first_account.is_primary = True
            logger.info(f"Set {first_account.display_name} as new primary account")
        
        await self.save()
        logger.info(f"Removed Bilibili account: {account.display_name} ({uid})")
        return True
    
    async def get_account(self, uid: str) -> Optional[BilibiliAccount]:
        """Get account by UID"""
        await self.load()
        return self._accounts.get(uid)
    
    async def get_account_by_label(self, label: str) -> Optional[BilibiliAccount]:
        """Get account by label"""
        await self.load()
        for acc in self._accounts.values():
            if acc.label == label:
                return acc
        return None
    
    async def get_primary_account(self) -> Optional[BilibiliAccount]:
        """Get primary account"""
        await self.load()
        for acc in self._accounts.values():
            if acc.is_primary:
                return acc
        # Fallback to first account
        if self._accounts:
            return next(iter(self._accounts.values()))
        return None
    
    async def get_all_accounts(self) -> List[BilibiliAccount]:
        """Get all accounts"""
        await self.load()
        return list(self._accounts.values())
    
    async def list_accounts(self) -> List[Dict[str, Any]]:
        """List accounts (safe info only, no credentials)"""
        await self.load()
        return [
            {
                "uid": acc.uid,
                "nickname": acc.nickname,
                "avatar": acc.avatar,
                "label": acc.label,
                "is_primary": acc.is_primary,
                "updated_at": acc.updated_at,
            }
            for acc in self._accounts.values()
        ]
    
    async def get_default_account(self) -> Optional[BilibiliAccount]:
        """Get default (primary) account"""
        return await self.get_primary_account()
    
    def get_account_sync(self, uid: str) -> Optional[BilibiliAccount]:
        """Synchronous get (assumes already loaded)"""
        return self._accounts.get(uid)
    
    def get_default_account_sync(self) -> Optional[BilibiliAccount]:
        """Synchronous get default (assumes already loaded)"""
        for acc in self._accounts.values():
            if acc.is_primary:
                return acc
        if self._accounts:
            return next(iter(self._accounts.values()))
        return None


# Global instance
bilibili_account_manager = BilibiliAccountManager()

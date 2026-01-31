"""
Douyin Multi-Account Manager

支持多账号管理，按 UID 去重
支持标签（唯一）和主账号功能
"""
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from douyin_auth import DouyinAccount


class DouyinAccountManager:
    """
    Manages multiple Douyin accounts
    
    Storage key: "douyin_accounts"
    Format: {uid: account_data, ...}
    """
    
    STORAGE_KEY = "douyin_accounts"
    
    def __init__(self):
        self._accounts: Dict[str, DouyinAccount] = {}
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
                    for uid, account_data in accounts_dict.items():
                        try:
                            account = DouyinAccount.from_dict(account_data)
                            self._accounts[uid] = account
                        except Exception as e:
                            logger.warning(f"Failed to load Douyin account {uid}: {e}")
                    
                    self._ensure_primary()
            
            self._loaded = True
            logger.info(f"Loaded {len(self._accounts)} Douyin accounts")
            
        except Exception as e:
            logger.error(f"Failed to load Douyin accounts: {e}")
            self._loaded = True
    
    def _ensure_primary(self):
        """Ensure exactly one primary account exists"""
        if not self._accounts:
            return
        
        primary_accounts = [a for a in self._accounts.values() if a.is_primary]
        
        if len(primary_accounts) == 0:
            first_account = next(iter(self._accounts.values()))
            first_account.is_primary = True
            logger.info(f"Set {first_account.nickname} as primary Douyin account")
        elif len(primary_accounts) > 1:
            for i, acc in enumerate(primary_accounts):
                if i > 0:
                    acc.is_primary = False
            logger.warning("Fixed multiple primary Douyin accounts")
    
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
            
            logger.debug(f"Saved {len(self._accounts)} Douyin accounts")
            
        except Exception as e:
            logger.error(f"Failed to save Douyin accounts: {e}")
    
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
        account: DouyinAccount, 
        label: Optional[str] = None,
        set_as_primary: bool = False
    ) -> bool:
        """
        Add or update an account
        
        Args:
            account: DouyinAccount to add
            label: Optional unique label
            set_as_primary: Set as primary account
            
        Returns:
            True if account was added/updated
        """
        await self.load()
        
        # Check label uniqueness
        if label and not self._is_label_unique(label, account.uid):
            logger.warning(f"Label '{label}' already exists")
            return False
        
        # Set label if provided
        if label:
            account.label = label
        
        # Handle primary status
        if set_as_primary or not self._accounts:
            # Remove primary from others
            for acc in self._accounts.values():
                acc.is_primary = False
            account.is_primary = True
        
        # Add/update account
        existing = self._accounts.get(account.uid)
        if existing:
            # Update existing - preserve label if not provided
            if not label and existing.label:
                account.label = existing.label
            if not set_as_primary:
                account.is_primary = existing.is_primary
            logger.info(f"Updated Douyin account: {account.nickname} ({account.uid})")
        else:
            logger.info(f"Added new Douyin account: {account.nickname} ({account.uid})")
        
        self._accounts[account.uid] = account
        await self.save()
        return True
    
    async def remove_account(self, uid: str) -> bool:
        """Remove an account by UID"""
        await self.load()
        
        if uid not in self._accounts:
            return False
        
        removed = self._accounts.pop(uid)
        logger.info(f"Removed Douyin account: {removed.nickname}")
        
        # Ensure primary
        if removed.is_primary and self._accounts:
            first = next(iter(self._accounts.values()))
            first.is_primary = True
            logger.info(f"Set {first.nickname} as new primary account")
        
        await self.save()
        return True
    
    async def set_primary(self, uid: str) -> bool:
        """Set an account as primary"""
        await self.load()
        
        if uid not in self._accounts:
            return False
        
        for acc in self._accounts.values():
            acc.is_primary = (acc.uid == uid)
        
        await self.save()
        return True
    
    async def update_label(self, uid: str, label: str) -> bool:
        """Update account label"""
        await self.load()
        
        if uid not in self._accounts:
            return False
        
        if not self._is_label_unique(label, uid):
            return False
        
        self._accounts[uid].label = label
        await self.save()
        return True
    
    async def get_account(self, uid: str) -> Optional[DouyinAccount]:
        """Get account by UID"""
        await self.load()
        return self._accounts.get(uid)
    
    async def get_primary(self) -> Optional[DouyinAccount]:
        """Get primary account"""
        await self.load()
        for acc in self._accounts.values():
            if acc.is_primary:
                return acc
        return next(iter(self._accounts.values()), None) if self._accounts else None
    
    async def list_accounts(self) -> List[Dict[str, Any]]:
        """List all accounts (safe format for API)"""
        await self.load()
        return [
            {
                "uid": acc.uid,
                "nickname": acc.nickname,
                "avatar": acc.avatar,
                "label": acc.label,
                "is_primary": acc.is_primary,
                "display_name": acc.display_name,
                "updated_at": acc.updated_at,
            }
            for acc in self._accounts.values()
        ]
    
    def get_account_sync(self, uid: str) -> Optional[DouyinAccount]:
        """Sync version for non-async contexts"""
        return self._accounts.get(uid)
    
    def list_accounts_sync(self) -> List[Dict[str, Any]]:
        """Sync version for non-async contexts"""
        return [
            {
                "uid": acc.uid,
                "nickname": acc.nickname,
                "avatar": acc.avatar,
                "label": acc.label,
                "is_primary": acc.is_primary,
                "display_name": acc.display_name,
                "updated_at": acc.updated_at,
            }
            for acc in self._accounts.values()
        ]


# Global instance
douyin_account_manager = DouyinAccountManager()

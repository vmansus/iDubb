"""
Xiaohongshu Multi-Account Manager

支持多账号管理，按 user_id 去重
支持标签（唯一）和主账号功能
"""
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from xiaohongshu_auth import XiaohongshuAccount


class XiaohongshuAccountManager:
    """
    Manages multiple Xiaohongshu accounts
    
    Storage key: "xiaohongshu_accounts"
    Format: {user_id: account_data, ...}
    """
    
    STORAGE_KEY = "xiaohongshu_accounts"
    
    def __init__(self):
        self._accounts: Dict[str, XiaohongshuAccount] = {}
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
                    for user_id, account_data in accounts_dict.items():
                        try:
                            account = XiaohongshuAccount.from_dict(account_data)
                            self._accounts[user_id] = account
                        except Exception as e:
                            logger.warning(f"Failed to load Xiaohongshu account {user_id}: {e}")
                    
                    self._ensure_primary()
            
            self._loaded = True
            logger.info(f"Loaded {len(self._accounts)} Xiaohongshu accounts")
            
        except Exception as e:
            logger.error(f"Failed to load Xiaohongshu accounts: {e}")
            self._loaded = True
    
    def _ensure_primary(self):
        """Ensure exactly one primary account exists"""
        if not self._accounts:
            return
        
        primary_accounts = [a for a in self._accounts.values() if a.is_primary]
        
        if len(primary_accounts) == 0:
            first_account = next(iter(self._accounts.values()))
            first_account.is_primary = True
            logger.info(f"Set {first_account.nickname} as primary Xiaohongshu account")
        elif len(primary_accounts) > 1:
            for i, acc in enumerate(primary_accounts):
                if i > 0:
                    acc.is_primary = False
            logger.warning("Fixed multiple primary Xiaohongshu accounts")
    
    async def save(self):
        """Save accounts to database"""
        try:
            from database.connection import get_session_factory
            from database.repository import CookieRepository
            
            accounts_dict = {
                user_id: account.to_dict() 
                for user_id, account in self._accounts.items()
            }
            
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = CookieRepository(session)
                await repo.save(
                    platform=self.STORAGE_KEY, 
                    cookie_data=json.dumps(accounts_dict, ensure_ascii=False)
                )
                await session.commit()
            
            logger.debug(f"Saved {len(self._accounts)} Xiaohongshu accounts")
            
        except Exception as e:
            logger.error(f"Failed to save Xiaohongshu accounts: {e}")
    
    def _is_label_unique(self, label: str, exclude_user_id: Optional[str] = None) -> bool:
        """Check if label is unique"""
        if not label:
            return True
        for user_id, acc in self._accounts.items():
            if user_id != exclude_user_id and acc.label == label:
                return False
        return True
    
    async def add_account(
        self, 
        account: XiaohongshuAccount, 
        label: Optional[str] = None,
        set_as_primary: bool = False
    ) -> bool:
        """
        Add or update an account
        
        Args:
            account: XiaohongshuAccount to add
            label: Optional unique label
            set_as_primary: Set as primary account
            
        Returns:
            True if account was added/updated
        """
        await self.load()
        
        # Check label uniqueness
        if label and not self._is_label_unique(label, account.user_id):
            logger.warning(f"Label '{label}' already exists")
            return False
        
        # Set label if provided
        if label:
            account.label = label
        
        # Handle primary status
        if set_as_primary or not self._accounts:
            for acc in self._accounts.values():
                acc.is_primary = False
            account.is_primary = True
        
        # Add/update account
        existing = self._accounts.get(account.user_id)
        if existing:
            if not label and existing.label:
                account.label = existing.label
            if not set_as_primary:
                account.is_primary = existing.is_primary
            logger.info(f"Updated Xiaohongshu account: {account.nickname} ({account.user_id})")
        else:
            logger.info(f"Added new Xiaohongshu account: {account.nickname} ({account.user_id})")
        
        self._accounts[account.user_id] = account
        await self.save()
        return True
    
    async def remove_account(self, user_id: str) -> bool:
        """Remove an account by user_id"""
        await self.load()
        
        if user_id not in self._accounts:
            return False
        
        removed = self._accounts.pop(user_id)
        logger.info(f"Removed Xiaohongshu account: {removed.nickname}")
        
        if removed.is_primary and self._accounts:
            first = next(iter(self._accounts.values()))
            first.is_primary = True
            logger.info(f"Set {first.nickname} as new primary account")
        
        await self.save()
        return True
    
    async def set_primary(self, user_id: str) -> bool:
        """Set an account as primary"""
        await self.load()
        
        if user_id not in self._accounts:
            return False
        
        for acc in self._accounts.values():
            acc.is_primary = (acc.user_id == user_id)
        
        await self.save()
        return True
    
    async def update_label(self, user_id: str, label: str) -> bool:
        """Update account label"""
        await self.load()
        
        if user_id not in self._accounts:
            return False
        
        if not self._is_label_unique(label, user_id):
            return False
        
        self._accounts[user_id].label = label
        await self.save()
        return True
    
    async def get_account(self, user_id: str) -> Optional[XiaohongshuAccount]:
        """Get account by user_id"""
        await self.load()
        return self._accounts.get(user_id)
    
    async def get_primary(self) -> Optional[XiaohongshuAccount]:
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
                "user_id": acc.user_id,
                "uid": acc.user_id,  # Alias for compatibility
                "nickname": acc.nickname,
                "avatar": acc.avatar,
                "label": acc.label,
                "is_primary": acc.is_primary,
                "display_name": acc.display_name,
                "updated_at": acc.updated_at,
            }
            for acc in self._accounts.values()
        ]
    
    def get_account_sync(self, user_id: str) -> Optional[XiaohongshuAccount]:
        """Sync version for non-async contexts"""
        return self._accounts.get(user_id)
    
    def list_accounts_sync(self) -> List[Dict[str, Any]]:
        """Sync version for non-async contexts"""
        return [
            {
                "user_id": acc.user_id,
                "uid": acc.user_id,
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
xiaohongshu_account_manager = XiaohongshuAccountManager()

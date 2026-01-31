"""
Database module for iDubb
Provides SQLite-based persistence for tasks, settings, and cookies
"""
from .models import Base, TaskModel, SettingsModel, StepResultModel, CookieModel, MetadataPresetModel, SubscriptionModel, ApiKeyModel
from .repository import TaskRepository, SettingsRepository, CookieRepository, MetadataPresetRepository, SubscriptionRepository, ApiKeyRepository
from .connection import get_db, init_db, close_db

__all__ = [
    "Base",
    "TaskModel",
    "SettingsModel",
    "StepResultModel",
    "CookieModel",
    "MetadataPresetModel",
    "SubscriptionModel",
    "ApiKeyModel",
    "TaskRepository",
    "SettingsRepository",
    "CookieRepository",
    "MetadataPresetRepository",
    "SubscriptionRepository",
    "ApiKeyRepository",
    "get_db",
    "init_db",
    "close_db",
]

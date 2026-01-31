"""
Terminology Management System

Provides consistent translation of domain-specific terms.
Based on VideoLingo's terminology management approach.
"""
from .manager import TerminologyManager
from .glossary import Glossary, GlossaryEntry

__all__ = [
    "TerminologyManager",
    "Glossary",
    "GlossaryEntry",
]

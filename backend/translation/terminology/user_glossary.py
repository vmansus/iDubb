"""
User Custom Glossary

Manages user-defined terminology/glossary entries.
Stored per-user and can be exported/imported.
"""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from loguru import logger


@dataclass
class GlossaryEntry:
    """A single glossary entry"""
    source: str  # Source language term
    target: str  # Target language translation
    note: str = ""  # Optional note/context
    category: str = "general"  # Category for organization
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlossaryEntry":
        return cls(
            source=data.get("source", ""),
            target=data.get("target", ""),
            note=data.get("note", ""),
            category=data.get("category", "general"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


@dataclass
class UserGlossary:
    """User's custom glossary"""
    name: str = "default"
    description: str = "User custom terminology"
    source_lang: str = "en"
    target_lang: str = "zh-CN"
    entries: List[GlossaryEntry] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "entries": [e.to_dict() for e in self.entries],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserGlossary":
        entries = [GlossaryEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            source_lang=data.get("source_lang", "en"),
            target_lang=data.get("target_lang", "zh-CN"),
            entries=entries,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )

    def get_terminology_dict(self) -> Dict[str, str]:
        """Get terminology as source -> target dict"""
        return {e.source: e.target for e in self.entries}


class UserGlossaryManager:
    """
    Manages user custom glossaries.
    
    Features:
    - CRUD operations for glossary entries
    - Import/export (JSON, CSV)
    - Persistence to file
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the glossary manager.
        
        Args:
            storage_path: Path to store glossary files
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Default to project data directory
            project_root = Path(__file__).parent.parent.parent
            self.storage_path = project_root / "data" / "glossaries"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._glossary: Optional[UserGlossary] = None

    def _get_glossary_file(self) -> Path:
        """Get path to the user glossary file"""
        return self.storage_path / "user_glossary.json"

    def load(self) -> UserGlossary:
        """Load the user glossary from file"""
        if self._glossary is not None:
            return self._glossary

        glossary_file = self._get_glossary_file()
        
        if glossary_file.exists():
            try:
                with open(glossary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._glossary = UserGlossary.from_dict(data)
                logger.info(f"Loaded user glossary with {len(self._glossary.entries)} entries")
            except Exception as e:
                logger.error(f"Failed to load user glossary: {e}")
                self._glossary = UserGlossary()
        else:
            self._glossary = UserGlossary()
        
        return self._glossary

    def save(self) -> bool:
        """Save the user glossary to file"""
        if self._glossary is None:
            return False

        try:
            self._glossary.updated_at = datetime.now().isoformat()
            glossary_file = self._get_glossary_file()
            
            with open(glossary_file, 'w', encoding='utf-8') as f:
                json.dump(self._glossary.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved user glossary with {len(self._glossary.entries)} entries")
            return True
        except Exception as e:
            logger.error(f"Failed to save user glossary: {e}")
            return False

    def get_entries(
        self, 
        category: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[GlossaryEntry]:
        """
        Get glossary entries with optional filtering.
        
        Args:
            category: Filter by category
            search: Search in source/target/note
            
        Returns:
            List of matching entries
        """
        glossary = self.load()
        entries = glossary.entries

        if category:
            entries = [e for e in entries if e.category == category]

        if search:
            search_lower = search.lower()
            entries = [
                e for e in entries
                if search_lower in e.source.lower()
                or search_lower in e.target.lower()
                or search_lower in e.note.lower()
            ]

        return entries

    def add_entry(
        self,
        source: str,
        target: str,
        note: str = "",
        category: str = "general"
    ) -> GlossaryEntry:
        """
        Add a new glossary entry.
        
        Args:
            source: Source term
            target: Target translation
            note: Optional note
            category: Category
            
        Returns:
            The created entry
        """
        glossary = self.load()
        
        # Check for duplicate
        for entry in glossary.entries:
            if entry.source.lower() == source.lower():
                # Update existing
                entry.target = target
                entry.note = note
                entry.category = category
                entry.updated_at = datetime.now().isoformat()
                self.save()
                return entry

        # Create new entry
        entry = GlossaryEntry(
            source=source,
            target=target,
            note=note,
            category=category,
        )
        glossary.entries.append(entry)
        self.save()
        
        return entry

    def update_entry(
        self,
        source: str,
        target: Optional[str] = None,
        note: Optional[str] = None,
        category: Optional[str] = None
    ) -> Optional[GlossaryEntry]:
        """
        Update an existing entry.
        
        Args:
            source: Source term (used as key)
            target: New target translation
            note: New note
            category: New category
            
        Returns:
            Updated entry or None if not found
        """
        glossary = self.load()
        
        for entry in glossary.entries:
            if entry.source.lower() == source.lower():
                if target is not None:
                    entry.target = target
                if note is not None:
                    entry.note = note
                if category is not None:
                    entry.category = category
                entry.updated_at = datetime.now().isoformat()
                self.save()
                return entry
        
        return None

    def delete_entry(self, source: str) -> bool:
        """
        Delete an entry by source term.
        
        Args:
            source: Source term to delete
            
        Returns:
            True if deleted, False if not found
        """
        glossary = self.load()
        
        for i, entry in enumerate(glossary.entries):
            if entry.source.lower() == source.lower():
                glossary.entries.pop(i)
                self.save()
                return True
        
        return False

    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        glossary = self.load()
        categories = set(e.category for e in glossary.entries)
        return sorted(categories)

    def get_terminology_dict(self) -> Dict[str, str]:
        """Get all entries as a source -> target dictionary"""
        glossary = self.load()
        return glossary.get_terminology_dict()

    def import_json(self, data: Dict[str, Any]) -> int:
        """
        Import entries from JSON data.
        
        Args:
            data: JSON data with entries list
            
        Returns:
            Number of entries imported
        """
        glossary = self.load()
        count = 0
        
        entries_data = data.get("entries", [])
        if not entries_data and isinstance(data, list):
            entries_data = data

        for entry_data in entries_data:
            if isinstance(entry_data, dict):
                source = entry_data.get("source", entry_data.get("src", ""))
                target = entry_data.get("target", entry_data.get("tgt", ""))
                if source and target:
                    self.add_entry(
                        source=source,
                        target=target,
                        note=entry_data.get("note", ""),
                        category=entry_data.get("category", "imported"),
                    )
                    count += 1

        return count

    def import_csv(self, csv_content: str) -> int:
        """
        Import entries from CSV content.
        
        Expected format: source,target,note,category
        
        Args:
            csv_content: CSV string
            
        Returns:
            Number of entries imported
        """
        import csv
        from io import StringIO
        
        count = 0
        reader = csv.reader(StringIO(csv_content))
        
        # Skip header if present
        header = next(reader, None)
        if header and header[0].lower() in ['source', 'src', '原文', '源语言']:
            pass  # It was a header, continue
        elif header:
            # First row was data, process it
            if len(header) >= 2:
                self.add_entry(
                    source=header[0],
                    target=header[1],
                    note=header[2] if len(header) > 2 else "",
                    category=header[3] if len(header) > 3 else "imported",
                )
                count += 1

        for row in reader:
            if len(row) >= 2:
                self.add_entry(
                    source=row[0],
                    target=row[1],
                    note=row[2] if len(row) > 2 else "",
                    category=row[3] if len(row) > 3 else "imported",
                )
                count += 1

        return count

    def export_json(self) -> Dict[str, Any]:
        """Export glossary as JSON"""
        glossary = self.load()
        return glossary.to_dict()

    def export_csv(self) -> str:
        """Export glossary as CSV"""
        import csv
        from io import StringIO
        
        glossary = self.load()
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['source', 'target', 'note', 'category'])
        
        # Data
        for entry in glossary.entries:
            writer.writerow([entry.source, entry.target, entry.note, entry.category])
        
        return output.getvalue()

    def clear(self) -> None:
        """Clear all entries"""
        glossary = self.load()
        glossary.entries = []
        self.save()

    def reload(self) -> UserGlossary:
        """Force reload from file"""
        self._glossary = None
        return self.load()


# Global instance
user_glossary_manager = UserGlossaryManager()

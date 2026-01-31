"""
Terminology Manager

Central manager for glossaries and terminology lookup.
Supports multiple glossaries, AI-based term extraction, and persistence.
"""
import asyncio
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
from loguru import logger

from .glossary import Glossary, GlossaryEntry, create_tech_glossary


class TerminologyManager:
    """
    Manages terminology for consistent translation.

    Features:
    - Multiple glossary support
    - Automatic term extraction from text
    - AI-powered term suggestion
    - Glossary persistence
    """

    def __init__(
        self,
        glossaries_dir: Optional[Path] = None,
        auto_load: bool = True
    ):
        """
        Initialize terminology manager.

        Args:
            glossaries_dir: Directory for storing glossaries
            auto_load: Auto-load glossaries from directory
        """
        self.glossaries: Dict[str, Glossary] = {}
        self.glossaries_dir = Path(glossaries_dir) if glossaries_dir else None

        # Load built-in glossaries
        self._load_builtin_glossaries()

        # Auto-load from directory
        if auto_load and self.glossaries_dir:
            self._load_from_directory()

    def _load_builtin_glossaries(self) -> None:
        """Load built-in glossaries"""
        tech_glossary = create_tech_glossary()
        self.glossaries["tech"] = tech_glossary
        logger.info(f"Loaded built-in tech glossary with {len(tech_glossary)} terms")

    def _load_from_directory(self) -> None:
        """Load all glossaries from directory"""
        if not self.glossaries_dir or not self.glossaries_dir.exists():
            return

        for file_path in self.glossaries_dir.glob("*.json"):
            glossary = Glossary.load(file_path)
            if glossary:
                self.glossaries[glossary.name] = glossary

    def add_glossary(self, glossary: Glossary) -> None:
        """Add a glossary to the manager"""
        self.glossaries[glossary.name] = glossary
        logger.info(f"Added glossary '{glossary.name}' with {len(glossary)} entries")

    def remove_glossary(self, name: str) -> bool:
        """Remove a glossary by name"""
        if name in self.glossaries:
            del self.glossaries[name]
            return True
        return False

    def get_glossary(self, name: str) -> Optional[Glossary]:
        """Get a glossary by name"""
        return self.glossaries.get(name)

    def list_glossaries(self) -> List[str]:
        """List all glossary names"""
        return list(self.glossaries.keys())

    def get_terminology_for_text(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-CN",
        glossary_names: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Get relevant terminology for a piece of text.

        Args:
            text: Text to find terminology for
            source_lang: Source language
            target_lang: Target language
            glossary_names: Specific glossaries to use (None = all)

        Returns:
            Dictionary of source term -> target term
        """
        terminology = {}

        glossaries_to_check = (
            [self.glossaries[n] for n in glossary_names if n in self.glossaries]
            if glossary_names
            else list(self.glossaries.values())
        )

        for glossary in glossaries_to_check:
            # Check language match
            if glossary.source_lang != source_lang or glossary.target_lang != target_lang:
                continue

            # Find matching entries
            matches = glossary.find_matches(text)
            for match in matches:
                terminology[match.source_term] = match.target_term

        return terminology

    def get_all_terms(
        self,
        source_lang: str = "en",
        target_lang: str = "zh-CN"
    ) -> Dict[str, str]:
        """
        Get all terms for a language pair.

        Args:
            source_lang: Source language
            target_lang: Target language

        Returns:
            Dictionary of all source term -> target term mappings
        """
        terminology = {}

        for glossary in self.glossaries.values():
            if glossary.source_lang == source_lang and glossary.target_lang == target_lang:
                terminology.update(glossary.to_dict_mapping())

        return terminology

    def add_term(
        self,
        source_term: str,
        target_term: str,
        glossary_name: str = "custom",
        source_lang: str = "en",
        target_lang: str = "zh-CN",
        category: Optional[str] = None
    ) -> bool:
        """
        Add a term to a glossary.

        Args:
            source_term: Source term
            target_term: Target translation
            glossary_name: Target glossary (created if doesn't exist)
            source_lang: Source language
            target_lang: Target language
            category: Optional category

        Returns:
            True if successful
        """
        # Create glossary if it doesn't exist
        if glossary_name not in self.glossaries:
            self.glossaries[glossary_name] = Glossary(
                name=glossary_name,
                description=f"Custom glossary for {source_lang} -> {target_lang}",
                source_lang=source_lang,
                target_lang=target_lang,
            )

        glossary = self.glossaries[glossary_name]
        glossary.add_entry(
            source_term=source_term,
            target_term=target_term,
            category=category
        )

        logger.info(f"Added term '{source_term}' -> '{target_term}' to '{glossary_name}'")
        return True

    def remove_term(self, source_term: str, glossary_name: str) -> bool:
        """Remove a term from a glossary"""
        if glossary_name not in self.glossaries:
            return False

        return self.glossaries[glossary_name].remove_entry(source_term)

    async def extract_terms_from_text(
        self,
        text: str,
        ai_engine: Optional[Any] = None,
        source_lang: str = "en"
    ) -> List[str]:
        """
        Extract potential terms from text that might need consistent translation.

        Args:
            text: Text to analyze
            ai_engine: Optional AI engine for smart extraction
            source_lang: Source language

        Returns:
            List of potential terms
        """
        import re

        potential_terms: Set[str] = set()

        # Pattern-based extraction
        # 1. Capitalized words/phrases (likely names, brands)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        potential_terms.update(capitalized)

        # 2. All-caps words (acronyms)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        potential_terms.update(acronyms)

        # 3. Technical terms (camelCase, snake_case)
        camel_case = re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', text)
        potential_terms.update(camel_case)

        snake_case = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text)
        potential_terms.update(snake_case)

        # 4. Words in quotes
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        potential_terms.update(quoted)

        # AI-powered extraction if available
        if ai_engine and hasattr(ai_engine, 'extract_terms'):
            try:
                ai_terms = await ai_engine.extract_terms(text, source_lang)
                potential_terms.update(ai_terms)
            except Exception as e:
                logger.warning(f"AI term extraction failed: {e}")

        # Filter out common words
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall", "can",
            "I", "you", "he", "she", "it", "we", "they", "this", "that", "these",
            "those", "what", "which", "who", "whom", "whose", "where", "when",
            "why", "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "also", "now", "here", "there",
        }

        filtered = [
            term for term in potential_terms
            if term.lower() not in common_words and len(term) > 1
        ]

        return sorted(filtered)

    async def suggest_translations(
        self,
        terms: List[str],
        ai_engine: Any,
        source_lang: str = "en",
        target_lang: str = "zh-CN"
    ) -> Dict[str, str]:
        """
        Use AI to suggest translations for terms.

        Args:
            terms: List of terms to translate
            ai_engine: AI engine for translation
            source_lang: Source language
            target_lang: Target language

        Returns:
            Dictionary of term -> suggested translation
        """
        if not terms or not ai_engine:
            return {}

        prompt = f"""Translate these terms from {source_lang} to {target_lang}.
For technical terms, use established translations. For names/brands, transliterate or keep original as appropriate.

Terms:
{chr(10).join(f'- {term}' for term in terms)}

Respond with ONLY a JSON object mapping each term to its translation:
{{"term1": "translation1", "term2": "translation2"}}"""

        try:
            if hasattr(ai_engine, 'complete'):
                response = await ai_engine.complete(prompt)
            else:
                response = await ai_engine.translate(
                    "\n".join(terms),
                    source_lang,
                    target_lang
                )

            # Try to parse as JSON
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    return json.loads(json_match.group())

        except Exception as e:
            logger.error(f"AI term suggestion failed: {e}")

        return {}

    def save_all(self) -> bool:
        """Save all glossaries to the glossaries directory"""
        if not self.glossaries_dir:
            logger.warning("No glossaries directory configured")
            return False

        self.glossaries_dir.mkdir(parents=True, exist_ok=True)

        success = True
        for name, glossary in self.glossaries.items():
            file_path = self.glossaries_dir / f"{name}.json"
            if not glossary.save(file_path):
                success = False

        return success

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded terminology"""
        total_terms = sum(len(g) for g in self.glossaries.values())

        return {
            "glossaries": len(self.glossaries),
            "total_terms": total_terms,
            "glossary_details": {
                name: {
                    "entries": len(g),
                    "source_lang": g.source_lang,
                    "target_lang": g.target_lang,
                }
                for name, g in self.glossaries.items()
            }
        }

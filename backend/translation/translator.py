"""
Translation Module supporting multiple engines:
- Google Translate (free)
- DeepL (API key required)
- OpenAI GPT (API key required)
- Anthropic Claude (API key required)
"""
import asyncio
import os
from typing import Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger

# Free translation libraries
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound

# Import config for proxy settings
from config import settings


def get_httpx_client_kwargs() -> dict:
    """Get httpx client kwargs including proxy if configured"""
    kwargs = {"timeout": 180.0}
    if settings.PROXY_URL:
        kwargs["proxy"] = settings.PROXY_URL
        logger.debug(f"Using proxy: {settings.PROXY_URL}")
    return kwargs


@dataclass
class TranslationResult:
    """Translation result"""
    success: bool
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    error: Optional[str] = None


class TranslationEngine(ABC):
    """Abstract base class for translation engines"""

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate text"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name"""
        pass


class GoogleEngine(TranslationEngine):
    """Google Translate engine (free)"""

    @property
    def name(self) -> str:
        return "google"

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        def do_translate():
            # Normalize language codes
            lang_map = {
                "zh": "zh-CN",
                "en": "en",
                "ja": "ja",
                "ko": "ko",
                "es": "es",
                "fr": "fr",
                "de": "de",
                "ru": "ru",
                "pt": "pt",
                "it": "it",
                "ar": "ar",
                "hi": "hi",
                "vi": "vi",
                "th": "th",
            }
            src = lang_map.get(source_lang, source_lang)
            tgt = lang_map.get(target_lang, target_lang)

            translator = GoogleTranslator(source=src, target=tgt)
            return translator.translate(text)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, do_translate)


class DeepLEngine(TranslationEngine):
    """DeepL translation engine"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPL_API_KEY")
        if not self.api_key:
            logger.warning("DeepL API key not configured")

    @property
    def name(self) -> str:
        return "deepl"

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        if not self.api_key:
            raise ValueError("DeepL API key not configured")

        try:
            import httpx

            # DeepL language code mapping
            lang_map = {
                "zh-CN": "ZH",
                "zh-TW": "ZH",
                "zh": "ZH",
                "en": "EN",
                "ja": "JA",
                "ko": "KO",
                "es": "ES",
                "fr": "FR",
                "de": "DE",
                "ru": "RU",
                "pt": "PT-BR",
                "it": "IT",
            }

            source = lang_map.get(source_lang, source_lang.upper())
            target = lang_map.get(target_lang, target_lang.upper())

            # Determine API endpoint (free vs pro)
            if self.api_key.endswith(":fx"):
                base_url = "https://api-free.deepl.com/v2/translate"
            else:
                base_url = "https://api.deepl.com/v2/translate"

            async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
                response = await client.post(
                    base_url,
                    data={
                        "auth_key": self.api_key,
                        "text": text,
                        "source_lang": source,
                        "target_lang": target,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()
                return result["translations"][0]["text"]

        except ImportError:
            logger.error("httpx not installed, run: pip install httpx")
            raise
        except Exception as e:
            logger.error(f"DeepL translation failed: {e}")
            raise


class OpenAIEngine(TranslationEngine):
    """OpenAI GPT translation engine with batch support"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            logger.warning("OpenAI API key not configured")

    @property
    def name(self) -> str:
        return "gpt"

    def _get_lang_name(self, lang_code: str) -> str:
        """Get language name from code"""
        lang_names = {
            "en": "English",
            "zh-CN": "Simplified Chinese",
            "zh-TW": "Traditional Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
        }
        return lang_names.get(lang_code, lang_code)

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        try:
            import httpx

            source_name = self._get_lang_name(source_lang)
            target_name = self._get_lang_name(target_lang)

            prompt = f"""Translate the following text from {source_name} to {target_name}.
Only output the translated text, nothing else. Preserve the original formatting, tone, and style.
IMPORTANT: Keep the same number of lines - each line in the input should correspond to exactly one line in the output.

Text to translate:
{text}"""

            # Estimate max_tokens based on input length
            # GPT-4 has max_tokens limit of 8192, GPT-4-turbo has higher limits
            # Use conservative limit to avoid 400 errors
            if "turbo" in self.model.lower() or "gpt-4o" in self.model.lower():
                max_output_tokens = 16000
            else:
                max_output_tokens = 4096  # Safe limit for gpt-4 base model
            estimated_tokens = min(max_output_tokens, max(1000, len(text) * 2))

            async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a professional translator. Translate accurately while preserving the original meaning, tone, and formatting. Always maintain the exact same number of lines as the input."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": estimated_tokens,
                    },
                    timeout=180.0,  # Longer timeout for large texts
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()

        except ImportError:
            logger.error("httpx not installed, run: pip install httpx")
            raise
        except Exception as e:
            logger.error(f"OpenAI translation failed: {e}")
            logger.error(f"Model: {self.model}, max_tokens: {estimated_tokens}, text_length: {len(text)}")
            raise

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """
        Translate multiple texts in a single API call.
        Much more efficient than translating one by one.
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        if not texts:
            return []

        # Single text - use simple translate
        if len(texts) == 1:
            result = await self.translate(texts[0], source_lang, target_lang)
            return [result]

        try:
            import httpx

            source_name = self._get_lang_name(source_lang)
            target_name = self._get_lang_name(target_lang)

            # Build numbered text block
            numbered_lines = []
            for i, text in enumerate(texts, 1):
                # Escape any existing brackets in text
                clean_text = text.replace('[', '【').replace(']', '】')
                numbered_lines.append(f"[{i}] {clean_text}")

            combined_text = "\n".join(numbered_lines)

            prompt = f"""Translate the following numbered lines from {source_name} to {target_name}.
IMPORTANT:
- Keep the exact same numbering format [1], [2], [3], etc.
- Each line must start with its number in brackets
- Translate ONLY the text after each number, preserve the number format exactly
- Do not add or remove any lines
- Do not merge lines together

Lines to translate:
{combined_text}"""

            async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a professional translator. Translate accurately while preserving the original meaning and tone. Always maintain the exact numbered format requested."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4000,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                result = response.json()
                response_text = result["choices"][0]["message"]["content"].strip()

            # Parse numbered response
            return self._parse_numbered_response(response_text, len(texts), texts)

        except Exception as e:
            logger.error(f"OpenAI batch translation failed: {e}, falling back to individual")
            # Fallback to individual translations
            results = []
            for text in texts:
                try:
                    translated = await self.translate(text, source_lang, target_lang)
                    results.append(translated)
                except Exception:
                    results.append(text)  # Keep original on failure
            return results

    def _parse_numbered_response(self, response: str, expected_count: int, originals: List[str]) -> List[str]:
        """Parse numbered response like [1] text [2] text ..."""
        import re

        results = [''] * expected_count

        # Try to parse [N] format
        pattern = r'\[(\d+)\]\s*([^\[]*?)(?=\[\d+\]|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for num_str, text in matches:
            try:
                num = int(num_str)
                if 1 <= num <= expected_count:
                    # Restore escaped brackets
                    clean_text = text.strip().replace('【', '[').replace('】', ']')
                    results[num - 1] = clean_text
            except ValueError:
                continue

        # Fill in any missing with originals
        for i, result in enumerate(results):
            if not result:
                logger.warning(f"Missing translation for line {i+1}, using original")
                results[i] = originals[i]

        return results


class DeepSeekEngine(TranslationEngine):
    """DeepSeek translation engine (OpenAI-compatible API)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        # DeepSeek API endpoint (no /v1/ prefix)
        # See: https://api-docs.deepseek.com/api/create-chat-completion
        self.base_url = "https://api.deepseek.com/chat/completions"
        if not self.api_key:
            logger.warning("DeepSeek API key not configured")

    @property
    def name(self) -> str:
        return "deepseek"

    def _get_lang_name(self, lang_code: str) -> str:
        """Get language name from code"""
        lang_names = {
            "en": "English",
            "zh-CN": "Simplified Chinese",
            "zh-TW": "Traditional Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
        }
        return lang_names.get(lang_code, lang_code)

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")

        try:
            import httpx

            source_name = self._get_lang_name(source_lang)
            target_name = self._get_lang_name(target_lang)

            prompt = f"""Translate the following text from {source_name} to {target_name}.
Only output the translated text, nothing else. Preserve the original formatting, tone, and style.
IMPORTANT: Keep the same number of lines - each line in the input should correspond to exactly one line in the output.

Text to translate:
{text}"""

            # DeepSeek has a max_tokens limit of 8192 for output
            # Estimate based on input but cap at 8000
            estimated_tokens = min(8000, max(2000, len(text) * 2))

            async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a professional translator. Translate accurately while preserving the original meaning, tone, and formatting. Always maintain the exact same number of lines as the input."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": estimated_tokens,
                    },
                    timeout=180.0,
                )

                # Log response for debugging
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"DeepSeek API error {response.status_code}: {error_text}")
                    raise Exception(f"DeepSeek API error {response.status_code}: {error_text}")

                result = response.json()
                return result["choices"][0]["message"]["content"].strip()

        except ImportError:
            logger.error("httpx not installed, run: pip install httpx")
            raise
        except Exception as e:
            logger.error(f"DeepSeek translation failed: {e}")
            raise


class ClaudeEngine(TranslationEngine):
    """Anthropic Claude translation engine with batch support"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            logger.warning("Anthropic API key not configured")

    @property
    def name(self) -> str:
        return "claude"

    def _get_lang_name(self, lang_code: str) -> str:
        """Get language name from code"""
        lang_names = {
            "en": "English",
            "zh-CN": "Simplified Chinese",
            "zh-TW": "Traditional Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
        }
        return lang_names.get(lang_code, lang_code)

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        try:
            import httpx

            source_name = self._get_lang_name(source_lang)
            target_name = self._get_lang_name(target_lang)

            prompt = f"""Translate the following text from {source_name} to {target_name}.
Only output the translated text, nothing else. Preserve the original formatting, tone, and style.
IMPORTANT: Keep the same number of lines - each line in the input should correspond to exactly one line in the output.

Text to translate:
{text}"""

            # Estimate max_tokens based on input length
            estimated_tokens = min(16000, max(2000, len(text) * 2))

            async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": estimated_tokens,
                        "system": "You are a professional translator. Translate accurately while preserving the original meaning, tone, and formatting. Always maintain the exact same number of lines as the input.",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                    },
                    timeout=180.0,  # Longer timeout for large texts
                )
                response.raise_for_status()
                result = response.json()
                return result["content"][0]["text"].strip()

        except ImportError:
            logger.error("httpx not installed, run: pip install httpx")
            raise
        except Exception as e:
            logger.error(f"Claude translation failed: {e}")
            raise

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """
        Translate multiple texts in a single API call.
        Much more efficient than translating one by one.
        """
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        if not texts:
            return []

        # Single text - use simple translate
        if len(texts) == 1:
            result = await self.translate(texts[0], source_lang, target_lang)
            return [result]

        try:
            import httpx

            source_name = self._get_lang_name(source_lang)
            target_name = self._get_lang_name(target_lang)

            # Build numbered text block
            numbered_lines = []
            for i, text in enumerate(texts, 1):
                # Escape any existing brackets in text
                clean_text = text.replace('[', '【').replace(']', '】')
                numbered_lines.append(f"[{i}] {clean_text}")

            combined_text = "\n".join(numbered_lines)

            prompt = f"""Translate the following numbered lines from {source_name} to {target_name}.
IMPORTANT:
- Keep the exact same numbering format [1], [2], [3], etc.
- Each line must start with its number in brackets
- Translate ONLY the text after each number, preserve the number format exactly
- Do not add or remove any lines
- Do not merge lines together

Lines to translate:
{combined_text}"""

            async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 4000,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                result = response.json()
                response_text = result["content"][0]["text"].strip()

            # Parse numbered response
            return self._parse_numbered_response(response_text, len(texts), texts)

        except Exception as e:
            logger.error(f"Claude batch translation failed: {e}, falling back to individual")
            # Fallback to individual translations
            results = []
            for text in texts:
                try:
                    translated = await self.translate(text, source_lang, target_lang)
                    results.append(translated)
                except Exception:
                    results.append(text)  # Keep original on failure
            return results

    def _parse_numbered_response(self, response: str, expected_count: int, originals: List[str]) -> List[str]:
        """Parse numbered response like [1] text [2] text ..."""
        import re

        results = [''] * expected_count

        # Try to parse [N] format
        pattern = r'\[(\d+)\]\s*([^\[]*?)(?=\[\d+\]|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for num_str, text in matches:
            try:
                num = int(num_str)
                if 1 <= num <= expected_count:
                    # Restore escaped brackets
                    clean_text = text.strip().replace('【', '[').replace('】', ']')
                    results[num - 1] = clean_text
            except ValueError:
                continue

        # Fill in any missing with originals
        for i, result in enumerate(results):
            if not result:
                logger.warning(f"Missing translation for line {i+1}, using original")
                results[i] = originals[i]

        return results


class Translator:
    """Multi-service translator supporting multiple engines"""

    SUPPORTED_ENGINES = {
        "google": {
            "name": "Google Translate",
            "description": "Free Google Translate API via deep-translator",
            "requires_api_key": False,
            "free": True,
        },
        "deepl": {
            "name": "DeepL",
            "description": "High-quality neural machine translation",
            "requires_api_key": True,
            "free": False,
        },
        "gpt": {
            "name": "OpenAI GPT",
            "description": "AI-powered translation using GPT models",
            "requires_api_key": True,
            "free": False,
        },
        "claude": {
            "name": "Anthropic Claude",
            "description": "AI-powered translation using Claude models",
            "requires_api_key": True,
            "free": False,
        },
        "deepseek": {
            "name": "DeepSeek",
            "description": "Cost-effective AI translation using DeepSeek models",
            "requires_api_key": True,
            "free": False,
        },
    }

    def __init__(
        self,
        engine: str = "google",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize translator

        Args:
            engine: Translation engine to use (google, deepl, gpt, claude)
            api_key: API key for paid engines
            model: Model name for AI engines (gpt-4, claude-3-sonnet, etc.)
        """
        self.engine_name = engine
        self._engine = self._create_engine(engine, api_key, model)
        logger.info(f"Initialized Translator with engine: {engine}")

    def _create_engine(
        self,
        engine: str,
        api_key: Optional[str],
        model: Optional[str]
    ) -> TranslationEngine:
        """Create the appropriate translation engine"""
        if engine == "google":
            return GoogleEngine()
        elif engine == "deepl":
            return DeepLEngine(api_key)
        elif engine == "gpt":
            return OpenAIEngine(api_key, model or "gpt-4")
        elif engine == "claude":
            return ClaudeEngine(api_key, model or "claude-3-sonnet-20240229")
        elif engine == "deepseek":
            return DeepSeekEngine(api_key, model or "deepseek-chat")
        else:
            logger.warning(f"Unknown engine {engine}, falling back to Google")
            return GoogleEngine()

    async def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-CN"
    ) -> TranslationResult:
        """
        Translate text

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            TranslationResult
        """
        try:
            if not text or not text.strip():
                return TranslationResult(
                    success=True,
                    original_text=text,
                    translated_text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )

            translated = await self._engine.translate(text, source_lang, target_lang)

            return TranslationResult(
                success=True,
                original_text=text,
                translated_text=translated or text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        except TranslationNotFound as e:
            logger.warning(f"Translation not found: {e}")
            return TranslationResult(
                success=False,
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Translation failed ({self.engine_name}): {e}")
            return TranslationResult(
                success=False,
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                error=str(e)
            )

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "en",
        target_lang: str = "zh-CN",
        batch_size: int = 20
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batches

        For GPT/Claude: Uses single API call per batch (much more efficient)
        For Google/DeepL: Uses parallel individual calls

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Number of texts per batch

        Returns:
            List of TranslationResults
        """
        if not texts:
            return []

        results = []

        # GPT and Claude support efficient batch translation in single API call
        if self.engine_name in ["gpt", "claude"] and hasattr(self._engine, 'translate_batch'):
            # Use larger batches for AI engines since they handle it in one call
            batch_size = min(batch_size, 25)  # 25 lines per API call
            delay = 0.5  # Small delay between batches

            logger.info(f"Using {self.engine_name} batch translation: {len(texts)} texts in batches of {batch_size}")

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size

                try:
                    logger.info(f"Translating batch {batch_num}/{total_batches} ({len(batch)} texts)")
                    translated_texts = await self._engine.translate_batch(batch, source_lang, target_lang)

                    for original, translated in zip(batch, translated_texts):
                        results.append(TranslationResult(
                            success=True,
                            original_text=original,
                            translated_text=translated,
                            source_lang=source_lang,
                            target_lang=target_lang,
                        ))

                except Exception as e:
                    logger.error(f"Batch translation failed: {e}")
                    # Fallback: mark all as failed with original text
                    for text in batch:
                        results.append(TranslationResult(
                            success=False,
                            original_text=text,
                            translated_text=text,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            error=str(e)
                        ))

                # Delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(delay)

        else:
            # Google/DeepL: use parallel individual calls
            if self.engine_name == "deepl":
                batch_size = min(batch_size, 10)
                delay = 0.3
            else:
                batch_size = min(batch_size, 10)
                delay = 0.5

            logger.info(f"Using {self.engine_name} parallel translation: {len(texts)} texts")

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = await asyncio.gather(*[
                    self.translate(text, source_lang, target_lang)
                    for text in batch
                ])
                results.extend(batch_results)

                # Delay between batches to avoid rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(delay)

        return results

    # Maximum lines per API call for AI engines (to avoid token limits)
    MAX_LINES_PER_REQUEST = 150  # Increased from 100 for fewer API calls

    async def _translate_text_chunk(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_retries: int = 3
    ) -> List[str]:
        """
        Translate a chunk of texts in one API call.
        Returns list of translated texts, same length as input.
        Includes retry logic for transient network errors.
        """
        full_text = "\n".join(texts)

        last_error = None
        for attempt in range(max_retries):
            result = await self.translate(full_text, source_lang, target_lang)

            if result.success:
                translated_lines = result.translated_text.split("\n")
                # Handle line count mismatch
                if len(translated_lines) != len(texts):
                    logger.warning(f"Chunk line count mismatch: {len(translated_lines)} vs {len(texts)}")
                    # Try to pad or truncate
                    if len(translated_lines) < len(texts):
                        # Pad with original texts
                        translated_lines.extend(texts[len(translated_lines):])
                    else:
                        # Truncate
                        translated_lines = translated_lines[:len(texts)]
                return [line.strip() or orig for line, orig in zip(translated_lines, texts)]
            else:
                last_error = result.error
                # Check if error is retryable (network/timeout issues)
                retryable_errors = [
                    "disconnected", "timeout", "connection", "reset",
                    "temporarily unavailable", "rate limit", "429", "503", "504"
                ]
                is_retryable = any(err in str(last_error).lower() for err in retryable_errors)

                if is_retryable and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    logger.warning(f"Chunk translation failed (attempt {attempt + 1}/{max_retries}): {last_error}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    break

        logger.warning(f"Chunk translation failed after {max_retries} attempts: {last_error}")
        return texts  # Return originals on failure

    async def translate_segments(
        self,
        segments: List,  # List[TranscriptSegment]
        source_lang: str = "en",
        target_lang: str = "zh-CN",
        cancel_check: Optional[callable] = None
    ) -> List:
        """
        Translate transcription segments

        For GPT/Claude/DeepSeek: Translates text in chunks if too long, otherwise in ONE request.
        For Google/DeepL: Falls back to batch translation.

        Args:
            segments: List of TranscriptSegment objects
            source_lang: Source language
            target_lang: Target language
            cancel_check: Optional callable that returns True if cancellation requested

        Returns:
            List of translated segments with same timing
        """
        from transcription.whisper_transcriber import TranscriptSegment

        if not segments:
            return []

        # Check for cancellation before starting
        if cancel_check and cancel_check():
            logger.info("Translation cancelled before starting")
            raise Exception("用户手动停止")

        texts = [seg.text for seg in segments]

        # For AI engines (GPT/Claude/DeepSeek), use smart chunking
        if self.engine_name in ["gpt", "claude", "deepseek"]:
            total_lines = len(texts)

            # Check if we need to split into multiple requests
            if total_lines <= self.MAX_LINES_PER_REQUEST:
                # Single request - translate all at once
                logger.info(f"Using {self.engine_name} full-text translation for {total_lines} segments (1 request)")
                translated_texts = await self._translate_text_chunk(texts, source_lang, target_lang)
            else:
                # Multiple requests - split into chunks and translate in parallel
                num_chunks = (total_lines + self.MAX_LINES_PER_REQUEST - 1) // self.MAX_LINES_PER_REQUEST
                parallel_limit = 3  # Process 3 chunks in parallel
                logger.info(f"Using {self.engine_name} parallel translation for {total_lines} segments ({num_chunks} chunks, {parallel_limit} parallel)")

                # Prepare all chunks
                chunks = []
                for i in range(0, total_lines, self.MAX_LINES_PER_REQUEST):
                    chunk = texts[i:i + self.MAX_LINES_PER_REQUEST]
                    chunks.append((i // self.MAX_LINES_PER_REQUEST + 1, chunk))

                # Translate chunks in parallel batches
                translated_texts = [None] * total_lines
                
                for batch_start in range(0, len(chunks), parallel_limit):
                    # Check for cancellation before each parallel batch
                    if cancel_check and cancel_check():
                        logger.info(f"Translation cancelled at batch starting chunk {batch_start + 1}")
                        raise Exception("用户手动停止")

                    batch_chunks = chunks[batch_start:batch_start + parallel_limit]
                    batch_end = min(batch_start + parallel_limit, len(chunks))
                    logger.info(f"Translating chunks {batch_start + 1}-{batch_end}/{num_chunks} in parallel")

                    async def translate_chunk(chunk_info):
                        chunk_num, chunk_texts = chunk_info
                        try:
                            return await self._translate_text_chunk(chunk_texts, source_lang, target_lang)
                        except Exception as e:
                            logger.error(f"Chunk {chunk_num} translation failed: {e}, using originals")
                            return chunk_texts

                    # Run parallel translations
                    results = await asyncio.gather(*[translate_chunk(c) for c in batch_chunks])

                    # Merge results in order
                    for (chunk_num, chunk_texts), result in zip(batch_chunks, results):
                        start_idx = (chunk_num - 1) * self.MAX_LINES_PER_REQUEST
                        for j, text in enumerate(result):
                            if start_idx + j < total_lines:
                                translated_texts[start_idx + j] = text

                    # Small delay between parallel batches to avoid rate limits
                    if batch_start + parallel_limit < len(chunks):
                        await asyncio.sleep(0.3)

            # Create translated segments
            translated_segments = []
            for seg, translated_text in zip(segments, translated_texts):
                translated_segments.append(TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=translated_text or seg.text
                ))
            return translated_segments

        # Fallback: batch translation (for Google/DeepL)
        translations = await self.translate_batch(texts, source_lang, target_lang)

        translated_segments = []
        for seg, trans in zip(segments, translations):
            translated_segments.append(TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=trans.translated_text if trans.success else seg.text
            ))

        return translated_segments

    @classmethod
    def get_available_engines(cls) -> List[dict]:
        """Get list of available translation engines"""
        return [
            {"id": key, **value}
            for key, value in cls.SUPPORTED_ENGINES.items()
        ]

    @staticmethod
    def get_supported_languages() -> dict:
        """Get supported language codes"""
        return {
            "en": "English",
            "zh-CN": "Chinese (Simplified)",
            "zh-TW": "Chinese (Traditional)",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
            "pt": "Portuguese",
            "it": "Italian",
            "ar": "Arabic",
            "hi": "Hindi",
            "vi": "Vietnamese",
            "th": "Thai",
        }

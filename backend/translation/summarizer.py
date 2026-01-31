"""
Video Content Summarizer

Pre-translation step that:
1. Summarizes video content to understand context
2. Extracts key terminology for consistent translation
3. Identifies style/tone for appropriate translation

This reduces token usage by computing context ONCE and reusing it.
Based on VideoLingo's approach.
"""
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class VideoContext:
    """Pre-computed context for translation"""
    summary: str = ""
    terminology: Dict[str, str] = field(default_factory=dict)
    style_notes: str = ""
    main_topics: List[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Convert to prompt-friendly format"""
        parts = []

        if self.summary:
            parts.append(f"Video Summary: {self.summary}")

        if self.terminology:
            terms = ", ".join(f'"{k}"→"{v}"' for k, v in self.terminology.items())
            parts.append(f"Key Terms: {terms}")

        if self.style_notes:
            parts.append(f"Style: {self.style_notes}")

        return "\n".join(parts)


class VideoSummarizer:
    """
    Summarizes video content before translation.

    This is a key optimization - we call the LLM once to understand
    the video, then reuse this context for all translation chunks.
    """

    # API endpoints for different engines
    API_ENDPOINTS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "gpt": "https://api.openai.com/v1/chat/completions",
        "deepseek": "https://api.deepseek.com/chat/completions",
        "claude": "https://api.anthropic.com/v1/messages",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        engine: str = "gpt",
        max_summary_chars: int = 500
    ):
        self.api_key = api_key
        self.model = model
        self.engine = engine.lower()
        self.max_summary_chars = max_summary_chars

    async def summarize(
        self,
        segments: List[Dict[str, Any]],
        source_lang: str = "en",
        target_lang: str = "zh-CN",
        custom_terms: Optional[Dict[str, str]] = None
    ) -> VideoContext:
        """
        Summarize video content and extract terminology.

        Args:
            segments: List of subtitle segments with 'text' field
            source_lang: Source language
            target_lang: Target language
            custom_terms: Pre-defined terminology to include

        Returns:
            VideoContext with summary, terminology, and style notes
        """
        # Combine all text (truncate if too long)
        all_text = "\n".join(seg.get("text", "") for seg in segments)

        # Truncate to reasonable length for summarization
        if len(all_text) > 3000:
            all_text = all_text[:3000] + "..."

        # Build prompt for summarization + terminology extraction
        prompt = f"""## Role
You are a video translation expert and terminology consultant, specializing in {source_lang} comprehension and {target_lang} expression optimization.

## Task
For the provided {source_lang} video transcript:
1. Summarize the main topic in two sentences
2. Extract professional terms/names with {target_lang} translations
3. Identify the overall style/tone of the content

## Steps
1. **Topic Summary**:
   - Quick scan for general understanding
   - Write two sentences: first for main topic, second for key point

2. **Term Extraction**:
   - Mark professional terms, technical jargon, and names
   - Provide {target_lang} translation or keep original if commonly used
   - Focus on terms that are: technical, names/brands, slang, or domain-specific
   - Extract no more than 15 key terms

3. **Style Analysis**:
   - Identify the tone: casual, formal, educational, entertainment, technical, etc.
   - This will guide translation style choices

## INPUT
<transcript lang="{source_lang}">
{all_text}
</transcript>

## Output (JSON only)
```json
{{
    "summary": "Two-sentence video summary describing the main topic and key point",
    "terminology": {{
        "{source_lang}_term": "{target_lang}_translation or original",
        "another_term": "translation"
    }},
    "style": "casual/formal/educational/technical/entertainment",
    "topics": ["main topic", "subtopic 1", "subtopic 2"]
}}
```

## Example
```json
{{
    "summary": "This video introduces AI applications in the medical field. It focuses on breakthroughs in medical imaging diagnosis and drug development.",
    "terminology": {{
        "Machine Learning": "机器学习",
        "CNN": "CNN",
        "neural network": "神经网络"
    }},
    "style": "educational",
    "topics": ["artificial intelligence", "medical imaging", "drug development"]
}}
```

Note: Start your answer with ```json and end with ```, do not add any other text."""

        try:
            result = await self._call_llm(prompt)
            context = self._parse_response(result)

            # Merge custom terms
            if custom_terms:
                context.terminology.update(custom_terms)

            logger.info(
                f"Video summarized: {len(context.summary)} chars, "
                f"{len(context.terminology)} terms, style={context.style_notes}"
            )
            return context

        except Exception as e:
            logger.warning(f"Summarization failed: {e}, using fallback")
            return VideoContext(
                summary="Video content",
                terminology=custom_terms or {},
                style_notes="general"
            )

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for summarization"""
        import httpx
        from config import settings

        # Build client kwargs with proxy if configured
        client_kwargs = {"timeout": 60.0}
        if settings.PROXY_URL:
            client_kwargs["proxy"] = settings.PROXY_URL

        # Get API endpoint for the engine
        api_url = self.API_ENDPOINTS.get(self.engine, self.API_ENDPOINTS["gpt"])

        async with httpx.AsyncClient(**client_kwargs) as client:
            if self.engine == "claude":
                # Anthropic Claude API has different format
                response = await client.post(
                    api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 1000,
                        "messages": [{"role": "user", "content": prompt}],
                        "system": "You are a video content analyzer. Extract key information for translation preparation."
                    },
                )
                response.raise_for_status()
                return response.json()["content"][0]["text"]
            else:
                # OpenAI-compatible API (GPT, DeepSeek)
                response = await client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a video content analyzer. Extract key information for translation preparation."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1000,
                    },
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

    def _parse_response(self, response: str) -> VideoContext:
        """Parse LLM response into VideoContext"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return VideoContext(
                    summary=data.get("summary", ""),
                    terminology=data.get("terminology", {}),
                    style_notes=data.get("style", ""),
                    main_topics=data.get("topics", [])
                )
        except json.JSONDecodeError:
            pass

        # Fallback: use response as summary
        return VideoContext(summary=response[:500])


class OptimizedTranslator:
    """
    Token-optimized translator using VideoLingo's approach.

    Key optimizations:
    1. Pre-compute context once via summarization
    2. Small chunks (5-10 lines) with sliding context window
    3. Optional reflection step (can be skipped for speed)
    4. Shared terminology across all chunks
    """

    # API endpoints for different engines
    API_ENDPOINTS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "gpt": "https://api.openai.com/v1/chat/completions",
        "deepseek": "https://api.deepseek.com/chat/completions",
        "claude": "https://api.anthropic.com/v1/messages",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        engine: str = "gpt",
        chunk_size: int = 8,
        context_before: int = 3,
        context_after: int = 2,
        enable_reflection: bool = True
    ):
        self.api_key = api_key
        self.model = model
        self.engine = engine.lower()
        self.chunk_size = chunk_size
        self.context_before = context_before
        self.context_after = context_after
        self.enable_reflection = enable_reflection

        self.summarizer = VideoSummarizer(api_key, model, engine)
        self._video_context: Optional[VideoContext] = None

    async def translate_segments(
        self,
        segments: List[Dict[str, Any]],
        source_lang: str = "en",
        target_lang: str = "zh-CN",
        custom_terms: Optional[Dict[str, str]] = None,
        progress_callback: Optional[callable] = None,
        fast_mode: bool = False,
        cancel_check: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Translate segments with optimized token usage.

        Args:
            segments: List of segments with 'text' field
            source_lang: Source language
            target_lang: Target language
            custom_terms: Pre-defined terminology
            progress_callback: Optional callback(current, total)
            fast_mode: Skip reflection step (50% fewer tokens)
            cancel_check: Optional callable that returns True if cancellation requested

        Returns:
            Translated segments
        """
        if not segments:
            return []

        # Check for cancellation before starting
        if cancel_check and cancel_check():
            logger.info("Translation cancelled before starting")
            raise Exception("用户手动停止")

        # Normalize segments to dicts (handle both dict and dataclass inputs)
        normalized_segments = []
        for seg in segments:
            if isinstance(seg, dict):
                normalized_segments.append(seg)
            elif hasattr(seg, '__dict__'):
                # Convert dataclass to dict
                normalized_segments.append({
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "text": getattr(seg, "text", "") or getattr(seg, "full_text", ""),
                })
            else:
                normalized_segments.append({"text": str(seg)})
        segments = normalized_segments

        # Step 1: Summarize content (ONE API call)
        logger.info("Step 1: Summarizing video content...")
        self._video_context = await self.summarizer.summarize(
            segments, source_lang, target_lang, custom_terms
        )

        # Step 2: Chunk segments
        texts = [seg.get("text", "") for seg in segments]
        chunks = self._create_chunks(texts)
        logger.info(f"Step 2: Created {len(chunks)} chunks from {len(texts)} segments")

        # Step 3: Translate each chunk
        translated_texts = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            # Check for cancellation before each chunk
            if cancel_check and cancel_check():
                logger.info(f"Translation cancelled at chunk {i+1}/{total_chunks}")
                raise Exception("用户手动停止")

            logger.info(f"Translating chunk {i+1}/{total_chunks} ({len(chunk['lines'])} lines)")

            # Get context lines
            prev_context = chunk.get("prev_context", [])
            next_context = chunk.get("next_context", [])

            # Translate chunk
            expected_count = len(chunk["lines"])
            if fast_mode or not self.enable_reflection:
                # Fast mode: single translation pass
                translations = await self._translate_chunk_fast(
                    chunk["lines"], prev_context, next_context,
                    source_lang, target_lang
                )
            else:
                # Quality mode: faithfulness → expressiveness
                translations = await self._translate_chunk_quality(
                    chunk["lines"], prev_context, next_context,
                    source_lang, target_lang
                )

            # Ensure we have exactly the expected number of translations
            if len(translations) != expected_count:
                logger.warning(
                    f"Chunk {i+1} translation count mismatch: expected {expected_count}, got {len(translations)}"
                )
                # Pad or trim to match expected count
                if len(translations) < expected_count:
                    # Use original text for missing translations
                    for j in range(len(translations), expected_count):
                        original_text = chunk["lines"][j] if j < len(chunk["lines"]) else ""
                        translations.append(original_text)
                        logger.warning(f"  Using original for line {j+1}: {original_text[:50]}...")
                else:
                    translations = translations[:expected_count]

            translated_texts.extend(translations)

            if progress_callback:
                progress_callback(i + 1, total_chunks)

        # Build result segments
        # Check for mismatch and log warning
        if len(translated_texts) != len(segments):
            logger.warning(
                f"Translation count mismatch! Expected {len(segments)}, got {len(translated_texts)}. "
                f"Missing {len(segments) - len(translated_texts)} translations will use original text."
            )

        result = []
        for i, seg in enumerate(segments):
            if i < len(translated_texts):
                translated_text = translated_texts[i]
            else:
                translated_text = seg.get("text", "")
                logger.warning(f"Segment {i+1} missing translation, using original: {translated_text[:50]}...")

            result.append({
                **seg,
                "text": translated_text,
                "original_text": seg.get("text", "")
            })

        return result

    def _create_chunks(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Create chunks with sliding context window"""
        chunks = []

        for i in range(0, len(texts), self.chunk_size):
            chunk_end = min(i + self.chunk_size, len(texts))
            chunk_lines = texts[i:chunk_end]

            # Get context before
            prev_start = max(0, i - self.context_before)
            prev_context = texts[prev_start:i] if i > 0 else []

            # Get context after
            next_end = min(len(texts), chunk_end + self.context_after)
            next_context = texts[chunk_end:next_end] if chunk_end < len(texts) else []

            chunks.append({
                "start_idx": i,
                "end_idx": chunk_end,
                "lines": chunk_lines,
                "prev_context": prev_context,
                "next_context": next_context
            })

        return chunks

    async def _translate_chunk_fast(
        self,
        lines: List[str],
        prev_context: List[str],
        next_context: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """Fast translation - single pass"""
        prompt = self._build_translation_prompt(
            lines, prev_context, next_context,
            source_lang, target_lang,
            mode="fast"
        )

        result = await self._call_llm(prompt)
        return self._parse_translations(result, len(lines))

    async def _translate_chunk_quality(
        self,
        lines: List[str],
        prev_context: List[str],
        next_context: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """Quality translation - faithfulness then expressiveness"""

        # Step 1: Faithful translation
        faithful_prompt = self._build_translation_prompt(
            lines, prev_context, next_context,
            source_lang, target_lang,
            mode="faithful"
        )
        faithful_result = await self._call_llm(faithful_prompt)
        faithful_translations = self._parse_translations(faithful_result, len(lines))

        # Step 2: Expressive refinement
        expressive_prompt = self._build_refinement_prompt(
            lines, faithful_translations, target_lang
        )
        expressive_result = await self._call_llm(expressive_prompt)

        return self._parse_translations(expressive_result, len(lines))

    def _build_translation_prompt(
        self,
        lines: List[str],
        prev_context: List[str],
        next_context: List[str],
        source_lang: str,
        target_lang: str,
        mode: str = "fast"
    ) -> str:
        """Build translation prompt with shared context"""

        # Include video context (computed once, reused for all chunks)
        context_str = self._video_context.to_prompt_context() if self._video_context else ""

        # Build context sections
        parts = []

        if context_str:
            parts.append(f"=== Video Context ===\n{context_str}")

        if prev_context:
            parts.append(f"=== Previous Lines (for context) ===\n" + "\n".join(prev_context))

        # Lines to translate
        numbered_lines = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
        parts.append(f"=== Translate These Lines ({source_lang} → {target_lang}) ===\n{numbered_lines}")

        if next_context:
            parts.append(f"=== Next Lines (for context) ===\n" + "\n".join(next_context))

        # Instructions based on mode
        if mode == "faithful":
            instruction = """Translate faithfully, preserving exact meaning.
Output format - JSON with line numbers:
{"1": "translation", "2": "translation", ...}"""
        else:
            instruction = """Translate naturally for the target audience.
Keep same number of lines. Be concise for subtitles.
Output format - JSON with line numbers:
{"1": "translation", "2": "translation", ...}"""

        parts.append(f"=== Instructions ===\n{instruction}")

        return "\n\n".join(parts)

    def _build_refinement_prompt(
        self,
        original_lines: List[str],
        translations: List[str],
        target_lang: str
    ) -> str:
        """Build refinement prompt for expressiveness"""

        pairs = "\n".join(
            f"{i+1}. {orig} → {trans}"
            for i, (orig, trans) in enumerate(zip(original_lines, translations))
        )

        return f"""Refine these translations for natural {target_lang} expression.
Keep meaning accurate but improve fluency and naturalness.
Subtitles should be concise.

Current translations:
{pairs}

Output refined translations as JSON:
{{"1": "refined", "2": "refined", ...}}"""

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for translation"""
        import httpx
        from config import settings

        client_kwargs = {"timeout": 120.0}
        if settings.PROXY_URL:
            client_kwargs["proxy"] = settings.PROXY_URL

        # Get API endpoint for the engine
        api_url = self.API_ENDPOINTS.get(self.engine, self.API_ENDPOINTS["gpt"])

        async with httpx.AsyncClient(**client_kwargs) as client:
            if self.engine == "claude":
                # Anthropic Claude API has different format
                response = await client.post(
                    api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 2000,
                        "messages": [{"role": "user", "content": prompt}],
                        "system": "You are a professional subtitle translator. Output only JSON."
                    },
                )
                response.raise_for_status()
                return response.json()["content"][0]["text"]
            else:
                # OpenAI-compatible API (GPT, DeepSeek)
                response = await client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a professional subtitle translator. Output only JSON."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 2000,
                    },
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

    def _parse_translations(self, response: str, expected_count: int) -> List[str]:
        """Parse JSON translation response"""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                # Extract translations in order
                translations = []
                missing_keys = []
                for i in range(1, expected_count + 1):
                    key = str(i)
                    if key in data:
                        translations.append(data[key])
                    elif i in data:  # Try integer key
                        translations.append(data[i])
                    else:
                        missing_keys.append(i)
                        translations.append(f"[翻译缺失 {i}]")

                if missing_keys:
                    logger.warning(f"Missing translation keys: {missing_keys}")

                return translations
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse translation JSON: {e}")
            logger.debug(f"Response was: {response[:500]}...")

        # Fallback: split by newlines
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        if len(lines) < expected_count:
            logger.warning(f"Fallback parsing: got {len(lines)} lines, expected {expected_count}")
            lines.extend(["[翻译缺失]"] * (expected_count - len(lines)))
        return lines[:expected_count]

    def get_stats(self) -> Dict[str, Any]:
        """Get translation statistics"""
        return {
            "model": self.model,
            "chunk_size": self.chunk_size,
            "reflection_enabled": self.enable_reflection,
            "context_window": f"{self.context_before} before, {self.context_after} after",
            "video_context": {
                "summary_length": len(self._video_context.summary) if self._video_context else 0,
                "terminology_count": len(self._video_context.terminology) if self._video_context else 0,
            }
        }

"""
AI-powered Video Metadata Generator

Generates:
- Title: Translated from source title or AI-optimized
- Description: AI summary of video content + source URL + custom signature
- Keywords: Extracted from video transcript

Supports multiple AI engines: OpenAI GPT, Claude, DeepSeek
"""
import asyncio
import os
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from loguru import logger

from config import settings


def get_httpx_client_kwargs() -> dict:
    """Get httpx client kwargs including proxy if configured"""
    kwargs = {"timeout": 60.0}
    if settings.PROXY_URL:
        kwargs["proxy"] = settings.PROXY_URL
    return kwargs


@dataclass
class VideoMetadataResult:
    """Result of metadata generation"""
    success: bool
    title: str = ""
    title_translated: str = ""  # Title translated to target language
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "title": self.title,
            "title_translated": self.title_translated,
            "description": self.description,
            "keywords": self.keywords,
            "error": self.error,
        }


class MetadataGenerator:
    """
    AI-powered video metadata generator

    Supports:
    - OpenAI GPT (gpt-4, gpt-4-turbo)
    - Anthropic Claude (claude-3-sonnet, claude-3-opus)
    - DeepSeek (deepseek-chat)
    """

    def __init__(
        self,
        engine: str = "gpt",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize metadata generator

        Args:
            engine: AI engine to use (gpt, claude, deepseek)
            api_key: API key for the engine
            model: Model name (optional, uses default for engine)
        """
        self.engine = engine.lower()
        self.api_key = api_key

        # Set default models
        default_models = {
            "gpt": "gpt-4",
            "claude": "claude-3-sonnet-20240229",
            "deepseek": "deepseek-chat",
        }
        self.model = model or default_models.get(self.engine, "gpt-4")

        if not self.api_key:
            logger.warning(f"No API key provided for {engine} metadata generator")

    async def generate(
        self,
        original_title: str,
        transcript: str,
        source_url: str = "",
        source_language: str = "en",
        target_language: str = "zh-CN",
        title_prefix: str = "",
        custom_signature: str = "",
        max_keywords: int = 10,
        platform: str = "generic"
    ) -> VideoMetadataResult:
        """
        Generate video metadata using AI

        Args:
            original_title: Original video title
            transcript: Video transcript text
            source_url: Original video URL
            source_language: Source language code
            target_language: Target language code
            title_prefix: Prefix to add before title (e.g., "[中字]")
            custom_signature: Custom signature to append to description
            max_keywords: Maximum number of keywords to generate

        Returns:
            VideoMetadataResult with generated metadata
        """
        if not self.api_key:
            return VideoMetadataResult(
                success=False,
                error=f"API key not configured for {self.engine}"
            )

        try:
            # Get language names for prompts
            lang_names = {
                "en": "English",
                "zh-CN": "Simplified Chinese",
                "zh-TW": "Traditional Chinese",
                "ja": "Japanese",
                "ko": "Korean",
            }
            target_lang_name = lang_names.get(target_language, target_language)
            source_lang_name = lang_names.get(source_language, source_language)

            # Truncate transcript if too long (keep first ~4000 chars for context)
            max_transcript_length = 4000
            truncated_transcript = transcript
            if len(transcript) > max_transcript_length:
                truncated_transcript = transcript[:max_transcript_length] + "..."

            # Build prompt for metadata generation
            prompt = self._build_prompt(
                original_title=original_title,
                transcript=truncated_transcript,
                source_lang_name=source_lang_name,
                target_lang_name=target_lang_name,
                max_keywords=max_keywords,
                platform=platform
            )

            # Call AI API
            response = await self._call_ai_api(prompt)

            if not response:
                return VideoMetadataResult(
                    success=False,
                    error="Empty response from AI"
                )

            # Parse response
            result = self._parse_response(response)

            # Apply title prefix if provided
            if title_prefix:
                if result.title:
                    result.title = f"{title_prefix}{result.title}"
                if result.title_translated:
                    result.title_translated = f"{title_prefix}{result.title_translated}"

            # Build final description with source URL and signature
            description_parts = [result.description]

            if source_url:
                description_parts.append(f"\n\n🔗 原视频: {source_url}")

            if custom_signature:
                description_parts.append(f"\n\n{custom_signature}")

            result.description = "".join(description_parts)
            result.success = True

            return result

        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return VideoMetadataResult(
                success=False,
                error=str(e)
            )

    def _build_prompt(
        self,
        original_title: str,
        transcript: str,
        source_lang_name: str,
        target_lang_name: str,
        max_keywords: int,
        platform: str = "generic"
    ) -> str:
        """Build the prompt for metadata generation based on platform"""
        # Determine if we need Chinese output
        is_chinese = target_lang_name in ["Simplified Chinese", "Traditional Chinese", "zh-CN", "zh-TW"]

        if is_chinese:
            # Platform-specific style guides
            platform_styles = {
                "douyin": """风格要求（抖音）：
   - 标题要短小精悍，有网感，能引起好奇心
   - 描述简短有力，适当使用emoji（1-3个）
   - 使用抖音热门话题标签格式：#话题
   - 语气活泼接地气，可以用"绝了"、"太可了"等网络用语
   - 描述控制在100字以内""",
                
                "bilibili": """风格要求（B站）：
   - 标题可以稍长，信息量大，专业感强
   - 描述详细完整，150-300字，介绍视频内容和亮点
   - 可以分段，使用【】标注重点
   - 语气可以轻松但要有内容深度
   - 关键词要精准，覆盖视频主题""",
                
                "xiaohongshu": """风格要求（小红书）：
   - 标题要有种草感，可以用！增强语气
   - 描述要像朋友分享，用第一人称
   - 大量使用emoji表情（5-10个）✨💕🌟
   - 适当分行，每行一个要点
   - 结尾可以加互动语句如「你们觉得呢？」""",
                
                "generic": """风格要求：
   - 描述视频的主题和目标观众
   - 说明观众能获得什么价值
   - 内容要吸引人且有信息量"""
            }
            
            style_guide = platform_styles.get(platform, platform_styles["generic"])
            
            return f"""你是一个专业的视频内容分析师，负责为视频创建SEO优化的元数据。

原标题: {original_title}

视频字幕（部分，仅供参考）:
{transcript}

目标平台: {platform}

{style_guide}

请用简体中文生成以下内容:

1. TITLE: 将原标题翻译成中文，根据平台风格优化，最多60个字符。

2. DESCRIPTION: 根据平台风格要求撰写描述。
   - 不要使用具体数字（不要说"7个技巧"、"10条建议"等）
   - 使用通用词汇："多个"、"一些"、"实用的"、"核心的"

3. KEYWORDS: {max_keywords}个搜索标签，用逗号分隔，使用中文关键词。

严格按照以下格式输出:
---
TITLE: [中文标题]
DESCRIPTION: [中文描述]
KEYWORDS: [关键词1], [关键词2], [关键词3]...
---"""
        else:
            return f"""Translate and create metadata for a video. ALL output MUST be in {target_lang_name}.

Original Title: {original_title}

Video Transcript (partial, for context only):
{transcript}

Generate ALL content in {target_lang_name}:

1. TITLE: Direct translation of the original title. Keep meaning intact. Max 60 chars.

2. DESCRIPTION: Write 2-3 sentences about the video (150-250 chars total).
   - Describe what the video is about and who it's for
   - Mention the value/benefit viewers will get
   - Be engaging and informative

3. KEYWORDS: {max_keywords} search tags in {target_lang_name}, comma separated.

Output format (exactly):
---
TITLE: [title in {target_lang_name}]
DESCRIPTION: [description in {target_lang_name}]
KEYWORDS: [k1], [k2], [k3]...
---"""

    async def _call_ai_api(self, prompt: str) -> str:
        """Call the appropriate AI API based on engine"""
        import httpx

        if self.engine == "gpt":
            return await self._call_openai(prompt)
        elif self.engine == "claude":
            return await self._call_claude(prompt)
        elif self.engine == "deepseek":
            return await self._call_deepseek(prompt)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        import httpx

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
                        {"role": "system", "content": "You are a professional video content analyst specializing in creating SEO-optimized metadata."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

    async def _call_claude(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        import httpx

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
                    "max_tokens": 1000,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                },
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"].strip()

    async def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek API"""
        import httpx

        async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
            response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a professional video content analyst specializing in creating SEO-optimized metadata."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

    def _parse_response(self, response: str) -> VideoMetadataResult:
        """Parse the AI response into structured metadata"""
        result = VideoMetadataResult(success=False)

        # Extract content between --- markers if present
        if "---" in response:
            parts = response.split("---")
            if len(parts) >= 2:
                response = parts[1].strip()

        # Parse TITLE
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if title_match:
            result.title_translated = title_match.group(1).strip()
            result.title = result.title_translated

        # Parse DESCRIPTION
        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=\nKEYWORDS:|\n---|\Z)', response, re.IGNORECASE | re.DOTALL)
        if desc_match:
            desc = desc_match.group(1).strip()
            # Remove number expressions like "7个", "10条", "5大", "80%"
            desc = re.sub(r'\d+[个条大种招步点项件款类篇章节期集]+', '', desc)
            desc = re.sub(r'\d+%[的人]?', '', desc)
            desc = re.sub(r'[分享揭示介绍讲解总结]?\d+[个条大种招步点项件款类篇章节期集]', '', desc)
            # Clean up double spaces
            desc = re.sub(r'\s+', ' ', desc).strip()
            # Remove leading punctuation
            desc = re.sub(r'^[，、,\s]+', '', desc)
            result.description = desc

        # Parse KEYWORDS
        keywords_match = re.search(r'KEYWORDS:\s*(.+?)(?:\n---|\Z)', response, re.IGNORECASE | re.DOTALL)
        if keywords_match:
            keywords_str = keywords_match.group(1).strip()
            # Split by comma and clean up
            keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
            result.keywords = keywords

        return result

    async def translate_title(
        self,
        title: str,
        source_language: str = "en",
        target_language: str = "zh-CN"
    ) -> str:
        """
        Translate a title using AI

        Args:
            title: Original title
            source_language: Source language code
            target_language: Target language code

        Returns:
            Translated title
        """
        if not self.api_key:
            return title

        lang_names = {
            "en": "English",
            "zh-CN": "Simplified Chinese",
            "zh-TW": "Traditional Chinese",
            "ja": "Japanese",
            "ko": "Korean",
        }
        target_lang_name = lang_names.get(target_language, target_language)

        prompt = f"""Translate the following video title to {target_lang_name}.
Keep it concise, engaging, and natural. Only output the translated title, nothing else.

Title: {title}"""

        try:
            response = await self._call_ai_api(prompt)
            return response.strip() if response else title
        except Exception as e:
            logger.error(f"Title translation failed: {e}")
            return title

    async def extract_keywords(
        self,
        transcript: str,
        target_language: str = "zh-CN",
        max_keywords: int = 10
    ) -> List[str]:
        """
        Extract keywords from transcript

        Args:
            transcript: Video transcript
            target_language: Target language for keywords
            max_keywords: Maximum number of keywords

        Returns:
            List of keywords
        """
        if not self.api_key:
            return []

        lang_names = {
            "en": "English",
            "zh-CN": "Simplified Chinese",
            "zh-TW": "Traditional Chinese",
            "ja": "Japanese",
            "ko": "Korean",
        }
        target_lang_name = lang_names.get(target_language, target_language)

        # Truncate transcript
        max_length = 3000
        truncated = transcript[:max_length] if len(transcript) > max_length else transcript

        prompt = f"""Extract {max_keywords} relevant keywords/tags from this video transcript.
Output keywords in {target_lang_name}, separated by commas. Only output the keywords, nothing else.

Transcript:
{truncated}"""

        try:
            response = await self._call_ai_api(prompt)
            if response:
                keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
                return keywords[:max_keywords]
            return []
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    async def generate_summary(
        self,
        transcript: str,
        target_language: str = "zh-CN",
        max_length: int = 200
    ) -> str:
        """
        Generate a summary of the video content

        Args:
            transcript: Video transcript
            target_language: Target language for summary
            max_length: Maximum summary length in characters

        Returns:
            Summary text
        """
        if not self.api_key:
            return ""

        lang_names = {
            "en": "English",
            "zh-CN": "Simplified Chinese",
            "zh-TW": "Traditional Chinese",
            "ja": "Japanese",
            "ko": "Korean",
        }
        target_lang_name = lang_names.get(target_language, target_language)

        # Truncate transcript
        max_transcript = 4000
        truncated = transcript[:max_transcript] if len(transcript) > max_transcript else transcript

        prompt = f"""Summarize this video content in {target_lang_name} in {max_length} characters or less.
Focus on the main topics and key takeaways. Make it engaging and informative.
Only output the summary, nothing else.

Transcript:
{truncated}"""

        try:
            response = await self._call_ai_api(prompt)
            return response.strip() if response else ""
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ""

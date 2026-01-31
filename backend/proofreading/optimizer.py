"""
AI-powered subtitle optimizer
Optimizes translated subtitles based on proofreading results
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import asyncio


@dataclass
class OptimizationConfig:
    """Configuration for subtitle optimization"""
    level: str = "moderate"  # minimal, moderate, aggressive
    max_segments_per_batch: int = 15
    timeout_seconds: int = 60


@dataclass
class OptimizationResult:
    """Result of subtitle optimization"""
    success: bool
    optimized_count: int
    total_segments: int
    segments: List[Dict[str, Any]]
    changes: List[Dict[str, Any]]  # List of changes made
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "optimized_count": self.optimized_count,
            "total_segments": self.total_segments,
            "segments": self.segments,
            "changes": self.changes,
            "error": self.error,
        }


class SubtitleOptimizer:
    """
    AI-powered subtitle optimizer.

    Optimizes subtitles based on proofreading results:
    - Fixes grammar issues
    - Improves naturalness
    - Maintains timeline (only changes text)
    """

    LEVEL_PROMPTS = {
        "minimal": "只修复明显的语法错误和错别字，保持原有风格不变",
        "moderate": "修复语法错误，改善不自然的表达，保持原意",
        "aggressive": "全面优化翻译质量，使表达更加地道自然，可以适当调整措辞",
    }

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

    async def optimize(
        self,
        segments: List[Dict[str, Any]],
        proofreading_result: Optional[Dict[str, Any]] = None,
        source_lang: str = "en",
        target_lang: str = "zh"
    ) -> OptimizationResult:
        """
        Optimize subtitles using AI.

        Args:
            segments: Subtitle segments with index, start_time, end_time,
                      original_text, translated_text
            proofreading_result: Optional proofreading results to guide optimization
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            OptimizationResult with optimized segments
        """
        logger.info(f"Starting subtitle optimization: {len(segments)} segments, level={self.config.level}")

        try:
            from settings_store import settings_store
            settings = settings_store.load()

            api_key = settings.translation.get_api_key()
            engine = settings.translation.engine

            if not api_key:
                return OptimizationResult(
                    success=False,
                    optimized_count=0,
                    total_segments=len(segments),
                    segments=segments,
                    changes=[],
                    error="No API key configured for AI optimization"
                )

            # Get segments that need optimization (have issues or all if no proofreading result)
            segments_to_optimize = self._get_segments_to_optimize(segments, proofreading_result)

            if not segments_to_optimize:
                logger.info("No segments need optimization")
                return OptimizationResult(
                    success=True,
                    optimized_count=0,
                    total_segments=len(segments),
                    segments=segments,
                    changes=[]
                )

            logger.info(f"Optimizing {len(segments_to_optimize)} segments with issues")

            # Optimize in batches
            all_changes = []
            optimized_texts = {}  # index -> optimized text

            batch_size = self.config.max_segments_per_batch
            # Build a map of segment index -> issues/suggestions from proofreading
            segment_issues_map = {}
            if proofreading_result:
                for seg_result in proofreading_result.get("segments", []):
                    seg_idx = seg_result.get("index")
                    if seg_idx is not None:
                        issues = seg_result.get("issues", [])
                        suggestions = [issue.get("suggestion", "") for issue in issues if issue.get("suggestion")]
                        segment_issues_map[seg_idx] = {
                            "issues": issues,
                            "suggestions": suggestions
                        }

            for batch_start in range(0, len(segments_to_optimize), batch_size):
                batch = segments_to_optimize[batch_start:batch_start + batch_size]

                batch_results = await self._optimize_batch(
                    batch,
                    proofreading_result,
                    source_lang,
                    target_lang,
                    api_key,
                    engine
                )

                for idx, new_text in batch_results.items():
                    if new_text and new_text != segments[idx].get("translated_text", ""):
                        optimized_texts[idx] = new_text
                        # Include rich information for comparison
                        issue_info = segment_issues_map.get(idx, {})
                        all_changes.append({
                            "index": idx,
                            "start_time": segments[idx].get("start_time", 0),  # Segment start time
                            "end_time": segments[idx].get("end_time", 0),  # Segment end time
                            "original_text": segments[idx].get("original_text", ""),  # Source language text
                            "translated_text": segments[idx].get("translated_text", ""),  # Before optimization
                            "optimized_text": new_text,  # After optimization
                            "suggestions": issue_info.get("suggestions", []),  # Proofreading suggestions
                            "issues": issue_info.get("issues", [])  # Full issue details
                        })

            # Build result segments (preserve all fields, only update translated_text)
            result_segments = []
            for seg in segments:
                new_seg = seg.copy()
                idx = seg.get("index", 0)
                if idx in optimized_texts:
                    new_seg["translated_text"] = optimized_texts[idx]
                result_segments.append(new_seg)

            logger.info(f"Optimization complete: {len(all_changes)} segments optimized")

            return OptimizationResult(
                success=True,
                optimized_count=len(all_changes),
                total_segments=len(segments),
                segments=result_segments,
                changes=all_changes
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                success=False,
                optimized_count=0,
                total_segments=len(segments),
                segments=segments,
                changes=[],
                error=str(e)
            )

    def _get_segments_to_optimize(
        self,
        segments: List[Dict[str, Any]],
        proofreading_result: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get segments that need optimization based on proofreading results"""
        if not proofreading_result:
            # No proofreading result, optimize all segments
            return segments

        # Get indices of segments with issues
        issue_indices = set()
        for seg_result in proofreading_result.get("segments", []):
            if seg_result.get("issues"):
                issue_indices.add(seg_result.get("index"))

        # Filter segments with issues
        return [seg for seg in segments if seg.get("index") in issue_indices]

    async def _optimize_batch(
        self,
        batch: List[Dict[str, Any]],
        proofreading_result: Optional[Dict[str, Any]],
        source_lang: str,
        target_lang: str,
        api_key: str,
        engine: str
    ) -> Dict[int, str]:
        """Optimize a batch of segments using AI"""
        results = {}

        # Build context with issues for each segment
        segments_info = []
        for seg in batch:
            idx = seg.get("index", 0)
            info = {
                "index": idx,
                "original": seg.get("original_text", ""),
                "translated": seg.get("translated_text", ""),
                "issues": []
            }

            # Add issues if available
            if proofreading_result:
                for seg_result in proofreading_result.get("segments", []):
                    if seg_result.get("index") == idx:
                        info["issues"] = [
                            f"{issue.get('type', 'unknown')}: {issue.get('message', '')}"
                            for issue in seg_result.get("issues", [])
                        ]
                        break

            segments_info.append(info)

        # Build prompt
        level_instruction = self.LEVEL_PROMPTS.get(self.config.level, self.LEVEL_PROMPTS["moderate"])

        segments_text = ""
        for info in segments_info:
            segments_text += f"[{info['index']}]\n"
            segments_text += f"原文: {info['original']}\n"
            segments_text += f"译文: {info['translated']}\n"
            if info['issues']:
                segments_text += f"问题: {'; '.join(info['issues'])}\n"
            segments_text += "\n"

        prompt = f"""请优化以下字幕翻译。原文语言是{source_lang}，目标语言是{target_lang}。

优化要求：{level_instruction}

{segments_text}

请直接返回优化后的译文，格式如下：
[段落编号] 优化后的译文

只返回需要修改的段落。如果某段落不需要修改，则不要返回该段落。
不要添加任何额外的解释或说明。

示例格式：
[0] 这是优化后的译文
[3] 另一段优化后的译文"""

        try:
            response = await self._call_ai(prompt, api_key, engine)
            results = self._parse_optimization_response(response, batch)
        except Exception as e:
            logger.error(f"Batch optimization failed: {e}")

        return results

    async def _call_ai(self, prompt: str, api_key: str, engine: str) -> str:
        """Call AI API to get optimization suggestions"""
        if engine in ["gpt", "openai"]:
            return await self._call_openai(prompt, api_key)
        elif engine == "deepseek":
            return await self._call_deepseek(prompt, api_key)
        elif engine in ["claude", "anthropic"]:
            return await self._call_anthropic(prompt, api_key)
        else:
            raise ValueError(f"Unsupported AI engine for optimization: {engine}")

    async def _call_openai(self, prompt: str, api_key: str) -> str:
        """Call OpenAI API"""
        import openai

        client = openai.OpenAI(api_key=api_key, timeout=self.config.timeout_seconds)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
        )

        return response.choices[0].message.content or ""

    async def _call_deepseek(self, prompt: str, api_key: str) -> str:
        """Call DeepSeek API"""
        import openai

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=self.config.timeout_seconds
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
        )

        return response.choices[0].message.content or ""

    async def _call_anthropic(self, prompt: str, api_key: str) -> str:
        """Call Anthropic API"""
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
        )

        return response.content[0].text if response.content else ""

    def _parse_optimization_response(
        self,
        response: str,
        batch: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        """Parse AI response to extract optimized texts"""
        results = {}

        if not response:
            return results

        import re

        # Parse lines like "[0] optimized text"
        pattern = r'\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        valid_indices = {seg.get("index") for seg in batch}

        for match in matches:
            try:
                idx = int(match[0])
                text = match[1].strip()
                if idx in valid_indices and text:
                    results[idx] = text
            except (ValueError, IndexError):
                continue

        return results


# Convenience functions

async def optimize_subtitles(
    segments: List[Dict[str, Any]],
    proofreading_result: Optional[Dict[str, Any]] = None,
    source_lang: str = "en",
    target_lang: str = "zh",
    level: str = "moderate"
) -> OptimizationResult:
    """
    Convenience function to optimize subtitles.

    Args:
        segments: Subtitle segments
        proofreading_result: Optional proofreading results
        source_lang: Source language
        target_lang: Target language
        level: Optimization level (minimal, moderate, aggressive)

    Returns:
        OptimizationResult
    """
    config = OptimizationConfig(level=level)
    optimizer = SubtitleOptimizer(config)
    return await optimizer.optimize(
        segments, proofreading_result, source_lang, target_lang
    )

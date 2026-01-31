"""
Subtitle validation checkers
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from loguru import logger

from .models import (
    ProofreadingIssue, IssueSeverity, IssueType,
    SegmentProofreadResult, ProofreadingConfig
)


class TimingValidator:
    """Validates subtitle timing and speech rate"""

    def __init__(self, config: ProofreadingConfig):
        self.config = config

    def validate(
        self,
        segments: List[Dict[str, Any]],
        translated_segments: List[Dict[str, Any]]
    ) -> List[ProofreadingIssue]:
        """Check timing issues in subtitles"""
        issues = []

        for i, (orig, trans) in enumerate(zip(segments, translated_segments)):
            start = orig.get("start", 0)
            end = orig.get("end", 0)
            duration = end - start

            if duration <= 0:
                issues.append(ProofreadingIssue(
                    segment_index=i,
                    issue_type=IssueType.OVERLAP_DETECTED,
                    severity=IssueSeverity.ERROR,
                    message="字幕时间段无效（持续时间<=0）",
                    original_text=orig.get("text", ""),
                    translated_text=trans.get("text", ""),
                    start_time=start,
                    end_time=end,
                ))
                continue

            # Check speech rate (characters per second)
            trans_text = trans.get("text", "")
            char_count = len(trans_text.replace(" ", ""))
            chars_per_second = char_count / duration if duration > 0 else 0

            if chars_per_second > self.config.max_chars_per_second:
                issues.append(ProofreadingIssue(
                    segment_index=i,
                    issue_type=IssueType.SPEECH_TOO_FAST,
                    severity=IssueSeverity.WARNING,
                    message=f"语速过快 ({chars_per_second:.1f} 字/秒)，可能需要拆分字幕",
                    original_text=orig.get("text", ""),
                    translated_text=trans_text,
                    start_time=start,
                    end_time=end,
                ))
            elif chars_per_second < self.config.min_chars_per_second and char_count > 0:
                issues.append(ProofreadingIssue(
                    segment_index=i,
                    issue_type=IssueType.SPEECH_TOO_SLOW,
                    severity=IssueSeverity.INFO,
                    message=f"语速较慢 ({chars_per_second:.1f} 字/秒)",
                    original_text=orig.get("text", ""),
                    translated_text=trans_text,
                    start_time=start,
                    end_time=end,
                ))

            # Check for gaps between segments
            if i > 0:
                prev_end = segments[i - 1].get("end", 0)
                gap = start - prev_end
                if gap > 5.0:  # More than 5 seconds gap
                    issues.append(ProofreadingIssue(
                        segment_index=i,
                        issue_type=IssueType.GAP_TOO_LONG,
                        severity=IssueSeverity.INFO,
                        message=f"与前一段字幕间隔较长 ({gap:.1f} 秒)",
                        start_time=start,
                        end_time=end,
                    ))
                elif gap < -0.1:  # Overlapping
                    issues.append(ProofreadingIssue(
                        segment_index=i,
                        issue_type=IssueType.OVERLAP_DETECTED,
                        severity=IssueSeverity.WARNING,
                        message=f"与前一段字幕重叠 ({-gap:.2f} 秒)",
                        start_time=start,
                        end_time=end,
                    ))

        return issues


class FormatValidator:
    """Validates subtitle formatting"""

    def __init__(self, config: ProofreadingConfig):
        self.config = config

    def validate(
        self,
        segments: List[Dict[str, Any]],
        translated_segments: List[Dict[str, Any]]
    ) -> List[ProofreadingIssue]:
        """Check formatting issues"""
        issues = []

        for i, (orig, trans) in enumerate(zip(segments, translated_segments)):
            orig_text = orig.get("text", "")
            trans_text = trans.get("text", "")

            # Check empty segment
            if not trans_text.strip():
                if orig_text.strip():
                    issues.append(ProofreadingIssue(
                        segment_index=i,
                        issue_type=IssueType.MISSING_TRANSLATION,
                        severity=IssueSeverity.ERROR,
                        message="翻译缺失（原文有内容但译文为空）",
                        original_text=orig_text,
                        translated_text=trans_text,
                        start_time=orig.get("start", 0),
                        end_time=orig.get("end", 0),
                    ))
                else:
                    issues.append(ProofreadingIssue(
                        segment_index=i,
                        issue_type=IssueType.EMPTY_SEGMENT,
                        severity=IssueSeverity.WARNING,
                        message="空字幕段",
                        start_time=orig.get("start", 0),
                        end_time=orig.get("end", 0),
                    ))
                continue

            # Check line length
            lines = trans_text.split("\n")
            for line in lines:
                if len(line) > self.config.max_line_length:
                    issues.append(ProofreadingIssue(
                        segment_index=i,
                        issue_type=IssueType.LINE_TOO_LONG,
                        severity=IssueSeverity.WARNING,
                        message=f"单行字幕过长 ({len(line)} 字符)，建议拆分",
                        original_text=orig_text,
                        translated_text=trans_text,
                        start_time=orig.get("start", 0),
                        end_time=orig.get("end", 0),
                    ))
                    break

            # Check for encoding issues (garbled text)
            if self._has_encoding_issues(trans_text):
                issues.append(ProofreadingIssue(
                    segment_index=i,
                    issue_type=IssueType.ENCODING_ERROR,
                    severity=IssueSeverity.ERROR,
                    message="可能存在编码问题（检测到乱码字符）",
                    original_text=orig_text,
                    translated_text=trans_text,
                    start_time=orig.get("start", 0),
                    end_time=orig.get("end", 0),
                ))

        return issues

    def _has_encoding_issues(self, text: str) -> bool:
        """Check if text has encoding issues"""
        # Check for common garbled patterns
        garbled_patterns = [
            r'[\ufffd]',  # Replacement character
            r'[锟斤拷]+',  # Common Chinese garbled text
            r'[\x00-\x08\x0b\x0c\x0e-\x1f]',  # Control characters
        ]
        for pattern in garbled_patterns:
            if re.search(pattern, text):
                return True
        return False


class TerminologyValidator:
    """Validates terminology consistency"""

    def __init__(self, config: ProofreadingConfig):
        self.config = config
        self.term_mapping: Dict[str, str] = {}

    def validate(
        self,
        segments: List[Dict[str, Any]],
        translated_segments: List[Dict[str, Any]]
    ) -> Tuple[List[ProofreadingIssue], float]:
        """Check terminology consistency and return issues with consistency score"""
        issues = []

        # Build term mapping from all segments
        term_translations: Dict[str, List[Tuple[int, str]]] = {}

        for i, (orig, trans) in enumerate(zip(segments, translated_segments)):
            orig_text = orig.get("text", "").lower()
            trans_text = trans.get("text", "")

            # Extract potential terms (simple approach: words that appear multiple times)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', orig_text)
            for word in words:
                if word not in term_translations:
                    term_translations[word] = []
                # Store the context (surrounding translated text)
                term_translations[word].append((i, trans_text))

        # Check for inconsistent translations
        inconsistent_terms = 0
        total_terms = 0

        for term, occurrences in term_translations.items():
            if len(occurrences) < 2:
                continue

            total_terms += 1

            # Simple check: if translations are very different, flag it
            translations = [t[1] for t in occurrences]
            unique_translations = set(translations)

            if len(unique_translations) > 1:
                # Check if translations are actually different (not just context)
                # This is a simple heuristic - in practice, we'd use more sophisticated matching
                first_occurrence = occurrences[0]
                for idx, trans in occurrences[1:]:
                    # Only flag if this is a technical term that should be consistent
                    if self._is_technical_term(term):
                        inconsistent_terms += 1
                        issues.append(ProofreadingIssue(
                            segment_index=idx,
                            issue_type=IssueType.INCONSISTENT_TERM,
                            severity=IssueSeverity.WARNING,
                            message=f"术语 '{term}' 的翻译可能不一致",
                            original_text=term,
                            translated_text=trans,
                            start_time=segments[idx].get("start", 0),
                            end_time=segments[idx].get("end", 0),
                        ))
                        break  # Only report once per term

        # Calculate consistency score
        if total_terms > 0:
            consistency_score = 1.0 - (inconsistent_terms / total_terms)
        else:
            consistency_score = 1.0

        return issues, consistency_score

    def _is_technical_term(self, word: str) -> bool:
        """Check if word is likely a technical term"""
        # Technical terms often have these patterns
        technical_patterns = [
            r'^[A-Z][a-z]+[A-Z]',  # CamelCase
            r'[A-Z]{2,}',  # Acronyms
            r'\d',  # Contains numbers
        ]
        for pattern in technical_patterns:
            if re.search(pattern, word):
                return True

        # Common technical words
        technical_words = {
            'api', 'sdk', 'cpu', 'gpu', 'ram', 'http', 'https', 'url',
            'json', 'xml', 'html', 'css', 'javascript', 'python',
            'database', 'server', 'client', 'algorithm', 'function',
            'variable', 'parameter', 'interface', 'module', 'framework',
        }
        return word.lower() in technical_words


class AIValidator:
    """Uses AI to validate translation quality"""

    def __init__(self, config: ProofreadingConfig):
        self.config = config

    async def validate(
        self,
        segments: List[Dict[str, Any]],
        translated_segments: List[Dict[str, Any]],
        source_lang: str = "en",
        target_lang: str = "zh"
    ) -> List[ProofreadingIssue]:
        """Use AI to check translation quality"""
        issues = []

        if not self.config.use_ai_validation:
            return issues

        try:
            from settings_store import settings_store
            settings = settings_store.load()

            api_key = settings.translation.get_api_key()
            engine = settings.translation.engine

            if not api_key or engine in ["google", "deepl"]:
                logger.debug("AI validation skipped: no suitable API key")
                return issues

            # Check if required AI module is available
            try:
                if engine in ["gpt", "openai", "deepseek"]:
                    import openai  # noqa: F401
                elif engine in ["claude", "anthropic"]:
                    import anthropic  # noqa: F401
            except ImportError as e:
                logger.warning(f"AI validation skipped: {e}")
                return issues

            # Batch segments for efficiency
            # Limit to first 100 segments to avoid timeout, increase batch size
            max_segments = min(100, len(segments))
            batch_size = 25
            total_batches = (max_segments + batch_size - 1) // batch_size

            for batch_idx, batch_start in enumerate(range(0, max_segments, batch_size)):
                batch_end = min(batch_start + batch_size, max_segments)
                logger.debug(f"AI validation batch {batch_idx + 1}/{total_batches} (segments {batch_start}-{batch_end})")

                batch_issues = await self._validate_batch(
                    segments[batch_start:batch_end],
                    translated_segments[batch_start:batch_end],
                    batch_start,
                    source_lang,
                    target_lang,
                    api_key,
                    engine
                )
                issues.extend(batch_issues)

        except Exception as e:
            logger.warning(f"AI validation failed: {e}")

        return issues

    async def _validate_batch(
        self,
        orig_batch: List[Dict[str, Any]],
        trans_batch: List[Dict[str, Any]],
        start_index: int,
        source_lang: str,
        target_lang: str,
        api_key: str,
        engine: str
    ) -> List[ProofreadingIssue]:
        """Validate a batch of segments using AI"""
        issues = []

        # Build prompt
        segments_text = ""
        for i, (orig, trans) in enumerate(zip(orig_batch, trans_batch)):
            segments_text += f"[{start_index + i}] 原文: {orig.get('text', '')}\n"
            segments_text += f"    译文: {trans.get('text', '')}\n\n"

        prompt = f"""请检查以下字幕翻译的质量。原文语言是{source_lang}，目标语言是{target_lang}。

{segments_text}

请检查是否存在以下问题：
1. 语法错误
2. 不自然的表达
3. 明显的翻译错误
4. 缺失的翻译

只返回有问题的段落，格式如下（如果没有问题则返回"无问题"）：
[段落编号] 问题类型: 问题描述 | 建议修改: 修改建议

例如：
[0] 语法错误: "他们是去学校" 应该是 "他们去学校" | 建议修改: 他们去学校
[3] 不自然: "这个是非常好的" 表达不自然 | 建议修改: 这非常好"""

        try:
            response_text = await self._call_ai(prompt, api_key, engine)
            issues = self._parse_ai_response(response_text, orig_batch, trans_batch, start_index)
        except Exception as e:
            logger.warning(f"AI batch validation failed: {e}")

        return issues

    async def _call_ai(self, prompt: str, api_key: str, engine: str) -> str:
        """Call AI API"""
        if engine in ["gpt", "openai"]:
            import openai
            client = openai.OpenAI(api_key=api_key, timeout=30.0)
            response = client.chat.completions.create(
                model=self.config.ai_model if "gpt" in self.config.ai_model else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content

        elif engine in ["claude", "anthropic"]:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif engine == "deepseek":
            import openai
            client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=30.0)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content

        return ""

    def _parse_ai_response(
        self,
        response: str,
        orig_batch: List[Dict[str, Any]],
        trans_batch: List[Dict[str, Any]],
        start_index: int
    ) -> List[ProofreadingIssue]:
        """Parse AI response into issues"""
        issues = []

        if "无问题" in response or not response.strip():
            return issues

        # Parse each line
        lines = response.strip().split("\n")
        for line in lines:
            match = re.match(r'\[(\d+)\]\s*(.+?):\s*(.+?)(?:\s*\|\s*建议修改:\s*(.+))?$', line)
            if match:
                idx = int(match.group(1))
                issue_type_str = match.group(2).strip()
                description = match.group(3).strip()
                suggestion = match.group(4).strip() if match.group(4) else ""

                # Map issue type
                if "语法" in issue_type_str:
                    issue_type = IssueType.GRAMMAR_ERROR
                    severity = IssueSeverity.WARNING
                elif "不自然" in issue_type_str:
                    issue_type = IssueType.UNNATURAL_PHRASING
                    severity = IssueSeverity.WARNING
                elif "错误" in issue_type_str or "翻译错误" in issue_type_str:
                    issue_type = IssueType.MISTRANSLATION
                    severity = IssueSeverity.ERROR
                elif "缺失" in issue_type_str:
                    issue_type = IssueType.MISSING_TRANSLATION
                    severity = IssueSeverity.ERROR
                else:
                    issue_type = IssueType.UNNATURAL_PHRASING
                    severity = IssueSeverity.WARNING

                # Get original segment info
                local_idx = idx - start_index
                if 0 <= local_idx < len(orig_batch):
                    orig = orig_batch[local_idx]
                    trans = trans_batch[local_idx]

                    issues.append(ProofreadingIssue(
                        segment_index=idx,
                        issue_type=issue_type,
                        severity=severity,
                        message=description,
                        original_text=orig.get("text", ""),
                        translated_text=trans.get("text", ""),
                        suggestion=suggestion,
                        start_time=orig.get("start", 0),
                        end_time=orig.get("end", 0),
                        auto_fixable=bool(suggestion),
                    ))

        return issues

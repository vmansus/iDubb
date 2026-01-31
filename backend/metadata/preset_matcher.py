"""
Metadata Preset Matcher
Provides AI-powered preset selection based on video content analysis
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger


# Minimum confidence threshold - below this, use default preset
MIN_CONFIDENCE_THRESHOLD = 0.15

# Default preset when no good match is found
DEFAULT_PRESET_ID = "chinese_subtitles"


@dataclass
class PresetMatch:
    """Result of preset matching"""
    preset_id: str
    confidence: float  # 0.0 to 1.0
    reason: str


@dataclass
class MatchResult:
    """Full result including all evaluated presets"""
    recommended: PresetMatch
    all_matches: List[PresetMatch]


# Keywords mapping for each builtin preset
# Note: chinese_subtitles is the default, so we don't need to match it with keywords
PRESET_KEYWORDS = {
    "tech_tutorial": {
        "positive": [
            # Tech/Programming keywords - weighted by specificity
            "programming", "code", "coding", "软件开发", "程序员",
            "AI", "人工智能", "machine learning", "深度学习", "neural",
            "python", "javascript", "typescript", "react", "vue", "angular",
            "kubernetes", "docker", "cloud", "aws", "azure", "gcp",
            "api", "backend", "frontend", "full stack", "devops",
            "algorithm", "算法", "数据结构", "leetcode",
            "tutorial", "教程", "how to", "learn", "学习",
            "linux", "unix", "terminal", "命令行",
            "database", "sql", "mongodb", "redis",
            "git", "github", "开源",
        ],
        "strong_positive": [
            # Strong indicators - if any of these match, high confidence
            "programming", "code", "coding", "python", "javascript",
            "kubernetes", "docker", "api", "algorithm", "leetcode",
            "tutorial", "教程", "开发", "developer",
        ],
        "negative": ["娱乐", "搞笑", "vlog", "游戏", "game", "react视频", "综艺"],
        "platforms": ["youtube"],
        "base_score": 0.1,  # Base score before keyword matching
    },
    "entertainment": {
        "positive": [
            "娱乐", "搞笑", "综艺", "entertainment", "funny", "comedy",
            "vlog", "日常", "daily",
            "游戏", "game", "gaming", "gameplay", "直播", "stream",
            "影视", "电影", "movie", "film", "剧", "drama",
            "音乐", "music", "mv", "song", "cover",
            "reaction", "react", "challenge", "挑战",
            "美食", "food", "cooking", "吃播", "mukbang",
            "旅游", "travel", "vlog", "trip",
            "生活", "lifestyle", "日常",
            "搞笑", "funny", "meme", "整活",
        ],
        "strong_positive": [
            "vlog", "游戏", "game", "综艺", "搞笑", "娱乐",
            "reaction", "challenge", "美食", "吃播",
        ],
        "negative": ["教程", "tutorial", "技术", "code", "programming", "algorithm"],
        "platforms": ["youtube", "tiktok", "bilibili"],
        "base_score": 0.1,
    },
    "bilingual_subtitles": {
        "positive": [
            "双语", "bilingual", "中英", "english chinese",
            "两种语言", "dual language", "中英文", "英中",
        ],
        "strong_positive": ["双语", "bilingual", "中英"],
        "negative": [],
        "platforms": ["youtube"],
        "base_score": 0.05,
    },
    "no_prefix": {
        "positive": [
            "原创", "original", "自制", "自己做", "my video",
            "原创内容", "自创", "独家",
        ],
        "strong_positive": ["原创", "original", "自制"],
        "negative": ["翻译", "字幕", "subtitle", "translated", "中字", "搬运"],
        "platforms": [],
        "base_score": 0.0,  # Only match if explicitly original content
    },
}


class PresetMatcher:
    """Matches video content to the most appropriate metadata preset"""

    def __init__(self):
        self.keywords = PRESET_KEYWORDS

    def match(
        self,
        video_info: Optional[Dict[str, Any]] = None,
        transcript: Optional[str] = None,
        presets: Optional[List[Dict[str, Any]]] = None,
    ) -> MatchResult:
        """
        Match video content to presets.

        Args:
            video_info: Video metadata (title, description, platform, tags, uploader)
            transcript: First portion of video transcript
            presets: List of available presets (if None, uses builtin preset IDs)

        Returns:
            MatchResult with recommended preset and all matches
        """
        # Build text corpus for matching
        corpus = self._build_corpus(video_info, transcript)
        corpus_lower = corpus.lower()

        # Log corpus for debugging
        logger.debug(f"Preset matching corpus (first 200 chars): {corpus_lower[:200]}...")

        # Get platform
        platform = ""
        if video_info:
            platform = video_info.get("platform", "").lower()

        # Score each preset (except chinese_subtitles which is default)
        matches = []

        for preset_id, config in self.keywords.items():
            score, reason = self._score_preset(corpus_lower, platform, config, preset_id)
            matches.append(PresetMatch(
                preset_id=preset_id,
                confidence=score,
                reason=reason
            ))
            logger.debug(f"Preset {preset_id}: score={score:.3f}, reason={reason}")

        # Sort by confidence (descending)
        matches.sort(key=lambda x: x.confidence, reverse=True)

        # Check if top match exceeds threshold
        top_match = matches[0] if matches else None

        if top_match and top_match.confidence >= MIN_CONFIDENCE_THRESHOLD:
            recommended = top_match
            logger.info(
                f"Preset match found: {recommended.preset_id} "
                f"(confidence: {recommended.confidence:.2f})"
            )
        else:
            # Fall back to default preset
            recommended = PresetMatch(
                preset_id=DEFAULT_PRESET_ID,
                confidence=0.5,
                reason=f"Default preset (top match {top_match.preset_id if top_match else 'none'} "
                       f"below threshold {MIN_CONFIDENCE_THRESHOLD})"
            )
            logger.info(
                f"No confident match, using default: {DEFAULT_PRESET_ID} "
                f"(top was {top_match.preset_id}={top_match.confidence:.2f} if top_match else 'none')"
            )

        # Add default preset to all_matches for completeness
        matches.append(PresetMatch(
            preset_id=DEFAULT_PRESET_ID,
            confidence=0.5,
            reason="Default preset for translation content"
        ))

        return MatchResult(
            recommended=recommended,
            all_matches=matches
        )

    def _build_corpus(
        self,
        video_info: Optional[Dict[str, Any]],
        transcript: Optional[str]
    ) -> str:
        """Build text corpus from video info and transcript"""
        parts = []

        if video_info:
            # Title is most important - add it twice for weight
            if video_info.get("title"):
                parts.append(video_info["title"])
                parts.append(video_info["title"])  # Double weight for title
            if video_info.get("description"):
                # Only use first 500 chars of description
                parts.append(video_info["description"][:500])
            if video_info.get("tags"):
                # Tags are good indicators
                parts.extend(video_info["tags"][:15])
            if video_info.get("uploader"):
                parts.append(video_info["uploader"])

        if transcript:
            # Only use first 1000 chars of transcript
            parts.append(transcript[:1000])

        return " ".join(parts)

    def _score_preset(
        self,
        corpus: str,
        platform: str,
        config: Dict[str, Any],
        preset_id: str
    ) -> tuple[float, str]:
        """Score a preset against the corpus"""
        positive_keywords = config.get("positive", [])
        strong_positive = config.get("strong_positive", [])
        negative_keywords = config.get("negative", [])
        platforms = config.get("platforms", [])
        base_score = config.get("base_score", 0.0)

        # Count keyword matches
        positive_matches = []
        strong_matches = []
        negative_matches = []

        for keyword in positive_keywords:
            if keyword.lower() in corpus:
                positive_matches.append(keyword)
                if keyword.lower() in [k.lower() for k in strong_positive]:
                    strong_matches.append(keyword)

        for keyword in negative_keywords:
            if keyword.lower() in corpus:
                negative_matches.append(keyword)

        # Calculate score
        # Strong matches contribute more
        strong_score = len(strong_matches) * 0.15  # Each strong match adds 0.15
        regular_score = (len(positive_matches) - len(strong_matches)) * 0.05  # Regular matches add 0.05

        # Negative matches reduce score significantly
        negative_penalty = len(negative_matches) * 0.2

        # Platform bonus (small)
        platform_bonus = 0.05 if platform and platform in platforms else 0

        # Final score
        score = base_score + strong_score + regular_score - negative_penalty + platform_bonus
        score = max(0, min(1, score))  # Clamp to [0, 1]

        # Build reason
        reason_parts = []
        if strong_matches:
            reason_parts.append(f"Strong: {', '.join(strong_matches[:3])}")
        if len(positive_matches) > len(strong_matches):
            other_matches = [m for m in positive_matches if m not in strong_matches][:3]
            if other_matches:
                reason_parts.append(f"Matched: {', '.join(other_matches)}")
        if negative_matches:
            reason_parts.append(f"Negative: {', '.join(negative_matches[:2])}")
        if platform and platform in platforms:
            reason_parts.append(f"Platform: {platform}")

        reason = "; ".join(reason_parts) if reason_parts else "No significant matches"

        return score, reason


# Global instance
preset_matcher = PresetMatcher()


async def select_preset_for_task(
    video_info: Optional[Dict[str, Any]] = None,
    transcript: Optional[str] = None,
) -> PresetMatch:
    """
    Convenience function to select the best preset for a task.

    Args:
        video_info: Video metadata
        transcript: Video transcript (first portion)

    Returns:
        PresetMatch with recommended preset
    """
    result = preset_matcher.match(video_info=video_info, transcript=transcript)

    # Log all matches for debugging
    logger.info(
        f"AI preset selection result: {result.recommended.preset_id} "
        f"(confidence: {result.recommended.confidence:.2f}, reason: {result.recommended.reason})"
    )
    for match in result.all_matches[:3]:
        logger.debug(f"  - {match.preset_id}: {match.confidence:.2f} ({match.reason})")

    return result.recommended

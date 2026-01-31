"""
Subtitle Length Controller

Ensures subtitles don't exceed maximum length.
Auto-splits long lines at natural break points.

Netflix standards:
- Max 42 characters per line
- Max 2 lines per subtitle
"""
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class LengthControlConfig:
    """Configuration for subtitle length control"""
    max_chars_per_line: int = 42  # Netflix standard
    max_lines: int = 2
    # Split preferences (in order of priority)
    split_on_punctuation: bool = True  # Prefer splitting at punctuation
    split_on_conjunctions: bool = True  # Split at conjunctions (和, 但是, etc.)
    balance_lines: bool = True  # Try to make lines similar length
    # AI rewrite for very long subtitles
    use_ai_rewrite: bool = False  # Use AI to condense very long subtitles
    ai_rewrite_threshold: int = 100  # Chars above which to use AI


@dataclass
class LengthControlResult:
    """Result of length control processing"""
    original: str
    processed: str
    was_modified: bool
    lines: List[str]
    total_chars: int
    modifications: List[str]


class SubtitleLengthController:
    """
    Controls subtitle length by splitting or condensing long lines.
    
    Strategies:
    1. Split at punctuation (，。！？；：、)
    2. Split at conjunctions (和、但是、因为、所以、然后)
    3. Split at natural phrase boundaries
    4. AI rewrite for very long subtitles (optional)
    """

    # Chinese punctuation marks (good split points)
    CN_PUNCTUATION = r'[，。！？；：、\n]'
    
    # Chinese conjunctions and connectors
    CN_CONJUNCTIONS = [
        '但是', '但', '然而', '不过', '可是',  # But/however
        '因为', '由于', '因此', '所以', '于是',  # Because/therefore
        '而且', '并且', '同时', '另外', '此外',  # And/moreover
        '或者', '还是', '要么',  # Or
        '如果', '假如', '要是', '倘若',  # If
        '虽然', '尽管', '即使',  # Although
        '然后', '接着', '随后',  # Then
        '那么', '那', '就',  # Then (result)
    ]
    
    # English conjunctions
    EN_CONJUNCTIONS = [
        'but', 'however', 'although', 'though',
        'because', 'since', 'therefore', 'so',
        'and', 'also', 'moreover', 'furthermore',
        'or', 'either', 'neither',
        'if', 'when', 'while', 'whereas',
        'then', 'after', 'before',
    ]

    def __init__(self, config: Optional[LengthControlConfig] = None):
        self.config = config or LengthControlConfig()

    def process(self, subtitle: str, target_lang: str = "zh") -> LengthControlResult:
        """
        Process a subtitle to ensure it meets length requirements.
        
        Args:
            subtitle: The subtitle text
            target_lang: Target language for smart splitting
            
        Returns:
            LengthControlResult with processed subtitle
        """
        original = subtitle.strip()
        modifications = []
        
        # Already short enough?
        if len(original) <= self.config.max_chars_per_line:
            return LengthControlResult(
                original=original,
                processed=original,
                was_modified=False,
                lines=[original],
                total_chars=len(original),
                modifications=[]
            )

        # Try to split
        is_chinese = self._is_chinese(original)
        lines = self._smart_split(original, is_chinese)
        
        # Check if we need further processing
        final_lines = []
        for line in lines:
            if len(line) <= self.config.max_chars_per_line:
                final_lines.append(line)
            else:
                # Line still too long, split again
                sub_lines = self._force_split(line, is_chinese)
                final_lines.extend(sub_lines)
                modifications.append(f"Force split: '{line[:20]}...'")

        # Limit to max lines
        if len(final_lines) > self.config.max_lines:
            # Keep first N lines, might lose some content
            modifications.append(f"Truncated from {len(final_lines)} to {self.config.max_lines} lines")
            final_lines = final_lines[:self.config.max_lines]

        # Balance line lengths if enabled
        if self.config.balance_lines and len(final_lines) == 2:
            final_lines = self._balance_lines(final_lines, is_chinese)

        # Fix orphan punctuation: if last line is only punctuation, merge with previous
        orphan_punctuation = '，。！？；：、…'
        while len(final_lines) > 1:
            last_line = final_lines[-1].strip()
            # Check if last line is only punctuation (1-2 chars, all punctuation)
            if len(last_line) <= 2 and all(c in orphan_punctuation for c in last_line):
                # Merge with previous line
                final_lines[-2] = final_lines[-2] + final_lines[-1]
                final_lines.pop()
                modifications.append("Merged orphan punctuation with previous line")
            else:
                break

        processed = '\n'.join(final_lines)
        
        return LengthControlResult(
            original=original,
            processed=processed,
            was_modified=processed != original,
            lines=final_lines,
            total_chars=sum(len(l) for l in final_lines),
            modifications=modifications
        )

    def _is_chinese(self, text: str) -> bool:
        """Check if text is primarily Chinese"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return chinese_chars > len(text) * 0.3

    def _smart_split(self, text: str, is_chinese: bool) -> List[str]:
        """
        Smart split text into lines at natural break points.
        """
        # First try: split at punctuation
        if self.config.split_on_punctuation:
            parts = re.split(f'({self.CN_PUNCTUATION})', text)
            if len(parts) > 1:
                # Recombine parts with their punctuation
                combined = []
                i = 0
                while i < len(parts):
                    part = parts[i]
                    # Check if next part is punctuation
                    if i + 1 < len(parts) and re.match(self.CN_PUNCTUATION, parts[i + 1]):
                        part += parts[i + 1]
                        i += 2
                    else:
                        i += 1
                    if part.strip():
                        combined.append(part.strip())
                
                # Try to merge into 2 lines of similar length
                if len(combined) >= 2:
                    return self._merge_to_n_lines(combined, self.config.max_lines)

        # Second try: split at conjunctions
        if self.config.split_on_conjunctions:
            conjunctions = self.CN_CONJUNCTIONS if is_chinese else self.EN_CONJUNCTIONS
            for conj in conjunctions:
                if conj in text:
                    idx = text.find(conj)
                    if idx > 5:  # Ensure first part is not too short
                        line1 = text[:idx].strip()
                        line2 = text[idx:].strip()
                        if line1 and line2:
                            return [line1, line2]

        # Third try: split at middle
        return self._split_at_middle(text, is_chinese)

    def _split_at_middle(self, text: str, is_chinese: bool) -> List[str]:
        """Split text roughly in the middle at a good break point. Never breaks English words."""
        target_len = len(text) // 2
        
        # Find best split point near middle
        best_pos = target_len
        best_score = -1000  # Start with very low score
        
        # Search window around middle
        search_start = max(5, target_len - 15)
        search_end = min(len(text) - 5, target_len + 15)
        
        for pos in range(search_start, search_end):
            score = 0
            char = text[pos] if pos < len(text) else ''
            prev_char = text[pos - 1] if pos > 0 else ''
            next_char = text[pos + 1] if pos + 1 < len(text) else ''
            
            # CRITICAL: Never break in the middle of an English word
            is_prev_latin = bool(re.match(r'[a-zA-Z0-9]', prev_char))
            is_curr_latin = bool(re.match(r'[a-zA-Z0-9]', char))
            if is_prev_latin and is_curr_latin:
                # This would break an English word - heavy penalty
                score -= 100
            
            # Prefer splitting after punctuation
            if re.match(self.CN_PUNCTUATION, prev_char):
                score += 10
            
            # Prefer splitting at spaces (for non-Chinese)
            if char == ' ':
                score += 8
            
            # Prefer splitting before conjunctions
            for conj in (self.CN_CONJUNCTIONS if is_chinese else self.EN_CONJUNCTIONS):
                if text[pos:].startswith(conj):
                    score += 8
                    break
            
            # Prefer positions closer to middle
            distance_penalty = abs(pos - target_len) * 0.1
            score -= distance_penalty
            
            if score > best_score:
                best_score = score
                best_pos = pos

        line1 = text[:best_pos].strip()
        line2 = text[best_pos:].strip()
        
        return [line1, line2] if line1 and line2 else [text]

    def _force_split(self, text: str, is_chinese: bool) -> List[str]:
        """Force split a line that's too long. Never breaks English words."""
        max_len = self.config.max_chars_per_line
        lines = []
        
        while len(text) > max_len:
            # Find best split point before max_len
            split_pos = max_len
            found_good_break = False
            
            # Look for punctuation or space to split at
            for pos in range(max_len - 1, max(0, max_len - 20), -1):
                char = text[pos]
                next_char = text[pos + 1] if pos + 1 < len(text) else ''
                prev_char = text[pos - 1] if pos > 0 else ''
                
                # Good break: after punctuation, or at space
                if re.match(self.CN_PUNCTUATION, char) or char == ' ':
                    split_pos = pos + 1
                    found_good_break = True
                    break
                
                # Also good: between CJK and non-CJK, but NOT in middle of English word
                is_curr_latin = bool(re.match(r'[a-zA-Z0-9]', char))
                is_next_latin = bool(re.match(r'[a-zA-Z0-9]', next_char))
                is_prev_latin = bool(re.match(r'[a-zA-Z0-9]', prev_char))
                
                # OK to break if current is NOT in the middle of a Latin word
                if not (is_curr_latin and is_next_latin):
                    if not (is_prev_latin and is_curr_latin):
                        # Safe to break here
                        split_pos = pos + 1
                        found_good_break = True
                        break
            
            # If no good break found, extend search or force break at safe point
            if not found_good_break:
                # Look for any non-Latin boundary
                for pos in range(max_len - 1, 0, -1):
                    char = text[pos]
                    next_char = text[pos + 1] if pos + 1 < len(text) else ''
                    is_curr_latin = bool(re.match(r'[a-zA-Z0-9]', char))
                    is_next_latin = bool(re.match(r'[a-zA-Z0-9]', next_char))
                    if not (is_curr_latin and is_next_latin):
                        split_pos = pos + 1
                        break
            
            lines.append(text[:split_pos].strip())
            text = text[split_pos:].strip()
        
        if text:
            lines.append(text)
        
        return lines

    def _merge_to_n_lines(self, parts: List[str], n: int) -> List[str]:
        """Merge multiple parts into n lines, balancing lengths"""
        if len(parts) <= n:
            return parts
        
        # Calculate total length
        total_len = sum(len(p) for p in parts)
        target_per_line = total_len / n
        
        lines = []
        current_line = ""
        
        for part in parts:
            if not current_line:
                current_line = part
            elif len(current_line) + len(part) <= target_per_line * 1.3:
                current_line += part
            else:
                lines.append(current_line)
                current_line = part
        
        if current_line:
            lines.append(current_line)
        
        # If we have more than n lines, merge smallest adjacent pairs
        while len(lines) > n:
            min_combined_len = float('inf')
            min_idx = 0
            for i in range(len(lines) - 1):
                combined = len(lines[i]) + len(lines[i + 1])
                if combined < min_combined_len:
                    min_combined_len = combined
                    min_idx = i
            lines[min_idx] = lines[min_idx] + lines[min_idx + 1]
            lines.pop(min_idx + 1)
        
        return lines

    def _balance_lines(self, lines: List[str], is_chinese: bool) -> List[str]:
        """Try to balance two lines to be similar length"""
        if len(lines) != 2:
            return lines
        
        line1, line2 = lines
        total = line1 + line2
        
        # If difference is small, keep as is
        if abs(len(line1) - len(line2)) < 5:
            return lines
        
        # Re-split at middle
        return self._split_at_middle(total, is_chinese)

    def process_batch(
        self, 
        subtitles: List[str], 
        target_lang: str = "zh"
    ) -> List[LengthControlResult]:
        """Process multiple subtitles"""
        return [self.process(sub, target_lang) for sub in subtitles]


# Convenience function
def control_subtitle_length(
    subtitle: str,
    max_chars: int = 42,
    max_lines: int = 2,
    target_lang: str = "zh"
) -> str:
    """
    Quick function to control subtitle length.
    
    Args:
        subtitle: Input subtitle text
        max_chars: Maximum characters per line
        max_lines: Maximum number of lines
        target_lang: Target language
        
    Returns:
        Processed subtitle text
    """
    config = LengthControlConfig(
        max_chars_per_line=max_chars,
        max_lines=max_lines
    )
    controller = SubtitleLengthController(config)
    result = controller.process(subtitle, target_lang)
    return result.processed

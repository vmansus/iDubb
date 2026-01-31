#!/usr/bin/env python3
"""
Post-process SRT file to split long subtitles into shorter, readable segments.

Usage:
    python resegment_srt.py input.srt output.srt [--max-chars 80]
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple


def parse_srt(content: str) -> List[dict]:
    """Parse SRT content into list of subtitle entries."""
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n\d+\n|\n*$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    subtitles = []
    for match in matches:
        idx, start, end, text = match
        subtitles.append({
            'index': int(idx),
            'start': start,
            'end': end,
            'text': text.strip().replace('\n', ' ')
        })
    return subtitles


def time_to_ms(time_str: str) -> int:
    """Convert SRT timestamp to milliseconds."""
    h, m, rest = time_str.split(':')
    s, ms = rest.split(',')
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def ms_to_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp."""
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_part = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_part:03d}"


def split_subtitle(sub: dict, max_chars: int = 80) -> List[dict]:
    """
    Split a long subtitle into shorter segments.
    
    Strategy:
    1. Split at sentence boundaries (. ! ?)
    2. If still too long, split at clause boundaries (, ; :)
    3. Distribute time proportionally
    """
    text = sub['text']
    
    if len(text) <= max_chars:
        return [sub]
    
    start_ms = time_to_ms(sub['start'])
    end_ms = time_to_ms(sub['end'])
    duration = end_ms - start_ms
    
    # Sentence-ending punctuation
    sentence_ends = ['.', '!', '?']
    clause_seps = [',', ';', ':']
    
    segments = []
    remaining = text
    char_offset = 0
    total_chars = len(text)
    
    while remaining:
        if len(remaining) <= max_chars:
            # Last piece
            seg_start = start_ms + int(duration * char_offset / total_chars)
            segments.append({
                'start': ms_to_time(seg_start),
                'end': sub['end'],
                'text': remaining.strip()
            })
            break
        
        # Find best split point
        best_split = -1
        
        # First, look for sentence end within max_chars
        for i, char in enumerate(remaining[:max_chars]):
            if char in sentence_ends and i >= 20:  # At least 20 chars
                best_split = i + 1
        
        # If no sentence end, look for clause separator
        if best_split < max_chars // 2:
            for i, char in enumerate(remaining[:max_chars]):
                if char in clause_seps and i >= max_chars // 3:
                    best_split = max(best_split, i + 1)
        
        # If still no good split, find last space
        if best_split < max_chars // 3:
            space_pos = remaining[:max_chars].rfind(' ')
            if space_pos > max_chars // 3:
                best_split = space_pos + 1
        
        # Fallback: hard split at max_chars
        if best_split <= 0:
            best_split = max_chars
        
        chunk = remaining[:best_split].strip()
        remaining = remaining[best_split:].strip()
        
        # Calculate proportional time
        seg_start = start_ms + int(duration * char_offset / total_chars)
        char_offset += len(chunk)
        seg_end = start_ms + int(duration * char_offset / total_chars)
        
        # Ensure minimum duration of 1 second
        if seg_end - seg_start < 1000 and remaining:
            seg_end = min(seg_start + 1500, end_ms)
        
        segments.append({
            'start': ms_to_time(seg_start),
            'end': ms_to_time(seg_end),
            'text': chunk
        })
    
    return segments


def resegment_srt(subtitles: List[dict], max_chars: int = 80) -> List[dict]:
    """Resegment all subtitles."""
    result = []
    
    for sub in subtitles:
        split_subs = split_subtitle(sub, max_chars)
        result.extend(split_subs)
    
    # Renumber
    for i, sub in enumerate(result, 1):
        sub['index'] = i
    
    return result


def format_srt(subtitles: List[dict]) -> str:
    """Format subtitles back to SRT format."""
    lines = []
    for sub in subtitles:
        lines.append(f"{sub['index']}")
        lines.append(f"{sub['start']} --> {sub['end']}")
        lines.append(sub['text'])
        lines.append('')
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python resegment_srt.py input.srt output.srt [--max-chars 80]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    max_chars = 80
    
    if '--max-chars' in sys.argv:
        idx = sys.argv.index('--max-chars')
        max_chars = int(sys.argv[idx + 1])
    
    # Read input
    content = input_path.read_text(encoding='utf-8')
    subtitles = parse_srt(content)
    
    print(f"Parsed {len(subtitles)} subtitles from {input_path}")
    
    # Find long ones
    long_count = sum(1 for s in subtitles if len(s['text']) > max_chars)
    print(f"Found {long_count} subtitles longer than {max_chars} chars")
    
    # Resegment
    result = resegment_srt(subtitles, max_chars)
    
    print(f"After resegmentation: {len(result)} subtitles")
    
    # Write output
    output_path.write_text(format_srt(result), encoding='utf-8')
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()

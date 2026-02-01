"""
Subtitle Burning Module - Embed subtitles into video
支持中文等CJK字符，通过指定字体文件解决乱码问题
"""
import asyncio
import subprocess
import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from loguru import logger

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Default font paths for different systems - ordered by preference
# Using list of tuples to maintain priority order
FONT_PATHS_LIST = [
    # macOS - PingFang has excellent CJK coverage and consistent sizing
    ("/System/Library/Fonts/PingFang.ttc", "PingFang SC"),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", "Songti SC"),
    ("/Library/Fonts/Arial Unicode.ttf", "Arial Unicode MS"),
    # Docker/Linux - Noto Sans CJK (best cross-platform CJK font)
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
    ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
    # WenQuanYi fonts (common in Linux)
    ("/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc", "WenQuanYi Micro Hei"),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "WenQuanYi Micro Hei"),
    # Source Han Sans
    ("/usr/share/fonts/opentype/adobe/SourceHanSansSC-Regular.otf", "Source Han Sans SC"),
    # Windows
    ("C:/Windows/Fonts/msyh.ttc", "Microsoft YaHei"),
    ("C:/Windows/Fonts/simsun.ttc", "SimSun"),
    # Project fonts directory (dynamic path)
    (str(DATA_DIR / "fonts/NotoSansCJK-Regular.ttc"), "Noto Sans CJK SC"),
    (str(DATA_DIR / "fonts/wqy-microhei.ttc"), "WenQuanYi Micro Hei"),
]

# Convert to dict for backward compatibility
FONT_PATHS = {path: name for path, name in FONT_PATHS_LIST}


def find_cjk_font() -> tuple:
    """Find available CJK font on the system - uses priority order"""
    for font_path, font_name in FONT_PATHS_LIST:
        if Path(font_path).exists():
            logger.info(f"Found CJK font: {font_name} at {font_path}")
            return font_path, font_name

    # Fallback: try to find any font in common locations
    import platform
    system = platform.system()

    if system == "Darwin":  # macOS
        # Try to find any Chinese font in system
        for fallback in ["/System/Library/Fonts/STHeiti Light.ttc",
                        "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
            if Path(fallback).exists():
                logger.info(f"Found fallback CJK font at {fallback}")
                return fallback, Path(fallback).stem

    logger.warning("No CJK font found, using system default - Chinese characters may display incorrectly")
    return None, "Sans"


@dataclass
class SubtitleStyle:
    """Subtitle styling options"""
    font_name: str = ""  # Will be auto-detected if empty
    font_file: Optional[str] = None  # Path to font file
    font_size: int = 24
    primary_color: str = "&H00FFFFFF"  # White (ASS format: &HAABBGGRR)
    outline_color: str = "&H00000000"  # Black outline
    back_color: str = "&H80000000"  # Semi-transparent background
    back_opacity: int = 0  # Background opacity (0-100), used to determine BorderStyle
    bold: bool = True
    italic: bool = False
    outline_width: int = 2
    shadow: int = 1
    shadow_color: str = "&H00000000"  # Shadow color (ASS format)
    margin_v: int = 30  # Vertical margin from bottom
    margin_h: int = 20  # Horizontal margin
    alignment: int = 2  # Bottom center (ASS alignment: 1-9)
    spacing: int = 0  # Character spacing
    scale_x: int = 100  # Horizontal scale
    scale_y: int = 100  # Vertical scale
    max_width: int = 90  # Maximum width as percentage of video width (50-100)

    def __post_init__(self):
        """Auto-detect font if not specified"""
        if not self.font_name or not self.font_file:
            font_path, font_name = find_cjk_font()
            if not self.font_name:
                self.font_name = font_name
            if not self.font_file:
                self.font_file = font_path

    @staticmethod
    def hex_to_ass_color(hex_color: str) -> str:
        """Convert hex color (#RRGGBB) to ASS format (&H00BBGGRR)

        ASS uses &HAABBGGRR format where AA is alpha (00=opaque, FF=transparent).
        Always output 8-digit format for compatibility with all renderers.
        """
        if not hex_color:
            return "&H00FFFFFF"
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        if len(hex_color) == 6:
            r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
            return f"&H00{b}{g}{r}"
        return "&H00FFFFFF"

    @staticmethod
    def hex_to_ass_back_color(hex_color: str, opacity: int = 0) -> str:
        """Convert hex color and opacity to ASS format (&HAABBGGRR)"""
        if not hex_color:
            hex_color = "#000000"
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        if len(hex_color) == 6:
            r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
            # Convert opacity (0-100) to alpha (00-FF, where 00=opaque, FF=transparent)
            alpha = int((100 - opacity) * 255 / 100)
            return f"&H{alpha:02X}{b}{g}{r}"
        return "&H80000000"

    @staticmethod
    def alignment_to_ass(alignment: str) -> int:
        """Convert alignment string to ASS alignment number"""
        # ASS alignment: 1-3 bottom, 4-6 middle, 7-9 top (left, center, right)
        alignment_map = {
            "top": 8,      # Top center
            "middle": 5,   # Middle center
            "bottom": 2,   # Bottom center
        }
        return alignment_map.get(alignment, 2)

    @classmethod
    def from_settings(cls, style_config: dict, margin_v: int = 30) -> "SubtitleStyle":
        """Create SubtitleStyle from settings config dict"""
        # Debug: log input config
        logger.info(f"[SubtitleStyle.from_settings] Input config: {style_config}")

        # Get alignment and convert to ASS format
        alignment_str = style_config.get("alignment", "bottom")
        alignment = cls.alignment_to_ass(alignment_str)

        # Get back opacity for BorderStyle determination
        back_opacity = style_config.get("back_opacity", 0)

        # Get colors - convert from hex to ASS format
        input_color = style_config.get("color", "#FFFFFF")
        input_outline_color = style_config.get("outline_color", "#000000")
        input_back_color = style_config.get("back_color", "#000000")
        input_shadow_color = style_config.get("shadow_color", "#000000")

        primary_color = cls.hex_to_ass_color(input_color)
        outline_color = cls.hex_to_ass_color(input_outline_color)
        back_color = cls.hex_to_ass_back_color(input_back_color, back_opacity)
        shadow_color = cls.hex_to_ass_color(input_shadow_color)

        max_width_value = style_config.get("max_width", 90)
        logger.info(f"[SubtitleStyle.from_settings] Color conversions: "
                   f"color {input_color} -> {primary_color}, "
                   f"outline {input_outline_color} -> {outline_color}, "
                   f"back {input_back_color}@{back_opacity}% -> {back_color}, "
                   f"shadow {input_shadow_color} -> {shadow_color}")
        logger.info(f"[SubtitleStyle.from_settings] max_width={max_width_value}%, font_size={style_config.get('font_size', 24)}")

        return cls(
            font_name=style_config.get("font_name", ""),
            font_size=style_config.get("font_size", 24),
            primary_color=primary_color,
            outline_color=outline_color,
            back_color=back_color,
            back_opacity=back_opacity,
            outline_width=style_config.get("outline_width", 2),
            bold=style_config.get("bold", True),
            italic=style_config.get("italic", False),
            shadow=style_config.get("shadow", 1),
            shadow_color=shadow_color,
            margin_v=style_config.get("margin_v", margin_v),
            margin_h=style_config.get("margin_h", 20),
            alignment=alignment,
            spacing=style_config.get("spacing", 0),
            scale_x=style_config.get("scale_x", 100),
            scale_y=style_config.get("scale_y", 100),
            max_width=style_config.get("max_width", 90),
        )


def adapt_style_for_vertical(style: SubtitleStyle, video_width: int, video_height: int) -> SubtitleStyle:
    """
    Adapt subtitle style for vertical video dimensions.

    For vertical videos (aspect ratio < 1), we need to:
    1. Reduce font size proportionally (narrower screen)
    2. Increase vertical margin to stay in lower third (avoid faces)
    3. Increase horizontal margin for better readability

    Args:
        style: Original SubtitleStyle
        video_width: Video width in pixels
        video_height: Video height in pixels

    Returns:
        Adapted SubtitleStyle for vertical video
    """
    if video_height == 0:
        return style

    aspect_ratio = video_width / video_height

    # Only adapt if video is vertical (aspect ratio < 0.9)
    if aspect_ratio >= 0.9:
        return style

    # Calculate adaptation factors
    # For 9:16 (0.5625), we want roughly 75% of normal font size
    # For 3:4 (0.75), we want roughly 85% of normal font size
    font_scale = 0.6 + (aspect_ratio * 0.4)  # Maps 0.5-0.9 to 0.8-0.96
    font_scale = max(0.7, min(1.0, font_scale))

    # Vertical margin should be higher for vertical videos (to stay in lower third)
    # Standard 16:9 videos use margin_v around 30-70
    # Vertical 9:16 should use margin_v around 100-200
    margin_scale = 1.5 + (1 - aspect_ratio)  # Maps 0.5-0.9 to 2.0-1.6

    logger.info(f"Adapting subtitle style for vertical video: "
               f"aspect_ratio={aspect_ratio:.2f}, font_scale={font_scale:.2f}, "
               f"margin_scale={margin_scale:.2f}")

    # Create new style with adapted values
    from dataclasses import replace
    return replace(
        style,
        font_size=max(12, int(style.font_size * font_scale)),
        margin_v=min(200, int(style.margin_v * margin_scale)),
        margin_h=max(style.margin_h, int(30 * (1 / aspect_ratio))),  # More padding for narrow screens
    )


class SubtitleBurner:
    """Burn subtitles into video using FFmpeg"""

    # Preferred FFmpeg paths (ffmpeg-full has libass for subtitle support)
    FFMPEG_PATHS = [
        "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",  # macOS Homebrew ffmpeg-full (Apple Silicon)
        "/usr/local/opt/ffmpeg-full/bin/ffmpeg",     # macOS Homebrew ffmpeg-full (Intel)
        "ffmpeg",                                     # System default
    ]

    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()
        # Find CJK font at initialization
        self.default_font_path, self.default_font_name = find_cjk_font()

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg with subtitle support (libass)"""
        for ffmpeg_path in self.FFMPEG_PATHS:
            try:
                # Check if ffmpeg exists and has subtitles filter
                result = subprocess.run(
                    [ffmpeg_path, "-filters"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and "subtitles" in result.stdout:
                    logger.info(f"Found FFmpeg with subtitle support: {ffmpeg_path}")
                    return ffmpeg_path
            except FileNotFoundError:
                continue

        # Fallback to system ffmpeg (may not have subtitle support)
        logger.warning("FFmpeg with subtitle support (libass) not found. "
                      "Install ffmpeg-full: brew install ffmpeg-full")
        return "ffmpeg"

    def _check_videotoolbox_support(self) -> bool:
        """Check if VideoToolbox hardware acceleration is available (macOS)"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-encoders"],
                capture_output=True,
                text=True
            )
            if "h264_videotoolbox" in result.stdout:
                logger.info("VideoToolbox hardware acceleration available")
                return True
        except Exception:
            pass
        logger.info("VideoToolbox not available, using CPU encoding")
        return False

    def _escape_ffmpeg_filter_path(self, path: str) -> str:
        """
        Escape a path for use in FFmpeg filter syntax.

        FFmpeg filter uses these special characters:
        - : as option delimiter
        - ' as string delimiter
        - \ as escape character
        - [ ] for filter graphs
        - , for filter chain separator

        All of these must be escaped with backslash.
        """
        # Convert Windows backslashes to forward slashes first
        path = path.replace("\\", "/")

        # Escape special characters for FFmpeg filter syntax
        # Order matters: escape backslash first if any remain
        path = path.replace("'", r"\'")
        path = path.replace(":", r"\:")
        path = path.replace("[", r"\[")
        path = path.replace("]", r"\]")
        # Don't escape comma here - it's handled at filter chain level

        return path

    def _create_styled_ass(
        self,
        srt_path: Path,
        output_path: Path,
        style: SubtitleStyle
    ) -> bool:
        """
        Convert SRT to ASS with embedded styles for fast rendering.

        Using SRT with force_style is extremely slow because FFmpeg applies
        styling at runtime. Converting to ASS with embedded styles makes
        subtitle rendering 10-100x faster.

        Args:
            srt_path: Input SRT file
            output_path: Output ASS file
            style: Subtitle styling

        Returns:
            True if successful
        """
        import re

        try:
            content = srt_path.read_text(encoding='utf-8')

            # Parse SRT format
            pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n\d+\n|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)

            if not matches:
                logger.warning(f"No subtitles found in {srt_path}")
                return False

            def srt_time_to_ass(time_str: str) -> str:
                """Convert SRT time (00:01:23,456) to ASS time (0:01:23.45)"""
                time_str = time_str.replace(',', '.')
                parts = time_str.rsplit('.', 1)
                if len(parts) == 2:
                    time_str = parts[0] + '.' + parts[1][:2]
                if time_str.startswith('0'):
                    time_str = time_str[1:]
                return time_str

            # Calculate horizontal margin based on max_width
            DEFAULT_PLAY_RES_X = 384
            max_width_pct = getattr(style, 'max_width', 90)
            if max_width_pct < 100:
                margin_h = int(DEFAULT_PLAY_RES_X * (100 - max_width_pct) / 2 / 100)
                margin_h = max(margin_h, style.margin_h)
            else:
                margin_h = style.margin_h

            # Determine border style based on back_opacity
            if style.back_opacity > 0:
                border_style = 3  # Opaque box
                box_outline = max(4, style.outline_width)
                actual_outline_color = style.back_color
                actual_shadow = style.shadow
            else:
                border_style = 1  # Outline + shadow
                box_outline = style.outline_width
                actual_outline_color = style.outline_color
                actual_shadow = style.shadow

            # Build ASS header
            ass_content = f"""[Script Info]
Title: Styled Subtitle
ScriptType: v4.00+
WrapStyle: 0
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},{style.primary_color},&H000000FF,{actual_outline_color},{style.back_color},{-1 if style.bold else 0},{-1 if style.italic else 0},0,0,{style.scale_x},{style.scale_y},{style.spacing},0,{border_style},{box_outline},{actual_shadow},{style.alignment},{margin_h},{margin_h},{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

            # Calculate max characters per line based on max_width
            # Estimate: Chinese chars ~2x width of Latin chars, avg ~1.5 chars per em
            # For font_size and max_width%, estimate chars that fit
            # Conservative estimate: 1080p video, font_size 24-28 = ~40-50 Chinese chars at 100% width
            base_chars = 45  # Approximate Chinese chars at 100% width for typical font
            max_chars_per_line = int(base_chars * max_width_pct / 100)
            max_chars_per_line = max(10, min(max_chars_per_line, 100))  # Clamp to reasonable range
            logger.debug(f"[ASS] max_width={max_width_pct}%, max_chars_per_line={max_chars_per_line}")

            def wrap_text(text: str, max_chars: int) -> str:
                """Wrap text to fit within max characters per line"""
                lines = []
                current_line = ""
                # Split by existing line breaks first
                for segment in text.split('\n'):
                    words = list(segment)  # Split into characters for CJK
                    for char in words:
                        if len(current_line) >= max_chars:
                            lines.append(current_line)
                            current_line = char
                        else:
                            current_line += char
                    if current_line:
                        lines.append(current_line)
                        current_line = ""
                if current_line:
                    lines.append(current_line)
                return '\\N'.join(lines)

            events = []
            for _, start, end, text in matches:
                start_ass = srt_time_to_ass(start)
                end_ass = srt_time_to_ass(end)
                # Always apply max_width wrapping
                text_clean = wrap_text(text.strip(), max_chars_per_line)
                logger.debug(f"[ASS] Original: '{text.strip()[:30]}...' -> Wrapped: '{text_clean[:30]}...' (max_chars={max_chars_per_line})")
                events.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text_clean}")

            ass_content += '\n'.join(events)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(ass_content, encoding='utf-8')

            logger.info(f"Created styled ASS file: {output_path} ({len(matches)} subtitles)")
            return True

        except Exception as e:
            logger.error(f"Failed to create styled ASS: {e}")
            return False

    def _build_subtitle_filter(
        self,
        subtitle_path: Path,
        style: SubtitleStyle = None
    ) -> str:
        """
        Build FFmpeg subtitle filter string with proper CJK font support

        Uses fontsdir to specify font location, which resolves Chinese character
        rendering issues (garbled/box characters)
        """
        style = style or SubtitleStyle()

        # Escape the subtitle path for FFmpeg filter syntax
        escaped_path = self._escape_ffmpeg_filter_path(str(subtitle_path))

        # Build ASS style override with all style parameters
        force_style = (
            f"FontName={style.font_name},"
            f"FontSize={style.font_size},"
            f"PrimaryColour={style.primary_color},"
            f"OutlineColour={style.outline_color},"
            f"BackColour={style.back_color},"
            f"Bold={1 if style.bold else 0},"
            f"Italic={1 if style.italic else 0},"
            f"Outline={style.outline_width},"
            f"Shadow={style.shadow},"
            f"MarginL={style.margin_h},"
            f"MarginR={style.margin_h},"
            f"MarginV={style.margin_v},"
            f"Alignment={style.alignment},"
            f"ScaleX={style.scale_x},"
            f"ScaleY={style.scale_y},"
            f"Spacing={style.spacing}"
        )

        # Build the filter
        # FFmpeg filter syntax: subtitles=filename:force_style='style'
        # Note: force_style value should be quoted with single quotes
        filter_str = f"subtitles={escaped_path}:force_style='{force_style}'"

        # Add fontsdir if we have a font file
        if style.font_file and Path(style.font_file).exists():
            escaped_fonts_dir = self._escape_ffmpeg_filter_path(str(Path(style.font_file).parent))
            filter_str = f"subtitles={escaped_path}:fontsdir={escaped_fonts_dir}:force_style='{force_style}'"

        return filter_str

    async def burn_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
        style: SubtitleStyle = None
    ) -> bool:
        """
        Burn subtitles into video

        Args:
            video_path: Input video file
            subtitle_path: Subtitle file (SRT, ASS, VTT)
            output_path: Output video file
            style: Subtitle styling options

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            style = style or SubtitleStyle()

            # Fix subtitle overlaps before burning (common in YouTube auto-generated subs)
            import tempfile
            import shutil
            temp_dir = Path(tempfile.mkdtemp())
            fixed_subtitle = temp_dir / f"fixed_{subtitle_path.name}"
            self.fix_subtitle_overlap(subtitle_path, fixed_subtitle)
            
            # Resegment long subtitles for better readability
            # NOTE: Disabled - resegmentation is now done in transcriber
            # resegmented_subtitle = temp_dir / f"resegmented_{subtitle_path.name}"
            # self._resegment_srt_file(fixed_subtitle, resegmented_subtitle, max_chars=80)
            # if resegmented_subtitle.exists():
            #     fixed_subtitle = resegmented_subtitle

            # === OPTIMIZATION: Convert SRT to ASS with embedded styles ===
            # Using force_style with SRT is extremely slow (100x+ slower)
            # Converting to ASS first makes subtitle rendering much faster
            styled_ass = temp_dir / "styled_subtitle.ass"
            ass_created = self._create_styled_ass(fixed_subtitle, styled_ass, style)

            if ass_created and styled_ass.exists():
                # Use optimized ASS path (same as dual subtitles)
                logger.info("Using optimized ASS-based single subtitle rendering")
                escaped_ass_path = self._escape_ffmpeg_filter_path(str(styled_ass))

                # Add fontsdir if we have a font file
                fonts_dir_option = ""
                if style.font_file and Path(style.font_file).exists():
                    escaped_fonts_dir = self._escape_ffmpeg_filter_path(str(Path(style.font_file).parent))
                    fonts_dir_option = f":fontsdir={escaped_fonts_dir}"

                subtitle_filter = f"subtitles={escaped_ass_path}{fonts_dir_option}"
            else:
                # Fallback to force_style (slow but works)
                logger.warning("ASS conversion failed, falling back to force_style (slower)")
                subtitle_filter = self._build_subtitle_filter(fixed_subtitle, style)

            # 检测是否支持硬件加速
            use_hw_accel = self._check_videotoolbox_support()

            if use_hw_accel:
                cmd = [
                    self.ffmpeg_path, "-y",
                    "-hwaccel", "videotoolbox",
                    "-hwaccel_output_format", "nv12",
                    "-i", str(video_path),
                    "-vf", subtitle_filter,
                    "-c:v", "h264_videotoolbox",
                    "-q:v", "75",  # Increased from 65 for smaller file size
                    "-maxrate", "4M",  # Limit max bitrate to prevent file bloat
                    "-bufsize", "8M",
                    "-allow_sw", "1",
                    "-realtime", "0",
                    "-c:a", "copy",
                    str(output_path)
                ]
            else:
                cmd = [
                    self.ffmpeg_path, "-y",
                    "-threads", "0",
                    "-i", str(video_path),
                    "-vf", subtitle_filter,
                    "-c:v", "libx264",
                    "-preset", "medium",  # Changed from fast for better compression
                    "-crf", "28",  # Changed from 23 for smaller file size
                    "-maxrate", "4M",  # Limit max bitrate to prevent file bloat
                    "-bufsize", "8M",  # Buffer size for rate control
                    "-c:a", "copy",
                    str(output_path)
                ]

            logger.debug(f"FFmpeg subtitle burn command: {' '.join(cmd)}")

            def run_burn():
                return subprocess.run(cmd, capture_output=True, text=True)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_burn)

            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)

            if result.returncode != 0:
                logger.error(f"Subtitle burn failed: {result.stderr}")
                return False

            logger.info(f"Burned subtitles into: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Subtitle burn failed: {e}")
            # Cleanup temp files if they exist
            if 'temp_dir' in locals():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            return False

    def calculate_dual_subtitle_positions(
        self,
        top_font_size: int,
        bottom_font_size: int,
        base_margin: int = 20
    ) -> tuple:
        """
        Calculate vertical margins for dual subtitles to avoid overlap.

        Args:
            top_font_size: Font size of the top subtitle
            bottom_font_size: Font size of the bottom subtitle
            base_margin: Base margin from screen edge

        Returns:
            (top_margin_v, bottom_margin_v) - margins for top and bottom subtitles
        """
        # Bottom subtitle margin (from screen bottom)
        bottom_margin_v = base_margin

        # Top subtitle needs enough margin to clear the bottom subtitle
        # Add some padding (1.5x font size for line height + extra spacing)
        spacing = int(bottom_font_size * 1.8) + 10
        top_margin_v = bottom_margin_v + spacing

        return top_margin_v, bottom_margin_v

    async def burn_dual_subtitles(
        self,
        video_path: Path,
        original_subtitle: Path,
        translated_subtitle: Path,
        output_path: Path,
        original_style: SubtitleStyle = None,
        translated_style: SubtitleStyle = None,
        chinese_on_top: bool = True,
        video_width: int = 0,
        video_height: int = 0
    ) -> bool:
        """
        Burn dual language subtitles with configurable positioning.

        Args:
            video_path: Input video
            original_subtitle: Original language subtitle file
            translated_subtitle: Translated subtitle file
            output_path: Output video
            original_style: Style for original subtitles
            translated_style: Style for translated subtitles
            chinese_on_top: If True, translated (Chinese) subtitle is on top
            video_width: Video width in pixels (for vertical video optimization)
            video_height: Video height in pixels (for vertical video optimization)
        """
        try:
            # Default styles for dual subtitles
            if original_style is None:
                original_style = SubtitleStyle(
                    font_size=20,
                    primary_color="&HCCCCCC",  # Light gray
                    bold=False,
                )

            if translated_style is None:
                translated_style = SubtitleStyle(
                    font_size=26,
                    primary_color="&HFFFFFF",  # White
                    bold=True,
                )

            # Determine which is on top based on chinese_on_top setting
            if chinese_on_top:
                # Chinese (translated) on top, original (English) on bottom
                top_style = translated_style
                bottom_style = original_style
                top_subtitle = translated_subtitle
                bottom_subtitle = original_subtitle
            else:
                # Original (English) on top, Chinese (translated) on bottom
                top_style = original_style
                bottom_style = translated_style
                top_subtitle = original_subtitle
                bottom_subtitle = translated_subtitle

            # Use preset's bottom margin_v, then calculate top margin to avoid overlap
            # This matches the frontend preview calculation in SubtitlePreview.tsx
            if bottom_style.margin_v != 30:
                # Preset has custom bottom margin, use it
                bottom_margin = bottom_style.margin_v
            else:
                # Use default
                bottom_margin = 20

            # Calculate top margin to match frontend preview:
            # top_margin = bottom_margin + bottom_font_size + padding (if background) + gap
            bottom_font_size = bottom_style.font_size
            # Check if bottom subtitle has background (back_color opacity > 0)
            # In ASS format, back_color has alpha in high byte, but SubtitleStyle stores back_color as &HAABBGGRR
            # For simplicity, just add a small padding buffer
            padding = 8  # Similar to frontend's padding * 2
            gap = 10  # Gap between subtitles (matches frontend)
            top_margin = bottom_margin + bottom_font_size + padding + gap

            # Update styles with calculated margins
            bottom_style.margin_v = bottom_margin
            top_style.margin_v = top_margin

            logger.info(f"Dual subtitles: top_margin={top_margin}, bottom_margin={bottom_margin}, "
                       f"chinese_on_top={chinese_on_top}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create temp directory for processing
            import tempfile
            temp_dir = Path(tempfile.mkdtemp())

            # Fix subtitle overlaps before merging
            fixed_original = temp_dir / "fixed_original.srt"
            fixed_translated = temp_dir / "fixed_translated.srt"

            self.fix_subtitle_overlap(original_subtitle, fixed_original)
            self.fix_subtitle_overlap(translated_subtitle, fixed_translated)

            # === OPTIMIZATION: Merge to single ASS with dual styles ===
            # This reduces FFmpeg filter processing from 2x to 1x (30-40% faster)
            merged_ass = temp_dir / "merged_dual.ass"

            merge_success = self.merge_srt_to_dual_ass(
                fixed_original,
                fixed_translated,
                merged_ass,
                original_style=original_style,
                translated_style=translated_style,
                translated_on_top=chinese_on_top,
                video_width=video_width,
                video_height=video_height
            )

            if merge_success and merged_ass.exists():
                # Use single ASS filter (optimized path)
                logger.info("Using optimized single-ASS dual subtitle rendering")
                escaped_ass_path = self._escape_ffmpeg_filter_path(str(merged_ass))

                # Add fontsdir if we have a font file
                fonts_dir_option = ""
                for style in [original_style, translated_style]:
                    if style.font_file and Path(style.font_file).exists():
                        escaped_fonts_dir = self._escape_ffmpeg_filter_path(str(Path(style.font_file).parent))
                        fonts_dir_option = f":fontsdir={escaped_fonts_dir}"
                        break

                subtitle_filter = f"subtitles={escaped_ass_path}{fonts_dir_option}"
                use_filter_complex = False
            else:
                # Fallback: use dual filter chain (original method)
                logger.warning("ASS merge failed, falling back to dual filter chain")
                filter1 = self._build_subtitle_filter(fixed_original, bottom_style)
                filter2 = self._build_subtitle_filter(fixed_translated, top_style)
                if not chinese_on_top:
                    filter1, filter2 = filter2, filter1
                subtitle_filter = f"[0:v]{filter1}[v1];[v1]{filter2}[vout]"
                use_filter_complex = True

            # Build FFmpeg command
            use_hw_accel = self._check_videotoolbox_support()

            if use_hw_accel:
                if use_filter_complex:
                    cmd = [
                        self.ffmpeg_path, "-y",
                        "-hwaccel", "videotoolbox",
                        "-hwaccel_output_format", "nv12",
                        "-i", str(video_path),
                        "-filter_complex", subtitle_filter,
                        "-map", "[vout]",
                        "-map", "0:a?",
                        "-c:v", "h264_videotoolbox",
                        "-q:v", "75",
                        "-maxrate", "4M",  # Limit max bitrate
                        "-bufsize", "8M",
                        "-allow_sw", "1",
                        "-realtime", "0",
                        "-c:a", "copy",
                        str(output_path)
                    ]
                else:
                    # Optimized single filter path
                    cmd = [
                        self.ffmpeg_path, "-y",
                        "-hwaccel", "videotoolbox",
                        "-hwaccel_output_format", "nv12",
                        "-i", str(video_path),
                        "-vf", subtitle_filter,
                        "-c:v", "h264_videotoolbox",
                        "-q:v", "75",
                        "-maxrate", "4M",  # Limit max bitrate
                        "-bufsize", "8M",
                        "-allow_sw", "1",
                        "-realtime", "0",
                        "-c:a", "copy",
                        str(output_path)
                    ]
            else:
                # CPU encoding fallback
                if use_filter_complex:
                    cmd = [
                        self.ffmpeg_path, "-y",
                        "-threads", "0",
                        "-i", str(video_path),
                        "-filter_complex", subtitle_filter,
                        "-map", "[vout]",
                        "-map", "0:a?",
                        "-c:v", "libx264",
                        "-preset", "medium",
                        "-crf", "28",
                        "-maxrate", "4M",  # Limit max bitrate
                        "-bufsize", "8M",
                        "-c:a", "copy",
                        str(output_path)
                    ]
                else:
                    # Optimized single filter path
                    cmd = [
                        self.ffmpeg_path, "-y",
                        "-threads", "0",
                        "-i", str(video_path),
                        "-vf", subtitle_filter,
                        "-c:v", "libx264",
                        "-preset", "medium",
                        "-crf", "28",
                        "-maxrate", "4M",  # Limit max bitrate
                        "-bufsize", "8M",
                        "-c:a", "copy",
                        str(output_path)
                    ]

            logger.debug(f"FFmpeg dual subtitle command: {' '.join(cmd)}")

            def run_burn():
                return subprocess.run(cmd, capture_output=True, text=True)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_burn)

            if result.returncode != 0:
                logger.error(f"Dual subtitle burn failed: {result.stderr}")
                # Cleanup temp files
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False

            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info(f"Burned dual subtitles into: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Dual subtitle burn failed: {e}")
            # Cleanup temp files if they exist
            if 'temp_dir' in locals():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            return False

    async def add_soft_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
        language: str = "chi"
    ) -> bool:
        """
        Add soft subtitles (can be toggled on/off by player)

        Args:
            video_path: Input video
            subtitle_path: Subtitle file
            output_path: Output video
            language: Subtitle language code
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                self.ffmpeg_path, "-y",
                "-i", str(video_path),
                "-i", str(subtitle_path),
                "-c:v", "copy",
                "-c:a", "copy",
                "-c:s", "mov_text",
                "-metadata:s:s:0", f"language={language}",
                str(output_path)
            ]

            def run_add():
                return subprocess.run(cmd, capture_output=True, text=True)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_add)

            if result.returncode != 0:
                logger.error(f"Soft subtitle add failed: {result.stderr}")
                return False

            logger.info(f"Added soft subtitles to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Soft subtitle add failed: {e}")
            return False

    def fix_subtitle_overlap(self, srt_path: Path, output_path: Path = None) -> Path:
        """
        Fix overlapping subtitle timings in SRT file.

        YouTube auto-generated subtitles often have overlapping times where
        the next subtitle starts before the previous one ends.
        This function adjusts end times to prevent overlap.

        Args:
            srt_path: Input SRT file with potential overlaps
            output_path: Output path (defaults to overwriting input)

        Returns:
            Path to the fixed SRT file
        """
        import re

        output_path = output_path or srt_path

        try:
            content = srt_path.read_text(encoding='utf-8')

            # Parse SRT format
            pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n\d+\n|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)

            if not matches:
                logger.warning(f"No subtitles found in {srt_path}")
                return srt_path

            def time_to_ms(time_str: str) -> int:
                """Convert SRT time string to milliseconds"""
                h, m, s_ms = time_str.split(':')
                s, ms = s_ms.split(',')
                return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

            def ms_to_time(ms: int) -> str:
                """Convert milliseconds to SRT time string"""
                h = ms // 3600000
                m = (ms % 3600000) // 60000
                s = (ms % 60000) // 1000
                ms_part = ms % 1000
                return f"{h:02d}:{m:02d}:{s:02d},{ms_part:03d}"

            # Parse and fix overlaps
            subtitles = []
            for idx, start, end, text in matches:
                subtitles.append({
                    'idx': int(idx),
                    'start_ms': time_to_ms(start),
                    'end_ms': time_to_ms(end),
                    'text': text.strip()
                })

            # Sort by start time
            subtitles.sort(key=lambda x: x['start_ms'])

            # Fix overlaps: if next subtitle starts before current ends,
            # adjust current end time to just before next starts
            fixed_count = 0
            min_gap = 50  # Minimum gap between subtitles in ms

            for i in range(len(subtitles) - 1):
                current = subtitles[i]
                next_sub = subtitles[i + 1]

                if current['end_ms'] > next_sub['start_ms'] - min_gap:
                    # Overlap detected - adjust end time
                    new_end = next_sub['start_ms'] - min_gap
                    if new_end > current['start_ms'] + 100:  # Ensure at least 100ms duration
                        current['end_ms'] = new_end
                        fixed_count += 1
                    else:
                        # Very short subtitle, just end it slightly before next
                        current['end_ms'] = current['start_ms'] + 100
                        fixed_count += 1

            if fixed_count > 0:
                logger.info(f"Fixed {fixed_count} overlapping subtitles in {srt_path.name}")

            # Write fixed SRT
            lines = []
            for i, sub in enumerate(subtitles, 1):
                lines.append(str(i))
                lines.append(f"{ms_to_time(sub['start_ms'])} --> {ms_to_time(sub['end_ms'])}")
                lines.append(sub['text'])
                lines.append('')

            output_path.write_text('\n'.join(lines), encoding='utf-8')
            return output_path

        except Exception as e:
            logger.error(f"Failed to fix subtitle overlap: {e}")
            return srt_path

    def _resegment_srt_file(self, srt_path: Path, output_path: Path, max_chars: int = 80) -> Path:
        """
        Resegment long subtitles into shorter, readable segments.
        
        Split at sentence boundaries first, then clause boundaries.
        This ensures subtitles are grammatically complete and readable.
        
        Args:
            srt_path: Input SRT file
            output_path: Output path for resegmented SRT
            max_chars: Maximum characters per subtitle segment
            
        Returns:
            Path to the resegmented SRT file
        """
        import re
        
        try:
            content = srt_path.read_text(encoding='utf-8')
            
            # Parse SRT format
            pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n\d+\n|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            if not matches:
                return srt_path
            
            def time_to_ms(time_str: str) -> int:
                h, m, s_ms = time_str.split(':')
                s, ms = s_ms.split(',')
                return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
            
            def ms_to_time(ms: int) -> str:
                h = ms // 3600000
                m = (ms % 3600000) // 60000
                s = (ms % 60000) // 1000
                ms_part = ms % 1000
                return f"{h:02d}:{m:02d}:{s:02d},{ms_part:03d}"
            
            # Parse subtitles
            subtitles = []
            for idx, start, end, text in matches:
                subtitles.append({
                    'start_ms': time_to_ms(start),
                    'end_ms': time_to_ms(end),
                    'text': text.strip().replace('\n', ' ')
                })
            
            # Resegment
            result = []
            split_count = 0
            
            for sub in subtitles:
                text = sub['text']
                
                if len(text) <= max_chars:
                    result.append(sub)
                    continue
                
                # Split long subtitle
                split_count += 1
                segments = self._split_text_by_sentences(
                    text, 
                    sub['start_ms'], 
                    sub['end_ms'], 
                    max_chars
                )
                result.extend(segments)
            
            if split_count > 0:
                logger.info(f"Resegmented {split_count} long subtitles ({len(subtitles)} -> {len(result)} total)")
            
            # Write output
            lines = []
            for i, sub in enumerate(result, 1):
                lines.append(str(i))
                lines.append(f"{ms_to_time(sub['start_ms'])} --> {ms_to_time(sub['end_ms'])}")
                lines.append(sub['text'])
                lines.append('')
            
            output_path.write_text('\n'.join(lines), encoding='utf-8')
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to resegment SRT: {e}")
            return srt_path
    
    def _split_text_by_sentences(
        self, 
        text: str, 
        start_ms: int, 
        end_ms: int, 
        max_chars: int
    ) -> list:
        """
        Split text at sentence/clause boundaries with proportional timing.
        
        Priority:
        1. Sentence-ending punctuation (. ! ? 。！？)
        2. Clause-separating punctuation (, ; : ，；：)
        3. Word boundaries (space)
        """
        duration = end_ms - start_ms
        total_chars = len(text)
        
        # Punctuation marks
        sentence_ends = ['.', '!', '?', '。', '！', '？']
        clause_seps = [',', ';', ':', '，', '；', '：', '、']
        
        result = []
        start_idx = 0
        
        while start_idx < total_chars:
            remaining = text[start_idx:]
            
            if len(remaining) <= max_chars:
                # Last piece
                seg_start = start_ms + int(duration * start_idx / total_chars)
                result.append({
                    'start_ms': seg_start,
                    'end_ms': end_ms,
                    'text': remaining.strip()
                })
                break
            
            # Find best split point within max_chars
            search_text = remaining[:max_chars]
            best_split = -1
            
            # First, look for sentence-ending punctuation
            for punct in sentence_ends:
                pos = search_text.rfind(punct)
                if pos >= 20:  # At least 20 chars before split
                    best_split = max(best_split, pos + 1)
            
            # If no sentence end found, look for clause separator
            if best_split < max_chars // 2:
                for punct in clause_seps:
                    pos = search_text.rfind(punct)
                    if pos >= max_chars // 3:
                        best_split = max(best_split, pos + 1)
            
            # If still no good split, find last space
            if best_split < max_chars // 3:
                space_pos = search_text.rfind(' ')
                if space_pos >= max_chars // 3:
                    best_split = space_pos + 1
            
            # Fallback: hard split at max_chars
            if best_split <= 0:
                best_split = max_chars
            
            chunk = remaining[:best_split].strip()
            
            # Calculate proportional timing
            seg_start = start_ms + int(duration * start_idx / total_chars)
            char_end = start_idx + best_split
            seg_end = start_ms + int(duration * char_end / total_chars)
            
            # Ensure minimum duration of 1 second
            if seg_end - seg_start < 1000:
                seg_end = min(seg_start + 1500, end_ms)
            
            if chunk:
                result.append({
                    'start_ms': seg_start,
                    'end_ms': seg_end,
                    'text': chunk
                })
            
            start_idx += best_split
        
        return result

    def merge_srt_to_dual_ass(
        self,
        original_srt: Path,
        translated_srt: Path,
        output_path: Path,
        original_style: SubtitleStyle = None,
        translated_style: SubtitleStyle = None,
        translated_on_top: bool = True,
        video_width: int = 0,
        video_height: int = 0
    ) -> bool:
        """
        Merge two SRT files into a single ASS file with dual styles.
        This optimization reduces FFmpeg filter processing from 2x to 1x.

        Args:
            original_srt: Original language SRT file
            translated_srt: Translated language SRT file
            output_path: Output ASS file path
            original_style: Style for original subtitles
            translated_style: Style for translated subtitles
            translated_on_top: If True, translated subtitle appears above original
            video_width: Video width in pixels (for vertical video optimization)
            video_height: Video height in pixels (for vertical video optimization)

        Returns:
            True if successful
        """
        import re

        try:
            original_style = original_style or SubtitleStyle(
                font_size=20, primary_color="&HCCCCCC", bold=False
            )
            translated_style = translated_style or SubtitleStyle(
                font_size=26, primary_color="&HFFFFFF", bold=True
            )

            # Debug: Log all style parameters
            logger.debug(f"[DEBUG] Original style: font_name={original_style.font_name}, font_size={original_style.font_size}, "
                       f"primary_color={original_style.primary_color}, outline_color={original_style.outline_color}, "
                       f"outline_width={original_style.outline_width}, back_color={original_style.back_color}, "
                       f"back_opacity={original_style.back_opacity}, bold={original_style.bold}, shadow={original_style.shadow}")
            logger.debug(f"[DEBUG] Translated style: font_name={translated_style.font_name}, font_size={translated_style.font_size}, "
                       f"primary_color={translated_style.primary_color}, outline_color={translated_style.outline_color}, "
                       f"outline_width={translated_style.outline_width}, back_color={translated_style.back_color}, "
                       f"back_opacity={translated_style.back_opacity}, bold={translated_style.bold}, shadow={translated_style.shadow}")

            # Determine which style is on top/bottom
            if translated_on_top:
                # Translated on top, original on bottom
                top_style = translated_style
                bottom_style = original_style
            else:
                # Original on top, translated on bottom
                top_style = original_style
                bottom_style = translated_style

            # Use preset's margin_v values for both styles
            # Always respect the preset value (no special case for default value 30)
            bottom_margin = bottom_style.margin_v
            
            # For top subtitle: if top_style has a different margin_v than bottom, use it
            # Otherwise, calculate to avoid overlap (for dual subtitle positioning)
            if top_style.margin_v != bottom_style.margin_v:
                # Preset explicitly set different margins for top and bottom
                top_margin = top_style.margin_v
                logger.info(f"Using preset margins: top={top_margin}, bottom={bottom_margin}")
            else:
                # Same margin_v for both - calculate top margin to avoid overlap
                # top_margin = bottom_margin + bottom_font_size + padding (if background) + gap
                bottom_font_size = bottom_style.font_size
                padding = 8  # Similar to frontend's padding * 2
                gap = 10  # Gap between subtitles (matches frontend)
                top_margin = bottom_margin + bottom_font_size + padding + gap
                logger.info(f"Calculated margins to avoid overlap: top={top_margin}, bottom={bottom_margin}")

            # Parse SRT pattern
            srt_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n\d+\n|\Z)'

            def parse_srt(srt_path: Path) -> list:
                content = srt_path.read_text(encoding='utf-8')
                matches = re.findall(srt_pattern, content, re.DOTALL)
                return [(start, end, text.strip()) for _, start, end, text in matches]

            def srt_time_to_ass(time_str: str) -> str:
                """Convert SRT time (00:01:23,456) to ASS time (0:01:23.45)"""
                time_str = time_str.replace(',', '.')
                # ASS uses centiseconds, truncate to 2 decimal places
                parts = time_str.rsplit('.', 1)
                if len(parts) == 2:
                    time_str = parts[0] + '.' + parts[1][:2]
                # Remove leading zero from hours if present
                if time_str.startswith('0'):
                    time_str = time_str[1:]
                return time_str

            def has_cjk(text: str) -> bool:
                """Check if text contains CJK (Chinese/Japanese/Korean) characters"""
                for char in text:
                    if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
                        return True
                    if '\u3040' <= char <= '\u30ff':  # Hiragana + Katakana
                        return True
                    if '\uac00' <= char <= '\ud7af':  # Korean Hangul
                        return True
                return False

            def srt_time_to_ms(time_str: str) -> int:
                """Convert SRT time string to milliseconds"""
                time_str = time_str.replace(',', '.')
                parts = time_str.split(':')
                h, m = int(parts[0]), int(parts[1])
                s_ms = parts[2].split('.')
                s = int(s_ms[0])
                ms = int(s_ms[1]) if len(s_ms) > 1 else 0
                return h * 3600000 + m * 60000 + s * 1000 + ms

            def ms_to_ass_time(ms: int) -> str:
                """Convert milliseconds to ASS time format (H:MM:SS.cc)"""
                h = ms // 3600000
                m = (ms % 3600000) // 60000
                s = (ms % 60000) // 1000
                cs = (ms % 1000) // 10  # centiseconds
                return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

            def wrap_cjk_text(text: str, chars_per_line: int) -> str:
                """
                Add line breaks to CJK text for proper wrapping.
                Tries to break at punctuation when possible.
                Never breaks in the middle of English words.

                Args:
                    text: CJK text to wrap
                    chars_per_line: Maximum characters per line

                Returns:
                    Text with \\N line breaks inserted
                """
                import re
                
                # Clean up any existing weird line breaks first
                text = text.replace('\n', '').replace('\\N', '').strip()
                
                if len(text) <= chars_per_line:
                    return text

                # Chinese punctuation marks - good places to break AFTER
                punctuation = '，。！？；：、）》」』】〉…—,.'
                # Punctuation that shouldn't be alone on a line
                orphan_punctuation = '，。！？；：、…'
                
                # Tokenize: split into CJK chars, English words, and punctuation
                # This ensures English words stay together
                tokens = []
                current_token = ""
                current_type = None  # 'cjk', 'latin', 'other'
                
                for char in text:
                    if re.match(r'[\u4e00-\u9fff]', char):
                        char_type = 'cjk'
                    elif re.match(r'[a-zA-Z0-9]', char):
                        char_type = 'latin'
                    else:
                        char_type = 'other'
                    
                    if char_type == current_type or (current_type == 'latin' and char_type == 'other' and char == ' '):
                        # Continue current token (allow space within latin for now, will handle later)
                        if current_type == 'latin' and char == ' ':
                            # Space breaks Latin words
                            if current_token:
                                tokens.append(current_token)
                            tokens.append(char)
                            current_token = ""
                            current_type = None
                        else:
                            current_token += char
                    else:
                        if current_token:
                            tokens.append(current_token)
                        current_token = char
                        current_type = char_type
                
                if current_token:
                    tokens.append(current_token)

                # Now wrap using tokens
                lines = []
                current_line = ""

                for token in tokens:
                    test_line = current_line + token
                    
                    if len(test_line) <= chars_per_line:
                        current_line = test_line
                    else:
                        # Would exceed limit
                        if current_line:
                            # Try to find a good break point in current_line
                            break_pos = -1
                            for j in range(len(current_line) - 1, max(len(current_line) - 8, -1), -1):
                                if current_line[j] in punctuation:
                                    break_pos = j + 1
                                    break
                            
                            if break_pos > 0 and break_pos < len(current_line):
                                lines.append(current_line[:break_pos])
                                current_line = current_line[break_pos:] + token
                            else:
                                lines.append(current_line)
                                current_line = token
                        else:
                            # Token itself is too long (rare), just add it
                            current_line = token

                if current_line:
                    lines.append(current_line)

                # Fix orphan punctuation: if last line is only punctuation, merge with previous
                while len(lines) > 1:
                    last_line = lines[-1].strip()
                    if len(last_line) <= 2 and all(c in orphan_punctuation for c in last_line):
                        lines[-2] = lines[-2] + lines[-1]
                        lines.pop()
                    else:
                        break

                return '\\N'.join(lines)

            def process_cjk_subtitle(start_str: str, end_str: str, text: str, chars_per_line: int) -> list:
                """
                Process CJK subtitle: wrap text and optionally split into time segments.

                For CJK text:
                1. Always add line breaks (\\N) if text exceeds chars_per_line
                2. Only split into sequential time segments if text is VERY long (3+ lines)

                Args:
                    start_str: SRT format start time
                    end_str: SRT format end time
                    text: Subtitle text
                    chars_per_line: Characters per line for wrapping

                Returns:
                    List of (start_ass, end_ass, segment_text) tuples
                """
                # Clean text
                text = text.replace('\n', ' ').replace('\\N', ' ').strip()

                # Short text - no processing needed
                if len(text) <= chars_per_line:
                    return [(srt_time_to_ass(start_str), srt_time_to_ass(end_str), text)]

                # Medium text (1-3 lines) - just wrap, don't split time
                # This covers most subtitles
                max_chars_before_split = chars_per_line * 3
                if len(text) <= max_chars_before_split:
                    wrapped = wrap_cjk_text(text, chars_per_line)
                    return [(srt_time_to_ass(start_str), srt_time_to_ass(end_str), wrapped)]

                # Very long text (4+ lines) - split into time segments
                # Each segment gets 2 lines max for readability
                start_ms = srt_time_to_ms(start_str)
                end_ms = srt_time_to_ms(end_str)
                total_duration = end_ms - start_ms

                chars_per_segment = chars_per_line * 2  # 2 lines per segment
                total_len = len(text)
                num_segments = (total_len + chars_per_segment - 1) // chars_per_segment

                segments = []
                for i in range(num_segments):
                    seg_start_char = i * chars_per_segment
                    seg_end_char = min((i + 1) * chars_per_segment, total_len)
                    seg_text = text[seg_start_char:seg_end_char]

                    # Wrap the segment text
                    wrapped_seg = wrap_cjk_text(seg_text, chars_per_line)

                    # Proportional time allocation
                    time_start = start_ms + int(total_duration * seg_start_char / total_len)
                    time_end = start_ms + int(total_duration * seg_end_char / total_len)

                    # Ensure minimum duration (1000ms for readability)
                    if time_end - time_start < 1000 and i < num_segments - 1:
                        time_end = min(time_start + 1000, end_ms)

                    segments.append((ms_to_ass_time(time_start), ms_to_ass_time(time_end), wrapped_seg))

                logger.debug(f"[DEBUG] Split long CJK subtitle: '{text[:20]}...' ({total_len} chars) "
                           f"-> {len(segments)} segments")
                return segments

            def process_subtitle(start_str: str, end_str: str, text: str, chars_per_line: int, is_cjk: bool) -> list:
                """
                Process subtitle based on language type.

                - English: No processing, ASS WrapStyle handles word wrapping
                - CJK: Manual line breaks needed (ASS WrapStyle doesn't work well for CJK)

                Args:
                    start_str: SRT format start time
                    end_str: SRT format end time
                    text: Subtitle text
                    chars_per_line: Characters per line (for CJK)
                    is_cjk: Whether this is CJK text

                Returns:
                    List of (start_ass, end_ass, segment_text) tuples
                """
                # Clean text - remove existing line breaks
                text = text.replace('\n', ' ').replace('\\N', ' ').strip()

                # Non-CJK (English etc): No processing, ASS WrapStyle handles wrapping
                if not is_cjk:
                    return [(srt_time_to_ass(start_str), srt_time_to_ass(end_str), text)]

                # CJK: Manual line breaks needed (libass doesn't auto-wrap CJK well)
                return process_cjk_subtitle(start_str, end_str, text, chars_per_line)

            # Parse both SRT files
            original_subs = parse_srt(original_srt)
            translated_subs = parse_srt(translated_srt)

            if not original_subs and not translated_subs:
                logger.warning("No subtitles found in either file")
                return False

            # ASS default resolution (when PlayResX/Y not specified)
            # Default 384x288 means font_size is relative to 288p, then scaled to actual video
            DEFAULT_PLAY_RES_X = 384
            DEFAULT_PLAY_RES_Y = 288

            # Calculate scale factors when using custom PlayRes
            # This ensures font sizes and margins look correct at any resolution
            actual_play_res_x = video_width if video_width > 0 else DEFAULT_PLAY_RES_X
            actual_play_res_y = video_height if video_height > 0 else DEFAULT_PLAY_RES_Y
            scale_x = actual_play_res_x / DEFAULT_PLAY_RES_X
            scale_y = actual_play_res_y / DEFAULT_PLAY_RES_Y

            logger.debug(f"[DEBUG] PlayRes scale factors: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}")

            def scale_font_size(size: int) -> int:
                """Scale font size to match PlayRes"""
                return int(size * scale_y)

            def scale_margin_h(margin: int) -> int:
                """Scale horizontal margin to match PlayRes"""
                return int(margin * scale_x)

            def scale_margin_v(margin: int) -> int:
                """Scale vertical margin to match PlayRes"""
                return int(margin * scale_y)

            def scale_outline(width: int) -> int:
                """Scale outline width to match PlayRes"""
                return max(1, int(width * scale_y))

            def calculate_margin_h(style: SubtitleStyle) -> int:
                """Calculate horizontal margin based on max_width percentage"""
                max_width_pct = getattr(style, 'max_width', 90)
                if max_width_pct < 100:
                    # Each side margin = (100% - max_width%) / 2 * actual_play_res_x
                    # max_width takes priority over margin_h
                    margin = int(actual_play_res_x * (100 - max_width_pct) / 2 / 100)
                    logger.info(f"[DEBUG] calculate_margin_h: max_width={max_width_pct}%, "
                               f"play_res_x={actual_play_res_x}, margin={margin}")
                    return margin
                # Only use margin_h when max_width is not set (100%)
                return scale_margin_h(style.margin_h)

            # Build ASS header with styles
            # For background box + text outline, we use dual-layer technique:
            # - Layer 0: Background box style (BorderStyle=3)
            # - Layer 1: Text outline style (BorderStyle=1)
            def build_style_line(name: str, style: SubtitleStyle, margin_v: int, is_outline_layer: bool = False) -> str:
                # BorderStyle: 1 = outline + shadow
                #              3 = opaque box (libass uses OutlineColour for box, not BackColour!)
                #
                # Dual-layer technique for background + outline:
                # - Layer 0 (is_outline_layer=False): BorderStyle=3 for background box
                # - Layer 1 (is_outline_layer=True): BorderStyle=1 for text outline (no shadow to avoid doubling)

                # Calculate horizontal margin based on max_width
                margin_h = calculate_margin_h(style)

                # Scale font size and margins to match PlayRes
                scaled_font_size = scale_font_size(style.font_size)
                scaled_margin_v = scale_margin_v(margin_v)
                scaled_shadow = max(1, int(style.shadow * scale_y)) if style.shadow > 0 else 0

                if style.back_opacity > 0 and not is_outline_layer:
                    # Background box layer (BorderStyle=3)
                    border_style = 3
                    # OutlineColour = box color, Outline = box padding
                    box_outline = scale_outline(max(4, style.outline_width) if style.outline_width > 0 else 4)
                    actual_outline_color = style.back_color  # Box fill color
                    actual_back_color = f"&H80{style.shadow_color[4:]}" if style.shadow > 0 else "&H00000000"
                    actual_shadow = scaled_shadow
                elif style.back_opacity > 0 and is_outline_layer:
                    # Text outline layer on top of background (BorderStyle=1)
                    # Only render outline, no shadow (shadow is on background layer)
                    border_style = 1
                    box_outline = scale_outline(style.outline_width)
                    actual_outline_color = style.outline_color
                    actual_back_color = "&H00000000"  # Transparent background
                    actual_shadow = 0  # No shadow on outline layer
                else:
                    # No background - standard outline + shadow (BorderStyle=1)
                    border_style = 1
                    box_outline = scale_outline(style.outline_width)
                    actual_outline_color = style.outline_color
                    actual_back_color = style.back_color
                    actual_shadow = scaled_shadow

                logger.debug(f"[DEBUG] build_style_line for {name}: font_size={scaled_font_size}, "
                           f"border_style={border_style}, outline={box_outline}, "
                           f"margin_h={margin_h}, margin_v={scaled_margin_v}")

                return (
                    f"Style: {name},{style.font_name},{scaled_font_size},"
                    f"{style.primary_color},&H000000FF,{actual_outline_color},{actual_back_color},"
                    f"{-1 if style.bold else 0},{-1 if style.italic else 0},0,0,"
                    f"{style.scale_x},{style.scale_y},{style.spacing},0,{border_style},"
                    f"{box_outline},{actual_shadow},{style.alignment},"
                    f"{margin_h},{margin_h},{scaled_margin_v},1"
                )

            # Build style lines
            # For styles with back_opacity > 0, we need TWO styles: background box + text outline
            translated_margin_v = top_margin if translated_on_top else bottom_margin
            original_margin_v = bottom_margin if translated_on_top else top_margin

            style_lines = []
            translated_needs_outline_layer = translated_style.back_opacity > 0 and translated_style.outline_width > 0
            original_needs_outline_layer = original_style.back_opacity > 0 and original_style.outline_width > 0

            # Base styles (background box or standard outline)
            translated_style_line = build_style_line("Translated", translated_style, translated_margin_v, is_outline_layer=False)
            original_style_line = build_style_line("Original", original_style, original_margin_v, is_outline_layer=False)
            style_lines.append(translated_style_line)
            style_lines.append(original_style_line)

            # Additional outline layer styles if needed
            if translated_needs_outline_layer:
                translated_outline_style = build_style_line("TranslatedOutline", translated_style, translated_margin_v, is_outline_layer=True)
                style_lines.append(translated_outline_style)
                logger.debug(f"[DEBUG] Created TranslatedOutline style for dual-layer rendering")

            if original_needs_outline_layer:
                original_outline_style = build_style_line("OriginalOutline", original_style, original_margin_v, is_outline_layer=True)
                style_lines.append(original_outline_style)
                logger.debug(f"[DEBUG] Created OriginalOutline style for dual-layer rendering")

            logger.debug(f"[DEBUG] Generated {len(style_lines)} styles")

            # Determine if video is vertical (aspect ratio < 1)
            is_vertical = video_width > 0 and video_height > 0 and (video_width / video_height) < 0.9
            if is_vertical:
                logger.debug(f"[DEBUG] Vertical video detected: {video_width}x{video_height}")

            # Build PlayRes header based on video dimensions
            playres_header = ""
            if video_width > 0 and video_height > 0:
                playres_header = f"PlayResX: {video_width}\nPlayResY: {video_height}\n"
                logger.debug(f"[DEBUG] Setting PlayRes to {video_width}x{video_height}")

            styles_str = '\n'.join(style_lines)
            ass_content = f"""[Script Info]
Title: Dual Subtitle (Merged)
ScriptType: v4.00+
{playres_header}WrapStyle: 0
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles_str}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

            events = []

            # Calculate characters per line for CJK text wrapping
            # English subtitles use ASS WrapStyle for word wrapping (no manual processing)
            #
            # For CJK, we calculate based on:
            # 1. Available width = video_width * max_width_pct / 100
            # 2. Scaled font size = font_size * scale_y (where scale_y = video_height / 288)
            # 3. CJK char width ≈ scaled_font_size (CJK chars are roughly square)
            # 4. chars_per_line = available_width / char_width

            def get_chars_per_line(style: SubtitleStyle) -> int:
                """Calculate how many CJK characters fit per line based on max_width and font_size"""
                max_width_pct = getattr(style, 'max_width', 90)

                # Available width in pixels
                available_width = actual_play_res_x * max_width_pct / 100

                # Scaled font size (font_size is scaled by scale_y to match PlayRes)
                scaled_font_size = style.font_size * scale_y

                # CJK character width ≈ font height (roughly square)
                # Add small safety factor (0.95) for spacing/rendering variations
                char_width = scaled_font_size * 0.95

                # Calculate chars per line
                chars = int(available_width / char_width) if char_width > 0 else 30

                # Reasonable bounds: at least 10 chars, at most 80
                result = max(10, min(chars, 80))

                logger.info(f"[DEBUG] get_chars_per_line: max_width={max_width_pct}%, "
                           f"available_width={available_width:.0f}px, "
                           f"font_size={style.font_size}, scaled={scaled_font_size:.0f}px, "
                           f"char_width={char_width:.0f}px, chars_per_line={result}")
                return result

            translated_chars_per_line = get_chars_per_line(translated_style)
            original_chars_per_line = get_chars_per_line(original_style)

            # Check if text contains CJK - only CJK text gets split
            # English relies on ASS WrapStyle for natural word wrapping
            translated_is_cjk = any(has_cjk(text) for _, _, text in translated_subs[:5]) if translated_subs else False
            original_is_cjk = any(has_cjk(text) for _, _, text in original_subs[:5]) if original_subs else False

            logger.debug(f"[DEBUG] Language detection: translated_is_cjk={translated_is_cjk}, original_is_cjk={original_is_cjk}")

            # Add translated subtitles
            # For dual-layer: Layer 0 = background box, Layer 1 = text outline
            # CJK subtitles get line breaks added; very long ones are split into time segments
            for start, end, text in translated_subs:
                # Process subtitle (wrap CJK text, optionally split very long ones)
                segments = process_subtitle(start, end, text, translated_chars_per_line, translated_is_cjk)
                for seg_start, seg_end, seg_text in segments:
                    # Base layer (background box or standard)
                    events.append(f"Dialogue: 0,{seg_start},{seg_end},Translated,,0,0,0,,{seg_text}")
                    # Outline layer if needed (rendered on top of background)
                    if translated_needs_outline_layer:
                        events.append(f"Dialogue: 1,{seg_start},{seg_end},TranslatedOutline,,0,0,0,,{seg_text}")

            # Add original subtitles
            # Use layers 2,3 to ensure they render below translated subtitles in the visual stack
            for start, end, text in original_subs:
                # Process subtitle (wrap CJK text, optionally split very long ones)
                segments = process_subtitle(start, end, text, original_chars_per_line, original_is_cjk)
                for seg_start, seg_end, seg_text in segments:
                    # Base layer (background box or standard)
                    events.append(f"Dialogue: 2,{seg_start},{seg_end},Original,,0,0,0,,{seg_text}")
                    # Outline layer if needed
                    if original_needs_outline_layer:
                        events.append(f"Dialogue: 3,{seg_start},{seg_end},OriginalOutline,,0,0,0,,{seg_text}")

            ass_content += '\n'.join(events)

            # Write ASS file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(ass_content, encoding='utf-8')

            logger.info(f"Created merged dual ASS file: {output_path} "
                       f"({len(translated_subs)} translated + {len(original_subs)} original subtitles)")
            return True

        except Exception as e:
            logger.error(f"Failed to merge SRT to dual ASS: {e}")
            return False

    def create_ass_from_srt(
        self,
        srt_path: Path,
        output_path: Path,
        style: SubtitleStyle = None
    ) -> bool:
        """
        Convert SRT to ASS format with custom styling

        Args:
            srt_path: Input SRT file
            output_path: Output ASS file
            style: Style settings
        """
        try:
            style = style or SubtitleStyle()

            # Read SRT content
            srt_content = srt_path.read_text(encoding="utf-8")

            # Parse SRT
            import re
            pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)"
            matches = re.findall(pattern, srt_content, re.DOTALL)

            # Build ASS file
            ass_header = f"""[Script Info]
Title: Converted Subtitles
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},{style.primary_color},&H000000FF,{style.outline_color},{style.back_color},{-1 if style.bold else 0},0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow},{style.alignment},10,10,{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
            events = []
            for _, start, end, text in matches:
                # Convert SRT time to ASS time
                start_ass = start.replace(",", ".")
                end_ass = end.replace(",", ".")
                # Remove hour leading zero for ASS format
                start_ass = start_ass[1:] if start_ass.startswith("0") else start_ass
                end_ass = end_ass[1:] if end_ass.startswith("0") else end_ass
                # Clean text
                text_clean = text.strip().replace("\n", "\\N")
                events.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text_clean}")

            ass_content = ass_header + "\n".join(events)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(ass_content, encoding="utf-8")

            logger.info(f"Created ASS file: {output_path}")
            return True

        except Exception as e:
            logger.error(f"ASS conversion failed: {e}")
            return False

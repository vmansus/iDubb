"""
AI Thumbnail Generator

Generates Chinese social media style thumbnails by:
1. Using AI (DeepSeek) to generate catchy Chinese titles
2. Overlaying text and decorations on original thumbnail using Pillow
"""
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import httpx
from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("Pillow not installed, thumbnail generation will be disabled")

from config import settings


@dataclass
class ThumbnailResult:
    """Result of thumbnail generation"""
    success: bool
    output_path: Optional[Path] = None
    title: str = ""
    error: Optional[str] = None
    generated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "title": self.title,
            "error": self.error,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


class ThumbnailGenerator:
    """
    AI-powered thumbnail generator for Chinese social media platforms

    Supported styles:
    - gradient_bar: Bottom gradient bar with eye-catching gold text
    - top_banner: Top banner with colored background
    - full_overlay: Semi-transparent overlay on entire image
    - corner_tag: Corner tag style
    """

    # Default fonts directory
    FONTS_DIR = Path(__file__).parent.parent / "data" / "fonts"

    # Default font for Chinese text (will be auto-detected)
    DEFAULT_FONTS = [
        "LXGWNeoXiHei.ttf",  # 霞鹜新晰黑 - Modern, clean
        "ZCOOLKuaiLe-Regular.ttf",  # 站酷快乐体 - Playful
        "LXGWWenKaiLite-Regular.ttf",  # 霞鹜文楷 - Elegant
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        style: str = "gradient_bar",
        font_name: Optional[str] = None,
        font_size: int = 72,
        text_color: str = "#FFD700",  # Bright gold for eye-catching Chinese social media style
        gradient_color: str = "#000000",
        gradient_opacity: float = 0.7,
    ):
        """
        Initialize thumbnail generator

        Args:
            api_key: DeepSeek API key for title generation
            style: Thumbnail style (gradient_bar, top_banner, full_overlay, corner_tag)
            font_name: Font file name (without path)
            font_size: Base font size
            text_color: Text color in hex format
            gradient_color: Gradient/overlay color in hex
            gradient_opacity: Opacity of gradient/overlay (0-1)
        """
        self.api_key = api_key
        self.style = style
        self.font_name = font_name
        self.font_size = font_size
        self.text_color = text_color
        self.gradient_color = gradient_color
        self.gradient_opacity = gradient_opacity

        self._font = None
        self._font_path = None

    def _get_font(self, size: int) -> Optional[Any]:
        """Get font for rendering text"""
        if not HAS_PIL:
            return None

        # Try to find a suitable font
        font_paths_to_try = []

        # If specific font requested, try it first
        if self.font_name:
            font_paths_to_try.append(self.FONTS_DIR / self.font_name)

        # Add default fonts
        for font_file in self.DEFAULT_FONTS:
            font_paths_to_try.append(self.FONTS_DIR / font_file)

        # Try system fonts as fallback
        system_font_dirs = [
            Path("/System/Library/Fonts"),  # macOS
            Path("/usr/share/fonts"),  # Linux
            Path("C:/Windows/Fonts"),  # Windows
        ]

        for font_path in font_paths_to_try:
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size)
                except Exception as e:
                    logger.warning(f"Failed to load font {font_path}: {e}")

        # Fallback to default font
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _hex_to_rgba(self, hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
        """Convert hex color to RGBA tuple"""
        rgb = self._hex_to_rgb(hex_color)
        return (*rgb, alpha)

    async def generate_title(
        self,
        original_title: str,
        description: str = "",
        keywords: list = None,
    ) -> str:
        """
        Generate a catchy Chinese title using DeepSeek AI

        Args:
            original_title: Original video title
            description: Video description
            keywords: Video keywords/tags

        Returns:
            Generated Chinese title
        """
        if not self.api_key:
            # Fallback: simple truncation of original title
            logger.warning("No API key for title generation, using truncated original")
            return original_title[:20] if len(original_title) > 20 else original_title

        keywords_str = ", ".join(keywords[:5]) if keywords else ""

        prompt = f"""你是一个中国社交媒体封面文案专家。根据以下视频信息，生成一个吸引点击的封面标题。

视频标题：{original_title}
视频描述：{description[:200] if description else '无'}
关键词：{keywords_str or '无'}

要求：
1. 标题不超过12个字（必须严格遵守）
2. 使用口语化、有冲击力的表达
3. 可以使用数字、疑问句、感叹句
4. 符合B站/抖音的标题风格
5. 突出视频的核心卖点或情绪价值

示例风格：
- "3分钟学会xxx"
- "这操作太秀了"
- "看完直接笑喷"
- "原来还能这样"
- "绝绝子！"

请只输出标题文字，不要其他任何内容。"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": "你是一个专业的中国社交媒体封面文案专家，擅长创作吸引眼球的短标题。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.8,
                        "max_tokens": 50,
                    },
                )
                response.raise_for_status()
                result = response.json()
                title = result["choices"][0]["message"]["content"].strip()

                # Clean up: remove quotes if present
                title = title.strip('"\'""''')

                # Ensure title is not too long
                if len(title) > 15:
                    title = title[:15]

                logger.info(f"Generated thumbnail title: {title}")
                return title

        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            # Fallback
            return original_title[:12] if len(original_title) > 12 else original_title

    def _create_gradient_bar(
        self,
        img: Image.Image,
        title: str,
        position: str = "bottom"
    ) -> Image.Image:
        """Create gradient bar style thumbnail"""
        width, height = img.size

        # Create a copy to work with
        result = img.copy().convert("RGBA")

        # Create gradient overlay
        gradient_height = int(height * 0.35)  # 35% of image height
        gradient = Image.new("RGBA", (width, gradient_height))

        # Draw gradient
        for y in range(gradient_height):
            # Calculate alpha (0 at top, max at bottom for bottom position)
            if position == "bottom":
                alpha = int((y / gradient_height) * 255 * self.gradient_opacity)
            else:
                alpha = int(((gradient_height - y) / gradient_height) * 255 * self.gradient_opacity)

            color = self._hex_to_rgba(self.gradient_color, alpha)
            for x in range(width):
                gradient.putpixel((x, y), color)

        # Paste gradient
        if position == "bottom":
            result.paste(gradient, (0, height - gradient_height), gradient)
        else:
            result.paste(gradient, (0, 0), gradient)

        # Draw text
        draw = ImageDraw.Draw(result)
        font = self._get_font(self.font_size)

        # Calculate text position
        if font:
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(title) * self.font_size
            text_height = self.font_size

        x = (width - text_width) // 2
        if position == "bottom":
            y = height - gradient_height // 2 - text_height // 2
        else:
            y = gradient_height // 2 - text_height // 2

        # Draw text with outline for better visibility
        text_color = self._hex_to_rgb(self.text_color)
        outline_color = (0, 0, 0)

        # Draw outline
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), title, font=font, fill=outline_color)

        # Draw main text
        draw.text((x, y), title, font=font, fill=text_color)

        return result.convert("RGB")

    def _create_top_banner(
        self,
        img: Image.Image,
        title: str,
    ) -> Image.Image:
        """Create top banner style thumbnail"""
        width, height = img.size
        result = img.copy().convert("RGBA")

        # Create banner
        banner_height = int(height * 0.20)
        banner = Image.new("RGBA", (width, banner_height))

        # Fill with gradient color
        banner_color = self._hex_to_rgba(self.gradient_color, int(255 * self.gradient_opacity))
        for y in range(banner_height):
            for x in range(width):
                banner.putpixel((x, y), banner_color)

        result.paste(banner, (0, 0), banner)

        # Draw text
        draw = ImageDraw.Draw(result)
        font = self._get_font(int(self.font_size * 0.8))

        if font:
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(title) * self.font_size
            text_height = self.font_size

        x = (width - text_width) // 2
        y = (banner_height - text_height) // 2

        text_color = self._hex_to_rgb(self.text_color)
        draw.text((x, y), title, font=font, fill=text_color)

        return result.convert("RGB")

    def _create_corner_tag(
        self,
        img: Image.Image,
        title: str,
    ) -> Image.Image:
        """Create corner tag style thumbnail"""
        width, height = img.size
        result = img.copy().convert("RGBA")
        draw = ImageDraw.Draw(result)

        # Use smaller font for corner tag
        font = self._get_font(int(self.font_size * 0.6))

        if font:
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(title) * self.font_size
            text_height = self.font_size

        # Add padding
        padding = 15
        tag_width = text_width + padding * 2
        tag_height = text_height + padding

        # Draw tag background (top-left corner)
        tag_color = self._hex_to_rgba("#FF4444", int(255 * 0.9))
        tag = Image.new("RGBA", (tag_width, tag_height), tag_color)
        result.paste(tag, (0, 0), tag)

        # Draw text
        text_color = self._hex_to_rgb(self.text_color)
        draw.text((padding, padding // 2), title, font=font, fill=text_color)

        return result.convert("RGB")

    async def generate(
        self,
        original_thumbnail_path: Path,
        output_path: Path,
        video_title: str,
        video_description: str = "",
        keywords: list = None,
        custom_title: str = None,
    ) -> ThumbnailResult:
        """
        Generate AI thumbnail

        Args:
            original_thumbnail_path: Path to original thumbnail image
            output_path: Path to save generated thumbnail
            video_title: Original video title
            video_description: Video description
            keywords: Video keywords/tags
            custom_title: Custom title to use (skips AI generation if provided)

        Returns:
            ThumbnailResult with success status and generated title
        """
        if not HAS_PIL:
            return ThumbnailResult(
                success=False,
                error="Pillow library not installed"
            )

        if not original_thumbnail_path.exists():
            return ThumbnailResult(
                success=False,
                error=f"Original thumbnail not found: {original_thumbnail_path}"
            )

        try:
            # Generate or use provided title
            if custom_title:
                title = custom_title
            else:
                title = await self.generate_title(
                    original_title=video_title,
                    description=video_description,
                    keywords=keywords or [],
                )

            # Load original image
            img = Image.open(original_thumbnail_path)

            # Resize to standard thumbnail size if needed (1280x720)
            target_size = (1280, 720)
            if img.size != target_size:
                # Calculate aspect ratio preserving resize
                img_ratio = img.width / img.height
                target_ratio = target_size[0] / target_size[1]

                if img_ratio > target_ratio:
                    # Image is wider, crop sides
                    new_height = img.height
                    new_width = int(new_height * target_ratio)
                    left = (img.width - new_width) // 2
                    img = img.crop((left, 0, left + new_width, new_height))
                else:
                    # Image is taller, crop top/bottom
                    new_width = img.width
                    new_height = int(new_width / target_ratio)
                    top = (img.height - new_height) // 2
                    img = img.crop((0, top, new_width, top + new_height))

                img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Apply style
            if self.style == "gradient_bar":
                result_img = self._create_gradient_bar(img, title, "bottom")
            elif self.style == "top_banner":
                result_img = self._create_top_banner(img, title)
            elif self.style == "corner_tag":
                result_img = self._create_corner_tag(img, title)
            else:
                # Default to gradient_bar
                result_img = self._create_gradient_bar(img, title, "bottom")

            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_img.save(output_path, "JPEG", quality=95)

            logger.info(f"Generated AI thumbnail: {output_path}")

            return ThumbnailResult(
                success=True,
                output_path=output_path,
                title=title,
                generated_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            return ThumbnailResult(
                success=False,
                error=str(e)
            )


# Convenience function for quick generation
async def generate_ai_thumbnail(
    original_path: Path,
    output_path: Path,
    video_title: str,
    api_key: str = None,
    style: str = "gradient_bar",
    **kwargs
) -> ThumbnailResult:
    """
    Quick function to generate AI thumbnail

    Args:
        original_path: Path to original thumbnail
        output_path: Path to save AI thumbnail
        video_title: Video title for AI title generation
        api_key: DeepSeek API key
        style: Thumbnail style
        **kwargs: Additional arguments for ThumbnailGenerator

    Returns:
        ThumbnailResult
    """
    generator = ThumbnailGenerator(
        api_key=api_key,
        style=style,
        **kwargs
    )
    return await generator.generate(
        original_thumbnail_path=original_path,
        output_path=output_path,
        video_title=video_title,
        **kwargs
    )

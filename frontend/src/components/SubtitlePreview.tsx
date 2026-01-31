import { useMemo, useRef, useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import type { SubtitleStyle } from '../types'

// Reference resolution for scaling calculations
// ASS default resolution is 384x288 (when PlayResX/Y not specified)
// Font sizes in ASS are relative to this resolution and scaled to actual video
const REFERENCE_HEIGHT = 288
// Minimum font size in preview to ensure readability
const MIN_PREVIEW_FONT_SIZE = 12

interface SubtitlePreviewProps {
  originalStyle: SubtitleStyle
  translatedStyle: SubtitleStyle
  chineseOnTop?: boolean  // Legacy prop name for backwards compatibility
  translatedOnTop?: boolean  // New generic prop name
  showOriginal?: boolean
  showTranslated?: boolean
  originalText?: string
  translatedText?: string
  isVertical?: boolean  // Support vertical video preview (9:16 aspect ratio)
}

/**
 * Live preview component for subtitle styles
 * Uses CSS to simulate ASS subtitle rendering
 */
export default function SubtitlePreview({
  originalStyle,
  translatedStyle,
  chineseOnTop,
  translatedOnTop,
  showOriginal = true,
  showTranslated = true,
  originalText,
  translatedText,
  isVertical = false,
}: SubtitlePreviewProps) {
  const { t } = useTranslation()
  const containerRef = useRef<HTMLDivElement>(null)
  const [scale, setScale] = useState(1)

  // Support both legacy and new prop names
  const isTranslatedOnTop = translatedOnTop ?? chineseOnTop ?? true

  // Use translation defaults if not provided
  const displayOriginalText = originalText ?? t('previewText.original')
  const displayTranslatedText = translatedText ?? t('previewText.translated')

  // Calculate scale based on container HEIGHT vs ASS default resolution (288)
  // This matches how ASS subtitles are scaled to actual video resolution
  useEffect(() => {
    const updateScale = () => {
      if (containerRef.current) {
        const containerHeight = containerRef.current.offsetHeight
        // Scale based on height ratio to ASS default resolution (288)
        // ASS scales font_size by (video_height / 288), we do the same with container
        // Example: font_size 24 in 300px container = 24 * (300/288) ≈ 25px
        const newScale = containerHeight / REFERENCE_HEIGHT
        setScale(newScale)
      }
    }

    updateScale()

    // Use ResizeObserver for responsive updates
    const resizeObserver = new ResizeObserver(updateScale)
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    return () => resizeObserver.disconnect()
  }, [])

  // Convert subtitle style to CSS with scaling
  const styleToCSS = (style: SubtitleStyle): React.CSSProperties => {
    // Apply scale to all pixel values
    // Use minimum font size to ensure readability in small preview
    const rawScaledFontSize = style.font_size * scale
    const scaledFontSize = Math.max(rawScaledFontSize, MIN_PREVIEW_FONT_SIZE)

    // Scale other values proportionally
    // Ensure minimum visibility while preventing excessive thickness
    const rawOutlineWidth = style.outline_width * scale
    const scaledOutlineWidth = style.outline_width > 0
      ? Math.max(Math.min(rawOutlineWidth, 2), 0.8) // Min 0.8px, max 2px
      : 0
    const rawShadowOffset = style.shadow * scale
    const scaledShadowOffset = style.shadow > 0
      ? Math.max(Math.min(rawShadowOffset, 3), 1) // Min 1px, max 3px
      : 0
    const scaledSpacing = style.spacing * scale
    const scaledPadding = Math.max(4 * scale, 3)

    // Generate text effects
    const outlineColor = style.outline_color
    const shadowColor = style.shadow_color

    // Use text-stroke for cleaner outline (better for CJK characters)
    const textStroke = scaledOutlineWidth > 0
      ? `${scaledOutlineWidth}px ${outlineColor}`
      : 'none'

    // Simple drop shadow only (no outline shadows to avoid visual chaos)
    const textShadow = scaledShadowOffset > 0
      ? `${scaledShadowOffset}px ${scaledShadowOffset}px ${scaledShadowOffset}px ${shadowColor}`
      : 'none'

    // Background with opacity
    const backOpacity = style.back_opacity / 100
    const backColor = style.back_color
    const backgroundColor = backOpacity > 0
      ? `${backColor}${Math.round(backOpacity * 255).toString(16).padStart(2, '0')}`
      : 'transparent'

    // Build transform combining centering and scaling
    const scaleX = style.scale_x / 100
    const scaleY = style.scale_y / 100
    // Only apply scale transform, centering is done via margin: auto
    const transform = scaleX !== 1 || scaleY !== 1 ? `scaleX(${scaleX}) scaleY(${scaleY})` : 'none'

    return {
      fontFamily: style.font_name || 'PingFang SC, Microsoft YaHei, sans-serif',
      fontSize: `${scaledFontSize}px`,
      color: style.color,
      fontWeight: style.bold ? 'bold' : 'normal',
      fontStyle: style.italic ? 'italic' : 'normal',
      WebkitTextStroke: textStroke,
      textShadow: textShadow,
      paintOrder: 'stroke fill' as const,
      letterSpacing: `${scaledSpacing}px`,
      transform,
      backgroundColor,
      padding: backgroundColor !== 'transparent' ? `${scaledPadding}px ${scaledPadding * 2}px` : '0',
      borderRadius: backgroundColor !== 'transparent' ? `${scaledPadding}px` : '0',
      textAlign: 'center' as const,
      position: 'absolute' as const,
      left: 0,
      right: 0,
      marginLeft: 'auto',
      marginRight: 'auto',
      maxWidth: `${style.max_width ?? 90}%`,
      whiteSpace: 'pre-wrap' as const,
      wordBreak: 'break-word' as const,
      WebkitFontSmoothing: 'antialiased' as const,
      MozOsxFontSmoothing: 'grayscale' as const,
      textRendering: 'optimizeLegibility' as const,
    }
  }

  // Calculate positions for dual subtitles (scaled)
  // In dual mode, both subtitles use their own margin_v independently
  const positions = useMemo(() => {
    const topStyle = isTranslatedOnTop ? translatedStyle : originalStyle
    const bottomStyle = isTranslatedOnTop ? originalStyle : translatedStyle

    // Bottom subtitle uses its own margin_v
    const bottomMargin = bottomStyle.margin_v * scale

    // Top subtitle also uses its own margin_v
    // But we need to ensure minimum clearance to avoid overlap
    const bottomFontSize = Math.max(bottomStyle.font_size * scale, MIN_PREVIEW_FONT_SIZE)
    const bottomHasBackground = bottomStyle.back_opacity > 0
    const bottomPadding = bottomHasBackground ? Math.max(4 * scale, 3) * 2 : 0
    const gap = 10 * scale
    const minTopMargin = bottomMargin + bottomFontSize + bottomPadding + gap

    // Use the top subtitle's margin_v, but ensure it doesn't overlap with bottom
    const topMargin = Math.max(topStyle.margin_v * scale, minTopMargin)

    return { topMargin, bottomMargin }
  }, [isTranslatedOnTop, originalStyle, translatedStyle, scale])

  const topSubtitle = isTranslatedOnTop ? { text: displayTranslatedText, style: translatedStyle } : { text: displayOriginalText, style: originalStyle }
  const bottomSubtitle = isTranslatedOnTop ? { text: displayOriginalText, style: originalStyle } : { text: displayTranslatedText, style: translatedStyle }

  // Use different aspect ratios for horizontal vs vertical videos
  // Vertical: 9:16 (0.5625), Horizontal: 16:9 (1.777...)
  const aspectRatioClass = isVertical ? 'aspect-[9/16]' : 'aspect-video'
  // For vertical preview, limit the width to avoid it being too tall
  const containerClass = isVertical
    ? `relative bg-gradient-to-b from-gray-700 via-gray-800 to-gray-900 rounded-lg overflow-hidden ${aspectRatioClass} max-w-[200px] mx-auto`
    : `relative w-full bg-gradient-to-b from-gray-700 via-gray-800 to-gray-900 rounded-lg overflow-hidden ${aspectRatioClass}`

  return (
    <div ref={containerRef} className={containerClass}>
      {/* Video frame simulation */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDQwIDQwIj48cmVjdCB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIGZpbGw9IiMxZjJlM2QiLz48cmVjdCB4PSIwIiB5PSIwIiB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIGZpbGw9IiMyNTM0NDMiLz48cmVjdCB4PSIyMCIgeT0iMjAiIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgZmlsbD0iIzI1MzQ0MyIvPjwvc3ZnPg==')] opacity-30" />

      {/* Simulated video content placeholder */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-gray-600 text-sm">{t('styleEditor.preview.videoArea')}</div>
      </div>

      {/* Subtitle container */}
      <div className="absolute inset-0">
        {/* Top subtitle (if showing both and dual mode) */}
        {showOriginal && showTranslated && (
          <div
            style={{
              ...styleToCSS(topSubtitle.style),
              bottom: `${positions.topMargin}px`,
              top: 'auto',
            }}
          >
            {topSubtitle.text}
          </div>
        )}

        {/* Bottom subtitle */}
        {((showOriginal && !showTranslated) || (showTranslated && !showOriginal)) ? (
          // Single subtitle mode
          <div
            style={{
              ...styleToCSS(showTranslated ? translatedStyle : originalStyle),
              bottom: `${(showTranslated ? translatedStyle : originalStyle).margin_v}px`,
            }}
          >
            {showTranslated ? displayTranslatedText : displayOriginalText}
          </div>
        ) : showOriginal && showTranslated ? (
          // Dual subtitle mode - bottom
          <div
            style={{
              ...styleToCSS(bottomSubtitle.style),
              bottom: `${positions.bottomMargin}px`,
            }}
          >
            {bottomSubtitle.text}
          </div>
        ) : null}
      </div>

      {/* Controls overlay */}
      <div className="absolute bottom-2 right-2 flex space-x-2 opacity-50">
        <div className="w-8 h-1 bg-white rounded" />
        <div className="w-2 h-2 bg-white rounded-full" />
      </div>
    </div>
  )
}

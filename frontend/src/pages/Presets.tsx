import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { settingsApi } from '../services/api'
import SubtitleStyleEditor from '../components/SubtitleStyleEditor'
import type { SubtitleStyle } from '../types'

// Default subtitle style
const DEFAULT_STYLE: SubtitleStyle = {
  font_name: '',
  font_size: 24,
  color: '#FFFFFF',
  bold: true,
  italic: false,
  outline_color: '#000000',
  outline_width: 2,
  shadow: 1,
  shadow_color: '#000000',
  alignment: 'bottom',
  margin_h: 20,
  margin_v: 60,
  max_width: 90,
  back_color: '#000000',
  back_opacity: 0,
  spacing: 0,
  scale_x: 100,
  scale_y: 100,
}

// Ensure all fields are present in a subtitle style
function ensureFullSubtitleStyle(style: Partial<SubtitleStyle> | undefined): SubtitleStyle {
  if (!style) return { ...DEFAULT_STYLE }
  return { ...DEFAULT_STYLE, ...style }
}

export default function Presets() {
  const { t, i18n } = useTranslation()

  // Fetch global settings for initial values
  const { data: globalSettings } = useQuery({
    queryKey: ['globalSettings'],
    queryFn: settingsApi.get,
  })

  // Local state for styles (not saved to global settings, just for preview)
  const [originalStyle, setOriginalStyle] = useState<SubtitleStyle>(DEFAULT_STYLE)
  const [translatedStyle, setTranslatedStyle] = useState<SubtitleStyle>(DEFAULT_STYLE)
  const [translatedOnTop, setTranslatedOnTop] = useState(true)
  const [sourceLanguage, setSourceLanguage] = useState('en')
  const [targetLanguage, setTargetLanguage] = useState('zh-CN')

  // Initialize from global settings
  useEffect(() => {
    if (globalSettings) {
      setOriginalStyle(ensureFullSubtitleStyle(globalSettings.subtitle.original_style))
      setTranslatedStyle(ensureFullSubtitleStyle(globalSettings.subtitle.translated_style))
      setTranslatedOnTop(globalSettings.subtitle.chinese_on_top ?? true)
      setSourceLanguage(globalSettings.subtitle.source_language || 'en')
      setTargetLanguage(globalSettings.subtitle.target_language || (i18n.language.startsWith('zh') ? 'zh-CN' : i18n.language))
    }
  }, [globalSettings, i18n.language])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">{t('nav.presets')}</h1>
          <p className="mt-1 text-sm text-gray-500">{t('presets.description')}</p>
        </div>
      </div>

      <SubtitleStyleEditor
        originalStyle={originalStyle}
        translatedStyle={translatedStyle}
        translatedOnTop={translatedOnTop}
        onOriginalStyleChange={setOriginalStyle}
        onTranslatedStyleChange={setTranslatedStyle}
        onTranslatedOnTopChange={setTranslatedOnTop}
        sourceLanguage={sourceLanguage}
        targetLanguage={targetLanguage}
        onSourceLanguageChange={setSourceLanguage}
        onTargetLanguageChange={setTargetLanguage}
      />
    </div>
  )
}

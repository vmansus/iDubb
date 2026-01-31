import { useState, useCallback, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  ChevronDown,
  ChevronUp,
  Save,
  Upload,
  Trash2,
  Plus,
  Check,
  Loader2,
  X,
  Search
} from 'lucide-react'
import { fontsApi, presetsApi } from '../services/api'
import type { SubtitleStyle, SubtitlePreset, FontInfo } from '../types'
import SubtitlePreview from './SubtitlePreview'

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
  margin_v: 30,
  max_width: 90,
  back_color: '#000000',
  back_opacity: 0,
  spacing: 0,
  scale_x: 100,
  scale_y: 100,
}

// Language to script mapping for font filtering
const LANGUAGE_SCRIPTS: Record<string, string[]> = {
  // CJK languages
  'zh': ['cjk', 'latin'],
  'zh-CN': ['cjk', 'latin'],
  'zh-TW': ['cjk', 'latin'],
  'zh-HK': ['cjk', 'latin'],
  'ja': ['cjk', 'latin'],
  'ko': ['cjk', 'latin'],
  // Arabic script languages
  'ar': ['arabic', 'latin'],
  'fa': ['arabic', 'latin'],  // Persian
  'ur': ['arabic', 'latin'],  // Urdu
  // Hebrew
  'he': ['hebrew', 'latin'],
  'iw': ['hebrew', 'latin'],
  // Thai
  'th': ['thai', 'latin'],
  // Devanagari languages
  'hi': ['devanagari', 'latin'],  // Hindi
  'mr': ['devanagari', 'latin'],  // Marathi
  'ne': ['devanagari', 'latin'],  // Nepali
  'sa': ['devanagari', 'latin'],  // Sanskrit
  // Cyrillic languages
  'ru': ['cyrillic', 'latin'],
  'uk': ['cyrillic', 'latin'],  // Ukrainian
  'bg': ['cyrillic', 'latin'],  // Bulgarian
  'sr': ['cyrillic', 'latin'],  // Serbian
  // Greek
  'el': ['greek', 'latin'],
  // Default (Latin-based languages)
  'en': ['latin'],
  'es': ['latin'],
  'fr': ['latin'],
  'de': ['latin'],
  'it': ['latin'],
  'pt': ['latin'],
  'nl': ['latin'],
  'pl': ['latin'],
  'vi': ['latin'],
  'id': ['latin'],
  'ms': ['latin'],
  'tr': ['latin'],
}

// Get required scripts for a language
function getRequiredScripts(langCode: string): string[] {
  // Check exact match first
  if (LANGUAGE_SCRIPTS[langCode]) {
    return LANGUAGE_SCRIPTS[langCode]
  }
  // Check base language code (e.g., 'zh' from 'zh-CN')
  const baseLang = langCode.split('-')[0].toLowerCase()
  if (LANGUAGE_SCRIPTS[baseLang]) {
    return LANGUAGE_SCRIPTS[baseLang]
  }
  // Default to Latin for unknown languages
  return ['latin']
}

// Get display name for a language
function getLanguageDisplayName(langCode: string): string {
  const names: Record<string, string> = {
    'zh': '中文', 'zh-CN': '简体中文', 'zh-TW': '繁體中文', 'zh-HK': '繁體中文',
    'ja': '日本語', 'ko': '한국어',
    'ar': 'العربية', 'fa': 'فارسی', 'ur': 'اردو',
    'he': 'עברית', 'iw': 'עברית',
    'th': 'ไทย',
    'hi': 'हिन्दी', 'mr': 'मराठी', 'ne': 'नेपाली',
    'ru': 'Русский', 'uk': 'Українська', 'bg': 'Български',
    'el': 'Ελληνικά',
    'en': 'English', 'es': 'Español', 'fr': 'Français', 'de': 'Deutsch',
    'it': 'Italiano', 'pt': 'Português', 'nl': 'Nederlands',
    'vi': 'Tiếng Việt', 'id': 'Bahasa Indonesia', 'ms': 'Bahasa Melayu',
    'tr': 'Türkçe', 'pl': 'Polski',
  }
  return names[langCode] || names[langCode.split('-')[0]] || langCode.toUpperCase()
}

interface SubtitleStyleEditorProps {
  originalStyle: SubtitleStyle
  translatedStyle: SubtitleStyle
  translatedOnTop: boolean
  onOriginalStyleChange: (style: SubtitleStyle) => void
  onTranslatedStyleChange: (style: SubtitleStyle) => void
  onTranslatedOnTopChange: (value: boolean) => void
  sourceLanguage?: string  // e.g., 'en', 'es', 'fr'
  targetLanguage?: string  // e.g., 'zh', 'ja', 'ko', 'ar'
  onSourceLanguageChange?: (lang: string) => void
  onTargetLanguageChange?: (lang: string) => void
  // Subtitle visibility control
  showOriginal?: boolean
  showTranslated?: boolean
  onShowOriginalChange?: (show: boolean) => void
  onShowTranslatedChange?: (show: boolean) => void
}

// Supported languages for subtitle styling
const SUPPORTED_LANGUAGES = [
  // CJK
  { code: 'zh-CN', name: '简体中文', nameEn: 'Chinese (Simplified)' },
  { code: 'zh-TW', name: '繁體中文', nameEn: 'Chinese (Traditional)' },
  { code: 'ja', name: '日本語', nameEn: 'Japanese' },
  { code: 'ko', name: '한국어', nameEn: 'Korean' },
  // Latin-based
  { code: 'en', name: 'English', nameEn: 'English' },
  { code: 'es', name: 'Español', nameEn: 'Spanish' },
  { code: 'fr', name: 'Français', nameEn: 'French' },
  { code: 'de', name: 'Deutsch', nameEn: 'German' },
  { code: 'it', name: 'Italiano', nameEn: 'Italian' },
  { code: 'pt', name: 'Português', nameEn: 'Portuguese' },
  { code: 'ru', name: 'Русский', nameEn: 'Russian' },
  // Arabic
  { code: 'ar', name: 'العربية', nameEn: 'Arabic' },
  // Thai
  { code: 'th', name: 'ไทย', nameEn: 'Thai' },
  // Hindi
  { code: 'hi', name: 'हिंदी', nameEn: 'Hindi' },
  // Vietnamese
  { code: 'vi', name: 'Tiếng Việt', nameEn: 'Vietnamese' },
  // Indonesian
  { code: 'id', name: 'Bahasa Indonesia', nameEn: 'Indonesian' },
]

// Style section component
interface StyleSectionProps {
  title: string
  defaultOpen?: boolean
  children: React.ReactNode
}

function StyleSection({ title, defaultOpen = true, children }: StyleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border border-gray-200 rounded-lg">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-2 flex items-center justify-between bg-gray-50 hover:bg-gray-100 transition-colors rounded-t-lg"
      >
        <span className="text-sm font-medium text-gray-700">{title}</span>
        {isOpen ? (
          <ChevronUp className="h-4 w-4 text-gray-500" />
        ) : (
          <ChevronDown className="h-4 w-4 text-gray-500" />
        )}
      </button>
      {isOpen && <div className="p-4 space-y-3">{children}</div>}
    </div>
  )
}

// Custom font selector with preview
interface FontSelectorProps {
  value: string
  onChange: (value: string) => void
  fonts: FontInfo[]
  language: string  // Language code like 'en', 'zh', 'ar', etc.
  recommendedFont?: FontInfo | null  // Recommended font for this language
}

function FontSelector({ value, onChange, fonts, language, recommendedFont }: FontSelectorProps) {
  const { t } = useTranslation()
  const [isOpen, setIsOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const dropdownRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  // Get required scripts for the language
  const requiredScripts = getRequiredScripts(language)
  // Get the primary script (non-Latin) if any
  const primaryScript = requiredScripts.find(s => s !== 'latin')
  const needsSpecialFont = !!primaryScript

  // Filter fonts based on language requirements
  const languageFilteredFonts = fonts.filter(font => {
    // If language only needs Latin, show all fonts
    if (!needsSpecialFont) {
      return true
    }
    // Otherwise, font MUST support the primary script (e.g., 'cjk' for Chinese)
    // A font that only supports 'latin' is NOT acceptable for CJK languages
    const fontScripts = font.supported_scripts || ['latin']
    return fontScripts.includes(primaryScript)
  })

  // Further filter by search query
  const filteredFonts = languageFilteredFonts.filter(font => {
    if (!searchQuery.trim()) return true
    const query = searchQuery.toLowerCase()
    return font.name.toLowerCase().includes(query)
  })

  // Check if recommended font is relevant for this language and matches search
  const showRecommended = needsSpecialFont && recommendedFont && primaryScript &&
    recommendedFont.supported_scripts?.includes(primaryScript) &&
    (!searchQuery.trim() || recommendedFont.name.toLowerCase().includes(searchQuery.toLowerCase()))

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
        setSearchQuery('')  // Clear search when closing
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Focus search input when dropdown opens
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      searchInputRef.current.focus()
    }
  }, [isOpen])

  // Sample text based on language script
  const getSampleText = () => {
    if (requiredScripts.includes('cjk')) return '中文字体'
    if (requiredScripts.includes('arabic')) return 'نص عربي'
    if (requiredScripts.includes('hebrew')) return 'טקסט עברי'
    if (requiredScripts.includes('thai')) return 'ข้อความไทย'
    if (requiredScripts.includes('devanagari')) return 'हिंदी पाठ'
    if (requiredScripts.includes('cyrillic')) return 'Русский'
    if (requiredScripts.includes('greek')) return 'Ελληνικά'
    return 'Font ABC'
  }
  const sampleText = getSampleText()

  return (
    <div ref={dropdownRef} className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm text-left flex items-center justify-between bg-white hover:border-gray-400"
      >
        <span
          style={{ fontFamily: value || 'inherit' }}
          className="truncate"
        >
          {value || t('styleEditor.fontSelector.autoSelect')}
        </span>
        <ChevronDown className={`h-4 w-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-80 overflow-hidden flex flex-col">
          {/* Search input - sticky at top */}
          <div className="p-2 border-b border-gray-200 sticky top-0 bg-white">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={t('styleEditor.fontSelector.searchPlaceholder')}
                className="w-full pl-8 pr-3 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:border-blue-500"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Scrollable content */}
          <div className="overflow-y-auto flex-1">
            {/* Auto option */}
            {!searchQuery && (
              <button
                type="button"
                onClick={() => { onChange(''); setIsOpen(false); setSearchQuery('') }}
                className={`w-full px-3 py-2 text-left text-sm hover:bg-blue-50 flex items-center justify-between ${
                  !value ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
                }`}
              >
                <span>{t('styleEditor.fontSelector.autoSelect')}</span>
                {!value && <Check className="h-4 w-4" />}
              </button>
            )}

            {/* Divider */}
            {!searchQuery && <div className="border-t border-gray-200 my-1" />}

            {/* Recommended font for CJK/special scripts */}
            {showRecommended && recommendedFont && (
              <>
                <button
                  type="button"
                  onClick={() => { onChange(recommendedFont.name); setIsOpen(false); setSearchQuery('') }}
                  className={`w-full px-3 py-2 text-left hover:bg-green-50 flex items-center justify-between ${
                    value === recommendedFont.name ? 'bg-green-50' : 'bg-green-50/30'
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <div
                      style={{ fontFamily: `"${recommendedFont.name}", sans-serif` }}
                      className="text-sm truncate font-medium"
                    >
                      {recommendedFont.name}
                    </div>
                    <div
                      style={{ fontFamily: `"${recommendedFont.name}", sans-serif` }}
                      className="text-xs text-gray-500 truncate"
                    >
                      {sampleText}
                    </div>
                  </div>
                  <div className="flex items-center ml-2 space-x-1">
                    <span className="text-[10px] text-green-600 bg-green-100 px-1.5 py-0.5 rounded font-medium">
                      {t('styleEditor.fontSelector.recommended')}
                    </span>
                    {value === recommendedFont.name && <Check className="h-4 w-4 text-green-600" />}
                  </div>
                </button>
                <div className="border-t border-gray-200 my-1" />
              </>
            )}

            {/* Font count hint */}
            <div className="px-3 py-1 text-xs text-gray-400">
              {searchQuery ? t('styleEditor.fontSelector.searchResults') : (needsSpecialFont ? getLanguageDisplayName(language) : t('styleEditor.fontSelector.allFonts'))} ({filteredFonts.length})
            </div>

            {/* Font list */}
            {filteredFonts.map((font) => (
              <button
                key={font.path}
                type="button"
                onClick={() => { onChange(font.name); setIsOpen(false); setSearchQuery('') }}
                className={`w-full px-3 py-2 text-left hover:bg-blue-50 flex items-center justify-between ${
                  value === font.name ? 'bg-blue-50' : ''
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div
                    style={{ fontFamily: `"${font.name}", sans-serif` }}
                    className="text-sm truncate"
                  >
                    {font.name}
                  </div>
                  <div
                    style={{ fontFamily: `"${font.name}", sans-serif` }}
                    className="text-xs text-gray-500 truncate"
                  >
                    {sampleText}
                  </div>
                </div>
                <div className="flex items-center ml-2 space-x-1">
                  {font.is_custom && (
                    <span className="text-[10px] text-orange-500 bg-orange-50 px-1 rounded">{t('styleEditor.custom')}</span>
                  )}
                  {font.supported_scripts && font.supported_scripts.filter(s => s !== 'latin').map(script => (
                    <span key={script} className="text-[10px] text-green-500 bg-green-50 px-1 rounded">
                      {script.toUpperCase()}
                    </span>
                  ))}
                  {value === font.name && <Check className="h-4 w-4 text-blue-600" />}
                </div>
              </button>
            ))}

            {filteredFonts.length === 0 && (
              <div className="px-3 py-4 text-sm text-gray-500 text-center">
                {searchQuery
                  ? t('styleEditor.fontSelector.noSearchResults', { query: searchQuery })
                  : t('styleEditor.fontSelector.noFonts', { language: getLanguageDisplayName(language) })
                }
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// Style panel for one language
interface StylePanelProps {
  style: SubtitleStyle
  onChange: (style: SubtitleStyle) => void
  fonts: FontInfo[]
  language: string  // Language code
  recommendedFont?: FontInfo | null  // Recommended font for CJK languages
}

function StylePanel({ style, onChange, fonts, language, recommendedFont }: StylePanelProps) {
  const { t } = useTranslation()
  const handleChange = useCallback(
    <K extends keyof SubtitleStyle>(key: K, value: SubtitleStyle[K]) => {
      onChange({ ...style, [key]: value })
    },
    [style, onChange]
  )

  return (
    <div className="space-y-4">
      {/* Text Settings */}
      <StyleSection title={t('styleEditor.sections.textStyle')} defaultOpen={true}>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.font')}</label>
            <FontSelector
              value={style.font_name || ''}
              onChange={(val) => handleChange('font_name', val)}
              fonts={fonts}
              language={language}
              recommendedFont={recommendedFont}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.fontSize')}</label>
            <input
              type="number"
              value={style.font_size}
              onChange={(e) => handleChange('font_size', parseInt(e.target.value) || 24)}
              min={12}
              max={72}
              className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.color')}</label>
            <div className="flex items-center space-x-2">
              <input
                type="color"
                value={style.color}
                onChange={(e) => handleChange('color', e.target.value)}
                className="w-10 h-8 flex-shrink-0 border border-gray-300 rounded cursor-pointer"
              />
              <input
                type="text"
                value={style.color}
                onChange={(e) => handleChange('color', e.target.value)}
                className="w-24 px-2 py-1.5 border border-gray-300 rounded text-sm font-mono"
              />
            </div>
          </div>
          <div className="flex items-end space-x-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={style.bold}
                onChange={(e) => handleChange('bold', e.target.checked)}
                className="rounded border-gray-300 text-blue-600"
              />
              <span className="ml-2 text-xs text-gray-600">{t('styleEditor.fields.bold')}</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={style.italic}
                onChange={(e) => handleChange('italic', e.target.checked)}
                className="rounded border-gray-300 text-blue-600"
              />
              <span className="ml-2 text-xs text-gray-600">{t('styleEditor.fields.italic')}</span>
            </label>
          </div>
        </div>
      </StyleSection>

      {/* Outline & Shadow */}
      <StyleSection title={t('styleEditor.sections.outlineShadow')} defaultOpen={true}>
        <div className="grid grid-cols-2 gap-3">
          {/* Row 1: Outline color and width */}
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.outlineColor')}</label>
            <div className="flex items-center space-x-2">
              <input
                type="color"
                value={style.outline_color}
                onChange={(e) => handleChange('outline_color', e.target.value)}
                className="w-10 h-8 flex-shrink-0 border border-gray-300 rounded cursor-pointer"
              />
              <input
                type="text"
                value={style.outline_color}
                onChange={(e) => handleChange('outline_color', e.target.value)}
                className="w-24 px-2 py-1.5 border border-gray-300 rounded text-sm font-mono"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.outlineWidth')}</label>
            <input
              type="range"
              value={style.outline_width}
              onChange={(e) => handleChange('outline_width', parseInt(e.target.value))}
              min={0}
              max={5}
              className="w-full"
            />
            <div className="text-xs text-gray-500 text-center">{style.outline_width}px</div>
          </div>
          {/* Row 2: Shadow color and intensity */}
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.shadowColor')}</label>
            <div className="flex items-center space-x-2">
              <input
                type="color"
                value={style.shadow_color}
                onChange={(e) => handleChange('shadow_color', e.target.value)}
                className="w-10 h-8 flex-shrink-0 border border-gray-300 rounded cursor-pointer"
              />
              <input
                type="text"
                value={style.shadow_color}
                onChange={(e) => handleChange('shadow_color', e.target.value)}
                className="w-24 px-2 py-1.5 border border-gray-300 rounded text-sm font-mono"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.shadow')}</label>
            <input
              type="range"
              value={style.shadow}
              onChange={(e) => handleChange('shadow', parseInt(e.target.value))}
              min={0}
              max={4}
              className="w-full"
            />
            <div className="text-xs text-gray-500 text-center">{style.shadow}</div>
          </div>
        </div>
      </StyleSection>

      {/* Position */}
      <StyleSection title={t('styleEditor.sections.position')} defaultOpen={true}>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.verticalAlign')}</label>
            <select
              value={style.alignment}
              onChange={(e) => handleChange('alignment', e.target.value as 'top' | 'middle' | 'bottom')}
              className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm"
            >
              <option value="bottom">{t('styleEditor.fields.bottom')}</option>
              <option value="middle">{t('styleEditor.fields.middle')}</option>
              <option value="top">{t('styleEditor.fields.top')}</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.maxWidth')}</label>
            <input
              type="range"
              value={style.max_width ?? 90}
              onChange={(e) => handleChange('max_width', parseInt(e.target.value))}
              min={50}
              max={100}
              className="w-full"
            />
            <div className="text-xs text-gray-500 text-center">{style.max_width ?? 90}%</div>
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.verticalMargin')}</label>
            <input
              type="range"
              value={style.margin_v}
              onChange={(e) => handleChange('margin_v', parseInt(e.target.value))}
              min={0}
              max={540}
              className="w-full"
            />
            <div className="text-xs text-gray-500 text-center">{style.margin_v}px</div>
          </div>
        </div>
      </StyleSection>

      {/* Advanced */}
      <StyleSection title={t('styleEditor.sections.advanced')} defaultOpen={false}>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.backgroundColor')}</label>
            <div className="flex items-center space-x-2">
              <input
                type="color"
                value={style.back_color}
                onChange={(e) => handleChange('back_color', e.target.value)}
                className="w-10 h-8 flex-shrink-0 border border-gray-300 rounded cursor-pointer"
              />
              <input
                type="text"
                value={style.back_color}
                onChange={(e) => handleChange('back_color', e.target.value)}
                className="w-24 px-2 py-1.5 border border-gray-300 rounded text-sm font-mono"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.backgroundOpacity')}</label>
            <input
              type="range"
              value={style.back_opacity}
              onChange={(e) => handleChange('back_opacity', parseInt(e.target.value))}
              min={0}
              max={100}
              className="w-full"
            />
            <div className="text-xs text-gray-500 text-center">{style.back_opacity}%</div>
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.letterSpacing')}</label>
            <input
              type="range"
              value={style.spacing}
              onChange={(e) => handleChange('spacing', parseInt(e.target.value))}
              min={-10}
              max={20}
              className="w-full"
            />
            <div className="text-xs text-gray-500 text-center">{style.spacing}px</div>
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.horizontalScale')}</label>
            <input
              type="range"
              value={style.scale_x}
              onChange={(e) => handleChange('scale_x', parseInt(e.target.value))}
              min={50}
              max={150}
              className="w-full"
            />
            <div className="text-xs text-gray-500 text-center">{style.scale_x}%</div>
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">{t('styleEditor.fields.verticalScale')}</label>
            <input
              type="range"
              value={style.scale_y}
              onChange={(e) => handleChange('scale_y', parseInt(e.target.value))}
              min={50}
              max={150}
              className="w-full"
            />
            <div className="text-xs text-gray-500 text-center">{style.scale_y}%</div>
          </div>
        </div>
      </StyleSection>
    </div>
  )
}

// Main editor component
export default function SubtitleStyleEditor({
  originalStyle,
  translatedStyle,
  translatedOnTop,
  onOriginalStyleChange,
  onTranslatedStyleChange,
  onTranslatedOnTopChange: _onTranslatedOnTopChange, // Reserved for future use
  sourceLanguage = 'en',
  targetLanguage = 'zh',
  onSourceLanguageChange,
  onTargetLanguageChange,
  showOriginal: propShowOriginal,
  showTranslated: propShowTranslated,
  onShowOriginalChange,
  onShowTranslatedChange,
}: SubtitleStyleEditorProps) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  // Use props if provided, otherwise use local state
  const [localShowOriginal, setLocalShowOriginal] = useState(propShowOriginal ?? true)
  const [localShowTranslated, setLocalShowTranslated] = useState(propShowTranslated ?? true)

  const showOriginal = propShowOriginal ?? localShowOriginal
  const showTranslated = propShowTranslated ?? localShowTranslated

  const handleShowOriginalChange = (value: boolean) => {
    if (onShowOriginalChange) {
      onShowOriginalChange(value)
    } else {
      setLocalShowOriginal(value)
    }
  }

  const handleShowTranslatedChange = (value: boolean) => {
    if (onShowTranslatedChange) {
      onShowTranslatedChange(value)
    } else {
      setLocalShowTranslated(value)
    }
  }

  // Determine active tab based on what's enabled
  const [activeTab, setActiveTab] = useState<'translated' | 'original'>('translated')

  // Auto-switch tab if current tab is disabled
  useEffect(() => {
    if (activeTab === 'translated' && !showTranslated && showOriginal) {
      setActiveTab('original')
    } else if (activeTab === 'original' && !showOriginal && showTranslated) {
      setActiveTab('translated')
    }
  }, [showOriginal, showTranslated, activeTab])

  const [customOriginalText, setCustomOriginalText] = useState('')
  const [customTranslatedText, setCustomTranslatedText] = useState('')
  const [savePresetName, setSavePresetName] = useState('')
  const [savePresetIsVertical, setSavePresetIsVertical] = useState(false)
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null)

  // Fetch fonts
  const { data: fontsData } = useQuery({
    queryKey: ['fonts'],
    queryFn: fontsApi.getAll,
  })

  // Fetch presets
  const { data: presetsData, isLoading: presetsLoading } = useQuery({
    queryKey: ['presets'],
    queryFn: presetsApi.getAll,
  })

  // Upload font mutation
  const uploadFontMutation = useMutation({
    mutationFn: fontsApi.upload,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['fonts'] })
    },
  })

  // Save preset mutation
  const savePresetMutation = useMutation({
    mutationFn: presetsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] })
      setShowSaveModal(false)
      setSavePresetName('')
      setSavePresetIsVertical(false)
    },
  })

  // Delete preset mutation
  const deletePresetMutation = useMutation({
    mutationFn: presetsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] })
    },
  })

  // Update preset mutation
  const updatePresetMutation = useMutation({
    mutationFn: ({ presetId, data }: { presetId: string; data: Parameters<typeof presetsApi.update>[1] }) =>
      presetsApi.update(presetId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] })
    },
  })

  // Apply preset
  const applyPreset = (preset: SubtitlePreset) => {
    // Apply subtitle mode
    const hasOriginal = preset.subtitle_mode === 'dual' || preset.subtitle_mode === 'original_only'
    const hasTranslated = preset.subtitle_mode === 'dual' || preset.subtitle_mode === 'translated_only'

    handleShowOriginalChange(hasOriginal)
    handleShowTranslatedChange(hasTranslated)

    // Set appropriate tab based on mode
    if (hasTranslated) {
      setActiveTab('translated')
    } else if (hasOriginal) {
      setActiveTab('original')
    }

    // Apply languages if available
    if (preset.source_language) {
      onSourceLanguageChange?.(preset.source_language)
    }
    if (preset.target_language) {
      onTargetLanguageChange?.(preset.target_language)
    }

    // Apply styles
    if (preset.original_style) {
      onOriginalStyleChange({ ...DEFAULT_STYLE, ...preset.original_style })
    }
    if (preset.translated_style) {
      onTranslatedStyleChange({ ...DEFAULT_STYLE, ...preset.translated_style })
    }
    setSelectedPresetId(preset.id)
  }

  // Check if a preset is currently active
  const isPresetActive = (preset: SubtitlePreset): boolean => {
    return selectedPresetId === preset.id
  }

  // Handle font upload
  const handleFontUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      uploadFontMutation.mutate(file)
    }
  }

  // Determine subtitle mode based on current selection
  const getSubtitleMode = (): 'dual' | 'original_only' | 'translated_only' | null => {
    if (showOriginal && showTranslated) return 'dual'
    if (showOriginal) return 'original_only'
    if (showTranslated) return 'translated_only'
    return null
  }

  // Handle save preset
  const handleSavePreset = () => {
    if (!savePresetName.trim()) return
    const subtitleMode = getSubtitleMode()
    if (!subtitleMode) return // Can't save without any subtitle selected

    savePresetMutation.mutate({
      name: savePresetName,
      description: '',
      is_vertical: savePresetIsVertical,
      subtitle_mode: subtitleMode,
      source_language: sourceLanguage,
      target_language: targetLanguage,
      original_style: showOriginal ? originalStyle : undefined,
      translated_style: showTranslated ? translatedStyle : undefined,
    })
  }

  // Handle update preset
  const handleUpdatePreset = (preset: SubtitlePreset) => {
    const subtitleMode = getSubtitleMode()
    if (!subtitleMode) return

    updatePresetMutation.mutate({
      presetId: preset.id,
      data: {
        name: preset.name,
        description: preset.description,
        is_vertical: preset.is_vertical ?? false,
        subtitle_mode: subtitleMode,
        source_language: sourceLanguage,
        target_language: targetLanguage,
        original_style: showOriginal ? originalStyle : undefined,
        translated_style: showTranslated ? translatedStyle : undefined,
      },
    })
  }

  // Check if save is allowed
  const canSavePreset = savePresetName.trim() && (showOriginal || showTranslated)

  const fonts = fontsData?.fonts || []
  const recommendedFont = fontsData?.recommended || null
  const presets = presetsData?.presets || []

  // Get the currently selected preset (for update button)
  const selectedPreset = presets.find(p => p.id === selectedPresetId)

  // Wrapper functions for style changes (keep preset selected so user can update it)
  const handleOriginalStyleChange = (style: SubtitleStyle) => {
    onOriginalStyleChange(style)
  }

  const handleTranslatedStyleChange = (style: SubtitleStyle) => {
    onTranslatedStyleChange(style)
  }

  return (
    <div className="flex gap-6">
      {/* Left Column: Sticky Preview */}
      <div className="w-[600px] flex-shrink-0">
        <div className="sticky top-4 space-y-4">
          {/* Live Preview */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h4 className="text-sm font-medium text-gray-900">{t('styleEditor.livePreview')}</h4>
                <p className="text-xs text-gray-400 mt-0.5">{t('styleEditor.previewNote')}</p>
              </div>
            </div>
            <div className="flex items-center space-x-3 mb-3">
              <label className="flex items-center text-xs text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showOriginal}
                  onChange={(e) => handleShowOriginalChange(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 mr-1"
                />
                {t('styleEditor.preview.showOriginal')}
              </label>
              <label className="flex items-center text-xs text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showTranslated}
                  onChange={(e) => handleShowTranslatedChange(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 mr-1"
                />
                {t('styleEditor.preview.showTranslated')}
              </label>
            </div>
            <SubtitlePreview
              originalStyle={originalStyle}
              translatedStyle={translatedStyle}
              chineseOnTop={translatedOnTop}
              showOriginal={showOriginal}
              showTranslated={showTranslated}
              originalText={customOriginalText || undefined}
              translatedText={customTranslatedText || undefined}
              isVertical={selectedPreset?.is_vertical ?? false}
            />
            {/* Custom text inputs for testing line wrap */}
            <div className="mt-3 space-y-2">
              <div>
                <label className="block text-xs text-gray-500 mb-1">{t('styleEditor.preview.testOriginal')}</label>
                <input
                  type="text"
                  value={customOriginalText}
                  onChange={(e) => setCustomOriginalText(e.target.value)}
                  placeholder={t('previewText.original')}
                  className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">{t('styleEditor.preview.testTranslated')}</label>
                <input
                  type="text"
                  value={customTranslatedText}
                  onChange={(e) => setCustomTranslatedText(e.target.value)}
                  placeholder={t('previewText.translated')}
                  className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* Right Column: Settings */}
      <div className="flex-1 min-w-0 space-y-4">
        {/* Preset Selector */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-gray-900">{t('styleEditor.presets')}</h4>
            <button
              onClick={() => setShowSaveModal(true)}
              className="flex items-center px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded"
            >
              <Plus className="h-3 w-3 mr-1" />
              {t('styleEditor.saveAsPreset')}
            </button>
          </div>
          {presetsLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {presets.map((preset) => {
                const isActive = isPresetActive(preset)
                return (
                  <button
                    key={preset.id}
                    onClick={() => applyPreset(preset)}
                    className={`relative p-3 border-2 rounded-lg transition-colors text-left group ${
                      isActive
                        ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                        : 'border-gray-200 hover:border-blue-400 hover:bg-blue-50'
                    }`}
                  >
                    {isActive && (
                      <div className="absolute -top-2 -right-2 bg-blue-500 rounded-full p-0.5">
                        <Check className="h-3 w-3 text-white" />
                      </div>
                    )}
                    <div className={`text-sm font-medium ${isActive ? 'text-blue-700' : 'text-gray-800'}`}>
                      {preset.name}
                    </div>
                    <div className="flex items-center space-x-1 mt-0.5">
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                        preset.subtitle_mode === 'dual'
                          ? 'bg-purple-100 text-purple-600'
                          : preset.subtitle_mode === 'translated_only'
                          ? 'bg-blue-100 text-blue-600'
                          : 'bg-green-100 text-green-600'
                      }`}>
                        {preset.subtitle_mode === 'dual'
                          ? t('styleEditor.presetMode.dual')
                          : preset.subtitle_mode === 'translated_only'
                          ? t('styleEditor.presetMode.translatedOnly')
                          : t('styleEditor.presetMode.originalOnly')
                        }
                      </span>
                    </div>
                    {preset.is_builtin && (
                      <span className="absolute top-1 right-1 text-[10px] text-gray-400">{t('styleEditor.builtIn')}</span>
                    )}
                    {!preset.is_builtin && !isActive && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          if (confirm(t('styleEditor.deletePreset'))) {
                            deletePresetMutation.mutate(preset.id)
                          }
                        }}
                        className="absolute top-1 right-1 p-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    )}
                  </button>
                )
              })}
            </div>
          )}

          {/* Update preset button - show when a custom preset is selected */}
          {selectedPreset && !selectedPreset.is_builtin && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <button
                onClick={() => handleUpdatePreset(selectedPreset)}
                disabled={updatePresetMutation.isPending}
                className="w-full px-3 py-2 bg-green-600 text-white rounded-md text-sm font-medium hover:bg-green-700 disabled:opacity-50 flex items-center justify-center"
              >
                {updatePresetMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Save className="h-4 w-4 mr-2" />
                )}
                {t('styleEditor.updatePreset', { name: selectedPreset.name })}
              </button>
            </div>
          )}
        </div>

        {/* Subtitle Mode Selection */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h4 className="text-sm font-medium text-gray-900 mb-3">{t('styleEditor.subtitleMode.title')}</h4>
          <div className="flex space-x-6">
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={showTranslated}
                onChange={(e) => handleShowTranslatedChange(e.target.checked)}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-700">{t('styleEditor.subtitleMode.showTranslated')}</span>
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={showOriginal}
                onChange={(e) => handleShowOriginalChange(e.target.checked)}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-700">{t('styleEditor.subtitleMode.showOriginal')}</span>
            </label>
          </div>
          {!showOriginal && !showTranslated && (
            <p className="mt-2 text-xs text-red-500">{t('styleEditor.subtitleMode.noSelectionWarning')}</p>
          )}
        </div>

        {/* Style Tabs - Only show if at least one subtitle is enabled */}
        {(showOriginal || showTranslated) && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          {/* Tab Headers - Only show both if dual mode */}
          {showOriginal && showTranslated ? (
            <div className="flex border-b border-gray-200">
              <button
                onClick={() => setActiveTab('translated')}
                className={`flex-1 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'translated'
                    ? 'border-blue-600 text-blue-600 bg-blue-50'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {t('styleEditor.tabs.translated')}
              </button>
              <button
                onClick={() => setActiveTab('original')}
                className={`flex-1 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'original'
                    ? 'border-blue-600 text-blue-600 bg-blue-50'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {t('styleEditor.tabs.original')}
              </button>
            </div>
          ) : (
            <div className="px-4 py-3 border-b border-gray-200">
              <span className="text-sm font-medium text-gray-700">
                {showTranslated ? t('styleEditor.tabs.translated') : t('styleEditor.tabs.original')}
              </span>
            </div>
          )}

          {/* Tab Content */}
          <div className="p-4">
            {/* Language Selector */}
            <div className="mb-4 pb-4 border-b border-gray-200">
              <label className="block text-xs text-gray-600 mb-1">
                {(showOriginal && showTranslated)
                  ? (activeTab === 'translated' ? t('styleEditor.language.targetLanguage') : t('styleEditor.language.sourceLanguage'))
                  : (showTranslated ? t('styleEditor.language.targetLanguage') : t('styleEditor.language.sourceLanguage'))
                }
              </label>
              <select
                value={(showOriginal && showTranslated)
                  ? (activeTab === 'translated' ? targetLanguage : sourceLanguage)
                  : (showTranslated ? targetLanguage : sourceLanguage)
                }
                onChange={(e) => {
                  if ((showOriginal && showTranslated && activeTab === 'translated') || (!showOriginal && showTranslated)) {
                    onTargetLanguageChange?.(e.target.value)
                  } else {
                    onSourceLanguageChange?.(e.target.value)
                  }
                }}
                className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm bg-white"
              >
                {SUPPORTED_LANGUAGES.map(lang => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name} ({lang.nameEn})
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-gray-400">
                {t('styleEditor.language.fontHint')}
              </p>
            </div>

            {/* Style Panel - Show appropriate panel based on mode */}
            {showOriginal && showTranslated ? (
              // Dual mode - show tab content
              activeTab === 'translated' ? (
                <StylePanel
                  style={translatedStyle}
                  onChange={handleTranslatedStyleChange}
                  fonts={fonts}
                  language={targetLanguage}
                  recommendedFont={recommendedFont}
                />
              ) : (
                <StylePanel
                  style={originalStyle}
                  onChange={handleOriginalStyleChange}
                  fonts={fonts}
                  language={sourceLanguage}
                  recommendedFont={recommendedFont}
                />
              )
            ) : showTranslated ? (
              // Translated only mode
              <StylePanel
                style={translatedStyle}
                onChange={handleTranslatedStyleChange}
                fonts={fonts}
                language={targetLanguage}
                recommendedFont={recommendedFont}
              />
            ) : (
              // Original only mode
              <StylePanel
                style={originalStyle}
                onChange={handleOriginalStyleChange}
                fonts={fonts}
                language={sourceLanguage}
                recommendedFont={recommendedFont}
              />
            )}
          </div>
        </div>
        )}

        {/* Font Upload */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h4 className="text-sm font-medium text-gray-900 mb-3">{t('styleEditor.customFonts.title')}</h4>
          <div className="flex items-center space-x-3">
            <label className="flex items-center px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg cursor-pointer transition-colors">
              <Upload className="h-4 w-4 mr-2 text-gray-600" />
              <span className="text-sm text-gray-700">{t('styleEditor.customFonts.uploadFont')}</span>
              <input
                type="file"
                accept=".ttf,.ttc,.otf"
                onChange={handleFontUpload}
                className="hidden"
              />
            </label>
            {uploadFontMutation.isPending && (
              <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
            )}
            {uploadFontMutation.isSuccess && (
              <span className="text-sm text-green-600">
                <Check className="h-4 w-4 inline mr-1" />
                {t('styleEditor.customFonts.uploaded')}
              </span>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {t('styleEditor.customFonts.supportedFormats')}
          </p>
        </div>
      </div>
      {/* End Right Column */}

      {/* Save Preset Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">{t('styleEditor.saveAsPreset')}</h3>
              <button
                onClick={() => setShowSaveModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('styleEditor.presetName')}
                </label>
                <input
                  type="text"
                  value={savePresetName}
                  onChange={(e) => setSavePresetName(e.target.value)}
                  placeholder={t('styleEditor.presetNamePlaceholder')}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
              <div>
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={savePresetIsVertical}
                    onChange={(e) => setSavePresetIsVertical(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 mr-2"
                  />
                  <span className="text-sm text-gray-700">{t('styleEditor.isVerticalPreset')}</span>
                </label>
                <p className="text-xs text-gray-500 mt-1 ml-6">{t('styleEditor.isVerticalPresetHint')}</p>
              </div>
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setShowSaveModal(false)}
                  className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
                >
                  {t('common.cancel')}
                </button>
                <button
                  onClick={handleSavePreset}
                  disabled={!canSavePreset || savePresetMutation.isPending}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 flex items-center"
                >
                  {savePresetMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                  ) : (
                    <Save className="h-4 w-4 mr-1" />
                  )}
                  {t('common.save')}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

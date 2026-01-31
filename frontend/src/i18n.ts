import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

import en from './locales/en.json'
import zhCN from './locales/zh-CN.json'

// Available languages configuration
export const LANGUAGES = [
  { code: 'en', name: 'English', nativeName: 'English', dir: 'ltr' },
  { code: 'zh-CN', name: 'Chinese (Simplified)', nativeName: '简体中文', dir: 'ltr' },
  { code: 'zh-TW', name: 'Chinese (Traditional)', nativeName: '繁體中文', dir: 'ltr' },
  { code: 'ja', name: 'Japanese', nativeName: '日本語', dir: 'ltr' },
  { code: 'ko', name: 'Korean', nativeName: '한국어', dir: 'ltr' },
  { code: 'es', name: 'Spanish', nativeName: 'Español', dir: 'ltr' },
  { code: 'fr', name: 'French', nativeName: 'Français', dir: 'ltr' },
  { code: 'de', name: 'German', nativeName: 'Deutsch', dir: 'ltr' },
  { code: 'ru', name: 'Russian', nativeName: 'Русский', dir: 'ltr' },
  { code: 'ar', name: 'Arabic', nativeName: 'العربية', dir: 'rtl' },
  { code: 'pt', name: 'Portuguese', nativeName: 'Português', dir: 'ltr' },
  { code: 'he', name: 'Hebrew', nativeName: 'עברית', dir: 'rtl' },
] as const

export type LanguageCode = typeof LANGUAGES[number]['code']

// Resources bundled with the app
const resources = {
  en: { translation: en },
  'zh-CN': { translation: zhCN },
  // Other languages will be loaded dynamically or added later
}

// Initialize i18next
i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    supportedLngs: LANGUAGES.map(l => l.code),

    detection: {
      // Order of language detection
      order: ['localStorage', 'navigator', 'htmlTag'],
      // Cache user language preference
      caches: ['localStorage'],
      lookupLocalStorage: 'i18nextLng',
    },

    interpolation: {
      escapeValue: false, // React already escapes
    },

    react: {
      useSuspense: false, // Disable suspense for SSR compatibility
    },
  })

// Helper to get language direction (LTR or RTL)
export function getLanguageDirection(langCode: string): 'ltr' | 'rtl' {
  const lang = LANGUAGES.find(l => l.code === langCode)
  return lang?.dir || 'ltr'
}

// Helper to check if language is RTL
export function isRTL(langCode: string): boolean {
  return getLanguageDirection(langCode) === 'rtl'
}

// Helper to get language info
export function getLanguageInfo(langCode: string) {
  return LANGUAGES.find(l => l.code === langCode) || LANGUAGES[0]
}

export default i18n

import { Outlet, Link, useLocation } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Home, PlusCircle, Settings, Video, Type, Rss, TrendingUp, Moon, Sun } from 'lucide-react'
import clsx from 'clsx'
import LanguageSelector from './LanguageSelector'
import { isRTL } from '../i18n'
import { useTheme } from '../contexts/ThemeContext'

export default function Layout() {
  const location = useLocation()
  const { t, i18n } = useTranslation()
  const rtl = isRTL(i18n.language)
  const { theme, toggleTheme } = useTheme()

  const navItems = [
    { path: '/', icon: Home, label: t('nav.dashboard') },
    { path: '/new', icon: PlusCircle, label: t('nav.newTask') },
    { path: '/discover', icon: TrendingUp, label: t('nav.discover') },
    { path: '/subscriptions', icon: Rss, label: t('nav.subscriptions') },
    { path: '/presets', icon: Type, label: t('nav.presets') },
    { path: '/settings', icon: Settings, label: t('nav.settings') },
  ]

  return (
    <div className={clsx('min-h-screen bg-gray-100 dark:bg-gray-900 transition-colors', rtl && 'rtl')}>
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm dark:shadow-gray-700/50">
        <div className="mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Video className="h-8 w-8 text-blue-600 dark:text-blue-400" />
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">iDubb</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                title={theme === 'dark' ? '切换到亮色模式' : '切换到暗色模式'}
              >
                {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </button>
              <LanguageSelector />
            </div>
          </div>
        </div>
      </header>

      {/* Sidebar - Fixed position, icon only */}
      <nav
        className={clsx(
          'fixed top-20 z-40 w-14',
          rtl ? 'right-4' : 'left-4'
        )}
      >
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm dark:shadow-gray-700/50 p-2">
          <ul className="space-y-1">
            {navItems.map(({ path, icon: Icon, label }) => (
              <li key={path}>
                <Link
                  to={path}
                  className={clsx(
                    'flex items-center justify-center p-3 rounded-lg transition-colors',
                    location.pathname === path
                      ? 'bg-blue-50 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300'
                      : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'
                  )}
                  title={label}
                >
                  <Icon className="h-5 w-5" />
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      {/* Main content - Centered */}
      <main className="py-8 px-4">
        <div className="max-w-4xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  )
}

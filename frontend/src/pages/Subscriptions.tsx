import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  Youtube,
  PlaySquare,
  Plus,
  RefreshCw,
  Trash2,
  ToggleLeft,
  ToggleRight,
  Clock,
  AlertCircle,
  CheckCircle,
  ExternalLink,
  Instagram,
  Music2,
  FolderDown,
  Settings2
} from 'lucide-react'
import { subscriptionApi, settingsApi } from '../services/api'
import type { Subscription } from '../types'
import AddSubscriptionModal from '../components/AddSubscriptionModal'
import BatchImportModal from '../components/BatchImportModal'

export default function Subscriptions() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [subscriptions, setSubscriptions] = useState<Subscription[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [checkingId, setCheckingId] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  // Edit modal state
  const [editingSubscription, setEditingSubscription] = useState<Subscription | null>(null)
  // Batch import modal state
  const [batchImportSubscription, setBatchImportSubscription] = useState<Subscription | null>(null)
  // User timezone
  const [userTimezone, setUserTimezone] = useState<string>('Asia/Shanghai')

  const fetchSubscriptions = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await subscriptionApi.list()
      setSubscriptions(data)
    } catch (err) {
      setError(t('subscriptions.loadError'))
      console.error('Failed to load subscriptions:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchSettings = async () => {
    try {
      const settings = await settingsApi.get()
      if (settings.processing?.timezone) {
        setUserTimezone(settings.processing.timezone)
      }
    } catch (err) {
      console.error('Failed to load settings:', err)
    }
  }

  useEffect(() => {
    fetchSubscriptions()
    fetchSettings()
  }, [])

  const handleToggleEnabled = async (subscription: Subscription) => {
    try {
      if (subscription.enabled) {
        await subscriptionApi.disable(subscription.id)
      } else {
        await subscriptionApi.enable(subscription.id)
      }
      // Refresh list
      fetchSubscriptions()
    } catch (err) {
      console.error('Failed to toggle subscription:', err)
    }
  }

  const handleCheckNow = async (subscriptionId: string) => {
    try {
      setCheckingId(subscriptionId)
      const result = await subscriptionApi.checkNow(subscriptionId)
      if (result.count > 0) {
        alert(t('subscriptions.newVideosFound', { count: result.count }))
      } else {
        alert(t('subscriptions.noNewVideos'))
      }
      fetchSubscriptions()
    } catch (err) {
      console.error('Failed to check subscription:', err)
      alert(t('subscriptions.checkFailed'))
    } finally {
      setCheckingId(null)
    }
  }

  const handleDelete = async (subscriptionId: string) => {
    if (!confirm(t('subscriptions.confirmDelete'))) {
      return
    }

    try {
      setDeletingId(subscriptionId)
      await subscriptionApi.delete(subscriptionId)
      fetchSubscriptions()
    } catch (err) {
      console.error('Failed to delete subscription:', err)
      alert(t('subscriptions.deleteFailed'))
    } finally {
      setDeletingId(null)
    }
  }

  const handleBatchTasksCreated = (taskIds: string[]) => {
    // Navigate to dashboard to see the created tasks
    if (taskIds.length > 0) {
      navigate('/')
    }
  }

  const getPlatformIcon = (platform: string) => {
    switch (platform) {
      case 'youtube':
        return <Youtube className="h-5 w-5 text-red-500" />
      case 'tiktok':
        return <Music2 className="h-5 w-5 text-gray-900" />
      case 'instagram':
        return <Instagram className="h-5 w-5 text-pink-500" />
      default:
        return <PlaySquare className="h-5 w-5 text-gray-500" />
    }
  }

  const formatRelativeTime = (dateString?: string) => {
    if (!dateString) return '-'
    const date = new Date(dateString)
    const now = new Date()
    const diffMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60))

    if (diffMinutes < 1) return t('subscriptions.justNow')
    if (diffMinutes < 60) return t('subscriptions.minutesAgo', { count: diffMinutes })
    if (diffMinutes < 1440) return t('subscriptions.hoursAgo', { count: Math.floor(diffMinutes / 60) })
    return t('subscriptions.daysAgo', { count: Math.floor(diffMinutes / 1440) })
  }

  const formatVideoDate = (dateString?: string) => {
    if (!dateString) return ''
    try {
      const date = new Date(dateString)
      return date.toLocaleString('zh-CN', {
        timeZone: userTimezone,
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      })
    } catch {
      return ''
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{t('subscriptions.title')}</h1>
          <p className="text-sm text-gray-500 mt-1">{t('subscriptions.description')}</p>
        </div>
        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <Plus className="h-4 w-4" />
          {t('subscriptions.addSubscription')}
        </button>
      </div>

      {/* Error message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-red-500" />
          <span className="text-red-700">{error}</span>
          <button
            onClick={fetchSubscriptions}
            className="ml-auto text-red-600 hover:text-red-800"
          >
            {t('common.retry')}
          </button>
        </div>
      )}

      {/* Subscriptions list */}
      {subscriptions.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm p-8 text-center">
          <PlaySquare className="h-12 w-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">{t('subscriptions.noSubscriptions')}</h3>
          <p className="text-gray-500 mb-4">{t('subscriptions.noSubscriptionsDesc')}</p>
          <button
            onClick={() => setIsModalOpen(true)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <Plus className="h-4 w-4" />
            {t('subscriptions.addFirst')}
          </button>
        </div>
      ) : (
        <div className="grid gap-4">
          {subscriptions.map((subscription) => (
            <div
              key={subscription.id}
              className={`bg-white rounded-lg shadow-sm p-4 ${!subscription.enabled ? 'opacity-60' : ''}`}
            >
              <div className="flex items-start gap-4">
                {/* Avatar */}
                <div className="flex-shrink-0">
                  {subscription.channel_avatar ? (
                    <img
                      src={subscription.channel_avatar}
                      alt={subscription.channel_name}
                      className="w-14 h-14 rounded-full object-cover"
                    />
                  ) : (
                    <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center">
                      {getPlatformIcon(subscription.platform)}
                    </div>
                  )}
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {getPlatformIcon(subscription.platform)}
                    <h3 className="text-lg font-medium text-gray-900 truncate">
                      {subscription.channel_name}
                    </h3>
                    {subscription.channel_url && (
                      <a
                        href={subscription.channel_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-gray-400 hover:text-blue-500"
                      >
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    )}
                  </div>

                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-gray-500">
                    <span className="flex items-center gap-1">
                      <Clock className="h-3.5 w-3.5" />
                      {t('subscriptions.checkEvery', { minutes: subscription.check_interval })}
                    </span>
                    {subscription.last_checked_at && (
                      <span>
                        {t('subscriptions.lastChecked')}: {formatRelativeTime(subscription.last_checked_at)}
                      </span>
                    )}
                    {subscription.auto_process && (
                      <span className="flex items-center gap-1 text-green-600">
                        <CheckCircle className="h-3.5 w-3.5" />
                        {t('subscriptions.autoProcess')}
                      </span>
                    )}
                  </div>

                  {/* Last video */}
                  {subscription.last_video_title && (
                    <div className="mt-2 text-sm text-gray-600 truncate">
                      <span className="text-gray-400">{t('subscriptions.lastVideo')}:</span>{' '}
                      {subscription.last_video_published_at && (
                        <span className="text-blue-500 mr-1">
                          [{formatVideoDate(subscription.last_video_published_at)}]
                        </span>
                      )}
                      {subscription.last_video_title}
                    </div>
                  )}

                  {/* Error */}
                  {subscription.last_error && (
                    <div className="mt-2 text-sm text-red-600 flex items-center gap-1">
                      <AlertCircle className="h-3.5 w-3.5" />
                      {subscription.last_error}
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2">
                  {/* Edit */}
                  <button
                    onClick={() => setEditingSubscription(subscription)}
                    className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                    title={t('subscriptions.editSettings')}
                  >
                    <Settings2 className="h-5 w-5" />
                  </button>

                  {/* Batch Import */}
                  {['youtube', 'tiktok'].includes(subscription.platform) && (
                    <button
                      onClick={() => setBatchImportSubscription(subscription)}
                      className="p-2 text-gray-500 hover:text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
                      title={t('subscriptions.batchImport.button')}
                    >
                      <FolderDown className="h-5 w-5" />
                    </button>
                  )}

                  <button
                    onClick={() => handleCheckNow(subscription.id)}
                    disabled={checkingId === subscription.id}
                    className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors disabled:opacity-50"
                    title={t('subscriptions.checkNow')}
                  >
                    <RefreshCw className={`h-5 w-5 ${checkingId === subscription.id ? 'animate-spin' : ''}`} />
                  </button>

                  <button
                    onClick={() => handleToggleEnabled(subscription)}
                    className={`p-2 rounded-lg transition-colors ${
                      subscription.enabled
                        ? 'text-green-600 hover:bg-green-50'
                        : 'text-gray-400 hover:bg-gray-50'
                    }`}
                    title={subscription.enabled ? t('subscriptions.disable') : t('subscriptions.enable')}
                  >
                    {subscription.enabled ? (
                      <ToggleRight className="h-5 w-5" />
                    ) : (
                      <ToggleLeft className="h-5 w-5" />
                    )}
                  </button>

                  <button
                    onClick={() => handleDelete(subscription.id)}
                    disabled={deletingId === subscription.id}
                    className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                    title={t('common.delete')}
                  >
                    <Trash2 className="h-5 w-5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Add/Edit Subscription Modal */}
      <AddSubscriptionModal
        isOpen={isModalOpen || !!editingSubscription}
        onClose={() => {
          setIsModalOpen(false)
          setEditingSubscription(null)
        }}
        onCreated={() => {
          setIsModalOpen(false)
          setEditingSubscription(null)
          fetchSubscriptions()
        }}
        editSubscription={editingSubscription || undefined}
      />

      {/* Batch Import Modal */}
      {batchImportSubscription && (
        <BatchImportModal
          isOpen={true}
          onClose={() => setBatchImportSubscription(null)}
          subscription={batchImportSubscription}
          onTasksCreated={handleBatchTasksCreated}
        />
      )}
    </div>
  )
}

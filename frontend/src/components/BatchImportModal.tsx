import { useState, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import {
  X,
  Calendar,
  Download,
  CheckSquare,
  Square,
  Loader2,
  AlertCircle,
  CheckCircle,
  Play,
  Clock,
  ExternalLink
} from 'lucide-react'
import { subscriptionApi } from '../services/api'
import type { Subscription, VideoItem, ProcessOptions } from '../types'

interface BatchImportModalProps {
  isOpen: boolean
  onClose: () => void
  subscription: Subscription
  onTasksCreated?: (taskIds: string[]) => void
}

export default function BatchImportModal({
  isOpen,
  onClose,
  subscription,
  onTasksCreated
}: BatchImportModalProps) {
  const { t } = useTranslation()

  // Date range state
  const [startDate, setStartDate] = useState(() => {
    // Default to 1 month ago
    const date = new Date()
    date.setMonth(date.getMonth() - 1)
    return date.toISOString().split('T')[0]
  })
  const [endDate, setEndDate] = useState(() => {
    // Default to today
    return new Date().toISOString().split('T')[0]
  })

  // Video fetching state
  const [videos, setVideos] = useState<VideoItem[]>([])
  const [selectedVideoIds, setSelectedVideoIds] = useState<Set<string>>(new Set())
  const [fetchLoading, setFetchLoading] = useState(false)
  const [fetchError, setFetchError] = useState<string | null>(null)

  // Task creation state
  const [createLoading, setCreateLoading] = useState(false)
  const [createResult, setCreateResult] = useState<{
    success: boolean
    created: number
    failed: number
    errors: string[]
  } | null>(null)

  // Computed date limits (max 3 months back)
  const minDate = useMemo(() => {
    const date = new Date()
    date.setMonth(date.getMonth() - 3)
    return date.toISOString().split('T')[0]
  }, [])

  const maxDate = useMemo(() => {
    return new Date().toISOString().split('T')[0]
  }, [])

  // Handle video selection
  const toggleVideoSelection = (videoId: string) => {
    setSelectedVideoIds(prev => {
      const newSet = new Set(prev)
      if (newSet.has(videoId)) {
        newSet.delete(videoId)
      } else {
        newSet.add(videoId)
      }
      return newSet
    })
  }

  const selectAll = () => {
    setSelectedVideoIds(new Set(videos.map(v => v.video_id)))
  }

  const deselectAll = () => {
    setSelectedVideoIds(new Set())
  }

  // Fetch videos in date range
  const handleFetchVideos = async () => {
    setFetchLoading(true)
    setFetchError(null)
    setVideos([])
    setSelectedVideoIds(new Set())
    setCreateResult(null)

    try {
      const response = await subscriptionApi.fetchHistoricalVideos(
        subscription.id,
        startDate,
        endDate,
        50
      )

      if (response.success) {
        setVideos(response.videos)
        // Auto-select all videos
        setSelectedVideoIds(new Set(response.videos.map(v => v.video_id)))
      } else {
        setFetchError(response.error || t('subscriptions.batchImport.fetchFailed'))
      }
    } catch (err) {
      console.error('Failed to fetch videos:', err)
      setFetchError(t('subscriptions.batchImport.fetchFailed'))
    } finally {
      setFetchLoading(false)
    }
  }

  // Create tasks for selected videos
  const handleCreateTasks = async () => {
    const selectedVideos = videos.filter(v => selectedVideoIds.has(v.video_id))
    if (selectedVideos.length === 0) return

    setCreateLoading(true)
    setCreateResult(null)

    try {
      // Use the subscription's process_options
      const response = await subscriptionApi.batchCreateTasks(
        subscription.id,
        selectedVideos,
        subscription.process_options as ProcessOptions
      )

      setCreateResult({
        success: response.success,
        created: response.created_count,
        failed: response.failed_count,
        errors: response.errors,
      })

      if (response.success && onTasksCreated) {
        onTasksCreated(response.task_ids)
      }
    } catch (err) {
      console.error('Failed to create tasks:', err)
      setCreateResult({
        success: false,
        created: 0,
        failed: selectedVideos.length,
        errors: [t('subscriptions.batchImport.createFailed')],
      })
    } finally {
      setCreateLoading(false)
    }
  }

  // Format duration
  const formatDuration = (seconds?: number) => {
    if (!seconds) return ''
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Format date
  const formatDate = (dateStr?: string) => {
    if (!dateStr) return ''
    const date = new Date(dateStr)
    return date.toLocaleDateString()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              {t('subscriptions.batchImport.title')}
            </h2>
            <p className="text-sm text-gray-500 mt-0.5">
              {subscription.channel_name}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Date Range Selection */}
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Calendar className="h-4 w-4 text-gray-500" />
              <span className="text-sm font-medium text-gray-700">
                {t('subscriptions.batchImport.dateRange')}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600">
                  {t('subscriptions.batchImport.from')}
                </label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  min={minDate}
                  max={endDate}
                  className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600">
                  {t('subscriptions.batchImport.to')}
                </label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  min={startDate}
                  max={maxDate}
                  className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <button
                onClick={handleFetchVideos}
                disabled={fetchLoading}
                className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {fetchLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Download className="h-4 w-4" />
                )}
                {t('subscriptions.batchImport.fetchVideos')}
              </button>
            </div>

            <p className="text-xs text-gray-500 mt-2">
              {t('subscriptions.batchImport.dateRangeHint')}
            </p>
          </div>

          {/* Fetch Error */}
          {fetchError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0" />
              <span className="text-sm text-red-700">{fetchError}</span>
            </div>
          )}

          {/* Video List */}
          {videos.length > 0 && (
            <div>
              {/* Selection Controls */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <span className="text-sm text-gray-600">
                    {t('subscriptions.batchImport.foundVideos', { count: videos.length })}
                  </span>
                  <span className="text-sm text-blue-600">
                    {t('subscriptions.batchImport.selectedCount', { count: selectedVideoIds.size })}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={selectAll}
                    className="text-sm text-blue-600 hover:text-blue-700"
                  >
                    {t('subscriptions.batchImport.selectAll')}
                  </button>
                  <span className="text-gray-300">|</span>
                  <button
                    onClick={deselectAll}
                    className="text-sm text-gray-600 hover:text-gray-700"
                  >
                    {t('subscriptions.batchImport.deselectAll')}
                  </button>
                </div>
              </div>

              {/* Video Grid */}
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {videos.map((video) => (
                  <div
                    key={video.video_id}
                    className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedVideoIds.has(video.video_id)
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300 bg-white'
                    }`}
                    onClick={() => toggleVideoSelection(video.video_id)}
                  >
                    {/* Checkbox */}
                    <div className="flex-shrink-0 pt-1">
                      {selectedVideoIds.has(video.video_id) ? (
                        <CheckSquare className="h-5 w-5 text-blue-600" />
                      ) : (
                        <Square className="h-5 w-5 text-gray-400" />
                      )}
                    </div>

                    {/* Thumbnail */}
                    {video.thumbnail_url && (
                      <div className="flex-shrink-0 w-32 h-18 relative">
                        <img
                          src={video.thumbnail_url}
                          alt={video.title}
                          className="w-full h-full object-cover rounded"
                        />
                        {video.duration && (
                          <span className="absolute bottom-1 right-1 bg-black/80 text-white text-xs px-1 rounded">
                            {formatDuration(video.duration)}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium text-gray-900 line-clamp-2">
                        {video.title}
                      </h4>
                      <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                        {video.published_at && (
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {formatDate(video.published_at)}
                          </span>
                        )}
                        <a
                          href={video.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1 text-blue-600 hover:text-blue-700"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <ExternalLink className="h-3 w-3" />
                          {t('subscriptions.batchImport.viewVideo')}
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Create Result */}
          {createResult && (
            <div className={`rounded-lg p-4 ${
              createResult.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
            }`}>
              <div className="flex items-center gap-2 mb-2">
                {createResult.success ? (
                  <CheckCircle className="h-5 w-5 text-green-600" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-red-600" />
                )}
                <span className={`font-medium ${createResult.success ? 'text-green-800' : 'text-red-800'}`}>
                  {createResult.success
                    ? t('subscriptions.batchImport.createSuccess', { count: createResult.created })
                    : t('subscriptions.batchImport.createPartialFail', {
                        created: createResult.created,
                        failed: createResult.failed
                      })
                  }
                </span>
              </div>
              {createResult.errors.length > 0 && (
                <ul className="text-sm text-red-700 list-disc list-inside">
                  {createResult.errors.slice(0, 3).map((err, i) => (
                    <li key={i}>{err}</li>
                  ))}
                  {createResult.errors.length > 3 && (
                    <li>{t('subscriptions.batchImport.moreErrors', { count: createResult.errors.length - 3 })}</li>
                  )}
                </ul>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-4 border-t bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900"
          >
            {t('common.close')}
          </button>
          <button
            onClick={handleCreateTasks}
            disabled={createLoading || selectedVideoIds.size === 0}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {createLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            {t('subscriptions.batchImport.createTasks', { count: selectedVideoIds.size })}
          </button>
        </div>
      </div>
    </div>
  )
}

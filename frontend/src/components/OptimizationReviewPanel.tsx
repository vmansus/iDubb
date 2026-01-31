import { useRef, useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Sparkles, AlertCircle, CheckCircle, Play } from 'lucide-react'
import VideoPlayer, { VideoPlayerHandle } from './VideoPlayer'

interface OptimizationChange {
  index: number
  start_time?: number
  end_time?: number
  original_text: string
  translated_text: string
  optimized_text: string
  suggestions: string[]
  issues: Array<{
    type?: string
    severity?: string
    message?: string
    suggestion?: string
  }>
}

interface OptimizationReviewPanelProps {
  videoUrl: string
  changes: OptimizationChange[]
  optimizedCount: number
  totalSegments: number
  onClose: () => void
}

export default function OptimizationReviewPanel({
  videoUrl,
  changes,
  optimizedCount,
  totalSegments,
  onClose
}: OptimizationReviewPanelProps) {
  const { t } = useTranslation()
  const videoPlayerRef = useRef<VideoPlayerHandle>(null)
  const changesContainerRef = useRef<HTMLDivElement>(null)

  const [currentTime, setCurrentTime] = useState(0)
  const [activeChangeIndex, setActiveChangeIndex] = useState<number | null>(null)

  // Format seconds to MM:SS or HH:MM:SS
  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    }
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  // Seek to specific time
  const handleSeekTo = (time: number, changeIdx: number) => {
    videoPlayerRef.current?.seekTo(time)
    setActiveChangeIndex(changeIdx)
  }

  // Update active change based on current video time
  useEffect(() => {
    if (changes.length === 0) return

    const currentChange = changes.findIndex((change) => {
      const start = change.start_time ?? 0
      const end = change.end_time ?? start + 5
      return currentTime >= start && currentTime < end
    })

    if (currentChange !== -1 && currentChange !== activeChangeIndex) {
      setActiveChangeIndex(currentChange)
      // Scroll to active change
      const changeElement = document.getElementById(`change-${currentChange}`)
      if (changeElement && changesContainerRef.current) {
        changeElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }
  }, [currentTime, changes, activeChangeIndex])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50'
      case 'error': return 'text-orange-600 bg-orange-50'
      case 'warning': return 'text-yellow-600 bg-yellow-50'
      default: return 'text-blue-600 bg-blue-50'
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-purple-50 to-white">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-100 rounded-lg">
            <Sparkles className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              {t('optimization.resultTitle')}
            </h2>
            <p className="text-sm text-gray-500">
              {t('optimization.resultSummary', { count: optimizedCount, total: totalSegments })}
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>
      </div>

      {/* Main content - Split view */}
      <div className="flex flex-col lg:flex-row" style={{ height: 'calc(100vh - 300px)', minHeight: '500px' }}>
        {/* Left side - Video Player */}
        <div className="lg:w-1/2 bg-black flex items-center justify-center">
          <VideoPlayer
            ref={videoPlayerRef}
            src={videoUrl}
            maxHeight="100%"
            showDownload={false}
            onTimeUpdate={setCurrentTime}
            className="w-full h-full"
          />
        </div>

        {/* Right side - Changes list */}
        <div
          ref={changesContainerRef}
          className="lg:w-1/2 overflow-y-auto bg-gray-50 p-4"
        >
          {changes.length === 0 ? (
            <div className="text-center py-12">
              <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
              <p className="text-gray-600">{t('optimization.noChangesNeeded')}</p>
            </div>
          ) : (
            <div className="space-y-4">
              {changes.map((change, idx) => (
                <div
                  key={idx}
                  id={`change-${idx}`}
                  className={`border rounded-xl overflow-hidden bg-white shadow-sm transition-all duration-300 ${
                    activeChangeIndex === idx
                      ? 'border-purple-400 ring-2 ring-purple-200'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  {/* Segment header */}
                  <div className={`px-4 py-3 border-b flex items-center justify-between ${
                    activeChangeIndex === idx ? 'bg-purple-50 border-purple-200' : 'bg-gray-50 border-gray-200'
                  }`}>
                    <div className="flex items-center gap-3">
                      <span className="font-medium text-gray-700">
                        {t('optimization.segment')} #{change.index + 1}
                      </span>
                      {/* Timestamp button */}
                      {change.start_time !== undefined && (
                        <button
                          onClick={() => handleSeekTo(change.start_time!, idx)}
                          className={`flex items-center gap-1 px-2 py-0.5 text-xs rounded-full transition-colors ${
                            activeChangeIndex === idx
                              ? 'bg-purple-200 text-purple-800 hover:bg-purple-300'
                              : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                          }`}
                          title={t('optimization.jumpToTime')}
                        >
                          <Play className="w-3 h-3" />
                          {formatTime(change.start_time)}
                          {change.end_time !== undefined && ` - ${formatTime(change.end_time)}`}
                        </button>
                      )}
                    </div>
                    {change.issues && change.issues.length > 0 && (
                      <div className="flex items-center gap-2">
                        {change.issues.filter(issue => issue && issue.type).slice(0, 2).map((issue, i) => (
                          <span
                            key={i}
                            className={`px-2 py-0.5 text-xs rounded-full ${getSeverityColor(issue.severity || 'info')}`}
                          >
                            {(issue.type || '').replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Comparison content */}
                  <div className="p-4 space-y-3">
                    {/* Original text (source language) */}
                    {change.original_text && (
                      <div>
                        <label className="block text-xs font-medium text-gray-500 mb-1">
                          {t('optimization.originalText')}
                        </label>
                        <p className="text-sm text-gray-600 bg-gray-50 rounded-lg px-3 py-2">
                          {change.original_text}
                        </p>
                      </div>
                    )}

                    {/* Before vs After comparison */}
                    <div className="grid grid-cols-1 gap-3">
                      {/* Before optimization */}
                      <div>
                        <label className="block text-xs font-medium text-red-500 mb-1">
                          {t('optimization.beforeOptimization')}
                        </label>
                        <div className="text-sm bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-gray-800">
                          {change.translated_text}
                        </div>
                      </div>

                      {/* After optimization */}
                      <div>
                        <label className="block text-xs font-medium text-green-500 mb-1">
                          {t('optimization.afterOptimization')}
                        </label>
                        <div className="text-sm bg-green-50 border border-green-200 rounded-lg px-3 py-2 text-gray-800">
                          {change.optimized_text}
                        </div>
                      </div>
                    </div>

                    {/* Proofreading suggestions */}
                    {change.suggestions && change.suggestions.length > 0 && (
                      <div>
                        <label className="block text-xs font-medium text-amber-500 mb-1">
                          {t('optimization.proofreadingSuggestions')}
                        </label>
                        <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                          <ul className="text-sm text-gray-700 space-y-1">
                            {change.suggestions.map((suggestion, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                                <span>{suggestion}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

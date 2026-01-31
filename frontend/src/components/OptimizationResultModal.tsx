import { useTranslation } from 'react-i18next'
import { X, Sparkles, AlertCircle, CheckCircle, Play } from 'lucide-react'

interface OptimizationChange {
  index: number
  start_time?: number        // Segment start time in seconds
  end_time?: number          // Segment end time in seconds
  original_text: string      // Source language text
  translated_text: string    // Before optimization
  optimized_text: string     // After optimization
  suggestions: string[]      // Proofreading suggestions
  issues: Array<{
    type: string
    severity: string
    message: string
    suggestion?: string
  }>
}

interface OptimizationResultModalProps {
  isOpen: boolean
  onClose: () => void
  changes: OptimizationChange[]
  optimizedCount: number
  totalSegments: number
  onSeekTo?: (time: number) => void  // Callback to seek video to specific time
}

export default function OptimizationResultModal({
  isOpen,
  onClose,
  changes,
  optimizedCount,
  totalSegments,
  onSeekTo
}: OptimizationResultModalProps) {
  const { t } = useTranslation()

  if (!isOpen) return null

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

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50'
      case 'error': return 'text-orange-600 bg-orange-50'
      case 'warning': return 'text-yellow-600 bg-yellow-50'
      default: return 'text-blue-600 bg-blue-50'
    }
  }

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-full items-center justify-center p-4">
        {/* Backdrop */}
        <div
          className="fixed inset-0 bg-black/50 transition-opacity"
          onClick={onClose}
        />

        {/* Modal */}
        <div className="relative bg-white rounded-xl shadow-xl max-w-4xl w-full max-h-[85vh] overflow-hidden">
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

          {/* Content */}
          <div className="overflow-y-auto max-h-[calc(85vh-140px)] p-6">
            {changes.length === 0 ? (
              <div className="text-center py-12">
                <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
                <p className="text-gray-600">{t('optimization.noChangesNeeded')}</p>
              </div>
            ) : (
              <div className="space-y-6">
                {changes.map((change, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 rounded-xl overflow-hidden bg-white shadow-sm"
                  >
                    {/* Segment header */}
                    <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="font-medium text-gray-700">
                          {t('optimization.segment')} #{change.index + 1}
                        </span>
                        {/* Timestamp with click to seek */}
                        {change.start_time !== undefined && (
                          <button
                            onClick={() => onSeekTo?.(change.start_time!)}
                            className="flex items-center gap-1 px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors"
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
                          {change.issues.filter(issue => issue && issue.type).map((issue, i) => (
                            <span
                              key={i}
                              className={`px-2 py-0.5 text-xs rounded-full ${getSeverityColor(issue.severity || 'info')}`}
                            >
                              {issue.type.replace(/_/g, ' ')}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Comparison content */}
                    <div className="p-4 space-y-4">
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
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex justify-end">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium"
            >
              {t('common.close')}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

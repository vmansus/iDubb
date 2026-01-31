import { useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Save,
  X,
  AlertTriangle,
  CheckCircle,
  Loader2,
  RotateCcw,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { taskApi } from '../services/api'
import type { ProofreadingResult, SegmentProofreadResult } from '../types'

interface SubtitleSegment {
  index: number
  start_time: number
  end_time: number
  original_text: string
  translated_text: string
}

interface SubtitleEditorProps {
  taskId: string
  proofreadingResult?: ProofreadingResult
  onClose: () => void
  onSaved?: () => void
}

export default function SubtitleEditor({
  taskId,
  proofreadingResult,
  onClose,
  onSaved
}: SubtitleEditorProps) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [segments, setSegments] = useState<SubtitleSegment[]>([])
  const [editedSegments, setEditedSegments] = useState<Set<number>>(new Set())
  const [showOnlyIssues, setShowOnlyIssues] = useState(false)
  const [expandedSegments, setExpandedSegments] = useState<Set<number>>(new Set())

  // Fetch subtitles
  const { data: subtitleData, isLoading, error } = useQuery({
    queryKey: ['subtitles', taskId],
    queryFn: () => taskApi.getSubtitles(taskId),
  })

  // Initialize segments when data is loaded
  useEffect(() => {
    if (subtitleData?.segments) {
      setSegments(subtitleData.segments)
    }
  }, [subtitleData])

  // Save subtitles mutation
  const saveMutation = useMutation({
    mutationFn: (updatedSegments: SubtitleSegment[]) =>
      taskApi.updateSubtitles(taskId, updatedSegments),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
      queryClient.invalidateQueries({ queryKey: ['subtitles', taskId] })
      setEditedSegments(new Set())
      onSaved?.()
    },
  })

  // Get proofreading issues for a segment
  const getSegmentIssues = useCallback((index: number): SegmentProofreadResult | undefined => {
    if (!proofreadingResult?.segments) return undefined
    return proofreadingResult.segments.find(s => s.index === index)
  }, [proofreadingResult])

  // Handle text change
  const handleTextChange = (index: number, newText: string) => {
    setSegments(prev => prev.map(seg =>
      seg.index === index ? { ...seg, translated_text: newText } : seg
    ))
    setEditedSegments(prev => new Set(prev).add(index))
  }

  // Reset segment to original
  const handleReset = (index: number) => {
    if (subtitleData?.segments) {
      const original = subtitleData.segments.find(s => s.index === index)
      if (original) {
        setSegments(prev => prev.map(seg =>
          seg.index === index ? { ...seg, translated_text: original.translated_text } : seg
        ))
        setEditedSegments(prev => {
          const newSet = new Set(prev)
          newSet.delete(index)
          return newSet
        })
      }
    }
  }

  // Apply suggestion from proofreading
  const applySuggestion = (index: number, suggestion: string) => {
    handleTextChange(index, suggestion)
  }

  // Toggle segment expansion
  const toggleExpanded = (index: number) => {
    setExpandedSegments(prev => {
      const newSet = new Set(prev)
      if (newSet.has(index)) {
        newSet.delete(index)
      } else {
        newSet.add(index)
      }
      return newSet
    })
  }

  // Format time
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 100)
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`
  }

  // Filter segments
  const displaySegments = showOnlyIssues
    ? segments.filter(seg => {
        const issues = getSegmentIssues(seg.index)
        return issues && issues.issues && issues.issues.length > 0
      })
    : segments

  // Count segments with issues
  const segmentsWithIssues = segments.filter(seg => {
    const issues = getSegmentIssues(seg.index)
    return issues && issues.issues && issues.issues.length > 0
  }).length

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500 mx-auto" />
          <p className="mt-4 text-gray-600">{t('subtitleEditor.loading')}</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8 max-w-md">
          <AlertTriangle className="w-8 h-8 text-red-500 mx-auto" />
          <p className="mt-4 text-gray-600 text-center">{t('subtitleEditor.loadError')}</p>
          <button
            onClick={onClose}
            className="mt-4 w-full px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg"
          >
            {t('common.close')}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              {t('subtitleEditor.title')}
            </h2>
            <p className="text-sm text-gray-500">
              {t('subtitleEditor.segmentCount', { count: segments.length })}
              {segmentsWithIssues > 0 && (
                <span className="ml-2 text-amber-600">
                  ({t('subtitleEditor.issuesCount', { count: segmentsWithIssues })})
                </span>
              )}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Toolbar */}
        <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
          <label className="flex items-center gap-2 text-sm text-gray-600">
            <input
              type="checkbox"
              checked={showOnlyIssues}
              onChange={(e) => setShowOnlyIssues(e.target.checked)}
              className="rounded border-gray-300"
            />
            {t('subtitleEditor.showOnlyIssues')}
          </label>

          {editedSegments.size > 0 && (
            <span className="text-sm text-amber-600">
              {t('subtitleEditor.unsavedChanges', { count: editedSegments.size })}
            </span>
          )}
        </div>

        {/* Subtitle List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {displaySegments.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              {showOnlyIssues
                ? t('subtitleEditor.noIssuesFound')
                : t('subtitleEditor.noSubtitles')}
            </div>
          ) : (
            displaySegments.map((segment) => {
              const segmentIssues = getSegmentIssues(segment.index)
              const hasIssues = segmentIssues && segmentIssues.issues && segmentIssues.issues.length > 0
              const isEdited = editedSegments.has(segment.index)
              const isExpanded = expandedSegments.has(segment.index)

              return (
                <div
                  key={segment.index}
                  className={`border rounded-lg overflow-hidden ${
                    hasIssues
                      ? 'border-amber-300 bg-amber-50/50'
                      : isEdited
                      ? 'border-blue-300 bg-blue-50/50'
                      : 'border-gray-200'
                  }`}
                >
                  {/* Segment Header */}
                  <div
                    className="flex items-center justify-between px-3 py-2 bg-gray-50 cursor-pointer"
                    onClick={() => toggleExpanded(segment.index)}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono text-gray-500 bg-gray-200 px-2 py-0.5 rounded">
                        #{segment.index + 1}
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatTime(segment.start_time)} → {formatTime(segment.end_time)}
                      </span>
                      {hasIssues && (
                        <span className="flex items-center gap-1 text-xs text-amber-600">
                          <AlertTriangle className="w-3 h-3" />
                          {segmentIssues!.issues.length}
                        </span>
                      )}
                      {isEdited && (
                        <span className="text-xs text-blue-600">
                          {t('subtitleEditor.edited')}
                        </span>
                      )}
                    </div>
                    {isExpanded ? (
                      <ChevronUp className="w-4 h-4 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-gray-400" />
                    )}
                  </div>

                  {/* Segment Content */}
                  {isExpanded && (
                    <div className="p-3 space-y-3">
                      {/* Original Text */}
                      <div>
                        <label className="text-xs font-medium text-gray-500 mb-1 block">
                          {t('subtitleEditor.originalText')}
                        </label>
                        <p className="text-sm text-gray-700 bg-gray-100 p-2 rounded">
                          {segment.original_text}
                        </p>
                      </div>

                      {/* Translated Text */}
                      <div>
                        <label className="text-xs font-medium text-gray-500 mb-1 block">
                          {t('subtitleEditor.translatedText')}
                        </label>
                        <textarea
                          value={segment.translated_text}
                          onChange={(e) => handleTextChange(segment.index, e.target.value)}
                          className="w-full text-sm p-2 border border-gray-300 rounded resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          rows={2}
                        />
                      </div>

                      {/* Issues */}
                      {hasIssues && (
                        <div className="space-y-2">
                          <label className="text-xs font-medium text-amber-600 mb-1 block">
                            {t('subtitleEditor.issues')}
                          </label>
                          {segmentIssues!.issues.map((issue, idx) => (
                            <div
                              key={idx}
                              className={`p-2 rounded text-xs ${
                                issue.severity === 'critical' ? 'bg-red-100 border-l-2 border-red-500' :
                                issue.severity === 'error' ? 'bg-orange-100 border-l-2 border-orange-500' :
                                issue.severity === 'warning' ? 'bg-yellow-100 border-l-2 border-yellow-500' :
                                'bg-blue-100 border-l-2 border-blue-500'
                              }`}
                            >
                              <p className="text-gray-700">{issue.message}</p>
                              {issue.suggestion && (
                                <div className="mt-2 flex items-center justify-between">
                                  <p className="text-gray-600">
                                    <span className="font-medium">{t('subtitleEditor.suggestion')}:</span> {issue.suggestion}
                                  </p>
                                  <button
                                    onClick={() => applySuggestion(segment.index, issue.suggestion!)}
                                    className="ml-2 px-2 py-1 text-xs bg-white hover:bg-gray-50 border border-gray-300 rounded"
                                  >
                                    {t('subtitleEditor.applySuggestion')}
                                  </button>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Actions */}
                      {isEdited && (
                        <div className="flex justify-end">
                          <button
                            onClick={() => handleReset(segment.index)}
                            className="flex items-center gap-1 px-2 py-1 text-xs text-gray-600 hover:bg-gray-100 rounded"
                          >
                            <RotateCcw className="w-3 h-3" />
                            {t('subtitleEditor.reset')}
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Collapsed Preview */}
                  {!isExpanded && (
                    <div className="px-3 py-2">
                      <p className="text-sm text-gray-600 truncate">
                        {segment.translated_text}
                      </p>
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-gray-200 bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
          >
            {t('common.cancel')}
          </button>

          <div className="flex items-center gap-3">
            {saveMutation.isSuccess && (
              <span className="flex items-center gap-1 text-sm text-green-600">
                <CheckCircle className="w-4 h-4" />
                {t('subtitleEditor.saved')}
              </span>
            )}
            <button
              onClick={() => saveMutation.mutate(segments)}
              disabled={editedSegments.size === 0 || saveMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {saveMutation.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              {t('subtitleEditor.save')}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

import { useState, useRef, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  ExternalLink,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  RefreshCw,
  Download,
  Play,
  FileVideo,
  FileAudio,
  FileText,
  Volume2,
  Square,
  Edit3,
  Sparkles,
  Eye
} from 'lucide-react'
import { taskApi, presetsApi, translationApi, ttsApi, transcriptionApi, configApi, videoApi, settingsApi, bilibiliApi } from '../services/api'
import type { Task, StepResult, TranslationEngine, TTSEngine, Language, Voice, DetailedVideoInfo, MetadataResult, UploadResult, SegmentProofreadResult, ProofreadingIssue, PlatformMetadata, PlatformMetadataMap } from '../types'
import type { WhisperModel } from '../services/api'
import ThumbnailSelector from '../components/ThumbnailSelector'
import SubtitleEditor from '../components/SubtitleEditor'
import OptimizationResultModal from '../components/OptimizationResultModal'
import OptimizationReviewPanel from '../components/OptimizationReviewPanel'
import VideoPlayer, { VideoPlayerHandle } from '../components/VideoPlayer'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8888'

// Map step names to display info (icons and file types)
const stepConfig: Record<string, { icon: React.ComponentType<{ className?: string }>; fileTypes: string[] }> = {
  download: {
    icon: FileVideo,
    fileTypes: ['video', 'audio']
  },
  transcribe: {
    icon: FileText,
    fileTypes: ['original_subtitle']
  },
  translate: {
    icon: FileText,
    fileTypes: ['translated_subtitle']
  },
  proofread: {
    icon: FileText,
    fileTypes: []
  },
  optimize: {
    icon: Sparkles,
    fileTypes: []
  },
  tts: {
    icon: Volume2,
    fileTypes: ['tts_audio']
  },
  process_video: {
    icon: FileVideo,
    fileTypes: ['final_video']
  },
  upload: {
    icon: ExternalLink,
    fileTypes: []
  },
}

const stepOrder = ['download', 'transcribe', 'translate', 'proofread', 'optimize', 'tts', 'process_video', 'upload']

// Format file size
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

export default function TaskDetail() {
  const { t } = useTranslation()
  const { taskId } = useParams<{ taskId: string }>()
  const queryClient = useQueryClient()

  const { data: task, isLoading } = useQuery<Task>({
    queryKey: ['task', taskId],
    queryFn: () => taskApi.get(taskId!),
    refetchInterval: (query) => {
      const data = query.state.data
      // Stop polling for terminal states (include 'completed' for backwards compatibility)
      if (['uploaded', 'completed', 'failed'].includes(data?.status || '')) {
        return false
      }
      // For pending_review: keep polling until metadata is generated
      if (data?.status === 'pending_review' && !data?.generated_metadata) {
        return 2000  // Continue polling until metadata is available
      }
      // Stop polling for other user-action-required states
      if (['paused', 'pending_review', 'pending_upload'].includes(data?.status || '')) {
        return false
      }
      return 2000
    },
    enabled: !!taskId,
  })

  const { data: filesInfo } = useQuery({
    queryKey: ['taskFiles', taskId],
    queryFn: () => taskApi.listFiles(taskId!),
    enabled: !!taskId,
    refetchInterval: 5000,
  })

  const retryMutation = useMutation({
    mutationFn: ({ stepName }: { stepName: string }) =>
      taskApi.retryStep(taskId!, stepName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
    },
  })

  const continueMutation = useMutation({
    mutationFn: ({ stepName }: { stepName: string }) =>
      taskApi.continueFromStep(taskId!, stepName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
    },
    onError: (error) => {
      console.error('Continue from step failed:', error)
      alert(`操作失败: ${error instanceof Error ? error.message : '未知错误'}`)
    },
  })

  const stopMutation = useMutation({
    mutationFn: () => taskApi.stopTask(taskId!),
    onSuccess: (data) => {
      // Immediately update cache with returned task data
      if (data.task) {
        queryClient.setQueryData(['task', taskId], data.task)
      }
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
      queryClient.invalidateQueries({ queryKey: ['taskFiles', taskId] })
    },
  })

  const updateOptionsMutation = useMutation({
    mutationFn: (options: Record<string, unknown>) =>
      taskApi.updateOptions(taskId!, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
      queryClient.invalidateQueries({ queryKey: ['taskOptions', taskId] })
    },
  })

  // State for inline editing
  const [editedOptions, setEditedOptions] = useState<Record<string, unknown>>({})

  // State for lazy loading video info (only fetch when user clicks dropdown)
  const [fetchVideoInfo, setFetchVideoInfo] = useState(false)

  // State for AI metadata generation
  const [generatedMetadata, setGeneratedMetadata] = useState<MetadataResult | null>(null)
  const [isEditingMetadata, setIsEditingMetadata] = useState(false)
  const [editedMetadata, setEditedMetadata] = useState<{
    title: string
    description: string
    keywords: string[]
  } | null>(null)
  const [metadataSaved, setMetadataSaved] = useState(false)
  
  // State for per-platform metadata (new format)
  const [platformMetadata, setPlatformMetadata] = useState<PlatformMetadataMap | null>(null)
  const [editedPlatformMetadata, setEditedPlatformMetadata] = useState<PlatformMetadataMap | null>(null)
  const [selectedMetadataPlatform, setSelectedMetadataPlatform] = useState<string>('douyin')
  
  // Helper to check if metadata is new per-platform format
  const isPlatformMetadataFormat = (metadata: Record<string, unknown> | null): boolean => {
    if (!metadata) return false
    return ['douyin', 'bilibili', 'xiaohongshu', 'generic'].some(
      (p) => metadata[p] && typeof metadata[p] === 'object' && (metadata[p] as Record<string, unknown>).title
    )
  }

  // Fetch global settings for metadata config
  const { data: globalSettings } = useQuery({
    queryKey: ['settings'],
    queryFn: settingsApi.get,
    staleTime: 60000,
  })

  // Load saved metadata from database
  // Refetch when task has generated_metadata but savedMetadata is empty
  const { data: savedMetadata } = useQuery({
    queryKey: ['task-metadata', taskId],
    queryFn: () => taskApi.loadMetadata(taskId!),
    enabled: !!taskId,
    staleTime: 5000,  // Reduced to allow faster refresh when metadata becomes available
    refetchInterval: (query) => {
      // If task has generated_metadata but we haven't loaded it yet, keep polling
      if (task?.generated_metadata && !query.state.data?.title) {
        return 2000
      }
      return false
    },
  })

  // Restore saved metadata when available
  useEffect(() => {
    if (savedMetadata?.success && !generatedMetadata && !platformMetadata) {
      // Check if it's new per-platform format
      const rawMetadata = savedMetadata as unknown as Record<string, unknown>
      if (isPlatformMetadataFormat(rawMetadata)) {
        // New format: per-platform metadata
        const platforms: PlatformMetadataMap = {}
        for (const p of ['douyin', 'bilibili', 'xiaohongshu', 'generic']) {
          if (rawMetadata[p] && typeof rawMetadata[p] === 'object') {
            const pm = rawMetadata[p] as PlatformMetadata
            platforms[p as keyof PlatformMetadataMap] = {
              title: pm.title || '',
              description: pm.description || '',
              keywords: pm.keywords || [],
            }
          }
        }
        setPlatformMetadata(platforms)
        setEditedPlatformMetadata(platforms)
        // Set default selected platform to first available
        const firstPlatform = Object.keys(platforms)[0]
        if (firstPlatform) setSelectedMetadataPlatform(firstPlatform)
        setMetadataSaved(true)
      } else if (savedMetadata.title) {
        // Old format: flat metadata
        setGeneratedMetadata({
          success: true,
          title: savedMetadata.title,
          title_translated: savedMetadata.title,
          description: savedMetadata.description || '',
          keywords: savedMetadata.keywords || [],
        })
        setEditedMetadata({
          title: savedMetadata.title,
          description: savedMetadata.description || '',
          keywords: savedMetadata.keywords || [],
        })
        setMetadataSaved(true)
      }
    }
  }, [savedMetadata, generatedMetadata, platformMetadata])

  // Reset thumbnail download state when task has local thumbnail
  useEffect(() => {
    if (task?.thumbnail_url) {
      setThumbnailDownloaded(false)
    }
  }, [task?.thumbnail_url])

  // Mutation for saving metadata
  const saveMetadataMutation = useMutation({
    mutationFn: (metadata: { title: string; description: string; keywords: string[] }) =>
      taskApi.saveMetadata(taskId!, metadata),
    onSuccess: () => {
      setMetadataSaved(true)
    },
  })

  // Mutation for approving metadata
  const approveMetadataMutation = useMutation({
    mutationFn: () => taskApi.approveMetadata(taskId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
    },
  })

  // Mutation for generating metadata
  const generateMetadataMutation = useMutation({
    mutationFn: () => taskApi.generateMetadata(taskId!, {
      source_language: taskOptions?.options?.source_language as string || 'en',
      target_language: taskOptions?.options?.target_language as string || 'zh-CN',
      include_source_url: globalSettings?.metadata?.include_source_url ?? true,
      max_keywords: globalSettings?.metadata?.max_keywords || 10,
      use_ai_preset_selection: globalSettings?.metadata?.default_use_ai_preset_selection ?? false,
      // Note: title_prefix and custom_signature are now handled via default preset in backend
    }),
    onSuccess: (data) => {
      setGeneratedMetadata(data)
      if (data.success) {
        const metadata = {
          title: data.title_translated || data.title,
          description: data.description,
          keywords: data.keywords,
        }
        setEditedMetadata(metadata)
        // Auto-save generated metadata
        saveMetadataMutation.mutate(metadata)
      }
    },
  })

  // State for thumbnail download success
  const [thumbnailDownloaded, setThumbnailDownloaded] = useState(false)

  // Mutation for re-downloading thumbnail
  const redownloadThumbnailMutation = useMutation({
    mutationFn: () => taskApi.redownloadThumbnail(taskId!),
    onSuccess: (data) => {
      if (data.success) {
        setThumbnailDownloaded(true)
        queryClient.invalidateQueries({ queryKey: ['task', taskId] })
        queryClient.invalidateQueries({ queryKey: ['taskFiles', taskId] })
      } else {
        alert(data.message || '封面下载失败')
      }
    },
    onError: (error) => {
      alert(`封面下载失败: ${error instanceof Error ? error.message : '未知错误'}`)
    },
  })

  // State for optimization job polling
  const [optimizationJobId, setOptimizationJobId] = useState<string | null>(null)

  // Mutation to START AI subtitle optimization (returns immediately)
  const optimizeSubtitlesMutation = useMutation({
    mutationFn: (level: 'minimal' | 'moderate' | 'aggressive' | undefined = undefined) => taskApi.startOptimization(taskId!, level),
    onSuccess: (data) => {
      // Store job ID and start polling
      setOptimizationJobId(data.job_id)
    },
    onError: (error) => {
      alert(`${t('taskDetail.proofreading.optimizeFailed')}: ${error instanceof Error ? error.message : t('common.unknownError')}`)
    },
  })

  // Poll for optimization job status
  useEffect(() => {
    if (!optimizationJobId) return

    let cancelled = false
    const pollInterval = 1000 // 1 second

    const pollStatus = async () => {
      try {
        const status = await taskApi.getOptimizationStatus(optimizationJobId)

        if (cancelled) return

        if (status.status === 'completed' && status.result) {
          // Show optimization result modal
          setOptimizationResult({
            changes: status.result.changes,
            optimizedCount: status.result.optimized_count,
            totalSegments: status.result.total_segments
          })
          setOptimizationJobId(null)
          // Refresh task data
          queryClient.invalidateQueries({ queryKey: ['task', taskId] })
          queryClient.invalidateQueries({ queryKey: ['taskFiles', taskId] })
        } else if (status.status === 'failed') {
          alert(`${t('taskDetail.proofreading.optimizeFailed')}: ${status.error || t('common.unknownError')}`)
          setOptimizationJobId(null)
        } else {
          // Still running, poll again
          setTimeout(pollStatus, pollInterval)
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to poll optimization status:', error)
          setOptimizationJobId(null)
        }
      }
    }

    pollStatus()

    return () => {
      cancelled = true
    }
  }, [optimizationJobId, queryClient, taskId, t])

  // Check if optimization is in progress
  const isOptimizing = optimizeSubtitlesMutation.isPending || !!optimizationJobId

  // Video player ref
  const videoPlayerRef = useRef<VideoPlayerHandle>(null)

  // UI state
  const [showSubtitleEditor, setShowSubtitleEditor] = useState(false)
  const [showOptimizationReview, setShowOptimizationReview] = useState(false)
  const [optimizationResult, setOptimizationResult] = useState<{
    changes: Array<{
      index: number
      start_time?: number
      end_time?: number
      original_text: string
      translated_text: string
      optimized_text: string
      suggestions: string[]
      issues: Array<{ type: string; severity: string; message: string; suggestion?: string }>
    }>
    optimizedCount: number
    totalSegments: number
  } | null>(null)

  // Handle seek to specific time in video
  const handleSeekTo = (time: number) => {
    videoPlayerRef.current?.seekTo(time)
  }

  // Fetch task options - always load to display in steps
  const { data: taskOptions } = useQuery({
    queryKey: ['taskOptions', taskId],
    queryFn: () => taskApi.getOptions(taskId!),
    enabled: !!taskId,
  })

  // Fetch subtitle presets - load for display and editing
  const { data: presetsData } = useQuery({
    queryKey: ['presets'],
    queryFn: presetsApi.getAll,
    enabled: !!taskId,
  })

  // Fetch Bilibili accounts
  const { data: bilibiliAccountsData } = useQuery({
    queryKey: ['bilibiliAccounts'],
    queryFn: bilibiliApi.listAccounts,
  })
  const bilibiliAccounts = bilibiliAccountsData?.accounts || []

  // Fetch translation engines
  const { data: translationEngines } = useQuery<TranslationEngine[]>({
    queryKey: ['translationEngines'],
    queryFn: translationApi.getEngines,
    enabled: !!taskId,
  })

  // Fetch TTS engines
  const { data: ttsEngines } = useQuery<TTSEngine[]>({
    queryKey: ['ttsEngines'],
    queryFn: ttsApi.getEngines,
    enabled: !!taskId,
  })

  // Fetch whisper models
  const { data: whisperModelsData } = useQuery({
    queryKey: ['whisperModels'],
    queryFn: () => transcriptionApi.getModels(),
    enabled: !!taskId,
  })

  // Fetch languages
  const { data: languages } = useQuery<Language[]>({
    queryKey: ['languages'],
    queryFn: configApi.getLanguages,
    enabled: !!taskId,
  })

  // Fetch TTS voices for current engine
  const currentTtsService = String(editedOptions.tts_service || taskOptions?.options?.tts_service || '')
  const currentTargetLang = String(editedOptions.target_language || taskOptions?.options?.target_language || '')
  const { data: ttsVoices } = useQuery<Voice[]>({
    queryKey: ['ttsVoices', currentTtsService, currentTargetLang],
    queryFn: () => ttsApi.getVoicesByEngine(currentTtsService, currentTargetLang),
    enabled: !!taskId && !!currentTtsService,
  })

  // Fetch video detailed info ONLY when user clicks to edit (lazy load)
  const sourceUrl = String(taskOptions?.options?.source_url || '')
  const { data: videoDetailedInfo, isLoading: isLoadingVideoInfo } = useQuery<DetailedVideoInfo>({
    queryKey: ['videoDetailedInfo', sourceUrl],
    queryFn: () => videoApi.getDetailedInfo(sourceUrl),
    enabled: !!sourceUrl && fetchVideoInfo, // Only fetch when user triggers it
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  })

  // Get available qualities from video detailed info
  const availableQualities = videoDetailedInfo?.formats
    ?.filter((f) => f.has_video && f.resolution)
    ?.map((f) => ({
      value: f.format_id,
      label: `${f.resolution} (${f.ext})${f.filesize_mb ? ` - ${f.filesize_mb.toFixed(1)}MB` : ''}`,
      quality: f.quality_label,
    }))
    ?.sort((a, b) => {
      const qualityOrder = ['2160p', '1440p', '1080p', '720p', '480p', '360p', '240p']
      return qualityOrder.indexOf(a.quality) - qualityOrder.indexOf(b.quality)
    }) || []

  // Get available subtitles from video detailed info
  const availableSubtitles = videoDetailedInfo?.subtitles || []

  if (isLoading) {
    return (
      <>
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
        {/* Keep optimization modal visible even during loading */}
        {optimizationResult && (
          <OptimizationResultModal
            isOpen={true}
            onClose={() => setOptimizationResult(null)}
            changes={optimizationResult.changes}
            optimizedCount={optimizationResult.optimizedCount}
            totalSegments={optimizationResult.totalSegments}
            onSeekTo={handleSeekTo}
          />
        )}
      </>
    )
  }

  if (!task) {
    return (
      <>
        <div className="text-center py-12">
          <p className="text-gray-500">{t('taskDetail.taskNotFound')}</p>
          <Link to="/" className="text-blue-600 hover:underline mt-2 inline-block">
            {t('taskDetail.backToList')}
          </Link>
        </div>
        {/* Keep optimization modal visible even when task not found */}
        {optimizationResult && (
          <OptimizationResultModal
            isOpen={true}
            onClose={() => setOptimizationResult(null)}
            changes={optimizationResult.changes}
            optimizedCount={optimizationResult.optimizedCount}
            totalSegments={optimizationResult.totalSegments}
            onSeekTo={handleSeekTo}
          />
        )}
      </>
    )
  }

  // Include 'completed' for backwards compatibility with existing database records
  const isUploaded = task.status === 'uploaded' || task.status === 'completed'
  const isPendingReview = task.status === 'pending_review'
  const isPendingUpload = task.status === 'pending_upload'
  const isFailed = task.status === 'failed'
  const isPaused = task.status === 'paused'
  // Helper: task is not actively processing (can edit options, view results, etc.)
  const isFinished = isUploaded || isPendingReview || isPendingUpload || isFailed || isPaused

  const getStepStatus = (stepName: string): StepResult | null => {
    return task.steps?.[stepName] || null
  }

  const renderStepIcon = (step: StepResult | null, _stepName: string) => {
    if (!step) return <Clock className="h-5 w-5" />

    switch (step.status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'running':
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />
      case 'skipped':
        return <Clock className="h-5 w-5 text-gray-400" />
      default:
        return <Clock className="h-5 w-5 text-gray-400" />
    }
  }

  const getStepBgColor = (step: StepResult | null) => {
    if (!step) return 'bg-gray-50'

    switch (step.status) {
      case 'completed':
        return 'bg-green-50 border-green-200'
      case 'running':
        return 'bg-blue-50 border-blue-200'
      case 'failed':
        return 'bg-red-50 border-red-200'
      case 'skipped':
        return 'bg-gray-50 border-gray-200'
      default:
        return 'bg-gray-50 border-gray-200'
    }
  }

  return (
    <div>
      <Link
        to="/"
        className="inline-flex items-center text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        {t('taskDetail.backToList')}
      </Link>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        {/* Header with Thumbnail */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex gap-4">
            {/* Thumbnail */}
            {(() => {
              const hasLocalThumbnail = !!task.thumbnail_url
              const thumbnailUrl = hasLocalThumbnail
                ? `${API_BASE}${task.thumbnail_url}`
                : task.video_info?.thumbnail_url
              const canRedownload = !hasLocalThumbnail && !!task.video_info?.thumbnail_url
              const isVertical = task.video_info?.is_vertical
              // Vertical: 9:16 ratio (w-16 h-28 = 64x112px), Horizontal: 16:9 ratio (w-40 h-24 = 160x96px)
              const sizeClass = isVertical ? 'w-16 h-28' : 'w-40 h-24'

              return (
                <div className="flex-shrink-0 relative group">
                  {thumbnailUrl ? (
                    <div className={`${sizeClass} rounded-lg overflow-hidden bg-gray-100`}>
                      <img
                        src={thumbnailUrl}
                        alt={task.video_info?.title || t('taskDetail.videoThumbnail')}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = 'none'
                        }}
                      />
                    </div>
                  ) : (
                    <div className={`${sizeClass} rounded-lg bg-gray-100 flex items-center justify-center`}>
                      <FileVideo className="h-10 w-10 text-gray-300" />
                    </div>
                  )}
                  {/* Re-download thumbnail button - for missing local thumbnails */}
                  {canRedownload && !thumbnailDownloaded && (
                    <button
                      onClick={() => redownloadThumbnailMutation.mutate()}
                      disabled={redownloadThumbnailMutation.isPending}
                      className={`absolute inset-0 flex items-center justify-center rounded-lg transition-opacity ${
                        redownloadThumbnailMutation.isPending
                          ? 'bg-black/70 opacity-100'
                          : 'bg-black/50 opacity-0 group-hover:opacity-100'
                      }`}
                      title={t('taskDetail.redownloadThumbnail')}
                    >
                      {redownloadThumbnailMutation.isPending ? (
                        <div className="flex flex-col items-center">
                          <Loader2 className="h-6 w-6 text-white animate-spin" />
                          <span className="text-white text-xs mt-1">下载中...</span>
                        </div>
                      ) : (
                        <Download className="h-6 w-6 text-white" />
                      )}
                    </button>
                  )}
                  {/* Show success state after download */}
                  {canRedownload && thumbnailDownloaded && (
                    <div className="absolute inset-0 flex items-center justify-center bg-green-500/70 rounded-lg">
                      <div className="flex flex-col items-center">
                        <CheckCircle className="h-6 w-6 text-white" />
                        <span className="text-white text-xs mt-1">已下载</span>
                      </div>
                    </div>
                  )}
                  {/* Show refresh button for existing thumbnails */}
                  {hasLocalThumbnail && !redownloadThumbnailMutation.isPending && (
                    <button
                      onClick={() => {
                        setThumbnailDownloaded(false)
                        redownloadThumbnailMutation.mutate()
                      }}
                      disabled={redownloadThumbnailMutation.isPending}
                      className="absolute bottom-1 right-1 p-1 bg-black/60 rounded text-white opacity-0 group-hover:opacity-100 transition-opacity"
                      title={t('taskDetail.redownloadThumbnail')}
                    >
                      <RefreshCw className="h-3 w-3" />
                    </button>
                  )}
                  {/* Show loading spinner on existing thumbnail during refresh */}
                  {hasLocalThumbnail && redownloadThumbnailMutation.isPending && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg">
                      <Loader2 className="h-6 w-6 text-white animate-spin" />
                    </div>
                  )}
                </div>
              )
            })()}

            {/* Title and Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between">
                <div className="min-w-0 flex-1">
                  <h2 className="text-xl font-semibold text-gray-900 truncate">
                    {task.video_info?.title || `${t('common.task')} ${task.task_id}`}
                  </h2>
                  <p className="mt-1 text-sm text-gray-500">
                    ID: {task.task_id} · {t('common.createdAt')} {new Date(task.created_at).toLocaleString()}
                  </p>
                  {task.video_info?.platform && (
                    <p className="mt-1 text-xs text-gray-400">
                      {t('taskDetail.platform')}: {task.video_info.platform}
                      {task.video_info.duration ? ` · ${Math.floor(task.video_info.duration / 60)}:${String(Math.floor(task.video_info.duration % 60)).padStart(2, '0')}` : ''}
                    </p>
                  )}
                </div>
                <div className="flex items-center space-x-3 ml-4">
                  {/* Stop button - show when processing */}
                  {!isFinished && (
                    <button
                      onClick={() => stopMutation.mutate()}
                      disabled={stopMutation.isPending}
                      className="inline-flex items-center px-3 py-1.5 text-sm bg-red-100 text-red-700 rounded-lg hover:bg-red-200 disabled:opacity-50 transition-colors"
                    >
                      {stopMutation.isPending ? (
                        <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
                      ) : (
                        <Square className="h-4 w-4 mr-1.5" />
                      )}
                      {t('taskDetail.stopTask')}
                    </button>
                  )}
                  <div
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      isUploaded
                        ? 'bg-green-100 text-green-700'
                        : isPendingUpload
                        ? 'bg-blue-100 text-blue-700'
                        : isPendingReview
                        ? 'bg-yellow-100 text-yellow-700'
                        : isFailed
                        ? 'bg-red-100 text-red-700'
                        : isPaused
                        ? 'bg-orange-100 text-orange-700'
                        : 'bg-blue-100 text-blue-700'
                    }`}
                  >
                    {t(`dashboard.status.${task.status === 'completed' ? 'uploaded' : task.status}`)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Overall Progress */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>{task.message}</span>
            <span>{task.progress}%</span>
          </div>
          <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                isFailed ? 'bg-red-500' : isUploaded ? 'bg-green-500' : isPendingReview ? 'bg-yellow-500' : isPendingUpload ? 'bg-blue-500' : 'bg-blue-500'
              }`}
              style={{ width: `${task.progress}%` }}
            />
          </div>
        </div>

        {/* Video Player - Show when final video is available */}
        {filesInfo?.final_video?.available && (
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-sm font-medium text-gray-700 mb-4">{t('taskDetail.videoPreview')}</h3>
            <VideoPlayer
              ref={videoPlayerRef}
              src={taskApi.getFileDownloadUrl(taskId!, 'final_video', task.updated_at)}
              maxHeight="500px"
              showDownload={true}
            />
          </div>
        )}

        {/* Steps */}
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-sm font-medium text-gray-700 mb-4">{t('taskDetail.processingSteps')}</h3>
          <div className="space-y-4">
            {stepOrder.map((stepName) => {
              const step = getStepStatus(stepName)
              const config = stepConfig[stepName]
              const Icon = config.icon
              const canRetry = step?.status === 'failed'
              // Can continue from: completed, skipped, or pending (for stalled tasks)
              // For upload step: only when metadata is approved
              const canContinue = step?.status === 'completed' || step?.status === 'skipped' ||
                (stepName === 'upload' && step?.status === 'pending' && isPendingUpload) ||
                (step?.status === 'pending' && isFinished && stepName !== 'upload')
              const hasFiles = config.fileTypes.length > 0

              // Check if transcribe step will be skipped (using existing subtitles)
              const willBeSkipped = stepName === 'transcribe' &&
                Boolean(editedOptions.use_existing_subtitles ?? taskOptions?.options?.use_existing_subtitles)

              return (
                <div
                  key={stepName}
                  className={`p-4 rounded-lg border ${willBeSkipped ? 'bg-gray-100 border-gray-300 opacity-60' : getStepBgColor(step)}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        {willBeSkipped ? <Clock className="h-5 w-5 text-gray-400" /> : renderStepIcon(step, stepName)}
                      </div>
                      <div>
                        <div className="flex items-center space-x-2">
                          <Icon className="h-4 w-4 text-gray-500" />
                          <span className={`font-medium ${willBeSkipped ? 'text-gray-500' : 'text-gray-900'}`}>
                            {t(`taskDetail.steps.${stepName === 'transcribe' && taskOptions?.options?.use_ocr ? 'transcribeOcr' : stepName}`)}
                          </span>
                          {willBeSkipped && (
                            <span className="px-1.5 py-0.5 text-xs bg-yellow-100 text-yellow-700 rounded">
                              {t('taskDetail.willSkip')}
                            </span>
                          )}
                        </div>
                        {step?.error && step.error !== '用户手动停止' && (
                          <p className="text-sm text-red-600 mt-1">{step.error}</p>
                        )}
                        {step?.error === '用户手动停止' && (
                          <p className="text-sm text-yellow-600 mt-1">{t('taskDetail.manuallyPaused')}</p>
                        )}
                        {step?.status === 'skipped' && Boolean(step.metadata?.skip_reason) && (
                          <p className="text-sm text-gray-500 mt-1">
                            {t('taskDetail.skipped')}: {String(step.metadata?.skip_reason)}
                          </p>
                        )}
                        {step?.completed_at && step.status === 'completed' && (
                          <p className="text-xs text-gray-500 mt-1">
                            {t('common.completedAt')} {new Date(step.completed_at).toLocaleTimeString()}
                          </p>
                        )}
                        {/* Inline step config - Editable for completed/failed/paused tasks */}
                        {taskOptions?.options && (
                          <div className="mt-2 flex flex-wrap gap-2 text-xs items-center">
                            {/* Download step - video quality / format_id */}
                            {stepName === 'download' && (() => {
                              // Prefer format_id (saved when using detailed info), fall back to video_quality
                              const currentFormatId = String(editedOptions.format_id ?? taskOptions.options.format_id ?? '')
                              const currentVideoQuality = String(editedOptions.video_quality ?? taskOptions.options.video_quality ?? '')
                              const savedQualityLabel = String(taskOptions.options.video_quality_label ?? '')
                              const currentValue = currentFormatId || currentVideoQuality
                              // Display priority: 1. From loaded qualities, 2. Saved label, 3. video_quality, 4. Raw value
                              const displayLabel = availableQualities.find(q => q.value === currentValue)?.label
                                || savedQualityLabel
                                || currentVideoQuality
                                || currentValue

                              return (isFinished || isFailed || isPaused) ? (
                                <select
                                  value={currentValue}
                                  onFocus={() => setFetchVideoInfo(true)} // Lazy load on focus
                                  onChange={(e) => {
                                    const selectedQuality = availableQualities.find(q => q.value === e.target.value)
                                    const newOpts = {
                                      ...editedOptions,
                                      format_id: e.target.value,
                                      video_quality: e.target.value,
                                      video_quality_label: selectedQuality?.label || e.target.value,
                                    }
                                    setEditedOptions(newOpts)
                                    updateOptionsMutation.mutate(newOpts)
                                  }}
                                  disabled={updateOptionsMutation.isPending}
                                  className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs min-w-24"
                                >
                                  {/* Always show current saved value first with saved label */}
                                  {!availableQualities.find(q => q.value === currentValue) && (
                                    <option value={currentValue}>
                                      {savedQualityLabel || currentVideoQuality || currentValue}
                                    </option>
                                  )}
                                  {/* Show loading state or available qualities */}
                                  {isLoadingVideoInfo ? (
                                    <option disabled>{t('common.loading')}...</option>
                                  ) : (
                                    availableQualities.map(opt => (
                                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))
                                  )}
                                </select>
                              ) : (
                                <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{displayLabel}</span>
                              )
                            })()}

                            {/* Transcribe step - whisper model, source language, use existing subtitles */}
                            {stepName === 'transcribe' && (
                              (isFinished || isFailed || isPaused) ? (
                                <>
                                  {/* Use existing subtitles option */}
                                  <select
                                    value={
                                      (editedOptions.use_existing_subtitles ?? taskOptions.options.use_existing_subtitles)
                                        ? String(editedOptions.subtitle_language ?? taskOptions.options.subtitle_language ?? 'auto')
                                        : 'none'
                                    }
                                    onFocus={() => setFetchVideoInfo(true)} // Lazy load subtitles on focus
                                    onChange={(e) => {
                                      if (e.target.value === 'none') {
                                        const newOpts = { ...editedOptions, use_existing_subtitles: false }
                                        setEditedOptions(newOpts)
                                        updateOptionsMutation.mutate(newOpts)
                                      } else {
                                        const newOpts = { ...editedOptions, use_existing_subtitles: true, subtitle_language: e.target.value }
                                        setEditedOptions(newOpts)
                                        updateOptionsMutation.mutate(newOpts)
                                      }
                                    }}
                                    disabled={updateOptionsMutation.isPending}
                                    className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs"
                                  >
                                    <option value="none">{t('taskDetail.options.useWhisper')}</option>
                                    {isLoadingVideoInfo ? (
                                      <option disabled>{t('common.loading')}...</option>
                                    ) : (
                                      availableSubtitles.map(sub => (
                                        <option key={sub.language} value={sub.language}>
                                          {t('taskDetail.options.useExistingSub')}: {sub.language_name}
                                          {sub.is_auto_generated ? ` (${t('taskDetail.options.autoGenerated')})` : ''}
                                        </option>
                                      ))
                                    )}
                                  </select>
                                  {/* Only show whisper options if not using existing subtitles */}
                                  {!(editedOptions.use_existing_subtitles ?? taskOptions.options.use_existing_subtitles) && (
                                    <>
                                      <select
                                        value={String(editedOptions.whisper_model ?? taskOptions.options.whisper_model ?? '')}
                                        onChange={(e) => {
                                          const newOpts = { ...editedOptions, whisper_model: e.target.value }
                                          setEditedOptions(newOpts)
                                          updateOptionsMutation.mutate(newOpts)
                                        }}
                                        disabled={updateOptionsMutation.isPending}
                                        className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs"
                                      >
                                        {whisperModelsData?.models?.map((m: WhisperModel) => (
                                          <option key={m.id} value={m.id}>{m.name}</option>
                                        ))}
                                      </select>
                                      <select
                                        value={String(editedOptions.source_language ?? taskOptions.options.source_language ?? '')}
                                        onChange={(e) => {
                                          const newOpts = { ...editedOptions, source_language: e.target.value }
                                          setEditedOptions(newOpts)
                                          updateOptionsMutation.mutate(newOpts)
                                        }}
                                        disabled={updateOptionsMutation.isPending}
                                        className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs"
                                      >
                                        {languages?.map(lang => (
                                          <option key={lang.code} value={lang.code}>{lang.name}</option>
                                        ))}
                                      </select>
                                    </>
                                  )}
                                </>
                              ) : (
                                <>
                                  {taskOptions.options.use_existing_subtitles ? (
                                    <span className="px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded">
                                      {t('taskDetail.options.useExistingSub')}: {String(taskOptions.options.subtitle_language || 'auto')}
                                    </span>
                                  ) : taskOptions.options.use_ocr ? (
                                    <>
                                      <span className="px-2 py-0.5 bg-amber-100 text-amber-700 rounded">🔍 {String(taskOptions.options.ocr_engine || 'paddleocr')}</span>
                                      <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{String(taskOptions.options.ocr_frame_interval || 0.5)}s</span>
                                    </>
                                  ) : (
                                    <>
                                      <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{String(taskOptions.options.whisper_model || '')}</span>
                                      <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{String(taskOptions.options.source_language || '')}</span>
                                    </>
                                  )}
                                </>
                              )
                            )}

                            {/* Translate step - translation engine & target language */}
                            {stepName === 'translate' && (
                              (isFinished || isFailed || isPaused) ? (
                                <>
                                  <select
                                    value={String(editedOptions.translation_engine ?? taskOptions.options.translation_engine ?? '')}
                                    onChange={(e) => {
                                      const newOpts = { ...editedOptions, translation_engine: e.target.value }
                                      setEditedOptions(newOpts)
                                      updateOptionsMutation.mutate(newOpts)
                                    }}
                                    disabled={updateOptionsMutation.isPending}
                                    className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs"
                                  >
                                    {translationEngines?.map((e: TranslationEngine) => (
                                      <option key={e.id} value={e.id}>{e.name}</option>
                                    ))}
                                  </select>
                                  <span className="text-gray-500">→</span>
                                  <select
                                    value={String(editedOptions.target_language ?? taskOptions.options.target_language ?? '')}
                                    onChange={(e) => {
                                      const newOpts = { ...editedOptions, target_language: e.target.value }
                                      setEditedOptions(newOpts)
                                      updateOptionsMutation.mutate(newOpts)
                                    }}
                                    disabled={updateOptionsMutation.isPending}
                                    className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs"
                                  >
                                    {languages?.map(lang => (
                                      <option key={lang.code} value={lang.code}>{lang.name}</option>
                                    ))}
                                  </select>
                                </>
                              ) : (
                                <>
                                  <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{String(taskOptions.options.translation_engine || '')}</span>
                                  <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{String(taskOptions.options.source_language || '')} → {String(taskOptions.options.target_language || '')}</span>
                                </>
                              )
                            )}

                            {/* TTS step - tts service, voice, and add_tts toggle */}
                            {stepName === 'tts' && (
                              (isFinished || isFailed || isPaused) ? (
                                <>
                                  <label className="flex items-center gap-1 cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={Boolean(editedOptions.add_tts ?? taskOptions.options.add_tts)}
                                      onChange={(e) => {
                                        const newOpts = { ...editedOptions, add_tts: e.target.checked }
                                        setEditedOptions(newOpts)
                                        updateOptionsMutation.mutate(newOpts)
                                      }}
                                      disabled={updateOptionsMutation.isPending}
                                      className="w-3 h-3"
                                    />
                                    <span className="text-gray-600">{t('newTask.addTts')}</span>
                                  </label>
                                  {(editedOptions.add_tts ?? taskOptions.options.add_tts) && (
                                    <>
                                      <select
                                        value={String(editedOptions.tts_service ?? taskOptions.options.tts_service ?? '')}
                                        onChange={(e) => {
                                          const newOpts = { ...editedOptions, tts_service: e.target.value }
                                          setEditedOptions(newOpts)
                                          updateOptionsMutation.mutate(newOpts)
                                        }}
                                        disabled={updateOptionsMutation.isPending}
                                        className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs"
                                      >
                                        {ttsEngines?.map((e: TTSEngine) => (
                                          <option key={e.id} value={e.id}>{e.name}</option>
                                        ))}
                                      </select>
                                      <select
                                        value={String(editedOptions.tts_voice ?? taskOptions.options.tts_voice ?? '')}
                                        onChange={(e) => {
                                          const newOpts = { ...editedOptions, tts_voice: e.target.value }
                                          setEditedOptions(newOpts)
                                          updateOptionsMutation.mutate(newOpts)
                                        }}
                                        disabled={updateOptionsMutation.isPending}
                                        className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs max-w-32"
                                      >
                                        {ttsVoices?.map((v: Voice) => (
                                          <option key={v.name} value={v.name}>{v.display_name}</option>
                                        ))}
                                      </select>
                                    </>
                                  )}
                                </>
                              ) : (
                                <>
                                  {taskOptions.options.add_tts ? (
                                    <>
                                      <span className="px-2 py-0.5 bg-green-100 text-green-600 rounded">{t('newTask.addTts')}</span>
                                      <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{String(taskOptions.options.tts_service || '')}</span>
                                      <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{String(taskOptions.options.tts_voice || '')}</span>
                                    </>
                                  ) : (
                                    <span className="px-2 py-0.5 bg-gray-100 text-gray-500 rounded">{t('taskDetail.skipped')}</span>
                                  )}
                                </>
                              )
                            )}

                            {/* Process video step - subtitle preset */}
                            {stepName === 'process_video' && (
                              (isFinished || isFailed || isPaused) ? (
                                <>
                                  {taskOptions.options.dual_subtitles && (
                                    <span className="px-2 py-0.5 bg-blue-100 text-blue-600 rounded">{t('taskDetail.options.dualSubtitles')}</span>
                                  )}
                                  {presetsData?.presets && (
                                    <select
                                      value={String(editedOptions.subtitle_preset ?? taskOptions.options.subtitle_preset ?? presetsData.presets[0]?.id ?? '')}
                                      onChange={(e) => {
                                        const newOpts = { ...editedOptions, subtitle_preset: e.target.value }
                                        setEditedOptions(newOpts)
                                        updateOptionsMutation.mutate(newOpts)
                                      }}
                                      disabled={updateOptionsMutation.isPending}
                                      className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs"
                                    >
                                      {presetsData.presets.map(preset => (
                                        <option key={preset.id} value={preset.id}>{preset.name}</option>
                                      ))}
                                    </select>
                                  )}
                                </>
                              ) : (
                                <>
                                  {taskOptions.options.dual_subtitles && (
                                    <span className="px-2 py-0.5 bg-blue-100 text-blue-600 rounded">{t('taskDetail.options.dualSubtitles')}</span>
                                  )}
                                  {taskOptions.options.subtitle_preset && (
                                    <span className="px-2 py-0.5 bg-purple-100 text-purple-600 rounded">
                                      {presetsData?.presets?.find(p => p.id === taskOptions.options.subtitle_preset)?.name || String(taskOptions.options.subtitle_preset)}
                                    </span>
                                  )}
                                </>
                              )
                            )}

                            {/* Upload step - platform checkboxes (only enabled after metadata approval) */}
                            {stepName === 'upload' && (
                              (isFinished || isFailed || isPaused) ? (
                                <div className="flex flex-wrap gap-2 items-center">
                                  {/* Show approval status hint */}
                                  {!task.metadata_approved && (
                                    <span className="text-xs text-yellow-600 mr-2">
                                      {t('taskDetail.metadata.approvalRequired')}
                                    </span>
                                  )}
                                  <label className={`flex items-center gap-1 px-2 py-0.5 bg-pink-50 rounded border border-pink-200 ${task.metadata_approved ? 'cursor-pointer' : 'opacity-50 cursor-not-allowed'}`}>
                                    <input
                                      type="checkbox"
                                      checked={Boolean(editedOptions.upload_bilibili ?? taskOptions.options.upload_bilibili)}
                                      onChange={(e) => {
                                        const newOpts = { ...editedOptions, upload_bilibili: e.target.checked }
                                        setEditedOptions(newOpts)
                                        updateOptionsMutation.mutate(newOpts)
                                      }}
                                      disabled={updateOptionsMutation.isPending || !task.metadata_approved}
                                      className="w-3 h-3"
                                    />
                                    <span className="text-pink-600">Bilibili</span>
                                  </label>
                                  {Boolean(editedOptions.upload_bilibili ?? taskOptions.options.upload_bilibili) && bilibiliAccounts.length > 0 && (
                                    <select
                                      value={String(editedOptions.bilibili_account_uid ?? taskOptions.options.bilibili_account_uid ?? '')}
                                      onChange={(e) => {
                                        const newOpts = { ...editedOptions, bilibili_account_uid: e.target.value }
                                        setEditedOptions(newOpts)
                                        updateOptionsMutation.mutate(newOpts)
                                      }}
                                      disabled={updateOptionsMutation.isPending || !task.metadata_approved}
                                      className="text-xs border border-gray-300 rounded px-1 py-0.5"
                                    >
                                      <option value="">
                                        {bilibiliAccounts.find(a => a.is_primary)?.label || '主账号'}
                                      </option>
                                      {bilibiliAccounts.filter(a => !a.is_primary).map((acc) => (
                                        <option key={acc.uid} value={acc.uid}>
                                          {acc.label || acc.nickname}
                                        </option>
                                      ))}
                                    </select>
                                  )}
                                  <label className={`flex items-center gap-1 px-2 py-0.5 bg-gray-100 rounded border border-gray-300 ${task.metadata_approved ? 'cursor-pointer' : 'opacity-50 cursor-not-allowed'}`}>
                                    <input
                                      type="checkbox"
                                      checked={Boolean(editedOptions.upload_douyin ?? taskOptions.options.upload_douyin)}
                                      onChange={(e) => {
                                        const newOpts = { ...editedOptions, upload_douyin: e.target.checked }
                                        setEditedOptions(newOpts)
                                        updateOptionsMutation.mutate(newOpts)
                                      }}
                                      disabled={updateOptionsMutation.isPending || !task.metadata_approved}
                                      className="w-3 h-3"
                                    />
                                    <span className="text-gray-800">抖音</span>
                                  </label>
                                  <label className={`flex items-center gap-1 px-2 py-0.5 bg-red-50 rounded border border-red-200 ${task.metadata_approved ? 'cursor-pointer' : 'opacity-50 cursor-not-allowed'}`}>
                                    <input
                                      type="checkbox"
                                      checked={Boolean(editedOptions.upload_xiaohongshu ?? taskOptions.options.upload_xiaohongshu)}
                                      onChange={(e) => {
                                        const newOpts = { ...editedOptions, upload_xiaohongshu: e.target.checked }
                                        setEditedOptions(newOpts)
                                        updateOptionsMutation.mutate(newOpts)
                                      }}
                                      disabled={updateOptionsMutation.isPending || !task.metadata_approved}
                                      className="w-3 h-3"
                                    />
                                    <span className="text-red-600">小红书</span>
                                  </label>
                                </div>
                              ) : (
                                <>
                                  {taskOptions.options.upload_bilibili && <span className="px-2 py-0.5 bg-pink-100 text-pink-600 rounded">Bilibili</span>}
                                  {taskOptions.options.upload_douyin && <span className="px-2 py-0.5 bg-black text-white rounded">抖音</span>}
                                  {taskOptions.options.upload_xiaohongshu && <span className="px-2 py-0.5 bg-red-100 text-red-600 rounded">小红书</span>}
                                  {!taskOptions.options.upload_bilibili && !taskOptions.options.upload_douyin && !taskOptions.options.upload_xiaohongshu && (
                                    <span className="px-2 py-0.5 bg-gray-100 text-gray-500 rounded">{t('taskDetail.options.noUpload')}</span>
                                  )}
                                </>
                              )
                            )}

                            {/* Loading indicator */}
                            {updateOptionsMutation.isPending && (
                              <Loader2 className="h-3 w-3 animate-spin text-gray-400" />
                            )}
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      {/* File Downloads */}
                      {hasFiles && filesInfo && (
                        <div className="flex items-center space-x-1">
                          {config.fileTypes.map((fileType) => {
                            const fileInfo = filesInfo[fileType]
                            if (!fileInfo?.available) return null

                            return (
                              <a
                                key={fileType}
                                href={taskApi.getFileDownloadUrl(taskId!, fileType)}
                                download
                                className="inline-flex items-center px-2 py-1 text-xs bg-white border border-gray-300 rounded hover:bg-gray-50"
                                title={`${t('common.download')} ${t(`taskDetail.fileTypes.${fileType}`)} (${formatFileSize(fileInfo.size)})`}
                              >
                                <Download className="h-3 w-3 mr-1" />
                                {t(`taskDetail.fileTypes.${fileType}`)}
                              </a>
                            )
                          })}
                        </div>
                      )}

                      {/* View Optimization Result Button (for optimize step when completed) */}
                      {stepName === 'optimize' && step?.status === 'completed' && task.optimization_result && (
                        <button
                          onClick={() => setShowOptimizationReview(true)}
                          className="inline-flex items-center px-3 py-1 text-sm bg-purple-100 text-purple-700 rounded hover:bg-purple-200"
                        >
                          <Eye className="h-4 w-4 mr-1" />
                          {t('taskDetail.optimization.viewResult')}
                        </button>
                      )}

                      {/* Retry Button (for failed steps) */}
                      {canRetry && (
                        <button
                          onClick={() => retryMutation.mutate({ stepName })}
                          disabled={retryMutation.isPending || continueMutation.isPending}
                          className="inline-flex items-center px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50"
                        >
                          <RefreshCw className={`h-4 w-4 mr-1 ${retryMutation.isPending ? 'animate-spin' : ''}`} />
                          {t('common.retry')}
                        </button>
                      )}

                      {/* Re-run from this step button (for completed/paused/failed tasks) */}
                      {/* For upload step, only enable when metadata is approved */}
                      {canContinue && (isFailed || isPaused || isFinished) && (
                        <button
                          onClick={() => continueMutation.mutate({ stepName })}
                          disabled={
                            continueMutation.isPending ||
                            retryMutation.isPending ||
                            (stepName === 'upload' && !task.metadata_approved)
                          }
                          className={`inline-flex items-center px-3 py-1 text-sm rounded disabled:opacity-50 ${
                            stepName === 'upload' && !task.metadata_approved
                              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                              : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                          }`}
                          title={stepName === 'upload' && !task.metadata_approved ? t('taskDetail.metadata.approvalRequired') : ''}
                        >
                          <Play className={`h-4 w-4 mr-1 ${continueMutation.isPending ? 'animate-spin' : ''}`} />
                          {stepName === 'upload' ? t('taskDetail.startUpload') : t('taskDetail.rerunFrom')}
                        </button>
                      )}
                    </div>
                  </div>

                </div>
              )
            })}
          </div>
        </div>

        {/* All Available Files */}
        {filesInfo && Object.values(filesInfo).some(f => f.available) && (
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-sm font-medium text-gray-700 mb-4">{t('taskDetail.downloadableFiles')}</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {Object.entries(filesInfo).map(([fileType, info]) => {
                if (!info.available) return null

                const getFileIcon = () => {
                  if (fileType.includes('video')) return FileVideo
                  if (fileType.includes('audio') || fileType === 'tts_audio') return FileAudio
                  return FileText
                }
                const FileIcon = getFileIcon()

                return (
                  <a
                    key={fileType}
                    href={taskApi.getFileDownloadUrl(taskId!, fileType)}
                    download
                    className="flex items-center p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition-colors"
                  >
                    <FileIcon className="h-8 w-8 text-gray-400 mr-3" />
                    <div>
                      <div className="text-sm font-medium text-gray-900">
                        {t(`taskDetail.fileTypes.${fileType}`)}
                      </div>
                      <div className="text-xs text-gray-500">
                        {formatFileSize(info.size)}
                      </div>
                    </div>
                    <Download className="h-4 w-4 text-gray-400 ml-auto" />
                  </a>
                )
              })}
            </div>
          </div>
        )}

        {/* Error */}
        {task.error && (
          <div className="p-6 border-b border-gray-200 bg-red-50">
            <h3 className="text-sm font-medium text-red-700 mb-2">{t('taskDetail.errorInfo')}</h3>
            <p className="text-sm text-red-600">{task.error}</p>
          </div>
        )}

        {/* Thumbnail Selection Section */}
        {/* Show when: process_video completed, OR task is pending_review/pending_upload/uploaded */}
        {(task.steps?.process_video?.status === 'completed' ||
          isPendingReview || isPendingUpload || isUploaded) && task.files?.thumbnail && (
          <div className="p-6 border-b border-gray-200">
            <ThumbnailSelector
              taskId={taskId!}
              onSelectionChange={(selected) => {
                // Optionally handle selection change
                console.log('Thumbnail selected:', selected)
              }}
              isVertical={task.video_info?.is_vertical}
            />
          </div>
        )}

        {/* AI Metadata Generation Section */}
        {/* Show when: transcribe/translate completed, OR task is pending_review/pending_upload/uploaded */}
        {(task.steps?.transcribe?.status === 'completed' ||
          task.steps?.translate?.status === 'completed' ||
          task.steps?.process_video?.status === 'completed' ||
          isPendingReview || isPendingUpload || isUploaded) && (
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-700">{t('taskDetail.metadata.title')}</h3>
              {!generatedMetadata && !platformMetadata && (
                <button
                  onClick={() => generateMetadataMutation.mutate()}
                  disabled={generateMetadataMutation.isPending}
                  className="inline-flex items-center px-3 py-1.5 text-sm bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 disabled:opacity-50 transition-colors"
                >
                  {generateMetadataMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
                      {t('taskDetail.metadata.generating')}
                    </>
                  ) : (
                    t('taskDetail.metadata.generate')
                  )}
                </button>
              )}
            </div>

            {generateMetadataMutation.isError && (
              <div className="mb-4 p-3 bg-red-50 rounded-lg text-sm text-red-600">
                {t('taskDetail.metadata.error')}: {(generateMetadataMutation.error as Error)?.message || t('common.unknownError')}
              </div>
            )}

            {/* Per-platform metadata (new format) */}
            {platformMetadata && Object.keys(platformMetadata).length > 0 && (
              <div className="space-y-4">
                {/* Platform Tabs */}
                <div className="flex border-b border-gray-200">
                  {Object.keys(platformMetadata).map((platform) => (
                    <button
                      key={platform}
                      onClick={() => setSelectedMetadataPlatform(platform)}
                      className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                        selectedMetadataPlatform === platform
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }`}
                    >
                      {platform === 'douyin' && '🎵 抖音'}
                      {platform === 'bilibili' && '📺 B站'}
                      {platform === 'xiaohongshu' && '📕 小红书'}
                      {platform === 'generic' && '📝 通用'}
                    </button>
                  ))}
                </div>

                {/* Selected Platform Metadata */}
                {editedPlatformMetadata?.[selectedMetadataPlatform as keyof PlatformMetadataMap] && (
                  <div className="space-y-4">
                    {/* Title */}
                    <div>
                      <label className="block text-xs font-medium text-gray-500 mb-1">
                        {t('taskDetail.metadata.generatedTitle')}
                      </label>
                      {isEditingMetadata ? (
                        <input
                          type="text"
                          value={editedPlatformMetadata[selectedMetadataPlatform as keyof PlatformMetadataMap]?.title || ''}
                          onChange={(e) => setEditedPlatformMetadata(prev => prev ? {
                            ...prev,
                            [selectedMetadataPlatform]: {
                              ...prev[selectedMetadataPlatform as keyof PlatformMetadataMap],
                              title: e.target.value
                            }
                          } : null)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                        />
                      ) : (
                        <p className="px-3 py-2 bg-gray-50 rounded-md text-sm">
                          {editedPlatformMetadata[selectedMetadataPlatform as keyof PlatformMetadataMap]?.title}
                        </p>
                      )}
                    </div>

                    {/* Description */}
                    <div>
                      <label className="block text-xs font-medium text-gray-500 mb-1">
                        {t('taskDetail.metadata.generatedDescription')}
                      </label>
                      {isEditingMetadata ? (
                        <textarea
                          value={editedPlatformMetadata[selectedMetadataPlatform as keyof PlatformMetadataMap]?.description || ''}
                          onChange={(e) => setEditedPlatformMetadata(prev => prev ? {
                            ...prev,
                            [selectedMetadataPlatform]: {
                              ...prev[selectedMetadataPlatform as keyof PlatformMetadataMap],
                              description: e.target.value
                            }
                          } : null)}
                          rows={4}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                        />
                      ) : (
                        <p className="px-3 py-2 bg-gray-50 rounded-md text-sm whitespace-pre-wrap">
                          {editedPlatformMetadata[selectedMetadataPlatform as keyof PlatformMetadataMap]?.description}
                        </p>
                      )}
                    </div>

                    {/* Keywords */}
                    <div>
                      <label className="block text-xs font-medium text-gray-500 mb-1">
                        {t('taskDetail.metadata.generatedKeywords')}
                      </label>
                      {isEditingMetadata ? (
                        <input
                          type="text"
                          value={editedPlatformMetadata[selectedMetadataPlatform as keyof PlatformMetadataMap]?.keywords?.join(', ') || ''}
                          onChange={(e) => setEditedPlatformMetadata(prev => prev ? {
                            ...prev,
                            [selectedMetadataPlatform]: {
                              ...prev[selectedMetadataPlatform as keyof PlatformMetadataMap],
                              keywords: e.target.value.split(',').map(k => k.trim()).filter(Boolean)
                            }
                          } : null)}
                          placeholder={t('taskDetail.metadata.keywordsPlaceholder')}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                        />
                      ) : (
                        <div className="flex flex-wrap gap-1.5">
                          {editedPlatformMetadata[selectedMetadataPlatform as keyof PlatformMetadataMap]?.keywords?.map((keyword, idx) => (
                            <span key={idx} className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
                              {keyword}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Action Buttons for Platform Metadata */}
                <div className="flex items-center gap-2 pt-2">
                  {isEditingMetadata ? (
                    <>
                      <button
                        onClick={() => {
                          if (editedPlatformMetadata) {
                            // Save all platform metadata
                            saveMetadataMutation.mutate(editedPlatformMetadata as unknown as { title: string; description: string; keywords: string[] })
                          }
                          setIsEditingMetadata(false)
                        }}
                        disabled={saveMetadataMutation.isPending}
                        className="px-3 py-1.5 text-sm bg-green-100 text-green-700 rounded-lg hover:bg-green-200 disabled:opacity-50"
                      >
                        {saveMetadataMutation.isPending ? t('common.saving') : t('common.save')}
                      </button>
                      <button
                        onClick={() => {
                          setEditedPlatformMetadata(platformMetadata)
                          setIsEditingMetadata(false)
                        }}
                        className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
                      >
                        {t('common.cancel')}
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => setIsEditingMetadata(true)}
                        className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
                      >
                        {t('taskDetail.metadata.edit')}
                      </button>
                      <button
                        onClick={() => {
                          setPlatformMetadata(null)
                          setEditedPlatformMetadata(null)
                          setMetadataSaved(false)
                          generateMetadataMutation.mutate()
                        }}
                        disabled={generateMetadataMutation.isPending}
                        className="px-3 py-1.5 text-sm bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 disabled:opacity-50"
                      >
                        <RefreshCw className={`h-3 w-3 inline mr-1 ${generateMetadataMutation.isPending ? 'animate-spin' : ''}`} />
                        {t('taskDetail.metadata.regenerate')}
                      </button>
                    </>
                  )}
                  {metadataSaved && !isEditingMetadata && !task?.metadata_approved && (
                    <span className="text-xs text-green-600 flex items-center gap-1">
                      <CheckCircle className="h-3 w-3" />
                      {t('taskDetail.metadata.saved')}
                    </span>
                  )}
                  {task?.metadata_approved && !isEditingMetadata && (
                    <span className="text-xs text-blue-600 flex items-center gap-1">
                      <CheckCircle className="h-3 w-3" />
                      {t('taskDetail.metadata.approved')}
                    </span>
                  )}
                </div>

                {/* Approve button for platform metadata */}
                {metadataSaved && isPendingReview && !isEditingMetadata && (
                  <div className="flex items-center gap-2 pt-2 border-t border-gray-100 mt-2">
                    <button
                      onClick={() => approveMetadataMutation.mutate()}
                      disabled={approveMetadataMutation.isPending}
                      className="px-4 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center gap-2"
                    >
                      {approveMetadataMutation.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <CheckCircle className="h-4 w-4" />
                      )}
                      {t('taskDetail.metadata.approve')} ({Object.keys(platformMetadata).length}个平台)
                    </button>
                    <span className="text-xs text-gray-500">
                      {t('taskDetail.metadata.approveHint')}
                    </span>
                  </div>
                )}

                {task?.metadata_approved && !isEditingMetadata && (
                  <div className="flex items-center gap-2 pt-2 border-t border-gray-100 mt-2">
                    <div className="flex items-center gap-2 px-4 py-2 bg-green-50 text-green-700 rounded-lg">
                      <CheckCircle className="h-4 w-4" />
                      <span className="text-sm font-medium">{t('taskDetail.metadata.alreadyApproved')}</span>
                    </div>
                  </div>
                )}

                <p className="text-xs text-gray-400 mt-2">
                  {t('taskDetail.metadata.usageHint')}
                </p>
              </div>
            )}

            {/* Legacy single metadata (old format) */}
            {generatedMetadata && !platformMetadata && (
              <div className="space-y-4">
                {/* Generated Title */}
                <div>
                  <label className="block text-xs font-medium text-gray-500 mb-1">
                    {t('taskDetail.metadata.generatedTitle')}
                  </label>
                  {isEditingMetadata ? (
                    <input
                      type="text"
                      value={editedMetadata?.title || ''}
                      onChange={(e) => setEditedMetadata(prev => prev ? {...prev, title: e.target.value} : null)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                    />
                  ) : (
                    <p className="px-3 py-2 bg-gray-50 rounded-md text-sm">
                      {editedMetadata?.title || generatedMetadata.title_translated || generatedMetadata.title}
                    </p>
                  )}
                </div>

                {/* Generated Description */}
                <div>
                  <label className="block text-xs font-medium text-gray-500 mb-1">
                    {t('taskDetail.metadata.generatedDescription')}
                  </label>
                  {isEditingMetadata ? (
                    <textarea
                      value={editedMetadata?.description || ''}
                      onChange={(e) => setEditedMetadata(prev => prev ? {...prev, description: e.target.value} : null)}
                      rows={4}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                    />
                  ) : (
                    <p className="px-3 py-2 bg-gray-50 rounded-md text-sm whitespace-pre-wrap">
                      {editedMetadata?.description || generatedMetadata.description}
                    </p>
                  )}
                </div>

                {/* Generated Keywords */}
                <div>
                  <label className="block text-xs font-medium text-gray-500 mb-1">
                    {t('taskDetail.metadata.generatedKeywords')}
                  </label>
                  {isEditingMetadata ? (
                    <input
                      type="text"
                      value={editedMetadata?.keywords?.join(', ') || ''}
                      onChange={(e) => setEditedMetadata(prev => prev ? {...prev, keywords: e.target.value.split(',').map(k => k.trim()).filter(Boolean)} : null)}
                      placeholder={t('taskDetail.metadata.keywordsPlaceholder')}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                    />
                  ) : (
                    <div className="flex flex-wrap gap-1.5">
                      {(editedMetadata?.keywords || generatedMetadata.keywords).map((keyword, idx) => (
                        <span key={idx} className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
                          {keyword}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex items-center gap-2 pt-2">
                  {isEditingMetadata ? (
                    <>
                      <button
                        onClick={() => {
                          if (editedMetadata) {
                            saveMetadataMutation.mutate(editedMetadata)
                          }
                          setIsEditingMetadata(false)
                        }}
                        disabled={saveMetadataMutation.isPending}
                        className="px-3 py-1.5 text-sm bg-green-100 text-green-700 rounded-lg hover:bg-green-200 disabled:opacity-50"
                      >
                        {saveMetadataMutation.isPending ? t('common.saving') : t('common.save')}
                      </button>
                      <button
                        onClick={() => {
                          setEditedMetadata({
                            title: generatedMetadata.title_translated || generatedMetadata.title,
                            description: generatedMetadata.description,
                            keywords: generatedMetadata.keywords,
                          })
                          setIsEditingMetadata(false)
                        }}
                        className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
                      >
                        {t('common.cancel')}
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => setIsEditingMetadata(true)}
                        className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
                      >
                        {t('taskDetail.metadata.edit')}
                      </button>
                      <button
                        onClick={() => {
                          setGeneratedMetadata(null)
                          setEditedMetadata(null)
                          setMetadataSaved(false)
                          generateMetadataMutation.mutate()
                        }}
                        disabled={generateMetadataMutation.isPending}
                        className="px-3 py-1.5 text-sm bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 disabled:opacity-50"
                      >
                        <RefreshCw className={`h-3 w-3 inline mr-1 ${generateMetadataMutation.isPending ? 'animate-spin' : ''}`} />
                        {t('taskDetail.metadata.regenerate')}
                      </button>
                    </>
                  )}
                  {metadataSaved && !isEditingMetadata && !task?.metadata_approved && (
                    <span className="text-xs text-green-600 flex items-center gap-1">
                      <CheckCircle className="h-3 w-3" />
                      {t('taskDetail.metadata.saved')}
                    </span>
                  )}
                  {task?.metadata_approved && !isEditingMetadata && (
                    <span className="text-xs text-blue-600 flex items-center gap-1">
                      <CheckCircle className="h-3 w-3" />
                      {t('taskDetail.metadata.approved')}
                    </span>
                  )}
                </div>

                {/* Approve button - show only when metadata saved and status is pending_review */}
                {metadataSaved && isPendingReview && !isEditingMetadata && (
                  <div className="flex items-center gap-2 pt-2 border-t border-gray-100 mt-2">
                    <button
                      onClick={() => approveMetadataMutation.mutate()}
                      disabled={approveMetadataMutation.isPending}
                      className="px-4 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center gap-2"
                    >
                      {approveMetadataMutation.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <CheckCircle className="h-4 w-4" />
                      )}
                      {t('taskDetail.metadata.approve')}
                    </button>
                    <span className="text-xs text-gray-500">
                      {t('taskDetail.metadata.approveHint')}
                    </span>
                  </div>
                )}

                {/* Show approved status when metadata is approved */}
                {task?.metadata_approved && !isEditingMetadata && (
                  <div className="flex items-center gap-2 pt-2 border-t border-gray-100 mt-2">
                    <div className="flex items-center gap-2 px-4 py-2 bg-green-50 text-green-700 rounded-lg">
                      <CheckCircle className="h-4 w-4" />
                      <span className="text-sm font-medium">{t('taskDetail.metadata.alreadyApproved')}</span>
                    </div>
                  </div>
                )}

                {/* Info about usage */}
                <p className="text-xs text-gray-400 mt-2">
                  {t('taskDetail.metadata.usageHint')}
                </p>
              </div>
            )}

            {!generatedMetadata && !platformMetadata && !generateMetadataMutation.isPending && (
              <p className="text-sm text-gray-500">
                {t('taskDetail.metadata.hint')}
              </p>
            )}
          </div>
        )}

        {/* Upload Results */}
        {Object.keys(task.upload_results).length > 0 && (
          <div className="p-6">
            <h3 className="text-sm font-medium text-gray-700 mb-4">{t('taskDetail.uploadResults')}</h3>
            <div className="space-y-3">
              {Object.entries(task.upload_results).map(([platform, result]: [string, UploadResult]) => (
                <div
                  key={platform}
                  className={`flex items-center justify-between p-3 rounded-lg ${
                    result.success ? 'bg-green-50' : 'bg-red-50'
                  }`}
                >
                  <div className="flex items-center">
                    {result.success ? (
                      <CheckCircle className="h-5 w-5 text-green-500 mr-3" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-500 mr-3" />
                    )}
                    <div>
                      <span className="font-medium capitalize">{platform}</span>
                      {result.error && (
                        <p className="text-sm text-red-600">{result.error}</p>
                      )}
                    </div>
                  </div>
                  {result.success && result.video_url && (
                    <a
                      href={result.video_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center text-blue-600 hover:text-blue-700"
                    >
                      {t('taskDetail.viewVideo')}
                      <ExternalLink className="h-4 w-4 ml-1" />
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Proofreading Results */}
        {task.proofreading_result && (
          <div className="p-6 border-t border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-700">{t('taskDetail.proofreading.title')}</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => optimizeSubtitlesMutation.mutate(undefined)}
                  disabled={isOptimizing}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-purple-600 hover:bg-purple-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title={t('taskDetail.proofreading.optimizeTooltip')}
                >
                  {isOptimizing ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                  {isOptimizing ? t('taskDetail.proofreading.optimizing') : t('taskDetail.proofreading.optimizeSubtitles')}
                </button>
                <button
                  onClick={() => setShowSubtitleEditor(true)}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                >
                  <Edit3 className="w-4 h-4" />
                  {t('taskDetail.proofreading.editSubtitles')}
                </button>
              </div>
            </div>

            {/* Overall Confidence */}
            <div className="mb-4 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">{t('taskDetail.proofreading.overallConfidence')}</span>
                <span className={`font-medium ${
                  task.proofreading_result.overall_confidence >= 0.8 ? 'text-green-600' :
                  task.proofreading_result.overall_confidence >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {Math.round(task.proofreading_result.overall_confidence * 100)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    task.proofreading_result.overall_confidence >= 0.8 ? 'bg-green-500' :
                    task.proofreading_result.overall_confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${task.proofreading_result.overall_confidence * 100}%` }}
                />
              </div>
            </div>

            {/* Pause Reason */}
            {task.proofreading_result.should_pause && task.proofreading_result.pause_reason && (
              <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <p className="text-sm text-amber-700">
                  <span className="font-medium">{t('taskDetail.proofreading.pauseReason')}:</span> {task.proofreading_result.pause_reason}
                </p>
              </div>
            )}

            {/* Issues Summary */}
            {task.proofreading_result.total_issues > 0 ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <XCircle className="h-4 w-4 text-amber-500" />
                  <span className="text-sm text-gray-700">
                    {t('taskDetail.proofreading.issuesCount', { count: task.proofreading_result.total_issues })}
                  </span>
                </div>

                {/* Issues by Severity */}
                {task.proofreading_result.issues_by_severity && (
                  <div>
                    <p className="text-xs text-gray-500 mb-2">{t('taskDetail.proofreading.issuesBySeverity')}</p>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(task.proofreading_result.issues_by_severity).map(([severity, count]: [string, number]) => (
                        count > 0 && (
                          <span
                            key={severity}
                            className={`px-2 py-1 text-xs rounded-full ${
                              severity === 'critical' ? 'bg-red-100 text-red-700' :
                              severity === 'error' ? 'bg-orange-100 text-orange-700' :
                              severity === 'warning' ? 'bg-yellow-100 text-yellow-700' :
                              'bg-blue-100 text-blue-700'
                            }`}
                          >
                            {t(`taskDetail.proofreading.severity.${severity}`)}: {count}
                          </span>
                        )
                      ))}
                    </div>
                  </div>
                )}

                {/* Issues by Type */}
                {task.proofreading_result.issues_by_type && (
                  <div>
                    <p className="text-xs text-gray-500 mb-2">{t('taskDetail.proofreading.issuesByType')}</p>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(task.proofreading_result.issues_by_type).map(([type, count]: [string, number]) => (
                        count > 0 && (
                          <span
                            key={type}
                            className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-700"
                          >
                            {t(`taskDetail.proofreading.issueType.${type}`, { defaultValue: type })}: {count}
                          </span>
                        )
                      ))}
                    </div>
                  </div>
                )}

                {/* Detailed Issues List */}
                {task.proofreading_result.segments && (
                  <div className="mt-4">
                    <p className="text-xs text-gray-500 mb-2">{t('taskDetail.proofreading.detailedIssues')}</p>
                    <div className="max-h-96 overflow-y-auto space-y-3">
                      {task.proofreading_result.segments
                        .filter((seg: SegmentProofreadResult) => seg.issues && seg.issues.length > 0)
                        .map((segment: SegmentProofreadResult, idx: number) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-xs font-medium text-gray-500">
                                #{segment.index + 1} | {segment.start_time?.toFixed(1)}s - {segment.end_time?.toFixed(1)}s
                              </span>
                              <span className={`text-xs px-2 py-0.5 rounded ${
                                segment.confidence >= 0.8 ? 'bg-green-100 text-green-700' :
                                segment.confidence >= 0.6 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
                              }`}>
                                {Math.round(segment.confidence * 100)}%
                              </span>
                            </div>
                            {segment.original_text && (
                              <p className="text-xs text-gray-600 mb-1">
                                <span className="font-medium">{t('taskDetail.proofreading.original')}:</span> {segment.original_text}
                              </p>
                            )}
                            {segment.translated_text && (
                              <p className="text-xs text-gray-800 mb-2">
                                <span className="font-medium">{t('taskDetail.proofreading.translated')}:</span> {segment.translated_text}
                              </p>
                            )}
                            {segment.issues.map((issue: ProofreadingIssue, issueIdx: number) => (
                              <div key={issueIdx} className={`mt-2 p-2 rounded text-xs ${
                                issue.severity === 'critical' ? 'bg-red-50 border-l-2 border-red-500' :
                                issue.severity === 'error' ? 'bg-orange-50 border-l-2 border-orange-500' :
                                issue.severity === 'warning' ? 'bg-yellow-50 border-l-2 border-yellow-500' :
                                'bg-blue-50 border-l-2 border-blue-500'
                              }`}>
                                <p className="text-gray-700">{issue.message}</p>
                                {issue.suggestion && (
                                  <p className="mt-1 text-gray-600">
                                    <span className="font-medium">{t('taskDetail.proofreading.suggestion')}:</span> {issue.suggestion}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle className="h-4 w-4" />
                <span className="text-sm">{t('taskDetail.proofreading.noIssues')}</span>
              </div>
            )}

            {/* Confirm and Continue Button - Show when task is paused waiting for optimization/confirmation */}
            {isPaused && !!task.steps?.optimize?.metadata?.waiting_for_user && (
              <div className="mt-6 pt-4 border-t border-gray-200">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-600">
                    {t('taskDetail.proofreading.confirmHint')}
                  </p>
                  <button
                    onClick={() => continueMutation.mutate({ stepName: 'tts' })}
                    disabled={continueMutation.isPending || isOptimizing}
                    className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                  >
                    {continueMutation.isPending ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Play className="w-4 h-4" />
                    )}
                    {t('taskDetail.proofreading.confirmAndContinue')}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Subtitle Editor Modal */}
      {showSubtitleEditor && (
        <SubtitleEditor
          taskId={taskId!}
          proofreadingResult={task.proofreading_result}
          onClose={() => setShowSubtitleEditor(false)}
          onSaved={() => {
            queryClient.invalidateQueries({ queryKey: ['task', taskId] })
          }}
        />
      )}

      {/* Optimization Result Modal (for backward compatibility) */}
      {optimizationResult && (
        <OptimizationResultModal
          isOpen={true}
          onClose={() => setOptimizationResult(null)}
          changes={optimizationResult.changes}
          optimizedCount={optimizationResult.optimizedCount}
          totalSegments={optimizationResult.totalSegments}
          onSeekTo={handleSeekTo}
        />
      )}

      {/* Optimization Review Panel (Split view with video) */}
      {showOptimizationReview && task.optimization_result && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
          <div className="w-full max-w-7xl">
            <OptimizationReviewPanel
              videoUrl={
                filesInfo?.final_video?.available
                  ? taskApi.getFileDownloadUrl(taskId!, 'final_video', task.updated_at)
                  : filesInfo?.video?.available
                    ? taskApi.getFileDownloadUrl(taskId!, 'video', task.updated_at)
                    : ''
              }
              changes={task.optimization_result.changes || []}
              optimizedCount={task.optimization_result.optimized_count || 0}
              totalSegments={task.optimization_result.total_segments || 0}
              onClose={() => setShowOptimizationReview(false)}
            />
          </div>
        </div>
      )}
    </div>
  )
}

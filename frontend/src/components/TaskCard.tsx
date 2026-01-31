import { useState, useRef, useEffect, memo } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { CheckCircle, XCircle, Clock, Loader2, Pause, Timer, Trash2, FileVideo, Play, X, Maximize, Volume2, VolumeX, CheckSquare, Square } from 'lucide-react'
import type { Task } from '../types'
import { taskApi } from '../services/api'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8888'

const statusConfig: Record<string, { icon: React.ComponentType<{ className?: string }>; color: string; bg: string; textColor: string; labelKey: string }> = {
  pending: { icon: Clock, color: 'text-gray-500', bg: 'bg-gray-100', textColor: 'text-gray-600', labelKey: 'dashboard.status.pending' },
  queued: { icon: Clock, color: 'text-orange-500', bg: 'bg-orange-50', textColor: 'text-orange-600', labelKey: 'dashboard.status.queued' },
  downloading: { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-50', textColor: 'text-blue-600', labelKey: 'dashboard.status.downloading' },
  transcribing: { icon: Loader2, color: 'text-purple-500', bg: 'bg-purple-50', textColor: 'text-purple-600', labelKey: 'dashboard.status.transcribing' },
  translating: { icon: Loader2, color: 'text-amber-500', bg: 'bg-amber-50', textColor: 'text-amber-600', labelKey: 'dashboard.status.translating' },
  generating_tts: { icon: Loader2, color: 'text-orange-500', bg: 'bg-orange-50', textColor: 'text-orange-600', labelKey: 'dashboard.status.generating_tts' },
  processing_video: { icon: Loader2, color: 'text-cyan-500', bg: 'bg-cyan-50', textColor: 'text-cyan-600', labelKey: 'dashboard.status.processing_video' },
  pending_review: { icon: Clock, color: 'text-yellow-500', bg: 'bg-yellow-50', textColor: 'text-yellow-600', labelKey: 'dashboard.status.pending_review' },
  pending_upload: { icon: Clock, color: 'text-blue-500', bg: 'bg-blue-50', textColor: 'text-blue-600', labelKey: 'dashboard.status.pending_upload' },
  uploading: { icon: Loader2, color: 'text-indigo-500', bg: 'bg-indigo-50', textColor: 'text-indigo-600', labelKey: 'dashboard.status.uploading' },
  uploaded: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50', textColor: 'text-green-600', labelKey: 'dashboard.status.uploaded' },
  // Legacy: 'completed' maps to 'uploaded' for backwards compatibility with existing database records
  completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50', textColor: 'text-green-600', labelKey: 'dashboard.status.uploaded' },
  failed: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-50', textColor: 'text-red-600', labelKey: 'dashboard.status.failed' },
  paused: { icon: Pause, color: 'text-amber-500', bg: 'bg-amber-50', textColor: 'text-amber-600', labelKey: 'dashboard.status.paused' },
}

// Step name keys for translation
const stepNameKeys: Record<string, string> = {
  download: 'dashboard.stepNames.download',
  transcribe: 'dashboard.stepNames.transcribe',
  translate: 'dashboard.stepNames.translate',
  tts: 'dashboard.stepNames.tts',
  process_video: 'dashboard.stepNames.process_video',
  upload: 'dashboard.stepNames.upload',
}

// Preview video component for progress bar hover - uses video element directly (no canvas/CORS issues)
const PreviewVideo = memo(function PreviewVideo({ src, seekTime, isVertical }: { src: string; seekTime: number; isVertical?: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isReady, setIsReady] = useState(false)
  const lastSeekRef = useRef<number>(0)

  useEffect(() => {
    const video = videoRef.current
    if (!video || !isReady) return

    // Only seek if the difference is significant (more than 0.3 second)
    if (Math.abs(lastSeekRef.current - seekTime) > 0.3) {
      lastSeekRef.current = seekTime
      video.currentTime = seekTime
    }
  }, [seekTime, isReady])

  // Adapt preview size for vertical videos
  const previewClass = isVertical
    ? "w-[90px] h-[160px] bg-gray-900 relative overflow-hidden"
    : "w-[160px] h-[90px] bg-gray-900 relative overflow-hidden"

  return (
    <div className={previewClass}>
      <video
        ref={videoRef}
        src={src}
        className="w-full h-full object-contain"
        muted
        playsInline
        preload="auto"
        onLoadedMetadata={() => {
          setIsReady(true)
          // Initial seek
          if (videoRef.current) {
            videoRef.current.currentTime = seekTime
          }
        }}
      />
      {/* Loading indicator - only show when not ready */}
      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <Loader2 className="w-4 h-4 text-white/50 animate-spin" />
        </div>
      )}
    </div>
  )
})

interface TaskCardProps {
  task: Task
  isSelected?: boolean
  onSelect?: () => void
  onDeleteRequest?: () => void
}

export default function TaskCard({ task, isSelected, onSelect, onDeleteRequest }: TaskCardProps) {
  const { t } = useTranslation()
  const [isVideoMode, setIsVideoMode] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [showPreview, setShowPreview] = useState(false)
  const [previewTime, setPreviewTime] = useState(0)
  const [previewPosition, setPreviewPosition] = useState(0)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const videoContainerRef = useRef<HTMLDivElement>(null)
  const progressBarRef = useRef<HTMLDivElement>(null)
  const config = statusConfig[task.status] || statusConfig.pending
  const Icon = config.icon
  // Include 'completed' for backwards compatibility with existing database records
  const isProcessing = !['uploaded', 'completed', 'pending_review', 'pending_upload', 'failed', 'pending', 'paused', 'queued'].includes(task.status)
  const isQueued = task.status === 'queued'
  const hasVideo = ['uploaded', 'completed', 'pending_review', 'pending_upload'].includes(task.status) && task.files?.final_video

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onDeleteRequest?.()
  }

  const handleSelect = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onSelect?.()
  }

  // Get completed steps count
  const completedSteps = task.steps
    ? Object.values(task.steps).filter(s => s.status === 'completed').length
    : 0
  const totalSteps = 6

  // Get failed step if any
  const failedStep = task.steps
    ? Object.values(task.steps).find(s => s.status === 'failed')
    : null

  // Thumbnail URL - prefer local thumbnail, fallback to video_info
  const thumbnailUrl = task.thumbnail_url
    ? `${API_BASE}${task.thumbnail_url}`
    : task.video_info?.thumbnail_url

  const [imageError, setImageError] = useState(false)
  // State for auto-detected vertical video (from thumbnail dimensions)
  const [detectedVertical, setDetectedVertical] = useState<boolean | null>(null)
  const thumbnailRef = useRef<HTMLImageElement>(null)

  const videoSrc = taskApi.getFileDownloadUrl(task.task_id, 'final_video')

  // Detect if video is vertical - prefer backend data, fallback to auto-detected
  const isVerticalVideo = task.video_info?.is_vertical ?? detectedVertical ?? false

  // Check thumbnail dimensions on mount and when thumbnailUrl changes
  useEffect(() => {
    const checkThumbnailDimensions = () => {
      const img = thumbnailRef.current
      if (img && img.naturalWidth && img.naturalHeight) {
        const aspectRatio = img.naturalWidth / img.naturalHeight
        const isVertical = aspectRatio < 0.9
        console.log(`[TaskCard] ${task.task_id.slice(0, 8)} thumbnail: ${img.naturalWidth}x${img.naturalHeight}, ratio=${aspectRatio.toFixed(2)}, isVertical=${isVertical}`)
        setDetectedVertical(isVertical)
      }
    }

    // Check immediately (for cached images)
    checkThumbnailDimensions()

    // Also check after a short delay (for images that load quickly)
    const timer = setTimeout(checkThumbnailDimensions, 100)
    return () => clearTimeout(timer)
  }, [thumbnailUrl, task.task_id])

  // Handle thumbnail load to detect vertical video from image dimensions
  const handleThumbnailLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget
    if (img.naturalWidth && img.naturalHeight) {
      const aspectRatio = img.naturalWidth / img.naturalHeight
      const isVertical = aspectRatio < 0.9
      console.log(`[TaskCard] ${task.task_id.slice(0, 8)} onLoad: ${img.naturalWidth}x${img.naturalHeight}, ratio=${aspectRatio.toFixed(2)}, isVertical=${isVertical}`)
      setDetectedVertical(isVertical)
    }
  }

  return (
    <Link
      to={`/task/${task.task_id}`}
      className={`group block bg-white rounded-xl shadow-sm border-2 transition-all duration-200 overflow-hidden ${
        isSelected ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-200 hover:border-gray-300 hover:shadow-lg'
      }`}
    >
      {/* Thumbnail / Video Section - adapt for vertical videos */}
      <div className={`relative bg-gradient-to-br from-gray-100 to-gray-200 overflow-hidden ${
        isVerticalVideo ? 'aspect-[9/16] max-h-[280px] mx-auto' : 'aspect-video'
      }`}>
        {/* Video Player Mode */}
        {isVideoMode && hasVideo ? (
          <div
            ref={videoContainerRef}
            className={`relative w-full h-full bg-black ${isFullscreen ? '' : 'overflow-hidden'}`}
          >
            <video
              ref={videoRef}
              src={videoSrc}
              className="w-full h-full object-contain"
              autoPlay
              muted={isMuted}
              playsInline
              onTimeUpdate={() => {
                if (videoRef.current) {
                  setCurrentTime(videoRef.current.currentTime)
                }
              }}
              onLoadedMetadata={() => {
                if (videoRef.current) {
                  setDuration(videoRef.current.duration)
                  // Auto-detect vertical video from video dimensions
                  if (detectedVertical === null) {
                    const aspectRatio = videoRef.current.videoWidth / videoRef.current.videoHeight
                    setDetectedVertical(aspectRatio < 0.9)
                  }
                }
              }}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                if (videoRef.current) {
                  if (isPlaying) {
                    videoRef.current.pause()
                  } else {
                    videoRef.current.play()
                  }
                }
              }}
            />

            {/* Custom Controls Overlay */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
              {/* Progress Bar with Preview */}
              <div
                ref={progressBarRef}
                className="relative mb-2 cursor-pointer group/progress"
                onMouseMove={(e) => {
                  e.stopPropagation()
                  if (!progressBarRef.current || !duration) return
                  const rect = progressBarRef.current.getBoundingClientRect()
                  const x = e.clientX - rect.left
                  const percentage = Math.max(0, Math.min(1, x / rect.width))
                  const time = percentage * duration
                  setPreviewTime(time)
                  setPreviewPosition(x)
                  setShowPreview(true)
                }}
                onMouseLeave={() => setShowPreview(false)}
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  if (!progressBarRef.current || !duration || !videoRef.current) return
                  const rect = progressBarRef.current.getBoundingClientRect()
                  const x = e.clientX - rect.left
                  const percentage = Math.max(0, Math.min(1, x / rect.width))
                  const time = percentage * duration
                  videoRef.current.currentTime = time
                  setCurrentTime(time)
                }}
              >
                {/* Preview Tooltip */}
                {showPreview && duration > 0 && (
                  <div
                    className="absolute bottom-full mb-2 pointer-events-none z-50"
                    style={{
                      left: `${Math.max(85, Math.min(previewPosition, (progressBarRef.current?.clientWidth || 0) - 85))}px`,
                      transform: 'translateX(-50%)',
                    }}
                  >
                    <div className="bg-black rounded overflow-hidden shadow-lg border border-white/30">
                      <PreviewVideo src={videoSrc} seekTime={previewTime} isVertical={isVerticalVideo} />
                      <div className="text-white text-xs text-center py-1 bg-black font-mono">
                        {formatDuration(previewTime)}
                      </div>
                    </div>
                  </div>
                )}
                {/* Progress Bar Background */}
                <div className="h-1 bg-white/30 rounded-full overflow-hidden group-hover/progress:h-1.5 transition-all">
                  <div
                    className="h-full bg-white rounded-full"
                    style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
                  />
                </div>
              </div>

              {/* Control Buttons */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-1">
                  {/* Play/Pause */}
                  <button
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      if (videoRef.current) {
                        if (isPlaying) {
                          videoRef.current.pause()
                        } else {
                          videoRef.current.play()
                        }
                      }
                    }}
                    className="p-1 text-white hover:bg-white/20 rounded transition-colors"
                  >
                    {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" fill="white" />}
                  </button>
                  {/* Mute/Unmute */}
                  <button
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      setIsMuted(!isMuted)
                      if (videoRef.current) {
                        videoRef.current.muted = !isMuted
                      }
                    }}
                    className="p-1 text-white hover:bg-white/20 rounded transition-colors"
                  >
                    {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                  </button>
                  {/* Time Display */}
                  <span className="text-white text-xs font-mono ml-1">
                    {formatDuration(currentTime)} / {formatDuration(duration)}
                  </span>
                </div>
                <div className="flex items-center space-x-1">
                  {/* Fullscreen */}
                  <button
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      if (videoContainerRef.current) {
                        if (document.fullscreenElement) {
                          document.exitFullscreen()
                        } else {
                          videoContainerRef.current.requestFullscreen()
                        }
                      }
                    }}
                    className="p-1 text-white hover:bg-white/20 rounded transition-colors"
                  >
                    <Maximize className="h-4 w-4" />
                  </button>
                  {/* Close */}
                  <button
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      setIsVideoMode(false)
                      if (videoRef.current) {
                        videoRef.current.pause()
                      }
                    }}
                    className="p-1 text-white hover:bg-white/20 rounded transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Thumbnail Mode */}
            {thumbnailUrl && !imageError ? (
              <img
                ref={thumbnailRef}
                src={thumbnailUrl}
                alt={task.video_info?.title || t('dashboard.taskCard.videoThumbnail')}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                onError={() => setImageError(true)}
                onLoad={handleThumbnailLoad}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <div className="text-center">
                  <FileVideo className="h-12 w-12 text-gray-300 mx-auto" />
                  <span className="text-xs text-gray-400 mt-1 block">{t('dashboard.taskCard.noThumbnail')}</span>
                </div>
              </div>
            )}

            {/* Status Badge - Overlay on thumbnail - z-20 above play overlay */}
            <div className="absolute top-2 left-2 z-20">
              <div className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full backdrop-blur-sm ${config.bg} bg-opacity-90 shadow-sm`}>
                <Icon className={`h-3.5 w-3.5 ${config.color} ${isProcessing ? 'animate-spin' : ''}`} />
                <span className={`text-xs font-medium ${config.textColor}`}>
                  {isQueued && task.queue_position !== undefined && task.queue_position > 0
                    ? t('dashboard.status.queuedWithPosition', { position: task.queue_position })
                    : t(config.labelKey)}
                </span>
              </div>
            </div>

            {/* Action buttons - top right - z-20 to stay above play overlay */}
            <div className={`absolute top-2 right-2 z-20 flex items-center space-x-1 transition-opacity ${isSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}>
              {/* Selection checkbox */}
              <button
                onClick={handleSelect}
                className={`p-1.5 rounded-md backdrop-blur-sm transition-all ${
                  isSelected
                    ? 'bg-blue-500 text-white'
                    : 'bg-black/30 text-white hover:bg-blue-500'
                }`}
                title={isSelected ? t('dashboard.taskCard.deselect') : t('dashboard.taskCard.select')}
              >
                {isSelected ? (
                  <CheckSquare className="h-4 w-4" />
                ) : (
                  <Square className="h-4 w-4" />
                )}
              </button>
              {/* Delete button */}
              <button
                onClick={handleDelete}
                disabled={isProcessing}
                className={`p-1.5 rounded-md backdrop-blur-sm transition-all bg-black/30 text-white hover:bg-red-500 ${
                  isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                title={t('dashboard.taskCard.deleteTask')}
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>

            {/* Processing overlay */}
            {isProcessing && (
              <div className="absolute inset-0 bg-black/20 flex items-center justify-center">
                <div className="bg-white/90 backdrop-blur-sm rounded-full p-3 shadow-lg">
                  <Loader2 className={`h-6 w-6 ${config.color} animate-spin`} />
                </div>
              </div>
            )}

            {/* Play icon for completed tasks with video - z-10 below action buttons */}
            {hasVideo && (
              <div
                className="absolute inset-0 z-10 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  setIsVideoMode(true)
                }}
              >
                <div className="bg-black/50 backdrop-blur-sm rounded-full p-3 hover:bg-black/70 transition-colors">
                  <Play className="h-8 w-8 text-white" fill="white" />
                </div>
              </div>
            )}

            {/* Duration badge if available */}
            {task.video_info?.duration && (
              <div className="absolute bottom-2 right-2 px-1.5 py-0.5 bg-black/70 backdrop-blur-sm rounded text-xs text-white font-medium">
                {formatDuration(task.video_info.duration)}
              </div>
            )}
          </>
        )}
      </div>

      {/* Content Section */}
      <div className="p-4">
        {/* Title */}
        <h3 className="text-sm font-semibold text-gray-900 line-clamp-2 leading-snug min-h-[2.5rem]">
          {task.video_info?.title || `${t('dashboard.taskCard.task')} ${task.task_id.slice(0, 8)}...`}
        </h3>

        {/* Step Progress Indicators */}
        {task.steps && (
          <div className="mt-3 flex items-center space-x-0.5">
            {Object.entries(task.steps).map(([stepName, step]) => {
              let bgColor = 'bg-gray-200'
              let animateClass = ''
              if (step.status === 'completed') bgColor = 'bg-green-500'
              else if (step.status === 'running') {
                bgColor = 'bg-blue-500'
                animateClass = 'animate-pulse'
              }
              else if (step.status === 'failed') bgColor = 'bg-red-500'
              else if (step.status === 'skipped') bgColor = 'bg-gray-300'

              return (
                <div
                  key={stepName}
                  className={`flex-1 h-1 rounded-full ${bgColor} ${animateClass} transition-colors`}
                  title={`${t(stepNameKeys[stepName]) || stepName}: ${step.status}`}
                />
              )
            })}
          </div>
        )}

        {/* Progress percentage for processing tasks */}
        {(isProcessing || task.status === 'paused') && (
          <div className="mt-2 flex items-center justify-between text-xs">
            <span className="text-gray-500">{completedSteps}/{totalSteps} {t('dashboard.taskCard.steps')}</span>
            <span className={`font-medium ${config.textColor}`}>{task.progress}%</span>
          </div>
        )}

        {/* Failed step indicator */}
        {failedStep && (
          <p className="mt-2 text-xs text-red-600 truncate">
            {t('dashboard.taskCard.failedStep')}: {t(stepNameKeys[failedStep.step_name]) || failedStep.step_name}
          </p>
        )}

        {/* Upload results - show for uploaded or completed (backwards compat) */}
        {(task.status === 'uploaded' || task.status === 'completed') && Object.keys(task.upload_results || {}).length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {Object.entries(task.upload_results).map(([platform, result]) => (
              <span
                key={platform}
                className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                  result.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                }`}
              >
                {platform}
                {result.success ? ' ✓' : ' ✗'}
              </span>
            ))}
          </div>
        )}

        {/* Footer - Time info */}
        <div className="mt-3 pt-3 border-t border-gray-100 flex items-center justify-between text-xs text-gray-400">
          <span>{new Date(task.created_at).toLocaleDateString()}</span>
          {task.total_time_formatted && (
            <span className="flex items-center space-x-1">
              <Timer className="h-3 w-3" />
              <span>{task.total_time_formatted}</span>
            </span>
          )}
        </div>
      </div>
    </Link>
  )
}

// Helper function to format duration
function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

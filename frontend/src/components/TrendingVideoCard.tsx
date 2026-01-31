import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Clock, Eye, ExternalLink, Plus, Check, Calendar, ChevronDown, Zap, FileText, Upload, Sparkles } from 'lucide-react'
import type { TrendingVideo } from '../types'

type ProcessingMode = 'full' | 'subtitle' | 'direct' | 'auto'

interface TrendingVideoCardProps {
  video: TrendingVideo
  selected: boolean
  onSelect: (videoId: string, selected: boolean) => void
  onCreateTask: (video: TrendingVideo) => void
  onQuickCreate?: (video: TrendingVideo, mode: ProcessingMode) => void
}

function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`
}

function formatViewCount(count: number): string {
  if (count >= 1000000) {
    return `${(count / 1000000).toFixed(1)}M`
  }
  if (count >= 1000) {
    return `${(count / 1000).toFixed(1)}K`
  }
  return count.toString()
}

function formatRelativeTime(dateString: string | undefined, t: (key: string, options?: Record<string, unknown>) => string): string {
  if (!dateString) return ''

  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / (1000 * 60))
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffMins < 60) {
    return t('discover.timeAgo.minutes', { count: diffMins })
  }
  if (diffHours < 24) {
    return t('discover.timeAgo.hours', { count: diffHours })
  }
  if (diffDays < 7) {
    return t('discover.timeAgo.days', { count: diffDays })
  }
  if (diffDays < 30) {
    return t('discover.timeAgo.weeks', { count: Math.floor(diffDays / 7) })
  }
  // Return formatted date for older videos
  return date.toLocaleDateString()
}

export default function TrendingVideoCard({
  video,
  selected,
  onSelect,
  onCreateTask,
  onQuickCreate,
}: TrendingVideoCardProps) {
  const { t } = useTranslation()
  const [showModeMenu, setShowModeMenu] = useState(false)

  const modeOptions: { mode: ProcessingMode; icon: React.ReactNode; label: string; desc: string }[] = [
    { mode: 'full', icon: <Zap className="w-3 h-3" />, label: t('newTask.modeFull'), desc: t('newTask.modeFullDesc') },
    { mode: 'subtitle', icon: <FileText className="w-3 h-3" />, label: t('newTask.modeSubtitle'), desc: t('newTask.modeSubtitleDesc') },
    { mode: 'direct', icon: <Upload className="w-3 h-3" />, label: t('newTask.modeDirect'), desc: t('newTask.modeDirectDesc') },
    { mode: 'auto', icon: <Sparkles className="w-3 h-3" />, label: t('newTask.modeAuto'), desc: t('newTask.modeAutoDesc') },
  ]

  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-shadow">
      {/* Thumbnail with duration and checkbox */}
      <div className="relative aspect-video bg-gray-200">
        {video.thumbnail_url ? (
          <img
            src={video.thumbnail_url}
            alt={video.title}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            No Thumbnail
          </div>
        )}

        {/* Duration badge */}
        <div className="absolute bottom-2 right-2 bg-black bg-opacity-80 text-white text-xs px-1.5 py-0.5 rounded">
          {formatDuration(video.duration)}
        </div>

        {/* Selection checkbox */}
        <button
          onClick={() => onSelect(video.video_id, !selected)}
          className={`absolute top-2 left-2 w-6 h-6 rounded-full flex items-center justify-center transition-colors ${
            selected
              ? 'bg-blue-600 text-white'
              : 'bg-white bg-opacity-80 text-gray-600 hover:bg-blue-100'
          }`}
        >
          {selected ? <Check className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
        </button>
      </div>

      {/* Content */}
      <div className="p-3">
        {/* Title */}
        <h3 className="font-medium text-gray-900 line-clamp-2 text-sm mb-1" title={video.title}>
          {video.title}
        </h3>

        {/* Channel */}
        <p className="text-xs text-gray-500 mb-2 truncate">
          {video.channel_name}
        </p>

        {/* Stats */}
        <div className="flex items-center flex-wrap gap-x-3 gap-y-1 text-xs text-gray-500 mb-3">
          <span className="flex items-center gap-1">
            <Eye className="w-3.5 h-3.5" />
            {formatViewCount(video.view_count)}
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-3.5 h-3.5" />
            {formatDuration(video.duration)}
          </span>
          {video.published_at && (
            <span className="flex items-center gap-1" title={new Date(video.published_at).toLocaleString()}>
              <Calendar className="w-3.5 h-3.5" />
              {formatRelativeTime(video.published_at, t)}
            </span>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={() => onCreateTask(video)}
            className="flex-1 px-3 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 rounded hover:bg-blue-100 transition-colors"
          >
            {t('discover.createTask')}
          </button>
          
          {/* Quick create dropdown */}
          {onQuickCreate && (
            <div className="relative">
              <button
                onClick={() => setShowModeMenu(!showModeMenu)}
                className="px-2 py-1.5 text-xs font-medium text-green-700 bg-green-50 rounded hover:bg-green-100 transition-colors flex items-center gap-1"
                title={t('discover.quickCreate')}
              >
                <Zap className="w-3 h-3" />
                <ChevronDown className="w-3 h-3" />
              </button>
              
              {showModeMenu && (
                <>
                  <div 
                    className="fixed inset-0 z-10" 
                    onClick={() => setShowModeMenu(false)}
                  />
                  <div className="absolute right-0 bottom-full mb-1 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-20">
                    {modeOptions.map(opt => (
                      <button
                        key={opt.mode}
                        onClick={() => {
                          onQuickCreate(video, opt.mode)
                          setShowModeMenu(false)
                        }}
                        className="w-full px-3 py-2 text-left hover:bg-gray-50 flex items-center gap-2"
                      >
                        <span className="text-gray-500">{opt.icon}</span>
                        <div>
                          <div className="text-xs font-medium text-gray-900">{opt.label}</div>
                          <div className="text-xs text-gray-500">{opt.desc}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
          
          <a
            href={video.video_url}
            target="_blank"
            rel="noopener noreferrer"
            className="px-2 py-1.5 text-gray-500 bg-gray-50 rounded hover:bg-gray-100 transition-colors"
            title={t('discover.openYouTube')}
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>
    </div>
  )
}

import { useState, useRef, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Image,
  Sparkles,
  Check,
  Loader2,
  AlertCircle,
  Edit3
} from 'lucide-react'
import { taskApi } from '../services/api'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8888'

interface ThumbnailSelectorProps {
  taskId: string
  onSelectionChange?: (selected: 'original' | 'ai_generated') => void
  isVertical?: boolean  // If true, use 9:16 aspect ratio for vertical videos
}

export default function ThumbnailSelector({ taskId, onSelectionChange, isVertical }: ThumbnailSelectorProps) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [customTitle, setCustomTitle] = useState('')
  const [showCustomTitleInput, setShowCustomTitleInput] = useState(false)
  const [imageVersion, setImageVersion] = useState(Date.now()) // Cache buster for images

  // Fetch thumbnail info
  const { data: thumbnailInfo, isLoading, error } = useQuery({
    queryKey: ['thumbnails', taskId],
    queryFn: () => taskApi.getThumbnails(taskId),
    refetchInterval: 5000, // Poll for updates
    staleTime: 2000,
  })

  // Generate AI thumbnail mutation
  const generateMutation = useMutation({
    mutationFn: (options?: { custom_title?: string; style?: string }) =>
      taskApi.generateAiThumbnail(taskId, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['thumbnails', taskId] })
      setImageVersion(Date.now()) // Force image refresh
      setShowCustomTitleInput(false)
      setCustomTitle('')
    },
  })

  // Select thumbnail mutation
  const selectMutation = useMutation({
    mutationFn: (selected: 'original' | 'ai_generated') =>
      taskApi.selectThumbnail(taskId, selected),
    onSuccess: (_, selected) => {
      queryClient.invalidateQueries({ queryKey: ['thumbnails', taskId] })
      onSelectionChange?.(selected)
    },
  })

  const handleSelect = (type: 'original' | 'ai_generated') => {
    if (thumbnailInfo?.selected === type) return
    selectMutation.mutate(type)
  }

  const handleGenerate = () => {
    const options = customTitle ? { custom_title: customTitle } : undefined
    generateMutation.mutate(options)
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-4">
        <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4 text-sm text-red-600 flex items-center gap-2">
        <AlertCircle className="w-4 h-4" />
        {t('thumbnail.loadError', 'Failed to load thumbnails')}
      </div>
    )
  }

  if (!thumbnailInfo?.original.exists) {
    return (
      <div className="p-4 text-sm text-gray-500 flex items-center gap-2">
        <Image className="w-4 h-4" />
        {t('thumbnail.noOriginal', 'No thumbnail available')}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-700 flex items-center gap-2">
          <Image className="w-4 h-4" />
          {t('thumbnail.title', 'Thumbnail Selection')}
        </h3>

        {/* Generate/Regenerate button */}
        <button
          onClick={() => {
            if (showCustomTitleInput) {
              handleGenerate()
            } else {
              generateMutation.mutate(undefined)
            }
          }}
          disabled={generateMutation.isPending}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-blue-600 hover:bg-blue-50 rounded-md transition-colors disabled:opacity-50"
        >
          {generateMutation.isPending ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              {t('thumbnail.generating', 'Generating...')}
            </>
          ) : (
            <>
              <Sparkles className="w-3.5 h-3.5" />
              {thumbnailInfo.ai_generated.exists
                ? t('thumbnail.regenerate', 'Regenerate AI')
                : t('thumbnail.generate', 'Generate AI Thumbnail')}
            </>
          )}
        </button>
      </div>

      {/* Custom title input (collapsible) */}
      {showCustomTitleInput && (
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={customTitle}
            onChange={(e) => setCustomTitle(e.target.value)}
            placeholder={t('thumbnail.customTitlePlaceholder', 'Enter custom title (optional)')}
            className="flex-1 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            maxLength={15}
          />
          <button
            onClick={handleGenerate}
            disabled={generateMutation.isPending}
            className="px-3 py-1.5 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50"
          >
            {t('common.confirm', 'Confirm')}
          </button>
          <button
            onClick={() => {
              setShowCustomTitleInput(false)
              setCustomTitle('')
            }}
            className="px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-100 rounded-md"
          >
            {t('common.cancel', 'Cancel')}
          </button>
        </div>
      )}

      {/* Thumbnails grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Original thumbnail */}
        <ThumbnailCard
          label={t('thumbnail.original', 'Original')}
          imageUrl={thumbnailInfo.original.url}
          isSelected={thumbnailInfo.selected === 'original'}
          onClick={() => handleSelect('original')}
          isLoading={selectMutation.isPending && selectMutation.variables === 'original'}
          cacheVersion={imageVersion}
          isVertical={isVertical}
        />

        {/* AI thumbnail */}
        <ThumbnailCard
          label={t('thumbnail.aiGenerated', 'AI Generated')}
          imageUrl={thumbnailInfo.ai_generated.url}
          aiTitle={thumbnailInfo.ai_generated.title}
          isSelected={thumbnailInfo.selected === 'ai_generated'}
          onClick={() => handleSelect('ai_generated')}
          isLoading={selectMutation.isPending && selectMutation.variables === 'ai_generated'}
          isEmpty={!thumbnailInfo.ai_generated.exists}
          onCustomize={() => setShowCustomTitleInput(true)}
          cacheVersion={imageVersion}
          isVertical={isVertical}
        />
      </div>

      {/* Generation error */}
      {generateMutation.isError && (
        <div className="p-2 text-xs text-red-600 bg-red-50 rounded-md flex items-center gap-2">
          <AlertCircle className="w-3.5 h-3.5" />
          {t('thumbnail.generateError', 'Failed to generate AI thumbnail')}
        </div>
      )}
    </div>
  )
}

// Individual thumbnail card component
interface ThumbnailCardProps {
  label: string
  imageUrl: string | null
  aiTitle?: string | null
  isSelected: boolean
  onClick: () => void
  isLoading?: boolean
  isEmpty?: boolean
  onCustomize?: () => void
  cacheVersion?: number
  isVertical?: boolean
}

function ThumbnailCard({
  label,
  imageUrl,
  aiTitle,
  isSelected,
  onClick,
  isLoading,
  isEmpty,
  onCustomize,
  cacheVersion,
  isVertical
}: ThumbnailCardProps) {
  const { t } = useTranslation()
  // Auto-detect vertical from image dimensions
  const [detectedVertical, setDetectedVertical] = useState<boolean | null>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const effectiveIsVertical = isVertical ?? detectedVertical ?? false

  const fullImageUrl = imageUrl ? `${API_BASE}${imageUrl}${imageUrl.includes('?') ? '&' : '?'}v=${cacheVersion || ''}` : null

  // Check image dimensions on mount and when imageUrl changes (handles cached images)
  useEffect(() => {
    const checkDimensions = () => {
      const img = imgRef.current
      if (img && img.naturalWidth && img.naturalHeight) {
        const aspectRatio = img.naturalWidth / img.naturalHeight
        setDetectedVertical(aspectRatio < 0.9)
      }
    }
    checkDimensions()
    const timer = setTimeout(checkDimensions, 100)
    return () => clearTimeout(timer)
  }, [fullImageUrl])

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget
    if (img.naturalWidth && img.naturalHeight) {
      const aspectRatio = img.naturalWidth / img.naturalHeight
      setDetectedVertical(aspectRatio < 0.9)
    }
  }

  return (
    <div
      onClick={!isEmpty ? onClick : undefined}
      className={`relative rounded-lg border-2 overflow-hidden transition-all cursor-pointer ${
        isSelected
          ? 'border-blue-500 ring-2 ring-blue-200'
          : isEmpty
          ? 'border-gray-200 border-dashed cursor-default'
          : 'border-gray-200 hover:border-gray-300'
      }`}
    >
      {/* Image area - adapt for vertical videos */}
      <div className={`bg-gray-100 relative ${effectiveIsVertical ? 'aspect-[9/16] max-h-[200px] mx-auto' : 'aspect-video'}`}>
        {fullImageUrl ? (
          <img
            ref={imgRef}
            src={fullImageUrl}
            alt={label}
            className="w-full h-full object-cover"
            onLoad={handleImageLoad}
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-400">
            <Sparkles className="w-8 h-8 mb-2" />
            <span className="text-xs">{t('thumbnail.notGenerated', 'Not generated yet')}</span>
          </div>
        )}

        {/* Selection indicator */}
        {isSelected && (
          <div className="absolute top-2 right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
            <Check className="w-4 h-4 text-white" />
          </div>
        )}

        {/* Loading overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-white/70 flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
          </div>
        )}
      </div>

      {/* Label and info */}
      <div className="p-2 bg-white border-t border-gray-100">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-700">{label}</span>
          {onCustomize && !isEmpty && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onCustomize()
              }}
              className="p-1 text-gray-400 hover:text-gray-600 rounded"
              title={t('thumbnail.customize', 'Customize title')}
            >
              <Edit3 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
        {aiTitle && (
          <p className="text-xs text-gray-500 mt-0.5 truncate" title={aiTitle}>
            "{aiTitle}"
          </p>
        )}
      </div>
    </div>
  )
}

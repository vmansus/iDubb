import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { RefreshCw, Loader2, CheckSquare, Square, AlertCircle, TrendingUp, Search, X } from 'lucide-react'
import { trendingApi, tiktokApi, taskApi } from '../services/api'
import TrendingVideoCard from '../components/TrendingVideoCard'
import type { TrendingVideo, YouTubeSearchResult, CreateTaskRequest } from '../types'

// Platform icons
const YouTubeIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
    <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
  </svg>
)

const TikTokIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
    <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5 20.1a6.34 6.34 0 0 0 10.86-4.43v-7a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1-.1z"/>
  </svg>
)

type Platform = 'youtube' | 'tiktok'

export default function Discover() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // Platform state
  const [platform, setPlatform] = useState<Platform>('youtube')

  // State
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedTag, setSelectedTag] = useState<string | null>(null)
  const [selectedVideoIds, setSelectedVideoIds] = useState<Set<string>>(new Set())
  const [isCreatingBatch, setIsCreatingBatch] = useState(false)

  // Search state (YouTube only)
  const [searchQuery, setSearchQuery] = useState('')
  const [isSearchMode, setIsSearchMode] = useState(false)
  const [searchResults, setSearchResults] = useState<YouTubeSearchResult[]>([])
  const [searchError, setSearchError] = useState<string | null>(null)

  // YouTube Queries
  const { data: categoriesData, isLoading: categoriesLoading } = useQuery({
    queryKey: ['trending-categories'],
    queryFn: trendingApi.getCategories,
    enabled: platform === 'youtube',
  })

  const { data: videosData, isLoading: videosLoading, error: videosError } = useQuery({
    queryKey: ['trending-videos', selectedCategory],
    queryFn: () => trendingApi.getVideos(selectedCategory || undefined),
    enabled: platform === 'youtube',
  })

  const { data: settingsData } = useQuery({
    queryKey: ['trending-settings'],
    queryFn: trendingApi.getSettings,
    enabled: platform === 'youtube',
  })

  // TikTok Queries
  const { data: tiktokTagsData, isLoading: tiktokTagsLoading } = useQuery({
    queryKey: ['tiktok-tags'],
    queryFn: tiktokApi.getTags,
    enabled: platform === 'tiktok',
  })

  const { data: tiktokVideosData, isLoading: tiktokVideosLoading, error: tiktokVideosError } = useQuery({
    queryKey: ['tiktok-videos', selectedTag],
    queryFn: () => tiktokApi.getVideos(selectedTag || undefined),
    enabled: platform === 'tiktok',
  })

  const { data: tiktokSettingsData } = useQuery({
    queryKey: ['tiktok-settings'],
    queryFn: tiktokApi.getSettings,
    enabled: platform === 'tiktok',
  })

  // YouTube Mutations
  const refreshYouTubeMutation = useMutation({
    mutationFn: (category?: string | null) => {
      if (category) {
        return trendingApi.refreshCategory(category)
      }
      return trendingApi.refresh()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trending-videos'] })
      queryClient.invalidateQueries({ queryKey: ['trending-settings'] })
    },
  })

  // TikTok Mutations
  const refreshTikTokMutation = useMutation({
    mutationFn: (tag?: string | null) => {
      if (tag) {
        return tiktokApi.refreshTag(tag)
      }
      return tiktokApi.refresh()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tiktok-videos'] })
      queryClient.invalidateQueries({ queryKey: ['tiktok-settings'] })
    },
  })

  // YouTube Search mutation
  const searchMutation = useMutation({
    mutationFn: (query: string) => trendingApi.searchYouTube(query, 20),
    onSuccess: (data) => {
      if (data.success) {
        setSearchResults(data.results)
        setSearchError(null)
        setIsSearchMode(true)
      } else {
        setSearchError(data.error || t('discover.searchError'))
        setSearchResults([])
      }
    },
    onError: (error: Error) => {
      setSearchError(error.message)
      setSearchResults([])
    },
  })

  // Filter categories by enabled ones in settings
  const enabledCategories = useMemo(() => {
    if (!categoriesData?.categories || !settingsData) return categoriesData?.categories || []
    return categoriesData.categories.filter(cat =>
      settingsData.enabled_categories.includes(cat.id)
    )
  }, [categoriesData, settingsData])

  // Filter TikTok tags by enabled ones
  const enabledTags = useMemo(() => {
    if (!tiktokTagsData?.tags || !tiktokSettingsData) return tiktokTagsData?.tags || []
    return tiktokTagsData.tags.filter((tag: { id: string; enabled: boolean }) =>
      tiktokSettingsData.enabled_tags?.includes(tag.id)
    )
  }, [tiktokTagsData, tiktokSettingsData])

  // Current videos based on platform
  const currentVideos = platform === 'youtube' ? videosData?.videos : tiktokVideosData?.videos
  const currentLoading = platform === 'youtube' ? videosLoading : tiktokVideosLoading
  const currentError = platform === 'youtube' ? videosError : tiktokVideosError
  const refreshMutation = platform === 'youtube' ? refreshYouTubeMutation : refreshTikTokMutation

  // Handlers
  const handleSelectVideo = (videoId: string, selected: boolean) => {
    const newSelected = new Set(selectedVideoIds)
    if (selected) {
      newSelected.add(videoId)
    } else {
      newSelected.delete(videoId)
    }
    setSelectedVideoIds(newSelected)
  }

  const handleSelectAll = () => {
    if (!currentVideos) return
    if (selectedVideoIds.size === currentVideos.length) {
      setSelectedVideoIds(new Set())
    } else {
      setSelectedVideoIds(new Set(currentVideos.map((v: TrendingVideo) => v.video_id)))
    }
  }

  // Search handlers
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      searchMutation.mutate(searchQuery.trim())
    }
  }

  const handleClearSearch = () => {
    setIsSearchMode(false)
    setSearchResults([])
    setSearchQuery('')
    setSearchError(null)
  }

  // Platform switch handler
  const handlePlatformChange = (newPlatform: Platform) => {
    setPlatform(newPlatform)
    setSelectedCategory(null)
    setSelectedTag(null)
    setSelectedVideoIds(new Set())
    setIsSearchMode(false)
    setSearchResults([])
    setSearchQuery('')
  }

  // Handle refresh based on platform
  const handleRefresh = () => {
    if (platform === 'youtube') {
      refreshYouTubeMutation.mutate(selectedCategory)
    } else {
      refreshTikTokMutation.mutate(selectedTag)
    }
  }

  // Batch create handler
  const handleBatchCreate = async () => {
    if (selectedVideoIds.size === 0) return

    setIsCreatingBatch(true)
    try {
      const result = await trendingApi.batchCreateTasks(Array.from(selectedVideoIds))
      if (result.success || result.created_count > 0) {
        setSelectedVideoIds(new Set())
        navigate('/tasks')
      }
    } catch (error) {
      console.error('Failed to create batch tasks:', error)
    } finally {
      setIsCreatingBatch(false)
    }
  }

  // Quick create handler - create task with specific processing mode
  const handleQuickCreate = async (video: TrendingVideo, mode: 'full' | 'subtitle' | 'direct' | 'auto') => {
    try {
      const taskData: CreateTaskRequest = {
        source_url: video.video_url,
        source_platform: video.platform || 'youtube',
        video_quality: '1080p',
        source_language: 'auto',
        target_language: 'zh-CN',
        processing_mode: mode,
        whisper_backend: 'faster',
        whisper_model: 'faster:small',
        whisper_device: 'auto',
        add_subtitles: mode !== 'direct',
        dual_subtitles: mode !== 'direct',
        use_existing_subtitles: true,
        add_tts: mode === 'full',
        tts_service: 'edge',
        tts_voice: 'zh-CN-XiaoxiaoNeural',
        replace_original_audio: false,
        original_audio_volume: 0.3,
        tts_audio_volume: 1.0,
        translation_engine: 'google',
        upload_bilibili: true,
        upload_douyin: true,
        upload_xiaohongshu: true,
        custom_tags: [],
        use_global_settings: false,
      }
      
      await taskApi.create(taskData)
      navigate('/tasks')
    } catch (error) {
      console.error('Failed to create quick task:', error)
    }
  }

  // Format last updated time
  const formatLastUpdated = (timestamp: string | undefined) => {
    if (!timestamp) return t('discover.neverUpdated')
    const date = new Date(timestamp)
    return date.toLocaleString()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{t('discover.title')}</h1>
          <p className="text-sm text-gray-500 mt-1">
            {t('discover.description')}
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshMutation.isPending}
          className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          title={platform === 'youtube' 
            ? (selectedCategory ? `刷新 ${selectedCategory} 类别` : '刷新全部类别')
            : (selectedTag ? `刷新 #${selectedTag}` : '刷新全部标签')
          }
        >
          {refreshMutation.isPending ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4" />
          )}
          {t('common.refresh')}
        </button>
      </div>

      {/* Platform Tabs */}
      <div className="flex gap-2 border-b border-gray-200">
        <button
          onClick={() => handlePlatformChange('youtube')}
          className={`flex items-center gap-2 px-4 py-2 border-b-2 transition-colors ${
            platform === 'youtube'
              ? 'border-red-500 text-red-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <YouTubeIcon />
          YouTube
        </button>
        <button
          onClick={() => handlePlatformChange('tiktok')}
          className={`flex items-center gap-2 px-4 py-2 border-b-2 transition-colors ${
            platform === 'tiktok'
              ? 'border-black text-black'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <TikTokIcon />
          TikTok
        </button>
      </div>

      {/* YouTube Search Box */}
      {platform === 'youtube' && (
        <form onSubmit={handleSearch} className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={t('discover.searchPlaceholder')}
              className="w-full pl-10 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={handleClearSearch}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          <button
            type="submit"
            disabled={searchMutation.isPending || !searchQuery.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {searchMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            {t('common.search')}
          </button>
        </form>
      )}

      {/* Search Error */}
      {searchError && (
        <div className="bg-red-50 text-red-700 rounded-lg p-4 flex items-center gap-2">
          <AlertCircle className="w-5 h-5" />
          {searchError}
          <button onClick={handleClearSearch} className="ml-auto text-red-500 hover:text-red-700">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Category/Tag Tabs */}
      {!isSearchMode && (
        <div className="flex flex-wrap gap-2">
          {platform === 'youtube' ? (
            // YouTube Categories
            <>
              <button
                onClick={() => setSelectedCategory(null)}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  selectedCategory === null
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {t('common.all')}
              </button>
              {categoriesLoading ? (
                <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
              ) : (
                enabledCategories.map((cat: { id: string; name: string }) => (
                  <button
                    key={cat.id}
                    onClick={() => setSelectedCategory(cat.id)}
                    className={`px-4 py-2 rounded-lg transition-colors ${
                      selectedCategory === cat.id
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {cat.name}
                  </button>
                ))
              )}
            </>
          ) : (
            // TikTok Tags
            <>
              <button
                onClick={() => setSelectedTag(null)}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  selectedTag === null
                    ? 'bg-black text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {t('common.all')}
              </button>
              {tiktokTagsLoading ? (
                <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
              ) : (
                enabledTags.map((tag: { id: string; name: string }) => (
                  <button
                    key={tag.id}
                    onClick={() => setSelectedTag(tag.id)}
                    className={`px-4 py-2 rounded-lg transition-colors ${
                      selectedTag === tag.id
                        ? 'bg-black text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {tag.name}
                  </button>
                ))
              )}
            </>
          )}
        </div>
      )}

      {/* Search Results Mode */}
      {isSearchMode && searchResults.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium text-gray-900">
              {t('discover.searchResults', { count: searchResults.length })}
            </h2>
            <button
              onClick={handleClearSearch}
              className="text-sm text-blue-600 hover:text-blue-700"
            >
              {t('discover.backToTrending')}
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {searchResults.map((video, index) => (
              <TrendingVideoCard
                key={video.video_id}
                video={{
                  ...video,
                  id: index,
                  category: 'search',
                  platform: 'youtube',
                  fetched_at: new Date().toISOString(),
                }}
                selected={selectedVideoIds.has(video.video_id)}
                onSelect={(_videoId: string, selected: boolean) => handleSelectVideo(video.video_id, selected)}
                onCreateTask={() => navigate(`/tasks/new?url=${encodeURIComponent(video.video_url)}`)}
                onQuickCreate={(v, mode) => handleQuickCreate({ ...v, video_url: video.video_url }, mode)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Selection Actions */}
      {selectedVideoIds.size > 0 && (
        <div className="sticky top-4 z-10 bg-white rounded-lg shadow-lg border border-gray-200 p-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={handleSelectAll}
              className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
            >
              {currentVideos && selectedVideoIds.size === currentVideos.length ? (
                <CheckSquare className="w-4 h-4" />
              ) : (
                <Square className="w-4 h-4" />
              )}
              {t('common.selectAll')}
            </button>
            <span className="text-sm text-gray-500">
              {t('discover.selectedCount', { count: selectedVideoIds.size })}
            </span>
          </div>
          <button
            onClick={handleBatchCreate}
            disabled={isCreatingBatch}
            className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
          >
            {isCreatingBatch ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <TrendingUp className="w-4 h-4" />
            )}
            {t('discover.createTasks')}
          </button>
        </div>
      )}

      {/* Videos Grid */}
      {!isSearchMode && (
        currentLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
          </div>
        ) : currentError ? (
          <div className="bg-red-50 text-red-700 rounded-lg p-4 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            {t('discover.loadError')}
          </div>
        ) : currentVideos?.length === 0 ? (
          <div className="text-center py-12">
            <TrendingUp className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              {t('discover.noVideos')}
            </h3>
            <p className="text-gray-500 mb-4">{t('discover.noVideosDesc')}</p>
            <button
              onClick={handleRefresh}
              disabled={refreshMutation.isPending}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {refreshMutation.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              {t('discover.fetchNow')}
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {currentVideos?.map((video: TrendingVideo) => (
              <TrendingVideoCard
                key={video.video_id}
                video={video}
                selected={selectedVideoIds.has(video.video_id)}
                onSelect={(_videoId: string, selected: boolean) => handleSelectVideo(video.video_id, selected)}
                onCreateTask={() => navigate(`/tasks/new?url=${encodeURIComponent(video.video_url)}`)}
                onQuickCreate={(v, mode) => handleQuickCreate(v, mode)}
              />
            ))}
          </div>
        )
      )}

      {/* Last Updated Info */}
      {!isSearchMode && (
        <div className="text-sm text-gray-500 text-center">
          {platform === 'youtube' 
            ? t('discover.lastUpdated', { time: formatLastUpdated(settingsData?.last_updated) })
            : t('discover.lastUpdated', { time: formatLastUpdated(tiktokSettingsData?.last_updated) })
          }
        </div>
      )}
    </div>
  )
}

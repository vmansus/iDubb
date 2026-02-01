import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Search, Loader2, ChevronDown, ChevronUp, Info, Upload, Link, X, File, Sparkles, Tag, FolderOpen, Plus, AlertCircle } from 'lucide-react'
import { taskApi, videoApi, configApi, settingsApi, translationApi, ttsApi, transcriptionApi, presetsApi, uploadApi, metadataPresetsApi, directoryApi, bilibiliApi } from '../services/api'
import type { CreateTaskRequest, VideoInfo, DetailedVideoInfo, TranslationEngine, TTSEngine, Voice, UploadResponse, MetadataPreset, Directory } from '../types'
import type { WhisperModel, TranscriptionEstimate, LocalVideoInfo } from '../services/api'

type SourceType = 'url' | 'local'

export default function NewTask() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [sourceType, setSourceType] = useState<SourceType>('url')
  const [url, setUrl] = useState('')
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null)
  const [detailedInfo, setDetailedInfo] = useState<DetailedVideoInfo | null>(null)
  const [detailedInfoLoading, setDetailedInfoLoading] = useState(false)
  const [detailedInfoError, setDetailedInfoError] = useState<string | null>(null)
  const [fetchingInfo, setFetchingInfo] = useState(false)
  const [videoFetchError, setVideoFetchError] = useState<string | null>(null)
  const [cookiesRefreshed, setCookiesRefreshed] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Local file upload state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState<number>(0)
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [localVideoInfo, setLocalVideoInfo] = useState<LocalVideoInfo | null>(null)
  const [fetchingLocalInfo, setFetchingLocalInfo] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Directory state
  const [directoryMode, setDirectoryMode] = useState<'none' | 'existing' | 'new'>('none')
  const [selectedDirectory, setSelectedDirectory] = useState('')
  const [newDirectoryName, setNewDirectoryName] = useState('')
  const [directories, setDirectories] = useState<Directory[]>([])
  const [directoryError, setDirectoryError] = useState<string | null>(null)
  const [loadingDirectories, setLoadingDirectories] = useState(false)

  const [options, setOptions] = useState<Partial<CreateTaskRequest>>({
    source_platform: 'auto',
    video_quality: '1080p',
    source_language: 'en',
    target_language: 'zh-CN',
    processing_mode: 'full',  // full | subtitle | direct | auto
    skip_translation: false,  // deprecated, use processing_mode instead
    whisper_backend: 'faster',
    whisper_model: 'faster:small',  // Default to small for balance of speed/quality
    whisper_device: 'auto',  // auto, cpu, cuda, mps
    use_ocr: false,  // OCR mode for text overlays
    ocr_engine: 'paddleocr',
    ocr_frame_interval: 0.5,
    add_subtitles: true,
    dual_subtitles: true,
    use_existing_subtitles: true,
    add_tts: true,
    tts_service: 'edge',
    tts_voice: 'zh-CN-XiaoxiaoNeural',
    voice_cloning_mode: 'disabled',
    replace_original_audio: false,
    original_audio_volume: 0.3,
    tts_audio_volume: 1.0,
    translation_engine: 'google',
    upload_bilibili: true,
    upload_douyin: true,
    upload_xiaohongshu: true,
    bilibili_account_uid: '',  // Empty = default account
    custom_tags: [],
    use_global_settings: true,
  })

  // Fetch global settings
  const { data: globalSettings } = useQuery({
    queryKey: ['globalSettings'],
    queryFn: settingsApi.get,
  })

  // Pre-fill URL from search params (e.g., from trending videos)
  useEffect(() => {
    const urlParam = searchParams.get('url')
    if (urlParam) {
      setUrl(urlParam)
      setSourceType('url')
    }
  }, [searchParams])

  // Apply global settings when loaded
  useEffect(() => {
    if (globalSettings && options.use_global_settings) {
      const globalSubtitlePreset = globalSettings.subtitle.default_preset
      setOptions((prev) => ({
        ...prev,
        video_quality: globalSettings.video.default_quality,
        use_existing_subtitles: globalSettings.video.prefer_existing_subtitles,
        add_subtitles: globalSettings.subtitle.enabled,
        dual_subtitles: globalSettings.subtitle.dual_subtitles,
        // Only override subtitle_preset if global setting has a value
        ...(globalSubtitlePreset ? { subtitle_preset: globalSubtitlePreset } : {}),
        add_tts: globalSettings.audio.generate_tts,
        replace_original_audio: globalSettings.audio.replace_original,
        original_audio_volume: globalSettings.audio.original_volume,
        tts_audio_volume: globalSettings.audio.tts_volume,
        tts_service: globalSettings.tts.engine,
        tts_voice: globalSettings.tts.voice,
        voice_cloning_mode: globalSettings.tts.voice_cloning_mode || 'disabled',
        tts_ref_audio: globalSettings.tts.ref_audio_path,
        tts_ref_text: globalSettings.tts.ref_audio_text,
        translation_engine: globalSettings.translation.engine,
        upload_bilibili: globalSettings.auto_upload_bilibili,
        upload_douyin: globalSettings.auto_upload_douyin,
        upload_xiaohongshu: globalSettings.auto_upload_xiaohongshu,
        use_ai_preset_selection: globalSettings.metadata?.default_use_ai_preset_selection ?? false,
      }))
    }
  }, [globalSettings, options.use_global_settings])

  const { data: languages } = useQuery({
    queryKey: ['languages'],
    queryFn: configApi.getLanguages,
  })

  // Fetch TTS engines
  const { data: ttsEngines } = useQuery({
    queryKey: ['ttsEngines'],
    queryFn: ttsApi.getEngines,
  })

  // Fetch voices for selected TTS engine
  const { data: voices } = useQuery({
    queryKey: ['voices', options.tts_service || 'edge', 'zh'],
    queryFn: () => ttsApi.getVoicesByEngine(options.tts_service || 'edge', 'zh'),
    enabled: !!options.tts_service,
  })

  // Check TTS engine health for local servers
  const { data: ttsHealth } = useQuery({
    queryKey: ['ttsHealth', options.tts_service],
    queryFn: () => ttsApi.checkHealth(options.tts_service || 'edge'),
    enabled: !!options.tts_service && options.tts_service !== 'edge',
  })

  const { data: platformStatus } = useQuery({
    queryKey: ['platformStatus'],
    queryFn: configApi.getPlatformStatus,
  })

  // Fetch Bilibili accounts for upload selection
  const { data: bilibiliAccountsData } = useQuery({
    queryKey: ['bilibiliAccounts'],
    queryFn: bilibiliApi.listAccounts,
  })
  const bilibiliAccounts = bilibiliAccountsData?.accounts || []

  const { data: translationEngines } = useQuery({
    queryKey: ['translationEngines'],
    queryFn: translationApi.getEngines,
  })

  // Fetch transcription models (all backends)
  const { data: transcriptionModels } = useQuery({
    queryKey: ['transcriptionModels'],
    queryFn: () => transcriptionApi.getModels(),  // Show all models (faster-whisper and OpenAI)
  })

  // Fetch subtitle presets
  const { data: presetsData } = useQuery({
    queryKey: ['presets'],
    queryFn: presetsApi.getAll,
  })

  // Fetch metadata presets (title prefix and signature combinations)
  const { data: metadataPresetsData } = useQuery({
    queryKey: ['metadataPresets'],
    queryFn: metadataPresetsApi.getAll,
  })

  // AI preset selection mutation
  const aiSelectPresetMutation = useMutation({
    mutationFn: metadataPresetsApi.aiSelect,
    onSuccess: (result) => {
      if (result.success && result.preset_id) {
        setOptions((prev) => ({
          ...prev,
          metadata_preset_id: result.preset_id,
        }))
      }
    },
  })

  // Set default subtitle_preset when presets load and no preset is selected
  useEffect(() => {
    if (presetsData?.presets?.length && !options.subtitle_preset) {
      setOptions((prev) => ({
        ...prev,
        subtitle_preset: presetsData.presets[0].id,
      }))
    }
  }, [presetsData, options.subtitle_preset])

  // Set default metadata_preset_id when metadata presets load
  useEffect(() => {
    if (metadataPresetsData?.presets?.length && !options.metadata_preset_id) {
      // Find the default preset or use the first one
      const defaultPreset = metadataPresetsData.presets.find(p => p.is_default) || metadataPresetsData.presets[0]
      setOptions((prev) => ({
        ...prev,
        metadata_preset_id: defaultPreset.id,
      }))
    }
  }, [metadataPresetsData, options.metadata_preset_id])

  // Load directories on mount
  useEffect(() => {
    const loadDirectories = async () => {
      setLoadingDirectories(true)
      try {
        const response = await directoryApi.list()
        setDirectories(response.directories)
      } catch (err) {
        console.error('Failed to load directories:', err)
      } finally {
        setLoadingDirectories(false)
      }
    }
    loadDirectories()
  }, [])

  // Fetch transcription time estimates when video info is available
  // Get video duration from either URL video or local video
  const videoDuration = videoInfo?.duration || localVideoInfo?.duration || 0

  const { data: transcriptionEstimates } = useQuery({
    queryKey: ['transcriptionEstimates', videoDuration],
    queryFn: () => transcriptionApi.getAllEstimates(videoDuration),  // All backends
    enabled: videoDuration > 0,
  })

  const createTaskMutation = useMutation({
    mutationFn: taskApi.create,
    onSuccess: (task) => {
      navigate(`/task/${task.task_id}`)
    },
  })

  const handleFetchInfo = async () => {
    if (!url) return
    setFetchingInfo(true)
    setDetailedInfo(null)
    setDetailedInfoError(null)
    setDetailedInfoLoading(true)
    setVideoFetchError(null)
    setCookiesRefreshed(false)
    try {
      // Fetch basic info first (required)
      const info = await videoApi.getInfo(url)
      setVideoInfo(info)
      setOptions((prev) => ({
        ...prev,
        custom_title: info.title,
        custom_tags: info.tags?.slice(0, 5) || [],
      }))

      // Try to fetch detailed info (optional, may fail for some videos)
      try {
        const detailed = await videoApi.getDetailedInfo(url)
        setDetailedInfo(detailed)
        setDetailedInfoError(null)
        // Check if cookies were auto-refreshed
        if (detailed.cookies_refreshed) {
          setCookiesRefreshed(true)
        }
        if (detailed.recommended_format) {
          // Find the recommended format's label
          const recommendedFormatInfo = detailed.formats?.find(
            (f) => f.format_id === detailed.recommended_format
          )
          const qualityLabel = recommendedFormatInfo
            ? `${recommendedFormatInfo.resolution} (${recommendedFormatInfo.ext})${recommendedFormatInfo.filesize_mb ? ` - ${recommendedFormatInfo.filesize_mb.toFixed(1)}MB` : ''}`
            : detailed.recommended_format
          setOptions((prev) => ({
            ...prev,
            format_id: detailed.recommended_format ?? undefined,
            video_quality_label: qualityLabel,
          }))
        }
      } catch (detailedError: unknown) {
        const errorMsg = detailedError instanceof Error ? detailedError.message : t('newTask.detailsUnavailable')
        console.warn('Could not fetch detailed info:', detailedError)
        setDetailedInfoError(errorMsg)
        // Continue without detailed info - basic info is enough
      }
    } catch (error) {
      console.error('Failed to fetch video info:', error)
      const errorMsg = error instanceof Error ? error.message : t('newTask.alertFetchFailed')
      setVideoFetchError(errorMsg)
    } finally {
      setFetchingInfo(false)
      setDetailedInfoLoading(false)
    }
  }

  // Handle file selection - automatically start upload
  const handleFileSelect = useCallback(async (file: File) => {
    // Validate file type
    const allowedTypes = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v']
    const ext = file.name.toLowerCase().substring(file.name.lastIndexOf('.'))
    if (!allowedTypes.includes(ext)) {
      setUploadError(t('newTask.uploadInvalidFormat', { formats: allowedTypes.join(', ') }))
      return
    }

    setUploadedFile(file)
    setUploadError(null)
    setUploadResult(null)
    setUploadProgress(0)
    setLocalVideoInfo(null)

    // Start upload immediately
    setUploading(true)
    try {
      const result = await uploadApi.uploadVideo(file, (progress) => {
        setUploadProgress(progress)
      })
      setUploadResult(result)
      setUploadProgress(100)

      // Fetch video info after successful upload
      setFetchingLocalInfo(true)
      try {
        const videoInfo = await uploadApi.getLocalVideoInfo(result.file_path)
        setLocalVideoInfo(videoInfo)

        // Set default quality and title based on video info
        if (videoInfo.available_qualities.length > 0) {
          setOptions((prev) => ({
            ...prev,
            video_quality: videoInfo.available_qualities[0],
            custom_title: videoInfo.title,
          }))
        }
      } catch (infoError) {
        console.error('Failed to get video info:', infoError)
      } finally {
        setFetchingLocalInfo(false)
      }
    } catch (error) {
      console.error('Upload failed:', error)
      setUploadError(error instanceof Error ? error.message : t('newTask.uploadFailed'))
    } finally {
      setUploading(false)
    }
  }, [t])

  // Handle drag and drop
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
  }, [])

  // Clear uploaded file
  const clearUpload = () => {
    setUploadedFile(null)
    setUploadResult(null)
    setUploadProgress(0)
    setUploadError(null)
    setLocalVideoInfo(null)
    setFetchingLocalInfo(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleSubmit = async () => {
    if (sourceType === 'url') {
      if (!url) {
        alert(t('newTask.alertEnterUrl'))
        return
      }
    } else {
      // Local file upload
      if (!uploadResult) {
        alert(t('newTask.alertUploadFirst'))
        return
      }
    }

    // Get directory name if specified
    let directory: string | undefined
    if (directoryMode === 'existing' && selectedDirectory) {
      directory = selectedDirectory
    } else if (directoryMode === 'new' && newDirectoryName.trim()) {
      // Create new directory first
      try {
        await directoryApi.create(newDirectoryName.trim())
        directory = newDirectoryName.trim()
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : t('newTask.createDirectoryFailed')
        setDirectoryError(errorMessage)
        return
      }
    }

    // Debug: Log the subtitle_preset being sent
    console.log('[SUBTITLE DEBUG] Submitting task with subtitle_preset:', options.subtitle_preset)

    if (sourceType === 'url') {
      createTaskMutation.mutate({
        source_url: url,
        source_platform: options.source_platform || 'auto',
        ...options,
        directory,
      } as CreateTaskRequest)
    } else {
      createTaskMutation.mutate({
        ...options,
        source_url: '',
        source_platform: 'local',
        local_file_path: uploadResult!.file_path,
        use_existing_subtitles: false,  // Local files don't have downloadable subtitles
        directory,
      } as CreateTaskRequest)
    }
  }

  // Get available qualities from detailed info
  const availableQualities = detailedInfo?.formats
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

  // Get available subtitles
  const availableSubtitles = detailedInfo?.subtitles || []

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-xl font-semibold text-gray-900 mb-6">{t('newTask.title')}</h2>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 space-y-6">
        {/* Source Type Toggle */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t('newTask.sourceType')}
          </label>
          <div className="flex rounded-lg border border-gray-300 overflow-hidden">
            <button
              type="button"
              onClick={() => setSourceType('url')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                sourceType === 'url'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              <Link className="h-4 w-4" />
              {t('newTask.sourceTypeUrl')}
            </button>
            <button
              type="button"
              onClick={() => setSourceType('local')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                sourceType === 'local'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              <Upload className="h-4 w-4" />
              {t('newTask.sourceTypeLocal')}
            </button>
          </div>
        </div>

        {/* Video URL Input */}
        {sourceType === 'url' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('newTask.sourceUrl')}
            </label>
            <div className="flex space-x-2">
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder={t('newTask.sourceUrlPlaceholder')}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={handleFetchInfo}
                disabled={fetchingInfo || !url}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 disabled:opacity-50"
              >
                {fetchingInfo ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Search className="h-5 w-5" />
                )}
              </button>
            </div>

            {/* Video fetch error message - inline instead of alert */}
            {videoFetchError && (
              <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                {videoFetchError}
              </div>
            )}

            {/* Cookies auto-refreshed success message */}
            {cookiesRefreshed && (
              <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg text-sm text-green-700">
                {t('newTask.cookiesAutoRefreshed')}
              </div>
            )}
          </div>
        )}

        {/* Local File Upload */}
        {sourceType === 'local' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('newTask.uploadVideo')}
            </label>

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.mkv,.avi,.mov,.webm,.flv,.wmv,.m4v"
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              className="hidden"
            />

            {!uploadedFile ? (
              /* Drop zone */
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors"
              >
                <Upload className="h-10 w-10 mx-auto text-gray-400 mb-3" />
                <p className="text-sm text-gray-600">{t('newTask.dropzoneText')}</p>
                <p className="text-xs text-gray-400 mt-1">{t('newTask.dropzoneFormats')}</p>
              </div>
            ) : (
              /* File selected */
              <div className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <File className="h-8 w-8 text-blue-500" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">{uploadedFile.name}</p>
                      <p className="text-xs text-gray-500">
                        {(uploadedFile.size / (1024 * 1024)).toFixed(1)} MB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={clearUpload}
                    className="p-1 text-gray-400 hover:text-gray-600"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>

                {/* Upload progress */}
                {uploading && (
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                      <span>{t('newTask.uploading')}</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Upload success - show video info */}
                {uploadResult && !uploading && (
                  <div className="mt-3">
                    {fetchingLocalInfo ? (
                      <div className="p-3 bg-blue-50 rounded flex items-center gap-2 text-sm text-blue-700">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        {t('newTask.analyzingVideo')}
                      </div>
                    ) : localVideoInfo ? (
                      <div className="p-3 bg-green-50 rounded">
                        <div className="text-sm text-green-700 font-medium mb-2">
                          {t('newTask.uploadSuccess')}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                          <div>
                            <span className="text-gray-500">{t('newTask.localVideoResolution')}:</span>{' '}
                            <span className="font-medium">{localVideoInfo.resolution_label}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">{t('newTask.localVideoDuration')}:</span>{' '}
                            <span className="font-medium">{localVideoInfo.duration_formatted}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">{t('newTask.localVideoCodec')}:</span>{' '}
                            <span className="font-medium">{localVideoInfo.codec.toUpperCase()}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">{t('newTask.localVideoSize')}:</span>{' '}
                            <span className="font-medium">{localVideoInfo.file_size_mb} MB</span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="p-2 bg-green-50 rounded text-sm text-green-700">
                        {t('newTask.uploadSuccess')}
                      </div>
                    )}
                  </div>
                )}

                {/* Upload error */}
                {uploadError && (
                  <div className="mt-3 p-2 bg-red-50 rounded text-sm text-red-700">
                    {uploadError}
                  </div>
                )}

              </div>
            )}
          </div>
        )}

        {/* Video Preview (URL mode) */}
        {sourceType === 'url' && videoInfo && (
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex space-x-4">
              <img
                src={videoInfo.thumbnail_url}
                alt={videoInfo.title}
                className="w-32 h-20 object-cover rounded"
              />
              <div className="flex-1 min-w-0">
                <h3 className="font-medium text-gray-900 truncate">
                  {videoInfo.title}
                </h3>
                <p className="text-sm text-gray-500">
                  {videoInfo.uploader} · {Math.floor(videoInfo.duration / 60)}{t('newTask.duration').toLowerCase()}
                </p>
                <span className="inline-block mt-1 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                  {videoInfo.platform}
                </span>
              </div>
            </div>

            {/* Available formats info */}
            {detailedInfoLoading && (
              <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span>{t('newTask.fetchingDetails')}</span>
                </div>
              </div>
            )}
            {detailedInfo && (
              <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
                <div className="flex flex-wrap gap-2">
                  <span>{t('newTask.qualityCount', { count: availableQualities.length })}</span>
                  <span>·</span>
                  <span>{t('newTask.audioTrackCount', { count: detailedInfo.audio_tracks?.length || 0 })}</span>
                  <span>·</span>
                  <span>{t('newTask.subtitleCount', { count: availableSubtitles.length })}</span>
                </div>
              </div>
            )}
            {detailedInfoError && !detailedInfo && (
              <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-yellow-600">
                <div className="flex items-center gap-1">
                  <Info className="h-3 w-3" />
                  <span>{t('newTask.detailsUnavailable')}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Video Quality Selection - URL mode with detailed info */}
        {sourceType === 'url' && detailedInfo && availableQualities.length > 0 && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('newTask.quality')}
            </label>
            <select
              value={options.format_id || ''}
              onChange={(e) => {
                const selectedQuality = availableQualities.find(q => q.value === e.target.value)
                setOptions({
                  ...options,
                  format_id: e.target.value,
                  video_quality_label: selectedQuality?.label || e.target.value,
                })
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              {availableQualities.map((q) => (
                <option key={q.value} value={q.value}>
                  {q.label}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Video Quality Selection - Local video with info */}
        {sourceType === 'local' && localVideoInfo && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('newTask.outputQuality')}
              <span className="text-xs text-gray-500 ml-2">
                ({t('newTask.maxQualityHint', { quality: localVideoInfo.resolution_label })})
              </span>
            </label>
            <select
              value={options.video_quality}
              onChange={(e) => setOptions({ ...options, video_quality: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              {localVideoInfo.available_qualities.map((q) => (
                <option key={q} value={q}>
                  {q === localVideoInfo.max_quality ? `${q} (${t('newTask.original')})` : q}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Default quality if no video info */}
        {((sourceType === 'url' && !detailedInfo) || (sourceType === 'local' && !localVideoInfo)) && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('newTask.quality')}
            </label>
            <select
              value={options.video_quality}
              onChange={(e) => setOptions({ ...options, video_quality: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="2160p">4K (2160p)</option>
              <option value="1080p">{t('newTask.qualityRecommended')}</option>
              <option value="720p">720p</option>
              <option value="480p">480p</option>
            </select>
          </div>
        )}

        {/* Processing Mode */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t('newTask.processingMode')}
          </label>
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setOptions({ ...options, processing_mode: 'full', skip_translation: false, add_tts: true })}
              className={`p-3 rounded-lg border-2 text-left transition-all ${
                options.processing_mode === 'full'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="font-medium text-gray-900">{t('newTask.modeFull')}</div>
              <div className="text-xs text-gray-500 mt-1">{t('newTask.modeFullDesc')}</div>
            </button>
            <button
              type="button"
              onClick={() => setOptions({ ...options, processing_mode: 'subtitle', skip_translation: false, add_tts: false, dual_subtitles: true })}
              className={`p-3 rounded-lg border-2 text-left transition-all ${
                options.processing_mode === 'subtitle'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="font-medium text-gray-900">{t('newTask.modeSubtitle')}</div>
              <div className="text-xs text-gray-500 mt-1">{t('newTask.modeSubtitleDesc')}</div>
            </button>
            <button
              type="button"
              onClick={() => setOptions({ ...options, processing_mode: 'direct', skip_translation: true, add_tts: false, add_subtitles: false })}
              className={`p-3 rounded-lg border-2 text-left transition-all ${
                options.processing_mode === 'direct'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="font-medium text-gray-900">{t('newTask.modeDirect')}</div>
              <div className="text-xs text-gray-500 mt-1">{t('newTask.modeDirectDesc')}</div>
            </button>
            <button
              type="button"
              onClick={() => setOptions({ ...options, processing_mode: 'auto', skip_translation: false, add_tts: false })}
              className={`p-3 rounded-lg border-2 text-left transition-all ${
                options.processing_mode === 'auto'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="font-medium text-gray-900">{t('newTask.modeAuto')}</div>
              <div className="text-xs text-gray-500 mt-1">{t('newTask.modeAutoDesc')}</div>
            </button>
          </div>
        </div>

        {/* Language Settings */}
        <div className={`grid gap-4 ${options.processing_mode === 'direct' ? 'grid-cols-1' : 'grid-cols-2'}`}>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('newTask.sourceLanguage')}
            </label>
            <select
              value={options.source_language}
              onChange={(e) => setOptions({ ...options, source_language: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              {languages?.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.name}
                </option>
              ))}
            </select>
          </div>
          {options.processing_mode !== 'direct' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {t('newTask.targetLanguage')}
              </label>
              <select
                value={options.target_language}
                onChange={(e) => setOptions({ ...options, target_language: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              >
                {languages?.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Translation Engine - only show when not in subtitles-only mode */}
        {options.processing_mode !== 'direct' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('newTask.translationEngine')}
            </label>
            <select
              value={options.translation_engine}
              onChange={(e) => setOptions({ ...options, translation_engine: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              {translationEngines?.map((engine: TranslationEngine) => (
                <option key={engine.id} value={engine.id}>
                  {engine.name} {engine.free ? `(${t('newTask.free')})` : ''}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* OCR Mode - for videos with text overlays instead of speech */}
        {options.processing_mode !== 'direct' && (
          <div className="p-3 bg-amber-50 rounded-md border border-amber-200">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.use_ocr || false}
                onChange={(e) => setOptions({ ...options, use_ocr: e.target.checked })}
                className="rounded border-gray-300 text-amber-600"
              />
              <span className="ml-2 text-sm font-medium text-amber-800">🔍 OCR 文字识别模式</span>
            </label>
            <p className="text-xs text-amber-600 mt-1 ml-6">
              适用于无人声、仅有画面文字的视频
            </p>
            {options.use_ocr && (
              <div className="mt-2 ml-6 flex gap-2">
                <select
                  value={options.ocr_engine || 'paddleocr'}
                  onChange={(e) => setOptions({ ...options, ocr_engine: e.target.value })}
                  className="px-2 py-1 border border-amber-300 rounded text-sm"
                >
                  <option value="paddleocr">🆓 PaddleOCR (免费)</option>
                  <option value="openai">OpenAI GPT-4o</option>
                  <option value="anthropic">Anthropic Claude</option>
                </select>
                <input
                  type="number"
                  value={options.ocr_frame_interval || 0.5}
                  onChange={(e) => setOptions({ ...options, ocr_frame_interval: parseFloat(e.target.value) || 0.5 })}
                  min="0.25" max="2" step="0.25"
                  className="w-16 px-2 py-1 border border-amber-300 rounded text-sm"
                  title="采样间隔(秒)"
                />
              </div>
            )}
          </div>
        )}

        {/* Transcription Model Selection - hide when OCR enabled */}
        {!options.use_ocr && (
        <>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t('newTask.transcriptionModel')}
          </label>
          <select
            value={options.whisper_model}
            onChange={(e) => setOptions({ ...options, whisper_model: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          >
            {/* Faster-Whisper models (recommended) */}
            <optgroup label={`⚡ ${t('newTask.fasterWhisper')}`}>
              {transcriptionModels?.models
                .filter((m: WhisperModel) => m.backend === 'faster')
                .map((model: WhisperModel) => {
                  const estimate = transcriptionEstimates?.estimates.find(
                    (e: TranscriptionEstimate) => e.model_id === model.id
                  )
                  return (
                    <option key={model.id} value={model.id}>
                      {model.name.replace('⚡ Faster-Whisper ', '')}
                      {estimate ? ` - ${t('newTask.estimated', { time: estimate.estimated_time_cpu_formatted })}` : ''}
                    </option>
                  )
                })}
            </optgroup>
            {/* WhisperX models (word-level alignment) */}
            <optgroup label={`🎯 ${t('newTask.whisperX')}`}>
              {transcriptionModels?.models
                .filter((m: WhisperModel) => m.backend === 'whisperx')
                .map((model: WhisperModel) => {
                  const estimate = transcriptionEstimates?.estimates.find(
                    (e: TranscriptionEstimate) => e.model_id === model.id
                  )
                  return (
                    <option key={model.id} value={model.id}>
                      {model.name.replace('🎯 WhisperX ', '')}
                      {estimate ? ` - ${t('newTask.estimated', { time: estimate.estimated_time_cpu_formatted })}` : ''}
                    </option>
                  )
                })}
            </optgroup>
            {/* OpenAI Whisper models */}
            <optgroup label={`🐢 ${t('newTask.openaiWhisper')}`}>
              {transcriptionModels?.models
                .filter((m: WhisperModel) => m.backend === 'openai')
                .map((model: WhisperModel) => {
                  const estimate = transcriptionEstimates?.estimates.find(
                    (e: TranscriptionEstimate) => e.model_id === model.id
                  )
                  return (
                    <option key={model.id} value={model.id}>
                      {model.name.replace('🐢 OpenAI Whisper ', '')}
                      {estimate ? ` - ${t('newTask.estimated', { time: estimate.estimated_time_cpu_formatted })}` : ''}
                    </option>
                  )
                })}
            </optgroup>
          </select>
          {/* Show model description */}
          {options.whisper_model && transcriptionModels?.models && (
            <div className="mt-2 text-xs text-gray-500">
              {transcriptionModels.models.find((m: WhisperModel) => m.id === options.whisper_model)?.description}
            </div>
          )}
          {/* Show detailed estimate for selected model - works for both URL and local videos */}
          {videoDuration > 0 && transcriptionEstimates && (
            <div className="mt-2 p-3 bg-blue-50 rounded-md">
              <div className="text-sm text-blue-800">
                <span className="font-medium">{t('newTask.videoDuration')}: </span>
                {t('newTask.durationFormat', { minutes: Math.floor(videoDuration / 60), seconds: Math.floor(videoDuration % 60) })}
              </div>
              {transcriptionEstimates.estimates.find(
                (e: TranscriptionEstimate) => e.model_id === options.whisper_model
              ) && (
                <div className="text-sm text-blue-800 mt-1">
                  <span className="font-medium">{t('newTask.estimatedTranscription')}: </span>
                  {transcriptionEstimates.estimates.find(
                    (e: TranscriptionEstimate) => e.model_id === options.whisper_model
                  )?.estimated_time_cpu_formatted}
                  <span className="text-blue-600 ml-2">{t('newTask.cpuMode')}</span>
                </div>
              )}
              {options.whisper_model?.startsWith('openai:') && (
                <div className="text-xs text-purple-600 mt-1">
                  💡 {t('newTask.openaiWhisperNote')}
                </div>
              )}
              {options.whisper_model?.startsWith('faster:') && (
                <div className="text-xs text-green-600 mt-1">
                  ⚡ {t('newTask.fasterWhisperNote')}
                </div>
              )}
              {options.whisper_model?.startsWith('whisperx:') && (
                <div className="text-xs text-orange-600 mt-1">
                  🎯 {t('newTask.whisperXNote')}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Device Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t('newTask.computeDevice')}
          </label>
          <select
            value={options.whisper_device}
            onChange={(e) => setOptions({ ...options, whisper_device: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          >
            <option value="auto">{t('newTask.autoDetect')}</option>
            <option value="cpu">{t('newTask.cpu')}</option>
            <option value="cuda">{t('newTask.cuda')}</option>
            <option value="mps">{t('newTask.mps')}</option>
          </select>
          <div className="mt-1 text-xs text-gray-500">
            {options.whisper_device === 'auto' && t('newTask.autoDetectDesc')}
            {options.whisper_device === 'cpu' && t('newTask.cpuDesc')}
            {options.whisper_device === 'cuda' && t('newTask.cudaDesc')}
            {options.whisper_device === 'mps' && t('newTask.mpsDesc')}
          </div>
          {options.whisper_device === 'mps' && options.whisper_model?.startsWith('faster:') && (
            <div className="mt-1 text-xs text-orange-600">
              ⚠️ {t('newTask.mpsWarning')}
            </div>
          )}
        </div>
        </>
        )}

        {/* Subtitle Options */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t('newTask.subtitleSettings')}
          </label>
          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.add_subtitles}
                onChange={(e) => setOptions({ ...options, add_subtitles: e.target.checked })}
                className="rounded border-gray-300 text-blue-600"
              />
              <span className="ml-2 text-sm text-gray-700">{t('newTask.addSubtitles')}</span>
            </label>
            {/* Dual subtitles - only available when not in subtitles-only mode */}
            {options.processing_mode !== 'direct' && (
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={options.dual_subtitles}
                  onChange={(e) => setOptions({ ...options, dual_subtitles: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!options.add_subtitles}
                />
                <span className="ml-2 text-sm text-gray-700">{t('newTask.dualSubtitles')}</span>
              </label>
            )}

            {/* Subtitle Preset Selection */}
            {options.add_subtitles && presetsData?.presets && presetsData.presets.length > 0 && (
              <div className="mt-2">
                <label className="block text-xs text-gray-500 mb-1">{t('newTask.subtitlePreset')}</label>
                <select
                  value={options.subtitle_preset || presetsData.presets[0].id}
                  onChange={(e) => setOptions({ ...options, subtitle_preset: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  {presetsData.presets.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name} {preset.is_builtin ? `(${t('styleEditor.builtIn')})` : ''}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Use existing subtitles option - show when detailed info is available */}
            {availableSubtitles.length > 0 && (
              <div className="mt-2 p-3 bg-blue-50 rounded-md">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={options.use_existing_subtitles}
                    onChange={(e) => setOptions({ ...options, use_existing_subtitles: e.target.checked })}
                    className="rounded border-gray-300 text-blue-600"
                    disabled={!options.add_subtitles}
                  />
                  <span className="ml-2 text-sm text-blue-800 font-medium">
                    {t('newTask.useExistingSubtitles')}
                  </span>
                </label>
                {options.use_existing_subtitles && (
                  <div className="mt-2 ml-6">
                    <select
                      value={options.subtitle_language || ''}
                      onChange={(e) => setOptions({ ...options, subtitle_language: e.target.value })}
                      className="w-full px-2 py-1 text-sm border border-blue-200 rounded-md bg-white"
                    >
                      <option value="">{t('newTask.autoSelect')}</option>
                      {availableSubtitles.map((sub) => (
                        <option key={sub.language} value={sub.language}>
                          {sub.language_name} {sub.is_auto_generated ? t('newTask.autoGenerated') : ''}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            )}

            {/* Show option even without detailed info (fallback) */}
            {videoInfo && !detailedInfo && !detailedInfoLoading && (
              <div className="mt-2 p-3 bg-gray-50 rounded-md">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={options.use_existing_subtitles}
                    onChange={(e) => setOptions({ ...options, use_existing_subtitles: e.target.checked })}
                    className="rounded border-gray-300 text-blue-600"
                    disabled={!options.add_subtitles}
                  />
                  <span className="ml-2 text-sm text-gray-700">
                    {t('newTask.useExistingSubtitlesFallback')}
                  </span>
                </label>
                <p className="mt-1 ml-6 text-xs text-gray-500">
                  {t('newTask.useExistingSubtitlesDesc')}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* TTS Options - only show when not in subtitles-only mode */}
        {options.processing_mode !== 'direct' && (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t('newTask.ttsSettings')}
          </label>
          <div className="space-y-3">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.add_tts}
                onChange={(e) => setOptions({ ...options, add_tts: e.target.checked })}
                className="rounded border-gray-300 text-blue-600"
              />
              <span className="ml-2 text-sm text-gray-700">{t('newTask.addTts')}</span>
            </label>
            {options.add_tts && (
              <>
                {/* TTS Engine Selection */}
                <div>
                  <label className="block text-xs text-gray-500 mb-1">{t('newTask.ttsEngine')}</label>
                  <select
                    value={options.tts_service}
                    onChange={(e) => setOptions({ ...options, tts_service: e.target.value, tts_voice: '' })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  >
                    {ttsEngines?.map((engine: TTSEngine) => (
                      <option key={engine.id} value={engine.id}>
                        {engine.name} {engine.free ? `(${t('newTask.free')})` : ''} {engine.supports_voice_cloning ? '🎙️' : ''}
                      </option>
                    ))}
                  </select>
                  {/* Show engine description */}
                  {ttsEngines && options.tts_service && (
                    <p className="text-xs text-gray-500 mt-1">
                      {ttsEngines.find((e: TTSEngine) => e.id === options.tts_service)?.description}
                    </p>
                  )}
                </div>

                {/* TTS Server Health Warning */}
                {options.tts_service !== 'edge' && ttsHealth && !ttsHealth.available && (
                  <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                    <p className="text-sm text-yellow-800">
                      ⚠️ {ttsHealth.message}
                    </p>
                  </div>
                )}

                {/* Voice Cloning Mode Selection */}
                {options.tts_service && ttsEngines?.find((e: TTSEngine) => e.id === options.tts_service)?.supports_voice_cloning && (
                  <div className="p-3 bg-purple-50 border border-purple-200 rounded-md space-y-3">
                    <label className="block text-sm font-medium text-purple-800">
                      🎙️ {t('newTask.voiceCloningMode')}
                    </label>
                    <div className="space-y-2">
                      <label className="flex items-center">
                        <input
                          type="radio"
                          name="voice_cloning_mode"
                          value="disabled"
                          checked={(options.voice_cloning_mode || 'disabled') === 'disabled'}
                          onChange={() => setOptions({ ...options, voice_cloning_mode: 'disabled' })}
                          className="text-purple-600"
                        />
                        <span className="ml-2 text-sm text-gray-700">{t('newTask.cloningDisabled')}</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="radio"
                          name="voice_cloning_mode"
                          value="video_audio"
                          checked={options.voice_cloning_mode === 'video_audio'}
                          onChange={() => setOptions({ ...options, voice_cloning_mode: 'video_audio' })}
                          className="text-purple-600"
                        />
                        <span className="ml-2 text-sm text-gray-700">{t('newTask.cloningVideoAudio')}</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="radio"
                          name="voice_cloning_mode"
                          value="custom"
                          checked={options.voice_cloning_mode === 'custom'}
                          onChange={() => setOptions({ ...options, voice_cloning_mode: 'custom' })}
                          className="text-purple-600"
                        />
                        <span className="ml-2 text-sm text-gray-700">{t('newTask.cloningCustom')}</span>
                      </label>
                    </div>
                    <p className="text-xs text-purple-600">
                      {options.voice_cloning_mode === 'video_audio'
                        ? t('newTask.cloningVideoAudioDesc')
                        : options.voice_cloning_mode === 'custom'
                          ? t('newTask.cloningCustomDesc')
                          : t('newTask.cloningDisabledDesc')}
                    </p>

                    {/* Custom Reference Audio */}
                    {options.voice_cloning_mode === 'custom' && (
                      <div className="space-y-2 pt-2 border-t border-purple-200">
                        <div>
                          <label className="block text-xs text-purple-700 mb-1">{t('newTask.refAudio')}</label>
                          <input
                            type="text"
                            value={options.tts_ref_audio || ''}
                            onChange={(e) => setOptions({ ...options, tts_ref_audio: e.target.value })}
                            placeholder={t('newTask.refAudioPlaceholder')}
                            className="w-full px-3 py-2 border border-purple-300 rounded-md text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-purple-700 mb-1">{t('newTask.refText')}</label>
                          <textarea
                            value={options.tts_ref_text || ''}
                            onChange={(e) => setOptions({ ...options, tts_ref_text: e.target.value })}
                            placeholder={t('newTask.refTextPlaceholder')}
                            rows={2}
                            className="w-full px-3 py-2 border border-purple-300 rounded-md text-sm"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Voice Selection - show when cloning disabled or engine doesn't support cloning */}
                {(!ttsEngines?.find((e: TTSEngine) => e.id === options.tts_service)?.supports_voice_cloning ||
                  (options.voice_cloning_mode || 'disabled') === 'disabled') && (
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">{t('newTask.voiceSelection')}</label>
                    <select
                      value={options.tts_voice}
                      onChange={(e) => setOptions({ ...options, tts_voice: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                    >
                      {voices?.map((voice: Voice) => (
                        <option key={voice.name} value={voice.name}>
                          {voice.display_name} ({voice.gender})
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={options.replace_original_audio}
                    onChange={(e) => setOptions({ ...options, replace_original_audio: e.target.checked })}
                    className="rounded border-gray-300 text-blue-600"
                  />
                  <span className="ml-2 text-sm text-gray-700">{t('newTask.replaceOriginalAudio')}</span>
                </label>

                {/* Audio volume controls */}
                {!options.replace_original_audio && (
                  <div className="p-3 bg-gray-50 rounded-md space-y-3">
                    <div>
                      <div className="flex justify-between text-sm text-gray-600 mb-1">
                        <span>{t('newTask.originalVolume')}</span>
                        <span>{Math.round((options.original_audio_volume || 0.3) * 100)}%</span>
                      </div>
                      <input
                        type="range"
                        value={options.original_audio_volume || 0.3}
                        onChange={(e) => setOptions({ ...options, original_audio_volume: parseFloat(e.target.value) })}
                        min={0}
                        max={1}
                        step={0.1}
                        className="w-full"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm text-gray-600 mb-1">
                        <span>{t('newTask.ttsVolume')}</span>
                        <span>{Math.round((options.tts_audio_volume || 1) * 100)}%</span>
                      </div>
                      <input
                        type="range"
                        value={options.tts_audio_volume || 1}
                        onChange={(e) => setOptions({ ...options, tts_audio_volume: parseFloat(e.target.value) })}
                        min={0}
                        max={1}
                        step={0.1}
                        className="w-full"
                      />
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
        )}

        {/* Metadata Preset Selection */}
        {metadataPresetsData?.presets && metadataPresetsData.presets.length > 0 && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('newTask.metadataPreset')}
            </label>
            <div className="space-y-2">
              {/* AI Selection option - shown first */}
              <div className="flex items-center gap-2">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={options.use_ai_preset_selection || false}
                    onChange={(e) => {
                      setOptions({ ...options, use_ai_preset_selection: e.target.checked })
                      if (e.target.checked && videoInfo) {
                        // Trigger AI selection when enabled and video info available
                        aiSelectPresetMutation.mutate({
                          video_info: { ...videoInfo } as Record<string, unknown>
                        })
                      }
                    }}
                    className="rounded border-gray-300 text-blue-600"
                  />
                  <span className="ml-2 text-sm text-gray-700 flex items-center gap-1">
                    <Sparkles className="w-4 h-4 text-blue-500" />
                    {t('newTask.useAiPresetSelection')}
                  </span>
                </label>
                {aiSelectPresetMutation.isPending && (
                  <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                )}
              </div>
              {options.use_ai_preset_selection && (
                <p className="text-xs text-gray-500 ml-6">
                  {t('newTask.aiPresetSelectionDesc')}
                </p>
              )}

              {/* Manual preset selection - hidden when AI selection is enabled */}
              {!options.use_ai_preset_selection && (
                <>
                  <select
                    value={options.metadata_preset_id || ''}
                    onChange={(e) => setOptions({ ...options, metadata_preset_id: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  >
                    {metadataPresetsData.presets.map((preset: MetadataPreset) => (
                      <option key={preset.id} value={preset.id}>
                        {preset.name}
                        {preset.is_default ? ` (${t('metadataPreset.default')})` : ''}
                        {preset.title_prefix ? ` - ${preset.title_prefix}` : ''}
                      </option>
                    ))}
                  </select>

                  {/* Preview of selected preset */}
                  {(() => {
                    const selectedPreset = metadataPresetsData.presets.find(
                      (p: MetadataPreset) => p.id === options.metadata_preset_id
                    )
                    if (selectedPreset) {
                      return (
                        <div className="p-2 bg-gray-50 rounded-md text-xs text-gray-600">
                          <div className="flex items-center gap-2">
                            <Tag className="w-3 h-3" />
                            <span className="font-medium">{t('newTask.presetPreview')}:</span>
                            {selectedPreset.title_prefix ? (
                              <span className="font-mono text-blue-600">{selectedPreset.title_prefix}</span>
                            ) : (
                              <span className="text-gray-400">{t('metadataPreset.noPrefix')}</span>
                            )}
                          </div>
                          {selectedPreset.custom_signature && (
                            <p className="mt-1 text-gray-500 truncate">
                              {selectedPreset.custom_signature.substring(0, 60)}...
                            </p>
                          )}
                        </div>
                      )
                    }
                    return null
                  })()}
                </>
              )}
            </div>
          </div>
        )}

        {/* Upload Platforms */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t('newTask.uploadPlatforms')}
          </label>
          <div className="space-y-2">
            <div className="space-y-2">
              <label className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    checked={options.upload_bilibili}
                    onChange={(e) => setOptions({ ...options, upload_bilibili: e.target.checked })}
                    className="rounded border-gray-300 text-blue-600"
                  />
                  <span className="ml-2 text-sm text-gray-700">Bilibili</span>
                </div>
                {bilibiliAccounts.length > 0 ? (
                  <span className="text-xs text-green-600">{bilibiliAccounts.length} 个账号</span>
                ) : platformStatus?.bilibili?.configured ? (
                  <span className="text-xs text-green-600">{t('newTask.configured')}</span>
                ) : (
                  <span className="text-xs text-gray-400">{t('newTask.notConfigured')}</span>
                )}
              </label>
              {options.upload_bilibili && bilibiliAccounts.length > 0 && (
                <select
                  value={options.bilibili_account_uid || ''}
                  onChange={(e) => setOptions({ ...options, bilibili_account_uid: e.target.value })}
                  className="ml-6 w-48 text-sm border border-gray-300 rounded-md py-1 px-2"
                >
                  <option value="">
                    {bilibiliAccounts.find(a => a.is_primary)?.label || '主账号'}（默认）
                  </option>
                  {bilibiliAccounts.filter(a => !a.is_primary).map((acc) => (
                    <option key={acc.uid} value={acc.uid}>
                      {acc.label || acc.nickname}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <label className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  checked={options.upload_douyin}
                  onChange={(e) => setOptions({ ...options, upload_douyin: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('newTask.platforms.douyin')}</span>
              </div>
              {platformStatus?.douyin && (
                <span className={`text-xs ${platformStatus.douyin.configured ? 'text-green-600' : 'text-gray-400'}`}>
                  {platformStatus.douyin.configured ? t('newTask.configured') : t('newTask.notConfigured')}
                </span>
              )}
            </label>
            <label className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  checked={options.upload_xiaohongshu}
                  onChange={(e) => setOptions({ ...options, upload_xiaohongshu: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('newTask.platforms.xiaohongshu')}</span>
              </div>
              {platformStatus?.xiaohongshu && (
                <span className={`text-xs ${platformStatus.xiaohongshu.configured ? 'text-green-600' : 'text-gray-400'}`}>
                  {platformStatus.xiaohongshu.configured ? t('newTask.configured') : t('newTask.notConfigured')}
                </span>
              )}
            </label>
          </div>
        </div>

        {/* Directory Selection (Optional) */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
            <FolderOpen className="h-4 w-4" />
            {t('newTask.directory')}
            <span className="text-xs text-gray-400">({t('newTask.optional')})</span>
          </label>
          <p className="text-xs text-gray-500 mb-2">{t('newTask.directoryDesc')}</p>

          {/* Directory mode selection */}
          <div className="flex rounded-lg border border-gray-200 overflow-hidden mb-3">
            <button
              type="button"
              onClick={() => setDirectoryMode('none')}
              className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                directoryMode === 'none'
                  ? 'bg-gray-500 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              {t('newTask.noDirectory')}
            </button>
            <button
              type="button"
              onClick={() => setDirectoryMode('existing')}
              className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                directoryMode === 'existing'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              {t('newTask.selectExisting')}
            </button>
            <button
              type="button"
              onClick={() => setDirectoryMode('new')}
              className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                directoryMode === 'new'
                  ? 'bg-green-500 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              <Plus className="h-3 w-3 inline mr-1" />
              {t('newTask.createNew')}
            </button>
          </div>

          {directoryMode === 'existing' && (
            <select
              value={selectedDirectory}
              onChange={(e) => setSelectedDirectory(e.target.value)}
              disabled={loadingDirectories || directories.length === 0}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {directories.length === 0 ? (
                <option value="">{loadingDirectories ? t('common.loading') : t('newTask.noDirectories')}</option>
              ) : (
                <>
                  <option value="">{t('newTask.selectDirectory')}</option>
                  {directories.map((dir) => (
                    <option key={dir.id} value={dir.name}>
                      {dir.name} ({dir.task_count} {t('newTask.tasksCount')})
                    </option>
                  ))}
                </>
              )}
            </select>
          )}

          {directoryMode === 'new' && (
            <input
              type="text"
              value={newDirectoryName}
              onChange={(e) => {
                setNewDirectoryName(e.target.value)
                setDirectoryError(null)
              }}
              placeholder={t('newTask.newDirectoryPlaceholder')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            />
          )}

          {directoryError && (
            <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
              <AlertCircle className="h-4 w-4" />
              {directoryError}
            </p>
          )}
        </div>

        {/* Advanced Options Toggle */}
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center text-sm text-gray-600 hover:text-gray-800"
        >
          {showAdvanced ? <ChevronUp className="h-4 w-4 mr-1" /> : <ChevronDown className="h-4 w-4 mr-1" />}
          {t('newTask.advanced')}
        </button>

        {/* Advanced Options */}
        {showAdvanced && (
          <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
            {/* Use Global Settings Toggle */}
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.use_global_settings}
                onChange={(e) => setOptions({ ...options, use_global_settings: e.target.checked })}
                className="rounded border-gray-300 text-blue-600"
              />
              <span className="ml-2 text-sm text-gray-700">{t('newTask.useGlobalSettings')}</span>
              <span title={t('newTask.useGlobalSettings')}><Info className="h-4 w-4 ml-1 text-gray-400" /></span>
            </label>

            {/* Custom Title */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {t('newTask.customTitle')}
              </label>
              <input
                type="text"
                value={options.custom_title || ''}
                onChange={(e) => setOptions({ ...options, custom_title: e.target.value })}
                placeholder={t('newTask.customTitlePlaceholder')}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
            </div>

            {/* Custom Description */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {t('newTask.customDescription')}
              </label>
              <textarea
                value={options.custom_description || ''}
                onChange={(e) => setOptions({ ...options, custom_description: e.target.value })}
                placeholder={t('newTask.customDescriptionPlaceholder')}
                rows={2}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
            </div>
          </div>
        )}

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={createTaskMutation.isPending || (sourceType === 'url' ? !url : !uploadResult)}
          className="w-full py-3 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        >
          {createTaskMutation.isPending ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin mr-2" />
              {t('newTask.submitting')}
            </>
          ) : (
            t('newTask.submit')
          )}
        </button>
      </div>
    </div>
  )
}

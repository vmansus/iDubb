import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import {
  X,
  Youtube,
  Search,
  Loader2,
  CheckCircle,
  AlertCircle,
  Settings,
  Instagram,
  Music2,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Volume2,
  Cpu,
  Mic
} from 'lucide-react'
import { subscriptionApi, translationApi, ttsApi, transcriptionApi, presetsApi, metadataPresetsApi, directoryApi, bilibiliApi, douyinApi, xiaohongshuApi, type BilibiliAccount, type DouyinAccount, type XiaohongshuAccount } from '../services/api'
import type { WhisperModel } from '../services/api'
import type {
  ChannelLookupResponse,
  ProcessOptions,
  TranslationEngine,
  TTSEngine,
  Voice,
  VoiceCloningMode,
  SubtitlePreset,
  MetadataPreset,
  Directory
} from '../types'

interface AddSubscriptionModalProps {
  isOpen: boolean
  onClose: () => void
  onCreated: () => void
  editSubscription?: import('../types').Subscription  // If provided, modal is in edit mode
}

type Platform = 'youtube' | 'tiktok' | 'instagram'

const SOURCE_LANGUAGES = [
  { code: 'auto', name: '自动检测' },
  { code: 'en', name: 'English' },
  { code: 'zh', name: '中文' },
  { code: 'ja', name: '日本語' },
  { code: 'ko', name: '한국어' },
  { code: 'es', name: 'Español' },
  { code: 'fr', name: 'Français' },
  { code: 'de', name: 'Deutsch' },
  { code: 'ru', name: 'Русский' },
]

const TARGET_LANGUAGES = [
  { code: 'zh-CN', name: '简体中文' },
  { code: 'zh-TW', name: '繁體中文' },
  { code: 'en', name: 'English' },
  { code: 'ja', name: '日本語' },
  { code: 'ko', name: '한국어' },
  { code: 'es', name: 'Español' },
  { code: 'fr', name: 'Français' },
  { code: 'de', name: 'Deutsch' },
]

const WHISPER_DEVICES = [
  { id: 'auto', name: '自动选择' },
  { id: 'cpu', name: 'CPU' },
  { id: 'cuda', name: 'GPU (CUDA)' },
  { id: 'mps', name: 'Apple Silicon (MPS)' },
]

export default function AddSubscriptionModal({
  isOpen,
  onClose,
  onCreated,
  editSubscription
}: AddSubscriptionModalProps) {
  const { t } = useTranslation()
  const isEditMode = !!editSubscription

  // Form state
  const [platform, setPlatform] = useState<Platform>('youtube')
  const [channelUrl, setChannelUrl] = useState('')
  const [checkInterval, setCheckInterval] = useState(60)
  const [autoProcess, setAutoProcess] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Directory state
  const [directoryMode, setDirectoryMode] = useState<'existing' | 'new'>('existing')
  const [selectedDirectory, setSelectedDirectory] = useState('')
  const [newDirectoryName, setNewDirectoryName] = useState('')
  const [directories, setDirectories] = useState<Directory[]>([])
  const [directoryError, setDirectoryError] = useState<string | null>(null)
  const [loadingDirectories, setLoadingDirectories] = useState(false)

  // Expanded sections
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    language: true,
    whisper: false,
    subtitle: false,
    translation: false,
    tts: false,
    metadata: false,
    upload: false,
  })

  // Process options - complete set
  const [processOptions, setProcessOptions] = useState<ProcessOptions>({
    // Processing mode
    processing_mode: 'full',  // full | subtitle | direct | auto
    source_language: 'auto',
    target_language: 'zh-CN',
    // Whisper settings
    whisper_backend: 'auto',
    whisper_model: 'auto',
    whisper_device: 'auto',
    // Video settings
    video_quality: 'best',
    // Subtitle settings
    add_subtitles: true,
    dual_subtitles: true,
    use_existing_subtitles: true,
    subtitle_preset: '',
    // Translation settings
    translation_engine: 'google',
    // TTS settings
    add_tts: true,
    tts_service: 'edge',
    tts_voice: 'zh-CN-XiaoxiaoNeural',
    voice_cloning_mode: 'disabled',
    replace_original_audio: false,
    original_audio_volume: 30,
    tts_audio_volume: 100,
    // Metadata settings
    metadata_preset_id: '',
    use_ai_preset_selection: false,
    // Upload settings
    upload_bilibili: false,
    upload_douyin: false,
    upload_xiaohongshu: false,
  })

  // Available options from API
  const [translationEngines, setTranslationEngines] = useState<TranslationEngine[]>([])
  const [ttsEngines, setTTSEngines] = useState<TTSEngine[]>([])
  const [ttsVoices, setTTSVoices] = useState<Voice[]>([])
  const [whisperModels, setWhisperModels] = useState<WhisperModel[]>([])
  const [subtitlePresets, setSubtitlePresets] = useState<SubtitlePreset[]>([])
  const [metadataPresets, setMetadataPresets] = useState<MetadataPreset[]>([])

  // Selected TTS engine info
  const [selectedTTSEngine, setSelectedTTSEngine] = useState<TTSEngine | null>(null)

  // Lookup state
  const [lookupResult, setLookupResult] = useState<ChannelLookupResponse | null>(null)
  const [isLookingUp, setIsLookingUp] = useState(false)
  const [lookupError, setLookupError] = useState<string | null>(null)

  // Submit state
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  // Platform accounts state
  const [bilibiliAccounts, setBilibiliAccounts] = useState<BilibiliAccount[]>([])
  const [douyinAccounts, setDouyinAccounts] = useState<DouyinAccount[]>([])
  const [xiaohongshuAccounts, setXiaohongshuAccounts] = useState<XiaohongshuAccount[]>([])

  // Load platform accounts
  const loadAccounts = async () => {
    try {
      const [biliRes, douyinRes, xhsRes] = await Promise.all([
        bilibiliApi.listAccounts().catch(() => ({ accounts: [] })),
        douyinApi.listAccounts().catch(() => ({ accounts: [] })),
        xiaohongshuApi.listAccounts().catch(() => ({ accounts: [] })),
      ])
      setBilibiliAccounts(biliRes.accounts)
      setDouyinAccounts(douyinRes.accounts)
      setXiaohongshuAccounts(xhsRes.accounts)
    } catch (err) {
      console.error('Failed to load accounts:', err)
    }
  }

  // Load available options when modal opens
  useEffect(() => {
    if (isOpen) {
      loadOptions()
      loadDirectories()
      loadAccounts()
    }
  }, [isOpen])

  // Initialize form values when in edit mode
  useEffect(() => {
    if (isOpen && editSubscription) {
      // Set basic fields
      setPlatform(editSubscription.platform)
      setCheckInterval(editSubscription.check_interval)
      setAutoProcess(editSubscription.auto_process)

      // Set directory
      if (editSubscription.directory) {
        setDirectoryMode('existing')
        setSelectedDirectory(editSubscription.directory)
      }

      // Set lookup result to show channel info
      setLookupResult({
        success: true,
        channel_id: editSubscription.channel_id,
        channel_name: editSubscription.channel_name,
        channel_url: editSubscription.channel_url || '',
        channel_avatar: editSubscription.channel_avatar,
      })

      // Set process options
      if (editSubscription.process_options) {
        setProcessOptions(prev => ({
          ...prev,
          ...editSubscription.process_options
        }))
      }

      // Show advanced options if there are custom settings
      setShowAdvanced(true)
    }
  }, [isOpen, editSubscription])

  const loadDirectories = async () => {
    setLoadingDirectories(true)
    try {
      const response = await directoryApi.list()
      setDirectories(response.directories)
      // Default to first directory if exists
      if (response.directories.length > 0 && !selectedDirectory) {
        setSelectedDirectory(response.directories[0].name)
      }
    } catch (err) {
      console.error('Failed to load directories:', err)
    } finally {
      setLoadingDirectories(false)
    }
  }

  // Load TTS voices when engine or language changes
  useEffect(() => {
    if (processOptions.tts_service && processOptions.target_language) {
      loadTTSVoices(processOptions.tts_service, processOptions.target_language)
    }
  }, [processOptions.tts_service, processOptions.target_language])

  // Update selected TTS engine when tts_service changes
  useEffect(() => {
    const engine = ttsEngines.find(e => e.id === processOptions.tts_service)
    setSelectedTTSEngine(engine || null)

    // Reset voice cloning mode if engine doesn't support it
    if (engine && !engine.supports_voice_cloning && processOptions.voice_cloning_mode !== 'disabled') {
      setProcessOptions(prev => ({ ...prev, voice_cloning_mode: 'disabled' }))
    }
  }, [processOptions.tts_service, ttsEngines])

  const loadOptions = async () => {
    try {
      const [
        engines,
        ttsEnginesList,
        modelsResponse,
        presetsResponse,
        metadataPresetsResponse
      ] = await Promise.all([
        translationApi.getEngines(),
        ttsApi.getEngines(),
        transcriptionApi.getModels(),
        presetsApi.getAll(),
        metadataPresetsApi.getAll(),
      ])
      setTranslationEngines(engines)
      setTTSEngines(ttsEnginesList)
      setWhisperModels(modelsResponse.models)
      setSubtitlePresets(presetsResponse.presets)
      setMetadataPresets(metadataPresetsResponse.presets)
    } catch (err) {
      console.error('Failed to load options:', err)
    }
  }

  const loadTTSVoices = async (engine: string, language: string) => {
    try {
      const voices = await ttsApi.getVoicesByEngine(engine, language)
      setTTSVoices(voices)
      // Auto-select first voice if current voice not in list
      if (voices.length > 0 && !voices.find(v => v.name === processOptions.tts_voice)) {
        setProcessOptions(prev => ({ ...prev, tts_voice: voices[0].name }))
      }
    } catch (err) {
      console.error('Failed to load voices:', err)
      setTTSVoices([])
    }
  }

  const platforms: { id: Platform; icon: React.ReactNode; label: string; color: string }[] = [
    { id: 'youtube', icon: <Youtube className="h-5 w-5" />, label: 'YouTube', color: 'text-red-500' },
    { id: 'tiktok', icon: <Music2 className="h-5 w-5" />, label: 'TikTok', color: 'text-gray-900' },
    { id: 'instagram', icon: <Instagram className="h-5 w-5" />, label: 'Instagram', color: 'text-pink-500' },
  ]

  const getPlaceholder = () => {
    switch (platform) {
      case 'youtube':
        return t('subscriptions.youtubeUrlPlaceholder')
      case 'tiktok':
        return t('subscriptions.tiktokUrlPlaceholder')
      case 'instagram':
        return t('subscriptions.instagramUrlPlaceholder')
    }
  }

  const getPlatformIcon = () => {
    const p = platforms.find(p => p.id === platform)
    return p ? <span className={p.color}>{p.icon}</span> : null
  }

  const handlePlatformChange = (newPlatform: Platform) => {
    setPlatform(newPlatform)
    setChannelUrl('')
    setLookupResult(null)
    setLookupError(null)
  }

  const handleLookup = async () => {
    if (!channelUrl.trim()) {
      setLookupError(t('subscriptions.enterChannelUrl'))
      return
    }

    try {
      setIsLookingUp(true)
      setLookupError(null)
      setLookupResult(null)

      const result = await subscriptionApi.lookup(platform, channelUrl.trim())

      if (result.success) {
        setLookupResult(result)
      } else {
        setLookupError(result.error || t('subscriptions.channelNotFound'))
      }
    } catch (err) {
      setLookupError(t('subscriptions.lookupFailed'))
      console.error('Lookup error:', err)
    } finally {
      setIsLookingUp(false)
    }
  }

  const handleSubmit = async () => {
    if (!lookupResult && !isEditMode) {
      setSubmitError(t('subscriptions.lookupFirst'))
      return
    }

    // Validate directory
    const directoryName = directoryMode === 'existing' ? selectedDirectory : newDirectoryName.trim()
    if (!directoryName) {
      setDirectoryError(t('subscriptions.directoryRequired'))
      return
    }

    try {
      setIsSubmitting(true)
      setSubmitError(null)
      setDirectoryError(null)

      // Create new directory if needed
      if (directoryMode === 'new') {
        try {
          await directoryApi.create(newDirectoryName.trim())
        } catch (err: unknown) {
          const errorMessage = err instanceof Error ? err.message : t('subscriptions.createDirectoryFailed')
          setDirectoryError(errorMessage)
          setIsSubmitting(false)
          return
        }
      }

      if (isEditMode && editSubscription) {
        // Update existing subscription
        await subscriptionApi.update(editSubscription.id, {
          directory: directoryName,
          check_interval: checkInterval,
          auto_process: autoProcess,
          process_options: autoProcess ? processOptions : undefined,
        })
      } else {
        // Create new subscription
        await subscriptionApi.create({
          platform,
          channel_url: channelUrl.trim(),
          directory: directoryName,
          check_interval: checkInterval,
          auto_process: autoProcess,
          process_options: autoProcess ? processOptions : undefined,
        })
      }

      // Reset form
      setChannelUrl('')
      setLookupResult(null)
      setPlatform('youtube')
      setCheckInterval(60)
      setAutoProcess(true)
      setShowAdvanced(false)
      setSelectedDirectory('')
      setNewDirectoryName('')
      setDirectoryMode('existing')

      onCreated()
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : t('subscriptions.createFailed')
      setSubmitError(errorMessage)
      console.error('Create error:', err)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    setChannelUrl('')
    setLookupResult(null)
    setLookupError(null)
    setSubmitError(null)
    setShowAdvanced(false)
    onClose()
  }

  const updateOption = <K extends keyof ProcessOptions>(key: K, value: ProcessOptions[K]) => {
    setProcessOptions(prev => ({ ...prev, [key]: value }))
  }

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  // Group whisper models by backend
  const groupedModels = whisperModels.reduce((acc, model) => {
    const backend = model.backend || 'other'
    if (!acc[backend]) acc[backend] = []
    acc[backend].push(model)
    return acc
  }, {} as Record<string, WhisperModel[]>)

  if (!isOpen) return null

  const renderSectionHeader = (
    section: string,
    icon: React.ReactNode,
    title: string,
    description?: string
  ) => (
    <button
      type="button"
      onClick={() => toggleSection(section)}
      className="w-full flex items-center justify-between py-2 text-left bg-transparent hover:bg-gray-50 rounded"
    >
      <div className="flex items-center gap-2">
        <span className="text-gray-500">{icon}</span>
        <span className="text-sm font-medium text-gray-700">{title}</span>
        {description && (
          <span className="text-xs text-gray-400">({description})</span>
        )}
      </div>
      {expandedSections[section] ? (
        <ChevronUp className="h-4 w-4 text-gray-400" />
      ) : (
        <ChevronDown className="h-4 w-4 text-gray-400" />
      )}
    </button>
  )

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-full items-center justify-center p-4">
        {/* Backdrop */}
        <div
          className="fixed inset-0 bg-black/50 transition-opacity"
          onClick={handleClose}
        />

        {/* Modal */}
        <div className="relative bg-white rounded-xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b flex-shrink-0">
            <h2 className="text-lg font-semibold">
              {isEditMode ? t('subscriptions.editSubscription') : t('subscriptions.addSubscription')}
            </h2>
            <button
              onClick={handleClose}
              className="p-1 text-gray-400 hover:text-gray-600 rounded"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Body - Scrollable */}
          <div className="p-4 space-y-4 overflow-y-auto flex-1">
            {/* Platform selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {t('subscriptions.selectPlatform')}
              </label>
              <div className="flex gap-2">
                {platforms.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => handlePlatformChange(p.id)}
                    className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg border-2 transition-colors ${
                      platform === p.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <span className={p.color}>{p.icon}</span>
                    <span className="text-sm font-medium">{p.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Channel URL input */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <span className="flex items-center gap-2">
                  {getPlatformIcon()}
                  {t('subscriptions.channelUrl')}
                </span>
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={channelUrl}
                  onChange={(e) => {
                    setChannelUrl(e.target.value)
                    setLookupResult(null)
                    setLookupError(null)
                  }}
                  placeholder={getPlaceholder()}
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  type="button"
                  onClick={handleLookup}
                  disabled={isLookingUp || !channelUrl.trim()}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLookingUp ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Search className="h-5 w-5" />
                  )}
                </button>
              </div>
              {lookupError && (
                <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                  <AlertCircle className="h-4 w-4" />
                  {lookupError}
                </p>
              )}
            </div>

            {/* Channel preview */}
            {lookupResult && (
              <div className="bg-gray-50 rounded-lg p-3 flex items-center gap-3">
                {lookupResult.channel_avatar ? (
                  <img
                    src={lookupResult.channel_avatar}
                    alt={lookupResult.channel_name}
                    className="w-12 h-12 rounded-full object-cover"
                  />
                ) : (
                  <div className="w-12 h-12 rounded-full bg-gray-200 flex items-center justify-center">
                    {getPlatformIcon()}
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 truncate">{lookupResult.channel_name}</p>
                  <p className="text-sm text-gray-500 truncate">{lookupResult.channel_id}</p>
                </div>
                <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0" />
              </div>
            )}

            {/* Directory selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {t('subscriptions.directory')}
                <span className="text-red-500 ml-1">*</span>
              </label>
              <p className="text-xs text-gray-500 mb-2">{t('subscriptions.directoryDesc')}</p>

              {/* Directory mode toggle */}
              <div className="flex rounded-lg border border-gray-200 overflow-hidden mb-3">
                <button
                  type="button"
                  onClick={() => setDirectoryMode('existing')}
                  className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                    directoryMode === 'existing'
                      ? 'bg-blue-500 text-white'
                      : 'bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  {t('subscriptions.selectExisting')}
                </button>
                <button
                  type="button"
                  onClick={() => setDirectoryMode('new')}
                  className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                    directoryMode === 'new'
                      ? 'bg-blue-500 text-white'
                      : 'bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  {t('subscriptions.createNew')}
                </button>
              </div>

              {directoryMode === 'existing' ? (
                <select
                  value={selectedDirectory}
                  onChange={(e) => setSelectedDirectory(e.target.value)}
                  disabled={loadingDirectories || directories.length === 0}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                >
                  {directories.length === 0 ? (
                    <option value="">{loadingDirectories ? t('common.loading') : t('subscriptions.noDirectories')}</option>
                  ) : (
                    <>
                      <option value="">{t('subscriptions.selectDirectory')}</option>
                      {directories.map((dir) => (
                        <option key={dir.id} value={dir.name}>
                          {dir.name} ({dir.task_count} {t('subscriptions.tasksCount')})
                        </option>
                      ))}
                    </>
                  )}
                </select>
              ) : (
                <input
                  type="text"
                  value={newDirectoryName}
                  onChange={(e) => {
                    setNewDirectoryName(e.target.value)
                    setDirectoryError(null)
                  }}
                  placeholder={t('subscriptions.newDirectoryPlaceholder')}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                />
              )}

              {directoryError && (
                <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                  <AlertCircle className="h-4 w-4" />
                  {directoryError}
                </p>
              )}
            </div>

            {/* Check interval */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {t('subscriptions.checkInterval')}
              </label>
              <select
                value={checkInterval}
                onChange={(e) => setCheckInterval(Number(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value={15}>{t('subscriptions.intervalMinutes', { count: 15 })}</option>
                <option value={30}>{t('subscriptions.intervalMinutes', { count: 30 })}</option>
                <option value={60}>{t('subscriptions.intervalHour', { count: 1 })}</option>
                <option value={120}>{t('subscriptions.intervalHours', { count: 2 })}</option>
                <option value={360}>{t('subscriptions.intervalHours', { count: 6 })}</option>
                <option value={720}>{t('subscriptions.intervalHours', { count: 12 })}</option>
                <option value={1440}>{t('subscriptions.intervalDay', { count: 1 })}</option>
              </select>
            </div>

            {/* Auto process toggle */}
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-900">{t('subscriptions.autoProcessLabel')}</p>
                <p className="text-sm text-gray-500">{t('subscriptions.autoProcessDesc')}</p>
              </div>
              <button
                type="button"
                onClick={() => setAutoProcess(!autoProcess)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  autoProcess ? 'bg-blue-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    autoProcess ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            {/* Advanced options */}
            {autoProcess && (
              <div className="border border-gray-200 rounded-lg">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-50"
                >
                  <span className="flex items-center gap-2 text-sm font-medium text-gray-700">
                    <Settings className="h-4 w-4" />
                    {t('subscriptions.processSettings')}
                  </span>
                  {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>

                {showAdvanced && (
                  <div className="px-4 pb-4 space-y-3 border-t border-gray-200">
                    {/* Processing Mode */}
                    <div className="pt-3">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        {t('newTask.processingMode')}
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() => updateOption('processing_mode', 'full')}
                          className={`p-2 rounded-lg border text-left text-xs ${
                            processOptions.processing_mode === 'full'
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:border-gray-300'
                          }`}
                        >
                          <div className="font-medium">{t('newTask.modeFull')}</div>
                          <div className="text-gray-500">{t('newTask.modeFullDesc')}</div>
                        </button>
                        <button
                          type="button"
                          onClick={() => updateOption('processing_mode', 'subtitle')}
                          className={`p-2 rounded-lg border text-left text-xs ${
                            processOptions.processing_mode === 'subtitle'
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:border-gray-300'
                          }`}
                        >
                          <div className="font-medium">{t('newTask.modeSubtitle')}</div>
                          <div className="text-gray-500">{t('newTask.modeSubtitleDesc')}</div>
                        </button>
                        <button
                          type="button"
                          onClick={() => updateOption('processing_mode', 'direct')}
                          className={`p-2 rounded-lg border text-left text-xs ${
                            processOptions.processing_mode === 'direct'
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:border-gray-300'
                          }`}
                        >
                          <div className="font-medium">{t('newTask.modeDirect')}</div>
                          <div className="text-gray-500">{t('newTask.modeDirectDesc')}</div>
                        </button>
                        <button
                          type="button"
                          onClick={() => updateOption('processing_mode', 'auto')}
                          className={`p-2 rounded-lg border text-left text-xs ${
                            processOptions.processing_mode === 'auto'
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:border-gray-300'
                          }`}
                        >
                          <div className="font-medium">{t('newTask.modeAuto')}</div>
                          <div className="text-gray-500">{t('newTask.modeAutoDesc')}</div>
                        </button>
                      </div>
                    </div>

                    {/* Language Settings - hide for direct mode */}
                    {processOptions.processing_mode !== 'direct' && (
                    <div className="pt-3">
                      {renderSectionHeader('language', <span className="text-base">🌐</span>, t('subscriptions.languageSettings'))}
                      {expandedSections.language && (
                        <div className="mt-3 grid grid-cols-2 gap-3 pl-6">
                          <div>
                            <label className="block text-xs text-gray-600 mb-1">
                              {t('newTask.sourceLanguage')}
                            </label>
                            <select
                              value={processOptions.source_language}
                              onChange={(e) => updateOption('source_language', e.target.value)}
                              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                            >
                              {SOURCE_LANGUAGES.map(lang => (
                                <option key={lang.code} value={lang.code}>{lang.name}</option>
                              ))}
                            </select>
                          </div>
                          <div>
                            <label className="block text-xs text-gray-600 mb-1">
                              {t('newTask.targetLanguage')}
                            </label>
                            <select
                              value={processOptions.target_language}
                              onChange={(e) => updateOption('target_language', e.target.value)}
                              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                            >
                              {TARGET_LANGUAGES.map(lang => (
                                <option key={lang.code} value={lang.code}>{lang.name}</option>
                              ))}
                            </select>
                          </div>
                        </div>
                      )}
                    </div>
                    )}

                    {/* Video Quality - Only for YouTube */}
                    {platform === 'youtube' && processOptions.processing_mode !== 'direct' && (
                      <div>
                        {renderSectionHeader('video', <span className="text-base">🎬</span>, t('subscriptions.videoSettings'))}
                        {expandedSections.video && (
                          <div className="mt-3 pl-6 space-y-2">
                            <label className="block text-xs text-gray-600 mb-1">
                              {t('subscriptions.preferredQuality')}
                            </label>
                            <select
                              value={processOptions.video_quality}
                              onChange={(e) => updateOption('video_quality', e.target.value)}
                              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                            >
                              <option value="best">{t('subscriptions.qualityBest')}</option>
                              <option value="2160p">4K (2160p)</option>
                              <option value="1080p">1080p</option>
                              <option value="720p">720p</option>
                              <option value="480p">480p</option>
                            </select>
                            <p className="text-xs text-gray-500">
                              {t('subscriptions.qualityFallbackHint')}
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Whisper Transcription Settings */}
                    <div>
                      {renderSectionHeader('whisper', <Cpu className="h-4 w-4" />, t('subscriptions.whisperSettings'))}
                      {expandedSections.whisper && (
                        <div className="mt-3 space-y-3 pl-6">
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-gray-600 mb-1">
                                {t('newTask.whisperModel')}
                              </label>
                              <select
                                value={processOptions.whisper_model}
                                onChange={(e) => {
                                  const value = e.target.value
                                  updateOption('whisper_model', value)
                                  // Auto-set backend based on model prefix
                                  if (value.startsWith('faster:')) {
                                    updateOption('whisper_backend', 'faster')
                                  } else if (value.startsWith('openai:')) {
                                    updateOption('whisper_backend', 'openai')
                                  }
                                }}
                                className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                              >
                                <option value="auto">{t('newTask.whisperAuto')}</option>
                                {Object.entries(groupedModels).map(([backend, models]) => (
                                  <optgroup key={backend} label={backend === 'faster' ? 'Faster Whisper' : 'OpenAI Whisper'}>
                                    {models.map(model => (
                                      <option key={model.id} value={model.id}>
                                        {model.name} - {model.quality}
                                      </option>
                                    ))}
                                  </optgroup>
                                ))}
                              </select>
                            </div>
                            <div>
                              <label className="block text-xs text-gray-600 mb-1">
                                {t('newTask.whisperDevice')}
                              </label>
                              <select
                                value={processOptions.whisper_device}
                                onChange={(e) => updateOption('whisper_device', e.target.value)}
                                className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                              >
                                {WHISPER_DEVICES.map(device => (
                                  <option key={device.id} value={device.id}>{device.name}</option>
                                ))}
                              </select>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Subtitle Settings */}
                    <div>
                      {renderSectionHeader('subtitle', <span className="text-base">📝</span>, t('subscriptions.subtitleSettings'))}
                      {expandedSections.subtitle && (
                        <div className="mt-3 space-y-3 pl-6">
                          <div className="space-y-2">
                            <label className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={processOptions.add_subtitles}
                                onChange={(e) => updateOption('add_subtitles', e.target.checked)}
                                className="rounded border-gray-300"
                              />
                              <span className="text-sm">{t('newTask.options.subtitles')}</span>
                            </label>
                            <label className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={processOptions.dual_subtitles}
                                onChange={(e) => updateOption('dual_subtitles', e.target.checked)}
                                className="rounded border-gray-300"
                                disabled={!processOptions.add_subtitles}
                              />
                              <span className="text-sm">{t('newTask.options.dualSubtitles')}</span>
                            </label>
                            {platform === 'youtube' && (
                              <label className="flex items-center gap-2">
                                <input
                                  type="checkbox"
                                  checked={processOptions.use_existing_subtitles}
                                  onChange={(e) => updateOption('use_existing_subtitles', e.target.checked)}
                                  className="rounded border-gray-300"
                                />
                                <span className="text-sm">{t('subscriptions.useExistingSubtitles')}</span>
                              </label>
                            )}
                          </div>

                          {/* Subtitle Preset Selection */}
                          {processOptions.add_subtitles && (
                            <div>
                              <label className="block text-xs text-gray-600 mb-1">
                                {t('subscriptions.subtitlePreset')}
                              </label>
                              <select
                                value={processOptions.subtitle_preset || ''}
                                onChange={(e) => updateOption('subtitle_preset', e.target.value)}
                                className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                              >
                                <option value="">{t('subscriptions.defaultPreset')}</option>
                                {subtitlePresets.map(preset => (
                                  <option key={preset.id} value={preset.id}>
                                    {preset.name} {preset.is_vertical && '(竖屏)'}
                                  </option>
                                ))}
                              </select>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Translation Settings */}
                    <div>
                      {renderSectionHeader('translation', <span className="text-base">🔄</span>, t('subscriptions.translationSettings'))}
                      {expandedSections.translation && (
                        <div className="mt-3 pl-6">
                          <label className="block text-xs text-gray-600 mb-1">
                            {t('newTask.translationEngine')}
                          </label>
                          <select
                            value={processOptions.translation_engine}
                            onChange={(e) => updateOption('translation_engine', e.target.value)}
                            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                          >
                            {translationEngines.map(engine => (
                              <option key={engine.id} value={engine.id}>
                                {engine.name} {engine.free && `(${t('newTask.free')})`}
                              </option>
                            ))}
                          </select>
                        </div>
                      )}
                    </div>

                    {/* TTS Settings */}
                    <div>
                      {renderSectionHeader('tts', <Mic className="h-4 w-4" />, t('subscriptions.ttsSettings'))}
                      {expandedSections.tts && (
                        <div className="mt-3 space-y-3 pl-6">
                          <label className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={processOptions.add_tts}
                              onChange={(e) => updateOption('add_tts', e.target.checked)}
                              className="rounded border-gray-300"
                            />
                            <span className="text-sm">{t('newTask.options.tts')}</span>
                          </label>

                          {processOptions.add_tts && (
                            <>
                              {/* TTS Engine Selection */}
                              <div>
                                <label className="block text-xs text-gray-600 mb-1">
                                  {t('subscriptions.ttsService')}
                                </label>
                                <select
                                  value={processOptions.tts_service}
                                  onChange={(e) => updateOption('tts_service', e.target.value)}
                                  className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                                >
                                  {ttsEngines.map(engine => (
                                    <option key={engine.id} value={engine.id}>
                                      {engine.name}
                                    </option>
                                  ))}
                                </select>
                              </div>

                              {/* Voice Cloning Mode - Show first if engine supports it */}
                              {selectedTTSEngine?.supports_voice_cloning && (
                                <div>
                                  <label className="block text-xs text-gray-600 mb-1">
                                    {t('subscriptions.voiceCloningMode')}
                                  </label>
                                  <select
                                    value={processOptions.voice_cloning_mode}
                                    onChange={(e) => updateOption('voice_cloning_mode', e.target.value as VoiceCloningMode)}
                                    className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                                  >
                                    <option value="disabled">{t('subscriptions.voiceCloningDisabled')}</option>
                                    <option value="video_audio">{t('subscriptions.voiceCloningVideoAudio')}</option>
                                  </select>
                                  <p className="text-xs text-gray-500 mt-1">
                                    {processOptions.voice_cloning_mode === 'video_audio'
                                      ? t('subscriptions.voiceCloningVideoAudioDesc')
                                      : t('subscriptions.voiceCloningDisabledDesc')}
                                  </p>
                                </div>
                              )}

                              {/* Voice Selection - Only show when NOT using voice cloning */}
                              {(!selectedTTSEngine?.supports_voice_cloning || processOptions.voice_cloning_mode === 'disabled') && (
                                <div>
                                  <label className="block text-xs text-gray-600 mb-1">
                                    {t('subscriptions.ttsVoice')}
                                  </label>
                                  <select
                                    value={processOptions.tts_voice}
                                    onChange={(e) => updateOption('tts_voice', e.target.value)}
                                    className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                                  >
                                    {ttsVoices.map(voice => (
                                      <option key={voice.name} value={voice.name}>
                                        {voice.display_name}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                              )}

                              {/* Audio Mixing */}
                              <div className="space-y-2">
                                <label className="flex items-center gap-2">
                                  <input
                                    type="checkbox"
                                    checked={processOptions.replace_original_audio}
                                    onChange={(e) => updateOption('replace_original_audio', e.target.checked)}
                                    className="rounded border-gray-300"
                                  />
                                  <span className="text-sm">{t('newTask.options.replaceTts')}</span>
                                </label>
                              </div>

                              {/* Volume Controls */}
                              {!processOptions.replace_original_audio && (
                                <div className="space-y-3 bg-gray-50 rounded-lg p-3">
                                  <div className="flex items-center gap-2 text-sm text-gray-600">
                                    <Volume2 className="h-4 w-4" />
                                    <span>{t('subscriptions.volumeSettings')}</span>
                                  </div>
                                  <div>
                                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                                      <span>{t('subscriptions.originalVolume')}</span>
                                      <span>{processOptions.original_audio_volume}%</span>
                                    </div>
                                    <input
                                      type="range"
                                      min="0"
                                      max="100"
                                      value={processOptions.original_audio_volume}
                                      onChange={(e) => updateOption('original_audio_volume', Number(e.target.value))}
                                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                    />
                                  </div>
                                  <div>
                                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                                      <span>{t('subscriptions.ttsVolume')}</span>
                                      <span>{processOptions.tts_audio_volume}%</span>
                                    </div>
                                    <input
                                      type="range"
                                      min="0"
                                      max="100"
                                      value={processOptions.tts_audio_volume}
                                      onChange={(e) => updateOption('tts_audio_volume', Number(e.target.value))}
                                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                    />
                                  </div>
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Metadata Settings */}
                    <div>
                      {renderSectionHeader('metadata', <Sparkles className="h-4 w-4" />, t('subscriptions.metadataSettings'))}
                      {expandedSections.metadata && (
                        <div className="mt-3 space-y-3 pl-6">
                          <div>
                            <label className="block text-xs text-gray-600 mb-1">
                              {t('subscriptions.metadataPreset')}
                            </label>
                            <select
                              value={processOptions.metadata_preset_id || ''}
                              onChange={(e) => updateOption('metadata_preset_id', e.target.value)}
                              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                            >
                              <option value="">{t('subscriptions.defaultPreset')}</option>
                              {metadataPresets.map(preset => (
                                <option key={preset.id} value={preset.id}>
                                  {preset.name} {preset.is_default && `(${t('common.default')})`}
                                </option>
                              ))}
                            </select>
                          </div>
                          <label className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={processOptions.use_ai_preset_selection}
                              onChange={(e) => updateOption('use_ai_preset_selection', e.target.checked)}
                              className="rounded border-gray-300"
                            />
                            <span className="text-sm flex items-center gap-1">
                              <Sparkles className="h-3.5 w-3.5 text-purple-500" />
                              {t('subscriptions.useAiPresetSelection')}
                            </span>
                          </label>
                          <p className="text-xs text-gray-500">
                            {t('subscriptions.useAiPresetSelectionDesc')}
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Upload Settings */}
                    <div>
                      {renderSectionHeader('upload', <span className="text-base">📤</span>, t('subscriptions.uploadSettings'))}
                      {expandedSections.upload && (
                        <div className="mt-3 pl-6">
                          <div className="space-y-3">
                            {/* Bilibili */}
                            <div className="flex items-center gap-3">
                              <label className="flex items-center gap-2">
                                <input
                                  type="checkbox"
                                  checked={processOptions.upload_bilibili}
                                  onChange={(e) => updateOption('upload_bilibili', e.target.checked)}
                                  className="rounded border-gray-300"
                                />
                                <span className="text-sm">{t('newTask.platforms.bilibili')}</span>
                              </label>
                              {processOptions.upload_bilibili && bilibiliAccounts.length > 1 && (
                                <select
                                  value={processOptions.bilibili_account_uid || ''}
                                  onChange={(e) => updateOption('bilibili_account_uid', e.target.value || undefined)}
                                  className="text-xs border border-gray-300 rounded px-2 py-1"
                                >
                                  <option value="">{bilibiliAccounts.find(a => a.is_primary)?.label || '默认账号'}</option>
                                  {bilibiliAccounts.filter(a => !a.is_primary).map((acc) => (
                                    <option key={acc.uid} value={acc.uid}>{acc.label || acc.nickname}</option>
                                  ))}
                                </select>
                              )}
                            </div>
                            {/* Douyin */}
                            <div className="flex items-center gap-3">
                              <label className="flex items-center gap-2">
                                <input
                                  type="checkbox"
                                  checked={processOptions.upload_douyin}
                                  onChange={(e) => updateOption('upload_douyin', e.target.checked)}
                                  className="rounded border-gray-300"
                                />
                                <span className="text-sm">{t('newTask.platforms.douyin')}</span>
                              </label>
                              {processOptions.upload_douyin && douyinAccounts.length > 1 && (
                                <select
                                  value={processOptions.douyin_account_id || ''}
                                  onChange={(e) => updateOption('douyin_account_id', e.target.value || undefined)}
                                  className="text-xs border border-gray-300 rounded px-2 py-1"
                                >
                                  <option value="">{douyinAccounts.find(a => a.is_primary)?.label || '默认账号'}</option>
                                  {douyinAccounts.filter(a => !a.is_primary).map((acc) => (
                                    <option key={acc.uid} value={acc.uid}>{acc.label || acc.nickname}</option>
                                  ))}
                                </select>
                              )}
                            </div>
                            {/* Xiaohongshu */}
                            <div className="flex items-center gap-3">
                              <label className="flex items-center gap-2">
                                <input
                                  type="checkbox"
                                  checked={processOptions.upload_xiaohongshu}
                                  onChange={(e) => updateOption('upload_xiaohongshu', e.target.checked)}
                                  className="rounded border-gray-300"
                                />
                                <span className="text-sm">{t('newTask.platforms.xiaohongshu')}</span>
                              </label>
                              {processOptions.upload_xiaohongshu && xiaohongshuAccounts.length > 1 && (
                                <select
                                  value={processOptions.xiaohongshu_account_id || ''}
                                  onChange={(e) => updateOption('xiaohongshu_account_id', e.target.value || undefined)}
                                  className="text-xs border border-gray-300 rounded px-2 py-1"
                                >
                                  <option value="">{xiaohongshuAccounts.find(a => a.is_primary)?.label || '默认账号'}</option>
                                  {xiaohongshuAccounts.filter(a => !a.is_primary).map((acc) => (
                                    <option key={acc.user_id} value={acc.user_id}>{acc.label || acc.nickname}</option>
                                  ))}
                                </select>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Submit error */}
            {submitError && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-red-500" />
                <span className="text-red-700 text-sm">{submitError}</span>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex justify-end gap-3 p-4 border-t flex-shrink-0">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            >
              {t('common.cancel')}
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!lookupResult || isSubmitting}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isSubmitting && <Loader2 className="h-4 w-4 animate-spin" />}
              {isEditMode ? t('common.save') : t('subscriptions.subscribe')}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

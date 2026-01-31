import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { CheckCircle, XCircle, Loader2, Eye, EyeOff, Save, RotateCcw, Settings2, Youtube, Cookie, Upload, Trash2, RefreshCw, Plus, Sparkles } from 'lucide-react'
import { configApi, settingsApi, translationApi, ttsApi, youtubeApi, storageApi, presetsApi, metadataPresetsApi } from '../services/api'
import type { GlobalSettings, TranslationEngine, TTSEngine, MetadataPreset, CreateMetadataPresetRequest } from '../types'
import { Link } from 'react-router-dom'
import MetadataPresetCard from '../components/MetadataPresetCard'
import MetadataPresetEditor from '../components/MetadataPresetEditor'
import BilibiliAccounts from '../components/BilibiliAccounts'
import DouyinAccounts from '../components/DouyinAccounts'
import XiaohongshuAccounts from '../components/XiaohongshuAccounts'

interface PlatformConfig {
  platform: string
  label: string
  fields: {
    key: string
    label: string
    placeholder: string
    type?: string
  }[]
}

// Platform config with translation keys - labels and placeholders are translated at render time
// All platforms now use a single cookies field for consistency
const platforms: PlatformConfig[] = [
  {
    platform: 'bilibili',
    label: 'settings.platforms.labels.bilibili',
    fields: [
      { key: 'cookies', label: 'Cookies', placeholder: 'settings.platforms.placeholders.fullCookieString', type: 'textarea' },
    ],
  },
  {
    platform: 'douyin',
    label: 'settings.platforms.labels.douyin',
    fields: [
      { key: 'cookies', label: 'Cookies', placeholder: 'settings.platforms.placeholders.fullCookieString', type: 'textarea' },
    ],
  },
  {
    platform: 'xiaohongshu',
    label: 'settings.platforms.labels.xiaohongshu',
    fields: [
      { key: 'cookies', label: 'Cookies', placeholder: 'settings.platforms.placeholders.fullCookieString', type: 'textarea' },
    ],
  },
]

// Video quality options with translation keys
const videoQualities = [
  { value: '2160p', labelKey: 'settings.video.qualities.4k' },
  { value: '1080p', labelKey: 'settings.video.qualities.1080p' },
  { value: '720p', labelKey: 'settings.video.qualities.720p' },
  { value: '480p', labelKey: 'settings.video.qualities.480p' },
  { value: '360p', labelKey: 'settings.video.qualities.360p' },
]

// StorageSettings component for configurable output directory
function StorageSettings() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [outputDir, setOutputDir] = useState('')
  const [hasStorageChanges, setHasStorageChanges] = useState(false)

  // Fetch storage settings
  const { data: storageSettings, isLoading } = useQuery({
    queryKey: ['storageSettings'],
    queryFn: storageApi.get,
  })

  // Initialize local state when data loads
  useEffect(() => {
    if (storageSettings) {
      setOutputDir(storageSettings.output_directory)
    }
  }, [storageSettings])

  // Save storage settings mutation
  const saveStorageMutation = useMutation({
    mutationFn: (dir: string) => storageApi.update(dir),
    onSuccess: () => {
      setHasStorageChanges(false)
      queryClient.invalidateQueries({ queryKey: ['storageSettings'] })
    },
  })

  const handleDirChange = (value: string) => {
    setOutputDir(value)
    setHasStorageChanges(value !== (storageSettings?.output_directory || ''))
  }

  const handleSaveStorage = () => {
    saveStorageMutation.mutate(outputDir)
  }

  const handleResetToDefault = () => {
    setOutputDir('')
    setHasStorageChanges(storageSettings?.output_directory !== '')
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-center py-4">
          <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.storage.title')}</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {t('settings.storage.outputDirectory')}
          </label>
          <input
            type="text"
            value={outputDir}
            onChange={(e) => handleDirChange(e.target.value)}
            placeholder={storageSettings?.default_directory || t('settings.storage.outputDirectoryPlaceholder')}
            className="w-full px-3 py-2 border border-gray-300 rounded-md font-mono text-sm"
          />
          <p className="mt-1 text-xs text-gray-500">
            {t('settings.storage.outputDirectoryDesc')}
          </p>
        </div>

        {/* Current effective directory */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">{t('settings.storage.currentEffective')}</span>
            <span className="text-sm font-mono text-gray-700 truncate max-w-xs" title={storageSettings?.effective_directory}>
              {storageSettings?.effective_directory}
            </span>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex items-center justify-between pt-2">
          <button
            onClick={handleResetToDefault}
            disabled={outputDir === ''}
            className="text-sm text-gray-600 hover:text-gray-800 disabled:opacity-50"
          >
            {t('settings.storage.useDefault')}
          </button>
          <button
            onClick={handleSaveStorage}
            disabled={!hasStorageChanges || saveStorageMutation.isPending}
            className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 flex items-center"
          >
            {saveStorageMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-1" />
            )}
            {t('common.save')}
          </button>
        </div>

        {/* Error message */}
        {saveStorageMutation.isError && (
          <div className="p-3 bg-red-50 text-red-700 rounded-md text-sm">
            <XCircle className="h-4 w-4 inline mr-1" />
            {t('common.saveFailed')}: {(saveStorageMutation.error as Error)?.message || t('common.unknownError')}
          </div>
        )}

        {/* Success message */}
        {saveStorageMutation.isSuccess && (
          <div className="p-3 bg-green-50 text-green-700 rounded-md text-sm">
            <CheckCircle className="h-4 w-4 inline mr-1" />
            {t('settings.storage.saved')}
          </div>
        )}
      </div>
    </div>
  )
}

// MetadataPresetsSection component for managing metadata presets
function MetadataPresetsSection() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [showEditor, setShowEditor] = useState(false)
  const [editingPreset, setEditingPreset] = useState<MetadataPreset | null>(null)

  // Fetch metadata presets
  const { data: presetsData, isLoading } = useQuery({
    queryKey: ['metadataPresets'],
    queryFn: metadataPresetsApi.getAll,
  })

  // Create preset mutation
  const createMutation = useMutation({
    mutationFn: metadataPresetsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['metadataPresets'] })
      setShowEditor(false)
      setEditingPreset(null)
    },
  })

  // Update preset mutation
  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: CreateMetadataPresetRequest }) =>
      metadataPresetsApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['metadataPresets'] })
      setShowEditor(false)
      setEditingPreset(null)
    },
  })

  // Delete preset mutation
  const deleteMutation = useMutation({
    mutationFn: metadataPresetsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['metadataPresets'] })
    },
  })

  // Set default mutation
  const setDefaultMutation = useMutation({
    mutationFn: metadataPresetsApi.setDefault,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['metadataPresets'] })
    },
  })

  const handleCreate = () => {
    setEditingPreset(null)
    setShowEditor(true)
  }

  const handleEdit = (preset: MetadataPreset) => {
    setEditingPreset(preset)
    setShowEditor(true)
  }

  const handleSave = (data: CreateMetadataPresetRequest) => {
    if (editingPreset) {
      updateMutation.mutate({ id: editingPreset.id, data })
    } else {
      createMutation.mutate(data)
    }
  }

  const handleDelete = (presetId: string) => {
    if (window.confirm(t('metadataPreset.confirmDelete'))) {
      deleteMutation.mutate(presetId)
    }
  }

  const handleSetDefault = (presetId: string) => {
    setDefaultMutation.mutate(presetId)
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
        </div>
      </div>
    )
  }

  const presets = presetsData?.presets || []

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-medium text-gray-900">{t('metadataPreset.title')}</h3>
          <p className="text-sm text-gray-500 mt-1">{t('metadataPreset.description')}</p>
        </div>
        <button
          onClick={handleCreate}
          className="px-3 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 flex items-center gap-1"
        >
          <Plus className="w-4 h-4" />
          {t('metadataPreset.addPreset')}
        </button>
      </div>

      {/* Presets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {presets.map((preset) => (
          <MetadataPresetCard
            key={preset.id}
            preset={preset}
            onEdit={() => handleEdit(preset)}
            onDelete={() => handleDelete(preset.id)}
            onSetDefault={() => handleSetDefault(preset.id)}
            isDeleting={deleteMutation.isPending && deleteMutation.variables === preset.id}
          />
        ))}
      </div>

      {presets.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <p>{t('metadataPreset.noPresets')}</p>
          <button
            onClick={handleCreate}
            className="mt-2 text-blue-600 hover:text-blue-700 text-sm"
          >
            {t('metadataPreset.createFirst')}
          </button>
        </div>
      )}

      {/* AI Selection Info */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg flex items-start gap-2">
        <Sparkles className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
        <div>
          <p className="text-sm text-blue-800 font-medium">{t('metadataPreset.aiSelectInfo')}</p>
          <p className="text-xs text-blue-600 mt-1">{t('metadataPreset.aiSelectDesc')}</p>
        </div>
      </div>

      {/* Editor Modal */}
      {showEditor && (
        <MetadataPresetEditor
          preset={editingPreset}
          onSave={handleSave}
          onCancel={() => {
            setShowEditor(false)
            setEditingPreset(null)
          }}
          isSaving={createMutation.isPending || updateMutation.isPending}
        />
      )}
    </div>
  )
}

export default function Settings() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'general' | 'video' | 'translation' | 'ai' | 'publish' | 'platforms' | 'youtube'>('general')
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({})
  const [credentials, setCredentials] = useState<Record<string, Record<string, string>>>({
    bilibili: {},
    douyin: {},
    xiaohongshu: {},
  })
  const [localSettings, setLocalSettings] = useState<GlobalSettings | null>(null)
  const [hasChanges, setHasChanges] = useState(false)
  const [cookieContent, setCookieContent] = useState('')
  const [selectedBrowser, setSelectedBrowser] = useState('chrome')

  // Fetch global settings
  const { data: globalSettings, isLoading: settingsLoading } = useQuery({
    queryKey: ['globalSettings'],
    queryFn: settingsApi.get,
  })

  // Fetch translation engines
  const { data: translationEngines } = useQuery({
    queryKey: ['translationEngines'],
    queryFn: translationApi.getEngines,
  })

  // Fetch TTS engines
  const { data: ttsEngines } = useQuery({
    queryKey: ['ttsEngines'],
    queryFn: ttsApi.getEngines,
  })

  // Fetch voices based on selected TTS engine
  const selectedTTSEngine = localSettings?.tts?.engine || 'edge'
  const { data: voices, isLoading: voicesLoading } = useQuery({
    queryKey: ['voices', selectedTTSEngine],
    queryFn: () => ttsApi.getVoicesByEngine(selectedTTSEngine),
    enabled: !!selectedTTSEngine,
  })

  const { data: platformStatus, isLoading: platformsLoading } = useQuery({
    queryKey: ['platformStatus'],
    queryFn: configApi.getPlatformStatus,
  })

  // Fetch YouTube cookie status
  const { data: cookieStatus, isLoading: cookieStatusLoading, refetch: refetchCookieStatus } = useQuery({
    queryKey: ['youtubeCookieStatus'],
    queryFn: youtubeApi.getCookieStatus,
  })

  // Fetch subtitle presets
  const { data: presetsData } = useQuery({
    queryKey: ['presets'],
    queryFn: presetsApi.getAll,
  })

  // Initialize local settings when data loads
  useEffect(() => {
    if (globalSettings && !localSettings) {
      setLocalSettings(globalSettings)
    }
  }, [globalSettings, localSettings])

  // Save settings mutation
  const saveSettingsMutation = useMutation({
    mutationFn: (settings: Partial<GlobalSettings>) => settingsApi.update(settings),
    onSuccess: (data) => {
      setLocalSettings(data)
      setHasChanges(false)
      queryClient.invalidateQueries({ queryKey: ['globalSettings'] })
      queryClient.invalidateQueries({ queryKey: ['trending-settings'] })
    },
  })

  // Reset settings mutation
  const resetSettingsMutation = useMutation({
    mutationFn: settingsApi.reset,
    onSuccess: (data) => {
      setLocalSettings(data)
      setHasChanges(false)
      queryClient.invalidateQueries({ queryKey: ['globalSettings'] })
      queryClient.invalidateQueries({ queryKey: ['trending-settings'] })
    },
  })

  const authMutation = useMutation({
    mutationFn: ({ platform, creds }: { platform: string; creds: Record<string, string> }) =>
      configApi.authenticatePlatform(platform, creds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['platformStatus'] })
    },
  })

  // Fetch platform cookie status (for auto-extract)
  const { data: platformCookieStatus } = useQuery({
    queryKey: ['platformCookieStatus'],
    queryFn: configApi.getPlatformCookieStatus,
  })

  // Platform cookie extraction mutation
  const extractPlatformCookiesMutation = useMutation({
    mutationFn: ({ platform, browser }: { platform: string; browser: string }) =>
      configApi.extractPlatformCookies(platform, browser),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['platformStatus'] })
      queryClient.invalidateQueries({ queryKey: ['platformCookieStatus'] })
    },
  })

  // YouTube cookie mutations
  const extractCookiesMutation = useMutation({
    mutationFn: (browser: string) => youtubeApi.extractCookies(browser),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['youtubeCookieStatus'] })
    },
  })

  const uploadCookiesMutation = useMutation({
    mutationFn: (content: string) => youtubeApi.uploadCookies(content),
    onSuccess: () => {
      setCookieContent('')
      queryClient.invalidateQueries({ queryKey: ['youtubeCookieStatus'] })
    },
  })

  const deleteCookiesMutation = useMutation({
    mutationFn: youtubeApi.deleteCookies,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['youtubeCookieStatus'] })
    },
  })

  const handleSettingChange = <K extends keyof GlobalSettings>(
    section: K,
    key: string,
    value: unknown
  ) => {
    if (!localSettings) return

    setLocalSettings((prev) => {
      if (!prev) return prev
      const sectionData = prev[section]
      // Handle case where section doesn't exist yet (e.g., trending)
      if (typeof sectionData === 'object' && sectionData !== null) {
        return {
          ...prev,
          [section]: {
            ...sectionData,
            [key]: value,
          },
        }
      }
      // Section is undefined or not an object - create new object with the key
      return {
        ...prev,
        [section]: {
          [key]: value,
        },
      }
    })
    setHasChanges(true)
  }

  const handleTopLevelChange = (key: string, value: unknown) => {
    if (!localSettings) return
    setLocalSettings((prev) => prev ? { ...prev, [key]: value } : prev)
    setHasChanges(true)
  }

  const handleSaveSettings = () => {
    if (localSettings) {
      saveSettingsMutation.mutate(localSettings)
    }
  }

  const handleAuthenticate = (platform: string) => {
    const creds = credentials[platform]
    if (Object.keys(creds).length === 0) {
      alert(t('settings.platforms.pleaseEnterCredentials'))
      return
    }
    authMutation.mutate({ platform, creds })
  }

  const toggleShowSecret = (key: string) => {
    setShowSecrets((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  if (settingsLoading || platformsLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">{t('settings.title')}</h2>
        {!['platforms'].includes(activeTab) && (
          <div className="flex space-x-2">
            <button
              onClick={() => resetSettingsMutation.mutate()}
              disabled={resetSettingsMutation.isPending}
              className="px-3 py-1.5 text-sm text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 flex items-center"
            >
              <RotateCcw className="h-4 w-4 mr-1" />
              {t('settings.resetDefault')}
            </button>
            <button
              onClick={handleSaveSettings}
              disabled={!hasChanges || saveSettingsMutation.isPending}
              className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center"
            >
              {saveSettingsMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Save className="h-4 w-4 mr-1" />
              )}
              {t('settings.saveSettings')}
            </button>
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex flex-wrap border-b border-gray-200 mb-6 gap-1">
        {[
          { id: 'general', icon: Settings2, label: 'settings.tabs.general' },
          { id: 'video', icon: null, label: 'settings.tabs.video' },
          { id: 'translation', icon: null, label: 'settings.tabs.translation' },
          { id: 'ai', icon: null, label: 'settings.tabs.ai' },
          { id: 'publish', icon: null, label: 'settings.tabs.publish' },
          { id: 'platforms', icon: null, label: 'settings.tabs.platforms' },
          { id: 'youtube', icon: Youtube, label: 'settings.tabs.youtube' },
        ].map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id as typeof activeTab)}
            className={`px-3 py-2 text-sm font-medium border-b-2 -mb-px whitespace-nowrap ${
              activeTab === id
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {Icon && <Icon className="h-4 w-4 inline mr-1" />}
            {t(label)}
          </button>
        ))}
      </div>

      {/* Global Settings Tab */}
      {/* General Tab - Storage & Processing */}
      {activeTab === 'general' && localSettings && (
        <div className="space-y-6">
          {/* General Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.general.title')}</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.general.timezone')}
                </label>
                <select
                  value={localSettings.processing?.timezone || 'Asia/Shanghai'}
                  onChange={(e) => handleSettingChange('processing', 'timezone', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="Asia/Shanghai">中国 (UTC+8)</option>
                  <option value="Asia/Tokyo">日本 (UTC+9)</option>
                  <option value="Asia/Seoul">韩国 (UTC+9)</option>
                  <option value="Asia/Singapore">新加坡 (UTC+8)</option>
                  <option value="Asia/Hong_Kong">香港 (UTC+8)</option>
                  <option value="America/New_York">美东 (UTC-5/-4)</option>
                  <option value="America/Los_Angeles">美西 (UTC-8/-7)</option>
                  <option value="Europe/London">伦敦 (UTC+0/+1)</option>
                  <option value="Europe/Paris">巴黎 (UTC+1/+2)</option>
                  <option value="Australia/Sydney">悉尼 (UTC+10/+11)</option>
                  <option value="UTC">UTC</option>
                </select>
                <p className="mt-1 text-xs text-gray-500">{t('settings.general.timezoneHint')}</p>
              </div>
            </div>
          </div>

          {/* Storage Settings */}
          <StorageSettings />

          {/* Processing Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.processing.title')}</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.processing.maxConcurrentTasks')}
                </label>
                <select
                  value={localSettings.processing?.max_concurrent_tasks || 2}
                  onChange={(e) => handleSettingChange('processing', 'max_concurrent_tasks', parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value={1}>1 ({t('settings.processing.sequential')})</option>
                  <option value={2}>2 ({t('settings.processing.recommended')})</option>
                  <option value={3}>3</option>
                  <option value={4}>4</option>
                </select>
                <p className="mt-1 text-xs text-gray-500">{t('settings.processing.maxConcurrentHint')}</p>
              </div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.processing?.use_gpu_lock ?? true}
                  onChange={(e) => handleSettingChange('processing', 'use_gpu_lock', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.processing.useGpuLock')}</span>
              </label>
              <p className="text-xs text-gray-500 -mt-2 ml-6">{t('settings.processing.gpuLockHint')}</p>
            </div>
          </div>
        </div>
      )}

      {/* Video Tab - Video, Transcription, Subtitle */}
      {activeTab === 'video' && localSettings && (
        <div className="space-y-6">
          {/* Video Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.sections.video')}</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.video.defaultQuality')}
                </label>
                <select
                  value={localSettings.video.default_quality}
                  onChange={(e) => handleSettingChange('video', 'default_quality', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  {videoQualities.map((q) => (
                    <option key={q.value} value={q.value}>
                      {t(q.labelKey)}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.video.maxDuration')}
                </label>
                <input
                  type="number"
                  value={localSettings.video.max_duration}
                  onChange={(e) => handleSettingChange('video', 'max_duration', parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.video.prefer_existing_subtitles}
                  onChange={(e) => handleSettingChange('video', 'prefer_existing_subtitles', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.video.preferExistingSubtitles')}</span>
              </label>
            </div>
          </div>

          {/* Transcription Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.transcription.title')}</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.transcription.backend')}
                </label>
                <select
                  value={localSettings.video.whisper_backend || 'faster'}
                  onChange={(e) => handleSettingChange('video', 'whisper_backend', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="auto">{t('settings.transcription.backends.auto')}</option>
                  <option value="faster">{t('settings.transcription.backends.faster')}</option>
                  <option value="openai">{t('settings.transcription.backends.openai')}</option>
                </select>
                <p className="mt-1 text-xs text-gray-500">{t('settings.transcription.backendHint')}</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.transcription.model')}
                </label>
                <select
                  value={localSettings.video.whisper_model || 'faster:small'}
                  onChange={(e) => handleSettingChange('video', 'whisper_model', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="auto">{t('settings.transcription.models.auto')}</option>
                  <option value="faster:tiny">Faster Whisper - Tiny ({t('settings.transcription.speed.fastest')})</option>
                  <option value="faster:base">Faster Whisper - Base</option>
                  <option value="faster:small">Faster Whisper - Small ({t('settings.transcription.speed.recommended')})</option>
                  <option value="faster:medium">Faster Whisper - Medium</option>
                  <option value="faster:large-v3">Faster Whisper - Large V3 ({t('settings.transcription.speed.best')})</option>
                  <option value="openai:tiny">OpenAI Whisper - Tiny</option>
                  <option value="openai:base">OpenAI Whisper - Base</option>
                  <option value="openai:small">OpenAI Whisper - Small</option>
                  <option value="openai:medium">OpenAI Whisper - Medium</option>
                  <option value="openai:large-v3">OpenAI Whisper - Large V3</option>
                </select>
                <p className="mt-1 text-xs text-gray-500">{t('settings.transcription.modelHint')}</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.transcription.device')}
                </label>
                <select
                  value={localSettings.video.whisper_device || 'auto'}
                  onChange={(e) => handleSettingChange('video', 'whisper_device', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="auto">{t('settings.transcription.devices.auto')}</option>
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA (NVIDIA GPU)</option>
                  <option value="mps">MPS (Apple Silicon)</option>
                </select>
                <p className="mt-1 text-xs text-gray-500">{t('settings.transcription.deviceHint')}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Translation Tab - Translation, TTS, Subtitle, Audio */}
      {activeTab === 'translation' && localSettings && (
        <div className="space-y-6">
          {/* Translation Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.sections.translation')}</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.translation.engine')}
                </label>
                <select
                  value={localSettings.translation.engine}
                  onChange={(e) => {
                    const newEngine = e.target.value;
                    handleSettingChange('translation', 'engine', newEngine);
                    // Auto-select default model for the new engine
                    const defaultModels: Record<string, string> = {
                      'gpt': 'gpt-4',
                      'claude': 'claude-3-sonnet',
                      'deepseek': 'deepseek-chat',
                    };
                    if (defaultModels[newEngine]) {
                      handleSettingChange('translation', 'model', defaultModels[newEngine]);
                    }
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  {translationEngines?.map((engine: TranslationEngine) => (
                    <option key={engine.id} value={engine.id}>
                      {engine.name} {engine.free ? t('settings.translation.free') : ''}
                    </option>
                  ))}
                </select>
                {translationEngines?.find((e: TranslationEngine) => e.id === localSettings.translation.engine)?.description && (
                  <p className="mt-1 text-xs text-gray-500">
                    {translationEngines.find((e: TranslationEngine) => e.id === localSettings.translation.engine)?.description}
                  </p>
                )}
              </div>

              {/* Show API key input for engines that require it */}
              {translationEngines?.find((e: TranslationEngine) => e.id === localSettings.translation.engine)?.requires_api_key && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t('settings.translation.apiKey')} ({localSettings.translation.engine === 'gpt' ? 'OpenAI' :
                              localSettings.translation.engine === 'claude' ? 'Anthropic' :
                              localSettings.translation.engine === 'deepseek' ? 'DeepSeek' :
                              localSettings.translation.engine === 'deepl' ? 'DeepL' : 'API'})
                  </label>
                  <div className="flex">
                    <input
                      type={showSecrets['translation_api_key'] ? 'text' : 'password'}
                      value={(() => {
                        const engine = localSettings.translation.engine;
                        const apiKeys = localSettings.translation.api_keys;
                        if (engine === 'gpt') return apiKeys?.openai || '';
                        if (engine === 'claude') return apiKeys?.anthropic || '';
                        if (engine === 'deepseek') return apiKeys?.deepseek || '';
                        if (engine === 'deepl') return apiKeys?.deepl || '';
                        return localSettings.translation.api_key || '';
                      })()}
                      onChange={(e) => {
                        const engine = localSettings.translation.engine;
                        const currentApiKeys = localSettings.translation.api_keys || {
                          openai: '', anthropic: '', deepseek: '', deepl: ''
                        };
                        let keyField = 'openai';
                        if (engine === 'gpt') keyField = 'openai';
                        else if (engine === 'claude') keyField = 'anthropic';
                        else if (engine === 'deepseek') keyField = 'deepseek';
                        else if (engine === 'deepl') keyField = 'deepl';

                        const newApiKeys = { ...currentApiKeys, [keyField]: e.target.value };
                        handleSettingChange('translation', 'api_keys', newApiKeys);
                      }}
                      placeholder={t('settings.translation.apiKeyPlaceholder', { provider: localSettings.translation.engine === 'gpt' ? 'OpenAI' :
                                        localSettings.translation.engine === 'claude' ? 'Anthropic' :
                                        localSettings.translation.engine === 'deepseek' ? 'DeepSeek' :
                                        'DeepL' })}
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-l-md font-mono text-sm"
                    />
                    <button
                      type="button"
                      onClick={() => toggleShowSecret('translation_api_key')}
                      className="px-3 py-2 border border-l-0 border-gray-300 rounded-r-md bg-gray-50 hover:bg-gray-100"
                    >
                      {showSecrets['translation_api_key'] ? (
                        <EyeOff className="h-4 w-4 text-gray-500" />
                      ) : (
                        <Eye className="h-4 w-4 text-gray-500" />
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* Model selection for GPT/Claude/DeepSeek */}
              {['gpt', 'claude', 'deepseek'].includes(localSettings.translation.engine) && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t('settings.translation.model')}
                  </label>
                  <select
                    value={localSettings.translation.model}
                    onChange={(e) => handleSettingChange('translation', 'model', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  >
                    {localSettings.translation.engine === 'gpt' ? (
                      <>
                        <option value="gpt-4">{t('settings.translation.models.gpt4')}</option>
                        <option value="gpt-4-turbo">{t('settings.translation.models.gpt4turbo')}</option>
                        <option value="gpt-3.5-turbo">{t('settings.translation.models.gpt35turbo')}</option>
                      </>
                    ) : localSettings.translation.engine === 'claude' ? (
                      <>
                        <option value="claude-3-opus">{t('settings.translation.models.claude3opus')}</option>
                        <option value="claude-3-sonnet">{t('settings.translation.models.claude3sonnet')}</option>
                        <option value="claude-3-haiku">{t('settings.translation.models.claude3haiku')}</option>
                      </>
                    ) : (
                      <>
                        <option value="deepseek-chat">{t('settings.translation.models.deepseekchat')}</option>
                        <option value="deepseek-reasoner">{t('settings.translation.models.deepseekReasoner')}</option>
                      </>
                    )}
                  </select>
                </div>
              )}

              {/* Translation Pipeline Mode - Only for AI engines */}
              {['gpt', 'claude', 'deepseek'].includes(localSettings.translation.engine) && (
                <div className="border-t pt-4 mt-4">
                  <h4 className="text-sm font-medium text-gray-900 mb-3">{t('settings.translation.pipelineSettings')}</h4>
                  
                  {/* Two-Step Mode Toggle */}
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <span className="text-sm font-medium text-gray-700">{t('settings.translation.twoStepMode')}</span>
                      <p className="text-xs text-gray-500">{t('settings.translation.twoStepModeDesc')}</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={localSettings.translation.use_two_step_mode ?? true}
                        onChange={(e) => handleSettingChange('translation', 'use_two_step_mode', e.target.checked)}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </label>
                  </div>

                  {/* Subtitle Alignment Toggle */}
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm font-medium text-gray-700">{t('settings.translation.enableAlignment')}</span>
                      <p className="text-xs text-gray-500">{t('settings.translation.enableAlignmentDesc')}</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={localSettings.translation.enable_alignment ?? false}
                        onChange={(e) => handleSettingChange('translation', 'enable_alignment', e.target.checked)}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </label>
                  </div>
                </div>
              )}

              {/* Post-processing Settings */}
              <div className="border-t pt-4 mt-4">
                <h4 className="text-sm font-medium text-gray-900 mb-3">{t('settings.translation.postProcessing')}</h4>
                
                {/* Length Control */}
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <span className="text-sm font-medium text-gray-700">{t('settings.translation.lengthControl')}</span>
                    <p className="text-xs text-gray-500">{t('settings.translation.lengthControlDesc')}</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={localSettings.translation.enable_length_control ?? true}
                      onChange={(e) => handleSettingChange('translation', 'enable_length_control', e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                {/* Max chars per line (only show if length control enabled) */}
                {(localSettings.translation.enable_length_control ?? true) && (
                  <div className="flex items-center justify-between mb-3 ml-4">
                    <span className="text-sm text-gray-600">{t('settings.translation.maxCharsPerLine')}</span>
                    <input
                      type="number"
                      value={localSettings.translation.max_chars_per_line ?? 42}
                      onChange={(e) => handleSettingChange('translation', 'max_chars_per_line', parseInt(e.target.value))}
                      min={20}
                      max={80}
                      className="w-20 px-2 py-1 border border-gray-300 rounded text-sm"
                    />
                  </div>
                )}

                {/* Localization */}
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <span className="text-sm font-medium text-gray-700">{t('settings.translation.localization')}</span>
                    <p className="text-xs text-gray-500">{t('settings.translation.localizationDesc')}</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={localSettings.translation.enable_localization ?? true}
                      onChange={(e) => handleSettingChange('translation', 'enable_localization', e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                {/* Custom Glossary */}
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <span className="text-sm font-medium text-gray-700">{t('settings.translation.useGlossary')}</span>
                    <p className="text-xs text-gray-500">{t('settings.translation.useGlossaryDesc')}</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={localSettings.translation.use_custom_glossary ?? true}
                      onChange={(e) => handleSettingChange('translation', 'use_custom_glossary', e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                {/* Manage Glossary Link */}
                <Link
                  to="/glossary"
                  className="inline-flex items-center text-sm text-blue-600 hover:text-blue-700 ml-4"
                >
                  {t('settings.translation.manageGlossary')} →
                </Link>
              </div>
            </div>
          </div>

          {/* TTS Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.tts.title')}</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.tts.engine')}
                </label>
                <select
                  value={localSettings.tts.engine}
                  onChange={(e) => {
                    handleSettingChange('tts', 'engine', e.target.value)
                    // Reset voice when engine changes
                    handleSettingChange('tts', 'voice', '')
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  {ttsEngines?.map((engine: TTSEngine) => (
                    <option key={engine.id} value={engine.id}>
                      {engine.name} {engine.free ? t('settings.translation.free') : ''}
                    </option>
                  ))}
                </select>
                {/* Engine description */}
                {ttsEngines?.find((e: TTSEngine) => e.id === localSettings.tts.engine) && (
                  <p className="mt-1 text-xs text-gray-500">
                    {ttsEngines.find((e: TTSEngine) => e.id === localSettings.tts.engine)?.description}
                  </p>
                )}
              </div>

              {/* Voice Cloning Mode */}
              {ttsEngines?.find((e: TTSEngine) => e.id === localSettings.tts.engine)?.supports_voice_cloning && (
                <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                  <label className="block text-sm font-medium text-purple-800 mb-2">
                    {t('settings.tts.voiceCloningMode')}
                  </label>
                  <div className="space-y-2">
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="voice_cloning_mode"
                        value="disabled"
                        checked={(localSettings.tts.voice_cloning_mode || 'disabled') === 'disabled'}
                        onChange={() => handleSettingChange('tts', 'voice_cloning_mode', 'disabled')}
                        className="text-purple-600"
                      />
                      <span className="ml-2 text-sm text-gray-700">{t('settings.tts.cloningDisabled')}</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="voice_cloning_mode"
                        value="video_audio"
                        checked={localSettings.tts.voice_cloning_mode === 'video_audio'}
                        onChange={() => handleSettingChange('tts', 'voice_cloning_mode', 'video_audio')}
                        className="text-purple-600"
                      />
                      <span className="ml-2 text-sm text-gray-700">{t('settings.tts.cloningVideoAudio')}</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="voice_cloning_mode"
                        value="custom"
                        checked={localSettings.tts.voice_cloning_mode === 'custom'}
                        onChange={() => handleSettingChange('tts', 'voice_cloning_mode', 'custom')}
                        className="text-purple-600"
                      />
                      <span className="ml-2 text-sm text-gray-700">{t('settings.tts.cloningCustom')}</span>
                    </label>
                  </div>
                  <p className="mt-2 text-xs text-purple-600">
                    {localSettings.tts.voice_cloning_mode === 'video_audio'
                      ? t('settings.tts.cloningVideoAudioDesc')
                      : localSettings.tts.voice_cloning_mode === 'custom'
                        ? t('settings.tts.cloningCustomDesc')
                        : t('settings.tts.cloningDisabledDesc')}
                  </p>

                  {/* Custom Reference Audio Upload */}
                  {localSettings.tts.voice_cloning_mode === 'custom' && (
                    <div className="mt-3 space-y-3 pt-3 border-t border-purple-200">
                      <div>
                        <label className="block text-sm font-medium text-purple-800 mb-1">
                          {t('settings.tts.refAudio')}
                        </label>
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={localSettings.tts.ref_audio_path || ''}
                            onChange={(e) => handleSettingChange('tts', 'ref_audio_path', e.target.value)}
                            placeholder={t('settings.tts.refAudioPlaceholder')}
                            className="flex-1 px-3 py-2 border border-purple-300 rounded-md text-sm"
                          />
                          <label className="px-3 py-2 bg-purple-600 text-white rounded-md cursor-pointer hover:bg-purple-700 text-sm">
                            <Upload className="w-4 h-4 inline mr-1" />
                            {t('common.upload')}
                            <input
                              type="file"
                              accept="audio/*"
                              className="hidden"
                              onChange={async (e) => {
                                const file = e.target.files?.[0]
                                if (file) {
                                  // TODO: Upload file and get path
                                  console.log('Upload ref audio:', file.name)
                                }
                              }}
                            />
                          </label>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-purple-800 mb-1">
                          {t('settings.tts.refText')}
                        </label>
                        <textarea
                          value={localSettings.tts.ref_audio_text || ''}
                          onChange={(e) => handleSettingChange('tts', 'ref_audio_text', e.target.value)}
                          placeholder={t('settings.tts.refTextPlaceholder')}
                          rows={2}
                          className="w-full px-3 py-2 border border-purple-300 rounded-md text-sm"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Default Voice (only shown when voice cloning is disabled) */}
              {(localSettings.tts.voice_cloning_mode || 'disabled') === 'disabled' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t('settings.tts.voice')}
                  </label>
                  <select
                    value={localSettings.tts.voice}
                    onChange={(e) => handleSettingChange('tts', 'voice', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                    disabled={voicesLoading}
                  >
                    {voicesLoading ? (
                      <option>{t('common.loading')}...</option>
                    ) : voices?.length ? (
                      voices.map((voice) => (
                        <option key={voice.name} value={voice.name}>
                          {voice.display_name} ({voice.gender})
                        </option>
                      ))
                    ) : (
                      <option>{t('settings.tts.noVoices')}</option>
                    )}
                  </select>
                </div>
              )}

              {/* TTS API key for engines that require it */}
              {ttsEngines?.find((e: TTSEngine) => e.id === localSettings.tts.engine)?.requires_api_key && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t('settings.translation.apiKey')}
                  </label>
                  <div className="flex">
                    <input
                      type={showSecrets['tts_api_key'] ? 'text' : 'password'}
                      value={localSettings.tts.api_key || ''}
                      onChange={(e) => handleSettingChange('tts', 'api_key', e.target.value)}
                      placeholder={t('settings.translation.apiKeyPlaceholder', { provider: 'API' })}
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-l-md font-mono text-sm"
                    />
                    <button
                      type="button"
                      onClick={() => toggleShowSecret('tts_api_key')}
                      className="px-3 py-2 border border-l-0 border-gray-300 rounded-r-md bg-gray-50 hover:bg-gray-100"
                    >
                      {showSecrets['tts_api_key'] ? (
                        <EyeOff className="h-4 w-4 text-gray-500" />
                      ) : (
                        <Eye className="h-4 w-4 text-gray-500" />
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Subtitle Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.subtitle.title')}</h3>
            <div className="space-y-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.subtitle.enabled}
                  onChange={(e) => handleSettingChange('subtitle', 'enabled', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.subtitle.enabled')}</span>
              </label>

              {/* Default Subtitle Preset Selection */}
              {localSettings.subtitle.enabled && (
                <div className="mt-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t('settings.subtitle.defaultPreset')}
                  </label>
                  <select
                    value={localSettings.subtitle.default_preset || ''}
                    onChange={(e) => handleSettingChange('subtitle', 'default_preset', e.target.value || undefined)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  >
                    <option value="">{t('newTask.defaultPreset')}</option>
                    {presetsData?.presets?.map((preset) => (
                      <option key={preset.id} value={preset.id}>
                        {preset.name} {preset.is_builtin ? `(${t('styleEditor.builtIn')})` : ''}
                      </option>
                    ))}
                  </select>
                  <p className="mt-1 text-xs text-gray-500">{t('settings.subtitle.defaultPresetDesc')}</p>
                  <Link
                    to="/presets"
                    className="mt-2 inline-flex items-center text-sm text-blue-600 hover:text-blue-700"
                  >
                    {t('settings.subtitle.managePresets')} →
                  </Link>
                </div>
              )}
            </div>
          </div>

          {/* Audio Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.audio.title')}</h3>
            <div className="space-y-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.audio.generate_tts}
                  onChange={(e) => handleSettingChange('audio', 'generate_tts', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.audio.generateTts')}</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.audio.replace_original}
                  onChange={(e) => handleSettingChange('audio', 'replace_original', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!localSettings.audio.generate_tts}
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.audio.replaceOriginal')}</span>
              </label>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t('settings.audio.originalVolume')} ({Math.round(localSettings.audio.original_volume * 100)}%)
                  </label>
                  <input
                    type="range"
                    value={localSettings.audio.original_volume}
                    onChange={(e) => handleSettingChange('audio', 'original_volume', parseFloat(e.target.value))}
                    min={0}
                    max={1}
                    step={0.1}
                    className="w-full"
                    disabled={localSettings.audio.replace_original}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t('settings.audio.ttsVolume')} ({Math.round(localSettings.audio.tts_volume * 100)}%)
                  </label>
                  <input
                    type="range"
                    value={localSettings.audio.tts_volume}
                    onChange={(e) => handleSettingChange('audio', 'tts_volume', parseFloat(e.target.value))}
                    min={0}
                    max={1}
                    step={0.1}
                    className="w-full"
                    disabled={!localSettings.audio.generate_tts}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI Tab - Metadata, Thumbnail, Proofreading */}
      {activeTab === 'ai' && localSettings && (
        <div className="space-y-6">
          {/* AI Metadata Generation Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.metadata.title')}</h3>
            <p className="text-sm text-gray-500 mb-4">{t('settings.metadata.description')}</p>
            <div className="space-y-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.metadata?.enabled ?? true}
                  onChange={(e) => handleSettingChange('metadata', 'enabled', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.metadata.enabled')}</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.metadata?.auto_generate ?? false}
                  onChange={(e) => handleSettingChange('metadata', 'auto_generate', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!localSettings.metadata?.enabled}
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.metadata.autoGenerate')}</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.metadata?.include_source_url ?? true}
                  onChange={(e) => handleSettingChange('metadata', 'include_source_url', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!localSettings.metadata?.enabled}
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.metadata.includeSourceUrl')}</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.metadata?.require_review ?? true}
                  onChange={(e) => handleSettingChange('metadata', 'require_review', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!localSettings.metadata?.enabled}
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.metadata.requireReview')}</span>
              </label>
              <p className="text-xs text-gray-500 ml-6 -mt-1">{t('settings.metadata.requireReviewHelp')}</p>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.metadata?.default_use_ai_preset_selection ?? false}
                  onChange={(e) => handleSettingChange('metadata', 'default_use_ai_preset_selection', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!localSettings.metadata?.enabled}
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.metadata.defaultUseAiPreset')}</span>
              </label>
              <p className="text-xs text-gray-500 ml-6 -mt-1">{t('settings.metadata.defaultUseAiPresetHelp')}</p>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.metadata.maxKeywords')}
                </label>
                <input
                  type="number"
                  value={localSettings.metadata?.max_keywords ?? 10}
                  onChange={(e) => handleSettingChange('metadata', 'max_keywords', parseInt(e.target.value))}
                  min={1}
                  max={20}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  disabled={!localSettings.metadata?.enabled}
                />
              </div>
            </div>
          </div>

          {/* AI Thumbnail Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.thumbnail.title')}</h3>
            <p className="text-sm text-gray-500 mb-4">{t('settings.thumbnail.description')}</p>
            <div className="space-y-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.thumbnail?.enabled ?? false}
                  onChange={(e) => handleSettingChange('thumbnail', 'enabled', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.thumbnail.enabled')}</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.thumbnail?.auto_generate ?? true}
                  onChange={(e) => handleSettingChange('thumbnail', 'auto_generate', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!localSettings.thumbnail?.enabled}
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.thumbnail.autoGenerate')}</span>
              </label>
              <p className="text-xs text-gray-500 ml-6 -mt-2">{t('settings.thumbnail.autoGenerateHelp')}</p>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.thumbnail?.default_use_ai ?? false}
                  onChange={(e) => handleSettingChange('thumbnail', 'default_use_ai', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!localSettings.thumbnail?.enabled}
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.thumbnail.defaultUseAi')}</span>
              </label>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('settings.thumbnail.style')}
                </label>
                <select
                  value={localSettings.thumbnail?.style ?? 'gradient_bar'}
                  onChange={(e) => handleSettingChange('thumbnail', 'style', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  disabled={!localSettings.thumbnail?.enabled}
                >
                  <option value="gradient_bar">{t('settings.thumbnail.styles.gradientBar')}</option>
                  <option value="top_banner">{t('settings.thumbnail.styles.topBanner')}</option>
                  <option value="corner_tag">{t('settings.thumbnail.styles.cornerTag')}</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('settings.thumbnail.fontSize')}
                </label>
                <input
                  type="number"
                  value={localSettings.thumbnail?.font_size ?? 72}
                  onChange={(e) => handleSettingChange('thumbnail', 'font_size', parseInt(e.target.value))}
                  min={24}
                  max={120}
                  className="w-32 px-3 py-2 border border-gray-300 rounded-md text-sm"
                  disabled={!localSettings.thumbnail?.enabled}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {t('settings.thumbnail.textColor')}
                  </label>
                  <input
                    type="color"
                    value={localSettings.thumbnail?.text_color ?? '#FFFFFF'}
                    onChange={(e) => handleSettingChange('thumbnail', 'text_color', e.target.value)}
                    className="w-16 h-10 border border-gray-300 rounded-md cursor-pointer"
                    disabled={!localSettings.thumbnail?.enabled}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {t('settings.thumbnail.gradientColor')}
                  </label>
                  <input
                    type="color"
                    value={localSettings.thumbnail?.gradient_color ?? '#000000'}
                    onChange={(e) => handleSettingChange('thumbnail', 'gradient_color', e.target.value)}
                    className="w-16 h-10 border border-gray-300 rounded-md cursor-pointer"
                    disabled={!localSettings.thumbnail?.enabled}
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('settings.thumbnail.gradientOpacity')}
                </label>
                <input
                  type="range"
                  value={(localSettings.thumbnail?.gradient_opacity ?? 0.7) * 100}
                  onChange={(e) => handleSettingChange('thumbnail', 'gradient_opacity', parseInt(e.target.value) / 100)}
                  min={0}
                  max={100}
                  className="w-full"
                  disabled={!localSettings.thumbnail?.enabled}
                />
                <p className="text-xs text-gray-500 mt-1">
                  {Math.round((localSettings.thumbnail?.gradient_opacity ?? 0.7) * 100)}%
                </p>
              </div>
            </div>
          </div>

          {/* AI Proofreading Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.proofreading.title')}</h3>
            <p className="text-sm text-gray-500 mb-4">{t('settings.proofreading.description')}</p>
            <div className="space-y-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.proofreading?.enabled ?? true}
                  onChange={(e) => handleSettingChange('proofreading', 'enabled', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.proofreading.enabled')}</span>
              </label>
              <p className="text-xs text-gray-500 ml-6 -mt-2">{t('settings.proofreading.enabledHelp')}</p>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.proofreading?.auto_pause ?? true}
                  onChange={(e) => handleSettingChange('proofreading', 'auto_pause', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!localSettings.proofreading?.enabled}
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.proofreading.autoPause')}</span>
              </label>
              <p className="text-xs text-gray-500 ml-6 -mt-2">{t('settings.proofreading.autoPauseHelp')}</p>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('settings.proofreading.minConfidence')}
                </label>
                <input
                  type="range"
                  value={(localSettings.proofreading?.min_confidence ?? 0.6) * 100}
                  onChange={(e) => handleSettingChange('proofreading', 'min_confidence', parseInt(e.target.value) / 100)}
                  min={0}
                  max={100}
                  className="w-full"
                  disabled={!localSettings.proofreading?.enabled || !localSettings.proofreading?.auto_pause}
                />
                <p className="text-xs text-gray-500 mt-1">
                  {Math.round((localSettings.proofreading?.min_confidence ?? 0.6) * 100)}% - {t('settings.proofreading.minConfidenceHelp')}
                </p>
              </div>

              <div className="border-t border-gray-200 pt-4 mt-4">
                <p className="text-sm font-medium text-gray-700 mb-3">校对检查项</p>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={localSettings.proofreading?.check_grammar ?? true}
                      onChange={(e) => handleSettingChange('proofreading', 'check_grammar', e.target.checked)}
                      className="rounded border-gray-300 text-blue-600"
                      disabled={!localSettings.proofreading?.enabled}
                    />
                    <span className="ml-2 text-sm text-gray-700">{t('settings.proofreading.checkGrammar')}</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={localSettings.proofreading?.check_terminology ?? true}
                      onChange={(e) => handleSettingChange('proofreading', 'check_terminology', e.target.checked)}
                      className="rounded border-gray-300 text-blue-600"
                      disabled={!localSettings.proofreading?.enabled}
                    />
                    <span className="ml-2 text-sm text-gray-700">{t('settings.proofreading.checkTerminology')}</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={localSettings.proofreading?.check_timing ?? true}
                      onChange={(e) => handleSettingChange('proofreading', 'check_timing', e.target.checked)}
                      className="rounded border-gray-300 text-blue-600"
                      disabled={!localSettings.proofreading?.enabled}
                    />
                    <span className="ml-2 text-sm text-gray-700">{t('settings.proofreading.checkTiming')}</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={localSettings.proofreading?.check_formatting ?? true}
                      onChange={(e) => handleSettingChange('proofreading', 'check_formatting', e.target.checked)}
                      className="rounded border-gray-300 text-blue-600"
                      disabled={!localSettings.proofreading?.enabled}
                    />
                    <span className="ml-2 text-sm text-gray-700">{t('settings.proofreading.checkFormatting')}</span>
                  </label>
                </div>
              </div>

              <label className="flex items-center mt-4">
                <input
                  type="checkbox"
                  checked={localSettings.proofreading?.use_ai_validation ?? true}
                  onChange={(e) => handleSettingChange('proofreading', 'use_ai_validation', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                  disabled={!localSettings.proofreading?.enabled}
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.proofreading.useAiValidation')}</span>
              </label>
              <p className="text-xs text-gray-500 ml-6 -mt-2">{t('settings.proofreading.useAiValidationHelp')}</p>

              {/* AI Optimization Settings */}
              <div className="border-t border-gray-200 pt-4 mt-4">
                <p className="text-sm font-medium text-gray-700 mb-3">{t('settings.proofreading.optimizationTitle')}</p>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={localSettings.proofreading?.auto_optimize ?? false}
                    onChange={(e) => handleSettingChange('proofreading', 'auto_optimize', e.target.checked)}
                    className="rounded border-gray-300 text-blue-600"
                    disabled={!localSettings.proofreading?.enabled}
                  />
                  <span className="ml-2 text-sm text-gray-700">{t('settings.proofreading.autoOptimize')}</span>
                </label>
                <p className="text-xs text-gray-500 ml-6 mt-1">{t('settings.proofreading.autoOptimizeHelp')}</p>

                <div className="mt-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {t('settings.proofreading.optimizationLevel')}
                  </label>
                  <select
                    value={localSettings.proofreading?.optimization_level ?? 'moderate'}
                    onChange={(e) => handleSettingChange('proofreading', 'optimization_level', e.target.value)}
                    className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                    disabled={!localSettings.proofreading?.enabled}
                  >
                    <option value="minimal">{t('settings.proofreading.levelMinimal')}</option>
                    <option value="moderate">{t('settings.proofreading.levelModerate')}</option>
                    <option value="aggressive">{t('settings.proofreading.levelAggressive')}</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">{t('settings.proofreading.optimizationLevelHelp')}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Publish Tab - Metadata Presets, Bilibili, Upload Platforms */}
      {activeTab === 'publish' && localSettings && (
        <div className="space-y-6">
          {/* Metadata Presets Section */}
          <MetadataPresetsSection />

          {/* Bilibili Upload Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.bilibili.title')}</h3>
            <p className="text-sm text-gray-500 mb-4">{t('settings.bilibili.description')}</p>
            <div className="space-y-4">
              {/* 自制 vs 转载 */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('settings.bilibili.copyrightType')}
                </label>
                <div className="flex space-x-4">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="bilibili_copyright"
                      checked={localSettings.bilibili?.is_original === true}
                      onChange={() => handleSettingChange('bilibili', 'is_original', true)}
                      className="border-gray-300 text-blue-600"
                    />
                    <span className="ml-2 text-sm text-gray-700">{t('settings.bilibili.original')}</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="bilibili_copyright"
                      checked={localSettings.bilibili?.is_original !== true}
                      onChange={() => handleSettingChange('bilibili', 'is_original', false)}
                      className="border-gray-300 text-blue-600"
                    />
                    <span className="ml-2 text-sm text-gray-700">{t('settings.bilibili.repost')}</span>
                  </label>
                </div>
                <p className="mt-1 text-xs text-gray-500">{t('settings.bilibili.copyrightHelp')}</p>
              </div>

              {/* AI智能匹配分区 */}
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.bilibili?.auto_match_partition ?? true}
                  onChange={(e) => handleSettingChange('bilibili', 'auto_match_partition', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.bilibili.autoMatchPartition')}</span>
              </label>
              <p className="text-xs text-gray-500 ml-6 -mt-2">{t('settings.bilibili.autoMatchPartitionHelp')}</p>

              {/* 默认分区选择 */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('settings.bilibili.defaultPartition')}
                </label>
                <select
                  value={localSettings.bilibili?.default_tid ?? 0}
                  onChange={(e) => handleSettingChange('bilibili', 'default_tid', parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  disabled={localSettings.bilibili?.auto_match_partition}
                >
                  <option value={0}>{t('settings.bilibili.autoMatch')}</option>
                  <option value={36}>{t('settings.bilibili.partitions.knowledge')}</option>
                  <option value={188}>{t('settings.bilibili.partitions.tech')}</option>
                  <option value={160}>{t('settings.bilibili.partitions.life')}</option>
                  <option value={4}>{t('settings.bilibili.partitions.gaming')}</option>
                  <option value={5}>{t('settings.bilibili.partitions.entertainment')}</option>
                  <option value={3}>{t('settings.bilibili.partitions.music')}</option>
                  <option value={129}>{t('settings.bilibili.partitions.dance')}</option>
                  <option value={1}>{t('settings.bilibili.partitions.animation')}</option>
                  <option value={181}>{t('settings.bilibili.partitions.film')}</option>
                  <option value={211}>{t('settings.bilibili.partitions.food')}</option>
                  <option value={217}>{t('settings.bilibili.partitions.animals')}</option>
                  <option value={223}>{t('settings.bilibili.partitions.car')}</option>
                  <option value={234}>{t('settings.bilibili.partitions.sports')}</option>
                  <option value={155}>{t('settings.bilibili.partitions.fashion')}</option>
                </select>
                <p className="mt-1 text-xs text-gray-500">{t('settings.bilibili.defaultPartitionHelp')}</p>
              </div>
            </div>
          </div>

          {/* Default Upload Platforms */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.sections.defaultUpload')}</h3>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.auto_upload_bilibili}
                  onChange={(e) => handleTopLevelChange('auto_upload_bilibili', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.upload.bilibili')}</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.auto_upload_douyin}
                  onChange={(e) => handleTopLevelChange('auto_upload_douyin', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.upload.douyin')}</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings.auto_upload_xiaohongshu}
                  onChange={(e) => handleTopLevelChange('auto_upload_xiaohongshu', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('settings.upload.xiaohongshu')}</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Platforms Tab */}
      {activeTab === 'platforms' && (
        <div className="space-y-6">
          {/* Bilibili Multi-Account Section */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <BilibiliAccounts />
          </div>

          {/* Douyin Multi-Account Section */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <DouyinAccounts />
          </div>

          {/* Xiaohongshu Multi-Account Section */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <XiaohongshuAccounts />
          </div>

          {platforms.map((config) => {
            const status = platformStatus?.[config.platform as keyof typeof platformStatus]

            return (
              <div
                key={config.platform}
                className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-gray-900">{t(config.label)}</h3>
                  <div className="flex items-center space-x-2">
                    {status?.configured ? (
                      <span className="flex items-center text-green-600 text-sm">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        {t('common.configured')}
                      </span>
                    ) : (
                      <span className="flex items-center text-gray-400 text-sm">
                        <XCircle className="h-4 w-4 mr-1" />
                        {t('common.notConfigured')}
                      </span>
                    )}
                  </div>
                </div>

                <div className="space-y-4">
                  {config.fields.map((field) => (
                    <div key={field.key}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {field.label}
                      </label>
                      <div className="relative">
                        {field.type === 'textarea' ? (
                          <textarea
                            value={credentials[config.platform]?.[field.key] || ''}
                            onChange={(e) =>
                              setCredentials((prev) => ({
                                ...prev,
                                [config.platform]: {
                                  ...prev[config.platform],
                                  [field.key]: e.target.value,
                                },
                              }))
                            }
                            placeholder={t(field.placeholder)}
                            rows={3}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
                          />
                        ) : (
                          <div className="flex">
                            <input
                              type={showSecrets[`${config.platform}-${field.key}`] ? 'text' : 'password'}
                              value={credentials[config.platform]?.[field.key] || ''}
                              onChange={(e) =>
                                setCredentials((prev) => ({
                                  ...prev,
                                  [config.platform]: {
                                    ...prev[config.platform],
                                    [field.key]: e.target.value,
                                  },
                                }))
                              }
                              placeholder={t(field.placeholder)}
                              className="flex-1 px-3 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
                            />
                            <button
                              type="button"
                              onClick={() => toggleShowSecret(`${config.platform}-${field.key}`)}
                              className="px-3 py-2 border border-l-0 border-gray-300 rounded-r-md bg-gray-50 hover:bg-gray-100"
                            >
                              {showSecrets[`${config.platform}-${field.key}`] ? (
                                <EyeOff className="h-4 w-4 text-gray-500" />
                              ) : (
                                <Eye className="h-4 w-4 text-gray-500" />
                              )}
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Auto Extract Section - Always visible */}
                <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-blue-800">{t('settings.platforms.autoExtract.title')}</p>
                      <p className="text-xs text-blue-600 mt-0.5">{t('settings.platforms.autoExtract.desc')}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      {platformCookieStatus?.in_docker ? (
                        <span className="text-xs text-gray-500">{t('settings.platforms.autoExtract.dockerNotSupported')}</span>
                      ) : platformCookieStatus?.available_browsers && platformCookieStatus.available_browsers.length > 0 ? (
                        <>
                          <select
                            className="px-2 py-1.5 border border-blue-300 rounded-md text-sm bg-white"
                            defaultValue={platformCookieStatus.available_browsers[0]}
                            id={`browser-select-${config.platform}`}
                          >
                            {platformCookieStatus.available_browsers.map((browser) => (
                              <option key={browser} value={browser}>
                                {browser.charAt(0).toUpperCase() + browser.slice(1)}
                              </option>
                            ))}
                          </select>
                          <button
                            onClick={() => {
                              const browserSelect = document.getElementById(`browser-select-${config.platform}`) as HTMLSelectElement
                              const browser = browserSelect?.value || 'chrome'
                              extractPlatformCookiesMutation.mutate({ platform: config.platform, browser })
                            }}
                            disabled={extractPlatformCookiesMutation.isPending}
                            className="px-3 py-1.5 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 flex items-center"
                          >
                            {extractPlatformCookiesMutation.isPending ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <>
                                <Cookie className="h-4 w-4 mr-1" />
                                {t('settings.platforms.autoExtract.button')}
                              </>
                            )}
                          </button>
                        </>
                      ) : (
                        <span className="text-xs text-gray-500">{t('settings.platforms.autoExtract.noBrowser')}</span>
                      )}
                    </div>
                  </div>
                  {extractPlatformCookiesMutation.isSuccess && extractPlatformCookiesMutation.variables?.platform === config.platform && (
                    <div className="mt-2 p-2 bg-green-100 text-green-700 rounded text-xs flex items-center">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      {extractPlatformCookiesMutation.data?.message || t('settings.platforms.autoExtract.success')}
                    </div>
                  )}
                  {extractPlatformCookiesMutation.isError && extractPlatformCookiesMutation.variables?.platform === config.platform && (
                    <div className="mt-2 p-2 bg-red-100 text-red-700 rounded text-xs flex items-center">
                      <XCircle className="h-3 w-3 mr-1" />
                      {(extractPlatformCookiesMutation.error as Error)?.message || t('settings.platforms.autoExtract.failed')}
                    </div>
                  )}
                </div>

                <div className="mt-4 flex items-center justify-between">
                  <p className="text-xs text-gray-500">
                    {t('settings.platforms.cookieHelp.other')}
                  </p>
                  <button
                    onClick={() => handleAuthenticate(config.platform)}
                    disabled={authMutation.isPending}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
                  >
                    {authMutation.isPending ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin inline mr-2" />
                        {t('common.verifying')}
                      </>
                    ) : (
                      t('settings.platforms.verifyAndSave')
                    )}
                  </button>
                </div>
              </div>
            )
          })}

          {/* Help Section */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="text-sm font-medium text-gray-900 mb-2">{t('settings.platforms.cookieHelp.title')}</h3>
            <ol className="text-sm text-gray-600 space-y-1 list-decimal list-inside">
              <li>{t('settings.platforms.cookieHelp.step1')}</li>
              <li>{t('settings.platforms.cookieHelp.step2')}</li>
              <li>{t('settings.platforms.cookieHelp.step3')}</li>
              <li>{t('settings.platforms.cookieHelp.step4')}</li>
              <li>{t('settings.platforms.cookieHelp.step5')}</li>
            </ol>
          </div>
        </div>
      )}

      {/* YouTube Cookies Tab */}
      {activeTab === 'youtube' && (
        <div className="space-y-6">
          {/* YouTube API Key */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
              <Youtube className="h-5 w-5 mr-2 text-red-500" />
              {t('settings.youtube.apiKey')}
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              {t('settings.youtube.apiKeyDesc')}
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  YouTube Data API v3 Key
                </label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <input
                      type={showSecrets['youtube_api_key'] ? 'text' : 'password'}
                      value={localSettings?.trending?.youtube_api_key || ''}
                      onChange={(e) => handleSettingChange('trending', 'youtube_api_key', e.target.value)}
                      placeholder={t('settings.youtube.apiKeyPlaceholder')}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md font-mono text-sm pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => toggleShowSecret('youtube_api_key')}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700"
                    >
                      {showSecrets['youtube_api_key'] ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {/* API Key status indicator */}
              <div className="flex items-center gap-2 text-sm">
                {localSettings?.trending?.youtube_api_key ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="text-green-700">{t('settings.youtube.apiKeyConfigured')}</span>
                  </>
                ) : (
                  <>
                    <XCircle className="h-4 w-4 text-gray-400" />
                    <span className="text-gray-500">{t('settings.youtube.apiKeyNotConfigured')}</span>
                  </>
                )}
              </div>

              <p className="text-xs text-gray-500">
                {t('settings.youtube.apiKeyHelp')}
                <a
                  href="https://console.cloud.google.com/apis/credentials"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline ml-1"
                >
                  Google Cloud Console
                </a>
              </p>
            </div>
          </div>

          {/* Trending Videos Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('discover.settings.title')}</h3>
            <div className="space-y-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={localSettings?.trending?.enabled ?? true}
                  onChange={(e) => handleSettingChange('trending', 'enabled', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{t('discover.settings.enabled')}</span>
              </label>
              <p className="text-xs text-gray-500 ml-6 -mt-2">{t('discover.settings.enabledDesc')}</p>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('discover.settings.updateInterval')}
                </label>
                <select
                  value={localSettings?.trending?.update_interval ?? 60}
                  onChange={(e) => handleSettingChange('trending', 'update_interval', parseInt(e.target.value))}
                  className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.trending?.enabled}
                >
                  <option value={30}>{t('discover.settings.intervalOptions.30')}</option>
                  <option value={60}>{t('discover.settings.intervalOptions.60')}</option>
                  <option value={120}>{t('discover.settings.intervalOptions.120')}</option>
                  <option value={360}>{t('discover.settings.intervalOptions.360')}</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('discover.settings.categories')}
                </label>
                <div className="space-y-2">
                  {['tech', 'gaming', 'lifestyle', 'music', 'entertainment', 'education', 'news', 'howto', 'comedy', 'film', 'sports'].map(cat => (
                    <label key={cat} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={(localSettings?.trending?.enabled_categories ?? ['tech', 'gaming', 'lifestyle']).includes(cat)}
                        onChange={(e) => {
                          const current = localSettings?.trending?.enabled_categories ?? ['tech', 'gaming', 'lifestyle']
                          const updated = e.target.checked
                            ? [...current, cat]
                            : current.filter(c => c !== cat)
                          handleSettingChange('trending', 'enabled_categories', updated)
                        }}
                        className="rounded border-gray-300 text-blue-600"
                        disabled={!localSettings?.trending?.enabled}
                      />
                      <span className="ml-2 text-sm text-gray-700">
                        {t(`discover.categories.${cat}`)}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('discover.settings.maxVideos')}
                </label>
                <input
                  type="number"
                  value={localSettings?.trending?.max_videos_per_category ?? 20}
                  onChange={(e) => handleSettingChange('trending', 'max_videos_per_category', parseInt(e.target.value))}
                  min={5}
                  max={50}
                  className="w-24 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.trending?.enabled}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('discover.settings.timeFilter')}
                </label>
                <select
                  value={localSettings?.trending?.time_filter ?? 'week'}
                  onChange={(e) => handleSettingChange('trending', 'time_filter', e.target.value)}
                  className="w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.trending?.enabled}
                >
                  <option value="hour">{t('discover.settings.timeOptions.hour')}</option>
                  <option value="today">{t('discover.settings.timeOptions.today')}</option>
                  <option value="week">{t('discover.settings.timeOptions.week')}</option>
                  <option value="month">{t('discover.settings.timeOptions.month')}</option>
                  <option value="year">{t('discover.settings.timeOptions.year')}</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('discover.settings.sortBy')}
                </label>
                <select
                  value={localSettings?.trending?.sort_by ?? 'view_count'}
                  onChange={(e) => handleSettingChange('trending', 'sort_by', e.target.value)}
                  className="w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.trending?.enabled}
                >
                  <option value="relevance">{t('discover.settings.sortOptions.relevance')}</option>
                  <option value="upload_date">{t('discover.settings.sortOptions.upload_date')}</option>
                  <option value="view_count">{t('discover.settings.sortOptions.view_count')}</option>
                  <option value="rating">{t('discover.settings.sortOptions.rating')}</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('discover.settings.minViewCount')}
                </label>
                <input
                  type="number"
                  value={localSettings?.trending?.min_view_count ?? 10000}
                  onChange={(e) => handleSettingChange('trending', 'min_view_count', parseInt(e.target.value))}
                  min={0}
                  max={10000000}
                  step={1000}
                  className="w-32 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.trending?.enabled}
                />
                <p className="text-xs text-gray-500 mt-1">{t('discover.settings.minViewCountDesc')}</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('discover.settings.maxDuration')}
                </label>
                <select
                  value={localSettings?.trending?.max_duration ?? 1800}
                  onChange={(e) => handleSettingChange('trending', 'max_duration', parseInt(e.target.value))}
                  className="w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.trending?.enabled}
                >
                  <option value={300}>{t('discover.settings.durationOptions.5min')}</option>
                  <option value={600}>{t('discover.settings.durationOptions.10min')}</option>
                  <option value={900}>{t('discover.settings.durationOptions.15min')}</option>
                  <option value={1200}>{t('discover.settings.durationOptions.20min')}</option>
                  <option value={1800}>{t('discover.settings.durationOptions.30min')}</option>
                  <option value={3600}>{t('discover.settings.durationOptions.60min')}</option>
                  <option value={0}>{t('discover.settings.durationOptions.unlimited')}</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">{t('discover.settings.maxDurationDesc')}</p>
              </div>

              {/* Exclude Shorts */}
              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    {t('discover.settings.excludeShorts', '排除短视频')}
                  </label>
                  <p className="text-xs text-gray-500 mt-1">
                    {t('discover.settings.excludeShortsDesc', '排除 YouTube Shorts（竖屏短视频）')}
                  </p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={localSettings?.trending?.exclude_shorts ?? true}
                    onChange={(e) => handleSettingChange('trending', 'exclude_shorts', e.target.checked)}
                    className="sr-only peer"
                    disabled={!localSettings?.trending?.enabled}
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600 peer-disabled:opacity-50"></div>
                </label>
              </div>
            </div>
          </div>

          {/* TikTok Discovery Settings */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-medium text-gray-900">TikTok 发现</h3>
                <p className="text-sm text-gray-500 mt-1">配置 TikTok 热门视频发现功能</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={localSettings?.tiktok?.enabled ?? false}
                  onChange={(e) => handleSettingChange('tiktok', 'enabled', e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  地区
                </label>
                <select
                  value={localSettings?.tiktok?.region_code ?? 'US'}
                  onChange={(e) => handleSettingChange('tiktok', 'region_code', e.target.value)}
                  className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.tiktok?.enabled}
                >
                  <option value="US">美国</option>
                  <option value="CN">中国</option>
                  <option value="JP">日本</option>
                  <option value="KR">韩国</option>
                  <option value="UK">英国</option>
                  <option value="DE">德国</option>
                  <option value="FR">法国</option>
                  <option value="TW">台湾</option>
                  <option value="HK">香港</option>
                  <option value="SG">新加坡</option>
                  <option value="TH">泰国</option>
                  <option value="VN">越南</option>
                  <option value="ID">印尼</option>
                  <option value="PH">菲律宾</option>
                  <option value="MY">马来西亚</option>
                  <option value="BR">巴西</option>
                  <option value="IN">印度</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  更新间隔（分钟）
                </label>
                <input
                  type="number"
                  value={localSettings?.tiktok?.update_interval ?? 60}
                  onChange={(e) => handleSettingChange('tiktok', 'update_interval', parseInt(e.target.value))}
                  min={15}
                  max={1440}
                  className="w-24 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.tiktok?.enabled}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  启用的标签
                </label>
                <div className="flex flex-wrap gap-2">
                  {['trending', 'fyp', 'viral', 'foryou', 'comedy', 'dance', 'music', 'food', 'travel', 'tech', 'gaming', 'fitness'].map(tag => (
                    <label key={tag} className="inline-flex items-center">
                      <input
                        type="checkbox"
                        checked={(localSettings?.tiktok?.enabled_tags ?? ['trending', 'fyp', 'viral']).includes(tag)}
                        onChange={(e) => {
                          const current = localSettings?.tiktok?.enabled_tags ?? ['trending', 'fyp', 'viral']
                          const updated = e.target.checked
                            ? [...current, tag]
                            : current.filter((t: string) => t !== tag)
                          handleSettingChange('tiktok', 'enabled_tags', updated)
                        }}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        disabled={!localSettings?.tiktok?.enabled}
                      />
                      <span className="ml-1 text-sm text-gray-700">#{tag}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  每个标签最大视频数
                </label>
                <input
                  type="number"
                  value={localSettings?.tiktok?.max_videos_per_tag ?? 20}
                  onChange={(e) => handleSettingChange('tiktok', 'max_videos_per_tag', parseInt(e.target.value))}
                  min={5}
                  max={50}
                  className="w-24 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.tiktok?.enabled}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  最小播放量
                </label>
                <input
                  type="number"
                  value={localSettings?.tiktok?.min_view_count ?? 10000}
                  onChange={(e) => handleSettingChange('tiktok', 'min_view_count', parseInt(e.target.value))}
                  min={0}
                  max={10000000}
                  step={1000}
                  className="w-32 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.tiktok?.enabled}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  最小点赞数
                </label>
                <input
                  type="number"
                  value={localSettings?.tiktok?.min_like_count ?? 1000}
                  onChange={(e) => handleSettingChange('tiktok', 'min_like_count', parseInt(e.target.value))}
                  min={0}
                  max={10000000}
                  step={100}
                  className="w-32 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.tiktok?.enabled}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  最大时长（秒）
                </label>
                <select
                  value={localSettings?.tiktok?.max_duration ?? 180}
                  onChange={(e) => handleSettingChange('tiktok', 'max_duration', parseInt(e.target.value))}
                  className="w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  disabled={!localSettings?.tiktok?.enabled}
                >
                  <option value={60}>1 分钟</option>
                  <option value={180}>3 分钟</option>
                  <option value={300}>5 分钟</option>
                  <option value={600}>10 分钟</option>
                  <option value={0}>不限制</option>
                </select>
              </div>
            </div>
          </div>

          {/* Cookie Status */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 flex items-center">
                <Cookie className="h-5 w-5 mr-2 text-red-500" />
                {t('settings.youtube.cookieStatus')}
              </h3>
              <button
                onClick={() => refetchCookieStatus()}
                disabled={cookieStatusLoading}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-md"
              >
                <RefreshCw className={`h-4 w-4 ${cookieStatusLoading ? 'animate-spin' : ''}`} />
              </button>
            </div>

            {cookieStatusLoading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
              </div>
            ) : cookieStatus ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">{t('settings.youtube.cookieFile')}</span>
                  <span className={`flex items-center text-sm ${cookieStatus.cookie_file_exists ? 'text-green-600' : 'text-gray-400'}`}>
                    {cookieStatus.cookie_file_exists ? (
                      <>
                        <CheckCircle className="h-4 w-4 mr-1" />
                        {t('settings.youtube.exists')}
                      </>
                    ) : (
                      <>
                        <XCircle className="h-4 w-4 mr-1" />
                        {t('common.notConfigured')}
                      </>
                    )}
                  </span>
                </div>

                {cookieStatus.validation && (
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-sm text-gray-600">{t('settings.youtube.validationStatus')}</span>
                    <span className={`flex items-center text-sm ${cookieStatus.validation.valid ? 'text-green-600' : 'text-red-500'}`}>
                      {cookieStatus.validation.valid ? (
                        <>
                          <CheckCircle className="h-4 w-4 mr-1" />
                          {t('settings.youtube.valid')} ({t('settings.youtube.youtubeCookieCount', { count: cookieStatus.validation.youtube_cookies })})
                        </>
                      ) : (
                        <>
                          <XCircle className="h-4 w-4 mr-1" />
                          {cookieStatus.validation.message}
                        </>
                      )}
                    </span>
                  </div>
                )}

                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">{t('settings.youtube.environment')}</span>
                  <span className="text-sm text-gray-700">
                    {cookieStatus.in_docker ? t('settings.youtube.docker') : t('settings.youtube.local')}
                  </span>
                </div>

                {cookieStatus.cookie_file_exists && (
                  <div className="pt-2">
                    <button
                      onClick={() => {
                        if (confirm(t('settings.youtube.deleteConfirm'))) {
                          deleteCookiesMutation.mutate()
                        }
                      }}
                      disabled={deleteCookiesMutation.isPending}
                      className="flex items-center px-3 py-2 text-sm text-red-600 border border-red-300 rounded-md hover:bg-red-50"
                    >
                      {deleteCookiesMutation.isPending ? (
                        <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4 mr-1" />
                      )}
                      {t('settings.youtube.deleteCookie')}
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-gray-500">{t('errors.serverError')}</p>
            )}
          </div>

          {/* Extract from Browser (only for non-Docker) */}
          {cookieStatus && !cookieStatus.in_docker && cookieStatus.available_browsers.length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.youtube.extractCookie')}</h3>
              <p className="text-sm text-gray-600 mb-4">
                {t('settings.youtube.extractDesc')}
              </p>

              <div className="flex items-center space-x-3">
                <select
                  value={selectedBrowser}
                  onChange={(e) => setSelectedBrowser(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  {cookieStatus.available_browsers.map((browser) => (
                    <option key={browser} value={browser}>
                      {browser.charAt(0).toUpperCase() + browser.slice(1)}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => extractCookiesMutation.mutate(selectedBrowser)}
                  disabled={extractCookiesMutation.isPending}
                  className="flex items-center px-4 py-2 bg-red-600 text-white rounded-md text-sm font-medium hover:bg-red-700 disabled:opacity-50"
                >
                  {extractCookiesMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      {t('common.extracting')}
                    </>
                  ) : (
                    <>
                      <Youtube className="h-4 w-4 mr-2" />
                      {t('settings.youtube.extractButton')}
                    </>
                  )}
                </button>
              </div>

              {extractCookiesMutation.isSuccess && (
                <div className="mt-3 p-3 bg-green-50 text-green-700 rounded-md text-sm">
                  <CheckCircle className="h-4 w-4 inline mr-1" />
                  {t('settings.youtube.extractSuccess')}
                </div>
              )}

              {extractCookiesMutation.isError && (
                <div className="mt-3 p-3 bg-red-50 text-red-700 rounded-md text-sm">
                  <XCircle className="h-4 w-4 inline mr-1" />
                  {t('settings.youtube.extractFailed')}: {(extractCookiesMutation.error as Error)?.message || t('common.unknownError')}
                </div>
              )}
            </div>
          )}

          {/* Manual Upload */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">{t('settings.youtube.manualUpload')}</h3>
            <p className="text-sm text-gray-600 mb-4">
              {t('settings.youtube.manualUploadDesc')}
            </p>

            <textarea
              value={cookieContent}
              onChange={(e) => setCookieContent(e.target.value)}
              placeholder={`# Netscape HTTP Cookie File
# https://curl.haxx.se/docs/http-cookies.html
# This file was generated by a browser extension

.youtube.com	TRUE	/	TRUE	1735689600	LOGIN_INFO	...
.youtube.com	TRUE	/	FALSE	1735689600	SID	...
...`}
              rows={8}
              className="w-full px-3 py-2 border border-gray-300 rounded-md font-mono text-xs"
            />

            <div className="mt-4 flex items-center space-x-3">
              <button
                onClick={() => uploadCookiesMutation.mutate(cookieContent)}
                disabled={!cookieContent.trim() || uploadCookiesMutation.isPending}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
              >
                {uploadCookiesMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    {t('common.uploading')}
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4 mr-2" />
                    {t('settings.youtube.uploadButton')}
                  </>
                )}
              </button>
              {cookieContent && (
                <button
                  onClick={() => setCookieContent('')}
                  className="px-3 py-2 text-sm text-gray-600 hover:text-gray-800"
                >
                  {t('common.clear')}
                </button>
              )}
            </div>

            {uploadCookiesMutation.isSuccess && (
              <div className="mt-3 p-3 bg-green-50 text-green-700 rounded-md text-sm">
                <CheckCircle className="h-4 w-4 inline mr-1" />
                {t('settings.youtube.uploadSuccess')}
              </div>
            )}

            {uploadCookiesMutation.isError && (
              <div className="mt-3 p-3 bg-red-50 text-red-700 rounded-md text-sm">
                <XCircle className="h-4 w-4 inline mr-1" />
                {t('settings.youtube.uploadFailed')}: {(uploadCookiesMutation.error as Error)?.message || t('common.unknownError')}
              </div>
            )}
          </div>

          {/* Help Section */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="text-sm font-medium text-gray-900 mb-2">{t('settings.youtube.whyNeeded')}</h3>
            <div className="text-sm text-gray-600 space-y-2">
              <p>
                {t('settings.youtube.whyNeededDesc')}
              </p>
              <p className="font-medium mt-3">{t('settings.youtube.howToGet')}</p>
              <ol className="list-decimal list-inside space-y-1 mt-1">
                <li>{t('settings.youtube.howToGetSteps.step1')}</li>
                <li>{t('settings.youtube.howToGetSteps.step2')}</li>
                <li>{t('settings.youtube.howToGetSteps.step3')}</li>
                <li>{t('settings.youtube.howToGetSteps.step4')}</li>
              </ol>
              <p className="mt-3 text-xs text-gray-500">
                {t('settings.youtube.cookieExpireNote')}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

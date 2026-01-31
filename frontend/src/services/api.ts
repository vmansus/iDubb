import axios from 'axios'
import type {
  Task,
  CreateTaskRequest,
  VideoInfo,
  Language,
  Voice,
  AllPlatformStatus,
  FileInfo,
  DetailedVideoInfo,
  GlobalSettings,
  TranslationEngine,
  TTSEngine,
  TTSHealthStatus,
  SubtitlePreset,
  SubtitleStyle,
  FontInfo,
  UploadResponse,
  MetadataResult
} from '../types'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000, // Increased for video info parsing
})

export const taskApi = {
  create: async (data: CreateTaskRequest): Promise<Task> => {
    const response = await api.post<Task>('/tasks', data)
    return response.data
  },

  get: async (taskId: string): Promise<Task> => {
    const response = await api.get<Task>(`/tasks/${taskId}`)
    return response.data
  },

  list: async (status?: string, directory?: string): Promise<Task[]> => {
    const params: Record<string, string> = {}
    if (status) params.status = status
    if (directory) params.directory = directory
    const response = await api.get<Task[]>('/tasks', { params })
    return response.data
  },

  // Step control
  retryStep: async (taskId: string, stepName: string): Promise<Task> => {
    const response = await api.post<Task>(`/tasks/${taskId}/retry/${stepName}`)
    return response.data
  },

  continueFromStep: async (taskId: string, stepName: string): Promise<Task> => {
    const response = await api.post<Task>(`/tasks/${taskId}/continue/${stepName}`)
    return response.data
  },

  pauseTask: async (taskId: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.post(`/tasks/${taskId}/pause`)
    return response.data
  },

  stopTask: async (taskId: string): Promise<{ success: boolean; message: string; task: Task }> => {
    const response = await api.post(`/tasks/${taskId}/stop`)
    return response.data
  },

  getOptions: async (taskId: string): Promise<{ task_id: string; status: string; options: Record<string, unknown> }> => {
    const response = await api.get(`/tasks/${taskId}/options`)
    return response.data
  },

  updateOptions: async (taskId: string, options: Record<string, unknown>): Promise<{ success: boolean; message: string; task: Task }> => {
    const response = await api.put(`/tasks/${taskId}/options`, options)
    return response.data
  },

  // Files
  listFiles: async (taskId: string): Promise<Record<string, FileInfo>> => {
    const response = await api.get<Record<string, FileInfo>>(`/tasks/${taskId}/files`)
    return response.data
  },

  getFileDownloadUrl: (taskId: string, fileType: string, cacheKey?: string | number): string => {
    const base = `/api/tasks/${taskId}/files/${fileType}`
    // Add cache key to bust browser cache when file is updated
    return cacheKey ? `${base}?v=${cacheKey}` : base
  },

  // AI Metadata generation
  generateMetadata: async (
    taskId: string,
    options?: {
      source_language?: string
      target_language?: string
      include_source_url?: boolean
      title_prefix?: string
      custom_signature?: string
      max_keywords?: number
      use_ai_preset_selection?: boolean
    }
  ): Promise<MetadataResult> => {
    const response = await api.post<MetadataResult>(`/tasks/${taskId}/generate-metadata`, {
      task_id: taskId,
      source_language: options?.source_language || 'en',
      target_language: options?.target_language || 'zh-CN',
      include_source_url: options?.include_source_url ?? true,
      title_prefix: options?.title_prefix || '',
      custom_signature: options?.custom_signature || '',
      max_keywords: options?.max_keywords || 10,
      use_ai_preset_selection: options?.use_ai_preset_selection ?? false,
    })
    return response.data
  },

  // Save generated/edited metadata to database
  saveMetadata: async (
    taskId: string,
    metadata: {
      title: string
      description: string
      keywords: string[]
    }
  ): Promise<{ success: boolean; message: string }> => {
    const response = await api.post(`/tasks/${taskId}/metadata`, metadata)
    return response.data
  },

  // Load saved metadata from database
  loadMetadata: async (taskId: string): Promise<{
    success: boolean
    title?: string
    description?: string
    keywords?: string[]
    generated_at?: string
    approved?: boolean
    message?: string
  }> => {
    const response = await api.get(`/tasks/${taskId}/metadata`)
    return response.data
  },

  // Approve metadata and optionally trigger upload
  approveMetadata: async (taskId: string): Promise<{
    success: boolean
    message: string
    uploading?: boolean
  }> => {
    const response = await api.post(`/tasks/${taskId}/metadata/approve`)
    return response.data
  },

  // Re-download thumbnail
  redownloadThumbnail: async (taskId: string): Promise<{
    success: boolean
    message: string
    thumbnail_path?: string
  }> => {
    const response = await api.post(`/tasks/${taskId}/redownload-thumbnail`)
    return response.data
  },

  // Get thumbnail information (original and AI)
  getThumbnails: async (taskId: string): Promise<ThumbnailInfo> => {
    const response = await api.get<ThumbnailInfo>(`/tasks/${taskId}/thumbnails`)
    return response.data
  },

  // Generate or regenerate AI thumbnail
  generateAiThumbnail: async (taskId: string, options?: {
    custom_title?: string
    style?: string
  }): Promise<{
    success: boolean
    message: string
    ai_thumbnail_path?: string
    ai_thumbnail_title?: string
    error?: string
  }> => {
    const response = await api.post(`/tasks/${taskId}/generate-ai-thumbnail`, options || {})
    return response.data
  },

  // Select which thumbnail to use for upload
  selectThumbnail: async (taskId: string, selected: 'original' | 'ai_generated'): Promise<{
    success: boolean
    message: string
  }> => {
    const response = await api.put(`/tasks/${taskId}/select-thumbnail`, { selected })
    return response.data
  },

  // Get subtitles for editing
  getSubtitles: async (taskId: string): Promise<{
    success: boolean
    segments: Array<{
      index: number
      start_time: number
      end_time: number
      original_text: string
      translated_text: string
    }>
    original_file?: string
    translated_file?: string
  }> => {
    const response = await api.get(`/tasks/${taskId}/subtitles`)
    return response.data
  },

  // Update subtitles after editing
  updateSubtitles: async (taskId: string, segments: Array<{
    index: number
    start_time: number
    end_time: number
    original_text: string
    translated_text: string
  }>): Promise<{
    success: boolean
    message: string
  }> => {
    const response = await api.put(`/tasks/${taskId}/subtitles`, { segments })
    return response.data
  },

  // Start AI subtitle optimization (background job)
  startOptimization: async (taskId: string, level?: 'minimal' | 'moderate' | 'aggressive'): Promise<{
    job_id: string
    status: string
    message: string
  }> => {
    const response = await api.post(`/tasks/${taskId}/optimize-subtitles`, level ? { level } : {})
    return response.data
  },

  // Get optimization job status
  getOptimizationStatus: async (jobId: string): Promise<{
    job_id: string
    status: 'pending' | 'running' | 'completed' | 'failed'
    task_id?: string
    result?: {
      success: boolean
      optimized_count: number
      total_segments: number
      changes: Array<{
        index: number
        original_text: string
        translated_text: string
        optimized_text: string
        suggestions: string[]
        issues: Array<{
          type: string
          severity: string
          message: string
          suggestion?: string
        }>
      }>
    }
    error?: string
  }> => {
    const response = await api.get(`/optimization-jobs/${jobId}`)
    return response.data
  },
}

// Thumbnail info type
export interface ThumbnailInfo {
  original: {
    url: string | null
    exists: boolean
  }
  ai_generated: {
    url: string | null
    exists: boolean
    title: string | null
  }
  selected: 'original' | 'ai_generated'
}

export const videoApi = {
  // Basic video info
  getInfo: async (url: string): Promise<VideoInfo> => {
    const response = await api.post<VideoInfo>('/video/info', null, {
      params: { url },
    })
    return response.data
  },

  // Detailed video info with formats, audio tracks, subtitles
  getDetailedInfo: async (url: string): Promise<DetailedVideoInfo> => {
    const response = await api.post<DetailedVideoInfo>('/video/detailed-info', null, {
      params: { url },
    })
    return response.data
  },
}

export interface LocalVideoInfo {
  success: boolean
  title: string
  duration: number
  duration_formatted: string
  width: number
  height: number
  resolution: string
  resolution_label: string
  max_quality: string
  available_qualities: string[]
  codec: string
  fps: number
  file_size: number
  file_size_mb: number
  has_audio: boolean
  platform: string
}

export const uploadApi = {
  // Upload a local video file
  uploadVideo: async (file: File, onProgress?: (progress: number) => void): Promise<UploadResponse> => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post<UploadResponse>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 600000, // 10 minutes for large files
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    })
    return response.data
  },

  // Get local video info (resolution, duration, etc.)
  getLocalVideoInfo: async (filePath: string): Promise<LocalVideoInfo> => {
    const response = await api.post<LocalVideoInfo>('/video/local-info', {
      file_path: filePath,
    })
    return response.data
  },
}

export const configApi = {
  getLanguages: async (): Promise<Language[]> => {
    const response = await api.get<Language[]>('/languages')
    return response.data
  },

  getVoices: async (language?: string): Promise<Voice[]> => {
    const params = language ? { language } : {}
    const response = await api.get<Voice[]>('/voices', { params })
    return response.data
  },

  getPlatformStatus: async (): Promise<AllPlatformStatus> => {
    const response = await api.get<AllPlatformStatus>('/platforms/status')
    return response.data
  },

  authenticatePlatform: async (platform: string, credentials: Record<string, string>) => {
    const response = await api.post('/platforms/authenticate', {
      platform,
      ...credentials,
    })
    return response.data
  },

  getPlatformCookieStatus: async (): Promise<{
    in_docker: boolean
    available_browsers: string[]
    platforms: Record<string, string[]>
  }> => {
    const response = await api.get('/platforms/cookie-status')
    return response.data
  },

  extractPlatformCookies: async (platform: string, browser: string = 'chrome'): Promise<{
    success: boolean
    message: string
    cookie_count?: number
    authenticated?: boolean
  }> => {
    const response = await api.post(`/platforms/${platform}/extract-cookies`, null, {
      params: { browser },
    })
    return response.data
  },
}

// Bilibili multi-account API
export interface BilibiliAccount {
  uid: string
  nickname: string
  avatar: string
  label: string
  is_primary: boolean
  updated_at: string
}

export const bilibiliApi = {
  // Generate QR code for login
  generateQRCode: async (): Promise<{
    qrcode_key: string
    qrcode_url: string
  }> => {
    const response = await api.get('/bilibili/qrcode')
    return response.data
  },

  // Poll QR code scan status
  pollQRCode: async (key: string, label?: string): Promise<{
    status: 'waiting' | 'scanned' | 'expired' | 'success' | 'error'
    message: string
    is_new?: boolean
    account?: {
      uid: string
      nickname: string
      avatar: string
      label: string
      is_primary: boolean
    }
  }> => {
    const response = await api.get('/bilibili/qrcode/poll', { params: { key, label } })
    return response.data
  },

  // List all accounts
  listAccounts: async (): Promise<{ accounts: BilibiliAccount[] }> => {
    const response = await api.get('/bilibili/accounts')
    return response.data
  },

  // Delete account
  deleteAccount: async (uid: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/bilibili/accounts/${uid}`)
    return response.data
  },

  // Add account by cookies (manual)
  addByCookies: async (sessdata: string, bili_jct: string, buvid3: string, label?: string): Promise<{
    success: boolean
    account?: {
      uid: string
      nickname: string
      avatar: string
      label: string
      is_primary: boolean
    }
  }> => {
    const response = await api.post('/bilibili/accounts/cookies', {
      sessdata,
      bili_jct,
      buvid3,
      label,
    })
    return response.data
  },

  // Update account label
  updateLabel: async (uid: string, label: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.put(`/bilibili/accounts/${uid}/label`, { label })
    return response.data
  },

  // Set account as primary
  setPrimary: async (uid: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.put(`/bilibili/accounts/${uid}/primary`)
    return response.data
  },
}

// Douyin account type
export interface DouyinAccount {
  uid: string
  nickname: string
  avatar: string
  label: string
  is_primary: boolean
  updated_at: string
}

export const douyinApi = {
  // Generate QR code for login
  generateQRCode: async (): Promise<{
    token: string
    qrcode_url: string
  }> => {
    const response = await api.get('/douyin/qrcode')
    return response.data
  },

  // Poll QR code scan status
  pollQRCode: async (token: string, label?: string): Promise<{
    status: 'waiting' | 'scanned' | 'expired' | 'success' | 'error'
    message: string
    is_new?: boolean
    account?: {
      uid: string
      nickname: string
      avatar: string
      label: string
      is_primary: boolean
    }
  }> => {
    const response = await api.get('/douyin/qrcode/poll', { params: { token, label } })
    return response.data
  },

  // List all accounts
  listAccounts: async (): Promise<{ accounts: DouyinAccount[] }> => {
    const response = await api.get('/douyin/accounts')
    return response.data
  },

  // Delete account
  deleteAccount: async (uid: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/douyin/accounts/${uid}`)
    return response.data
  },

  // Add account by cookies (manual)
  addByCookies: async (cookies: string, label?: string): Promise<{
    success: boolean
    account?: {
      uid: string
      nickname: string
      avatar: string
      label: string
      is_primary: boolean
    }
  }> => {
    const response = await api.post('/douyin/accounts/cookies', { cookies, label })
    return response.data
  },

  // Update account label
  updateLabel: async (uid: string, label: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.put(`/douyin/accounts/${uid}/label`, { label })
    return response.data
  },

  // Set account as primary
  setPrimary: async (uid: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.put(`/douyin/accounts/${uid}/primary`)
    return response.data
  },
}

// Xiaohongshu account type
export interface XiaohongshuAccount {
  user_id: string
  uid: string
  nickname: string
  avatar: string
  label: string
  is_primary: boolean
  updated_at: string
}

export const xiaohongshuApi = {
  // Generate QR code for login
  generateQRCode: async (): Promise<{
    qr_id: string
    qrcode_url: string
  }> => {
    const response = await api.get('/xiaohongshu/qrcode')
    return response.data
  },

  // Poll QR code scan status
  pollQRCode: async (qr_id: string, label?: string): Promise<{
    status: 'waiting' | 'scanned' | 'expired' | 'success' | 'error'
    message: string
    is_new?: boolean
    account?: {
      uid: string
      user_id: string
      nickname: string
      avatar: string
      label: string
      is_primary: boolean
    }
  }> => {
    const response = await api.get('/xiaohongshu/qrcode/poll', { params: { qr_id, label } })
    return response.data
  },

  // List all accounts
  listAccounts: async (): Promise<{ accounts: XiaohongshuAccount[] }> => {
    const response = await api.get('/xiaohongshu/accounts')
    return response.data
  },

  // Delete account
  deleteAccount: async (user_id: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/xiaohongshu/accounts/${user_id}`)
    return response.data
  },

  // Add account by cookies (manual)
  addByCookies: async (cookies: string, label?: string): Promise<{
    success: boolean
    account?: {
      uid: string
      user_id: string
      nickname: string
      avatar: string
      label: string
      is_primary: boolean
    }
  }> => {
    const response = await api.post('/xiaohongshu/accounts/cookies', { cookies, label })
    return response.data
  },

  // Update account label
  updateLabel: async (user_id: string, label: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.put(`/xiaohongshu/accounts/${user_id}/label`, { label })
    return response.data
  },

  // Set account as primary
  setPrimary: async (user_id: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.put(`/xiaohongshu/accounts/${user_id}/primary`)
    return response.data
  },
}

// Global settings API
export const settingsApi = {
  get: async (): Promise<GlobalSettings> => {
    const response = await api.get<GlobalSettings>('/settings')
    return response.data
  },

  update: async (updates: Partial<GlobalSettings>): Promise<GlobalSettings> => {
    const response = await api.put<GlobalSettings>('/settings', updates)
    return response.data
  },

  reset: async (): Promise<GlobalSettings> => {
    const response = await api.post<GlobalSettings>('/settings/reset')
    return response.data
  },
}

// Translation engines API
export const translationApi = {
  getEngines: async (): Promise<TranslationEngine[]> => {
    const response = await api.get<TranslationEngine[]>('/translation/engines')
    return response.data
  },
}

// Transcription models API
export interface WhisperModel {
  id: string
  name: string
  backend: string
  size_mb: number
  vram_mb: number
  quality: string
  speed_factor_cpu: number
  speed_factor_gpu: number
  recommended_for: string[]
  description: string
}

export interface TranscriptionModelsResponse {
  models: WhisperModel[]
  default: string
  recommended_cpu: string
  recommended_gpu: string
}

export interface TranscriptionEstimate {
  model_id: string
  model_name: string
  duration_seconds: number
  duration_formatted: string
  estimated_time_cpu_seconds: number
  estimated_time_gpu_seconds: number
  estimated_time_cpu_formatted: string
  estimated_time_gpu_formatted: string
  device: string
  estimated_time_seconds: number
  estimated_time_formatted: string
  quality?: string
  description?: string
  recommended_for?: string[]
}

export interface AllEstimatesResponse {
  duration_seconds: number
  estimates: TranscriptionEstimate[]
  recommended_model: string
}

export const transcriptionApi = {
  getModels: async (backend?: string): Promise<TranscriptionModelsResponse> => {
    const params: Record<string, string> = {}
    if (backend) params.backend = backend
    const response = await api.get<TranscriptionModelsResponse>('/transcription/models', { params })
    return response.data
  },

  getEstimate: async (duration: number, modelId: string = 'faster:small', device: string = 'cpu'): Promise<TranscriptionEstimate> => {
    const response = await api.get<TranscriptionEstimate>('/transcription/estimate', {
      params: { duration, model_id: modelId, device }
    })
    return response.data
  },

  getAllEstimates: async (duration: number, backend?: string): Promise<AllEstimatesResponse> => {
    const params: Record<string, string | number> = { duration }
    if (backend) params.backend = backend
    const response = await api.get<AllEstimatesResponse>('/transcription/estimates', { params })
    return response.data
  },
}

// TTS engines API
export const ttsApi = {
  getEngines: async (): Promise<TTSEngine[]> => {
    const response = await api.get<TTSEngine[]>('/tts/engines')
    return response.data
  },

  getVoicesByEngine: async (engine: string, language?: string): Promise<Voice[]> => {
    const params = language ? { language } : {}
    const response = await api.get<Voice[]>(`/tts/voices/${engine}`, { params })
    return response.data
  },

  checkHealth: async (engine: string): Promise<TTSHealthStatus> => {
    const response = await api.get<TTSHealthStatus>(`/tts/health/${engine}`)
    return response.data
  },
}

// Storage settings API
export interface StorageSettings {
  output_directory: string
  effective_directory: string
  default_directory: string
}

export const storageApi = {
  get: async (): Promise<StorageSettings> => {
    const response = await api.get<StorageSettings>('/settings/storage')
    return response.data
  },

  update: async (outputDirectory: string): Promise<StorageSettings> => {
    const response = await api.put<StorageSettings>('/settings/storage', {
      output_directory: outputDirectory,
    })
    return response.data
  },
}

// YouTube cookies API
export interface CookieStatus {
  cookie_file_exists: boolean
  in_docker: boolean
  available_browsers: string[]
  validation: {
    valid: boolean
    message: string
    youtube_cookies: number
  } | null
}

export interface CookieUploadResult {
  success: boolean
  message: string
  validation?: {
    valid: boolean
    message: string
    youtube_cookies: number
  }
}

export const youtubeApi = {
  getCookieStatus: async (): Promise<CookieStatus> => {
    const response = await api.get<CookieStatus>('/youtube/cookies/status')
    return response.data
  },

  extractCookies: async (browser: string = 'chrome'): Promise<CookieUploadResult> => {
    const response = await api.post<CookieUploadResult>('/youtube/cookies/extract', null, {
      params: { browser },
    })
    return response.data
  },

  uploadCookies: async (cookiesContent: string): Promise<CookieUploadResult> => {
    const response = await api.post<CookieUploadResult>('/youtube/cookies/upload', null, {
      params: { cookies_content: cookiesContent },
    })
    return response.data
  },

  deleteCookies: async (): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete('/youtube/cookies')
    return response.data
  },
}

// Fonts API
export interface FontsResponse {
  fonts: FontInfo[]
  recommended: FontInfo
  total: number
}

export const fontsApi = {
  getAll: async (): Promise<FontsResponse> => {
    const response = await api.get<FontsResponse>('/fonts')
    return response.data
  },

  upload: async (file: File): Promise<{ success: boolean; message: string; font: FontInfo }> => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post('/fonts/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  delete: async (fontName: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/fonts/${fontName}`)
    return response.data
  },
}

// Subtitle Presets API
export interface PresetsResponse {
  presets: SubtitlePreset[]
  builtin_count: number
  custom_count: number
}

export interface CreatePresetRequest {
  name: string
  description: string
  is_vertical?: boolean
  subtitle_mode: 'dual' | 'original_only' | 'translated_only'
  source_language?: string
  target_language?: string
  original_style?: SubtitleStyle
  translated_style?: SubtitleStyle
}

export const presetsApi = {
  getAll: async (): Promise<PresetsResponse> => {
    const response = await api.get<PresetsResponse>('/subtitle-presets')
    return response.data
  },

  create: async (preset: CreatePresetRequest): Promise<{ success: boolean; message: string; preset: SubtitlePreset }> => {
    const response = await api.post('/subtitle-presets', preset)
    return response.data
  },

  update: async (presetId: string, preset: CreatePresetRequest): Promise<{ success: boolean; message: string; preset: SubtitlePreset }> => {
    const response = await api.put(`/subtitle-presets/${presetId}`, preset)
    return response.data
  },

  delete: async (presetId: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/subtitle-presets/${presetId}`)
    return response.data
  },
}

// Metadata Presets API (title prefix and signature combinations)
import type {
  MetadataPreset,
  MetadataPresetsListResponse,
  CreateMetadataPresetRequest,
  AIPresetSelectResponse,
  Subscription,
  CreateSubscriptionRequest,
  UpdateSubscriptionRequest,
  ChannelLookupResponse,
  SubscriptionPlatform,
  NewVideosResponse,
  TrendingVideosListResponse,
  TrendingCategoriesResponse,
  TrendingSettings,
  TrendingRefreshResponse,
  TrendingBatchCreateResponse,
  ProcessOptions,
  YouTubeSearchResponse
} from '../types'

export const metadataPresetsApi = {
  getAll: async (): Promise<MetadataPresetsListResponse> => {
    const response = await api.get<MetadataPresetsListResponse>('/metadata-presets')
    return response.data
  },

  get: async (presetId: string): Promise<MetadataPreset> => {
    const response = await api.get<MetadataPreset>(`/metadata-presets/${presetId}`)
    return response.data
  },

  getDefault: async (): Promise<MetadataPreset> => {
    const response = await api.get<MetadataPreset>('/metadata-presets/default')
    return response.data
  },

  create: async (preset: CreateMetadataPresetRequest): Promise<{ success: boolean; message: string; preset: MetadataPreset }> => {
    const response = await api.post('/metadata-presets', preset)
    return response.data
  },

  update: async (presetId: string, preset: CreateMetadataPresetRequest): Promise<{ success: boolean; message: string; preset: MetadataPreset }> => {
    const response = await api.put(`/metadata-presets/${presetId}`, preset)
    return response.data
  },

  delete: async (presetId: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/metadata-presets/${presetId}`)
    return response.data
  },

  setDefault: async (presetId: string): Promise<{ success: boolean; message: string; preset: MetadataPreset }> => {
    const response = await api.post(`/metadata-presets/${presetId}/set-default`)
    return response.data
  },

  aiSelect: async (options?: {
    task_id?: string
    video_info?: Record<string, unknown>
    transcript_snippet?: string
  }): Promise<AIPresetSelectResponse> => {
    const response = await api.post<AIPresetSelectResponse>('/metadata-presets/ai-select', options || {})
    return response.data
  },
}

// Subscriptions API
export const subscriptionApi = {
  // Get supported platforms
  getPlatforms: async (): Promise<{ platforms: SubscriptionPlatform[] }> => {
    const response = await api.get('/subscriptions/platforms')
    return response.data
  },

  // Look up channel information
  lookup: async (platform: string, url: string): Promise<ChannelLookupResponse> => {
    const response = await api.post<ChannelLookupResponse>('/subscriptions/lookup', null, {
      params: { platform, url }
    })
    return response.data
  },

  // List all subscriptions
  list: async (options?: {
    platform?: string
    enabled_only?: boolean
    limit?: number
  }): Promise<Subscription[]> => {
    const response = await api.get<Subscription[]>('/subscriptions', { params: options })
    return response.data
  },

  // Get single subscription
  get: async (subscriptionId: string): Promise<Subscription> => {
    const response = await api.get<Subscription>(`/subscriptions/${subscriptionId}`)
    return response.data
  },

  // Create subscription
  create: async (request: CreateSubscriptionRequest): Promise<Subscription> => {
    const response = await api.post<Subscription>('/subscriptions', request)
    return response.data
  },

  // Update subscription
  update: async (subscriptionId: string, request: UpdateSubscriptionRequest): Promise<Subscription> => {
    const response = await api.put<Subscription>(`/subscriptions/${subscriptionId}`, request)
    return response.data
  },

  // Delete subscription
  delete: async (subscriptionId: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/subscriptions/${subscriptionId}`)
    return response.data
  },

  // Check for new videos now
  checkNow: async (subscriptionId: string): Promise<NewVideosResponse> => {
    const response = await api.post<NewVideosResponse>(`/subscriptions/${subscriptionId}/check`)
    return response.data
  },

  // Enable subscription
  enable: async (subscriptionId: string): Promise<{ success: boolean; enabled: boolean }> => {
    const response = await api.post(`/subscriptions/${subscriptionId}/enable`)
    return response.data
  },

  // Disable subscription
  disable: async (subscriptionId: string): Promise<{ success: boolean; enabled: boolean }> => {
    const response = await api.post(`/subscriptions/${subscriptionId}/disable`)
    return response.data
  },

  // Fetch historical videos in date range
  fetchHistoricalVideos: async (
    subscriptionId: string,
    startDate: string,
    endDate: string,
    maxVideos: number = 50
  ): Promise<import('../types').FetchHistoricalVideosResponse> => {
    const response = await api.post(`/subscriptions/${subscriptionId}/fetch-videos`, {
      start_date: startDate,
      end_date: endDate,
      max_videos: maxVideos,
    }, { timeout: 120000 }) // Longer timeout for video fetching
    return response.data
  },

  // Batch create tasks from videos
  batchCreateTasks: async (
    subscriptionId: string,
    videos: import('../types').VideoItem[],
    processOptions?: import('../types').ProcessOptions
  ): Promise<import('../types').BatchCreateTasksResponse> => {
    const response = await api.post(`/subscriptions/${subscriptionId}/batch-create-tasks`, {
      videos,
      process_options: processOptions,
    })
    return response.data
  },
}

// Trending Videos API
export const trendingApi = {
  // Get trending videos by category
  getVideos: async (category?: string, limit: number = 50): Promise<TrendingVideosListResponse> => {
    const params: Record<string, string | number> = { limit }
    if (category) params.category = category
    const response = await api.get<TrendingVideosListResponse>('/trending/videos', { params })
    return response.data
  },

  // Get available categories
  getCategories: async (): Promise<TrendingCategoriesResponse> => {
    const response = await api.get<TrendingCategoriesResponse>('/trending/categories')
    return response.data
  },

  // Get trending settings
  getSettings: async (): Promise<TrendingSettings> => {
    const response = await api.get<TrendingSettings>('/trending/settings')
    return response.data
  },

  // Update trending settings
  updateSettings: async (settings: Partial<TrendingSettings>): Promise<TrendingSettings> => {
    const response = await api.put<TrendingSettings>('/trending/settings', settings)
    return response.data
  },

  // Manually refresh all trending videos
  refresh: async (): Promise<TrendingRefreshResponse> => {
    const response = await api.post<TrendingRefreshResponse>('/trending/refresh')
    return response.data
  },

  // Refresh trending videos for a specific category only
  refreshCategory: async (category: string): Promise<TrendingRefreshResponse> => {
    const response = await api.post<TrendingRefreshResponse>(`/trending/refresh/${category}`)
    return response.data
  },

  // Batch create tasks from trending videos
  batchCreateTasks: async (
    videoIds: string[],
    processOptions?: ProcessOptions
  ): Promise<TrendingBatchCreateResponse> => {
    const response = await api.post<TrendingBatchCreateResponse>('/trending/batch-create', {
      video_ids: videoIds,
      process_options: processOptions
    })
    return response.data
  },

  // Search YouTube videos
  searchYouTube: async (query: string, maxResults: number = 20): Promise<YouTubeSearchResponse> => {
    const response = await api.post<YouTubeSearchResponse>('/youtube/search', {
      query,
      max_results: maxResults,
    })
    return response.data
  },
}

// TikTok Discovery API
export const tiktokApi = {
  // Get available tags
  getTags: async () => {
    const response = await api.get('/tiktok/tags')
    return response.data
  },

  // Get TikTok videos
  getVideos: async (tag?: string, limit: number = 50) => {
    const params: Record<string, string | number> = { limit }
    if (tag) params.tag = tag
    const response = await api.get('/tiktok/videos', { params })
    return response.data
  },

  // Get TikTok settings
  getSettings: async () => {
    const response = await api.get('/tiktok/settings')
    return response.data
  },

  // Update TikTok settings
  updateSettings: async (settings: Record<string, unknown>) => {
    const response = await api.put('/tiktok/settings', settings)
    return response.data
  },

  // Refresh all TikTok videos
  refresh: async () => {
    const response = await api.post('/tiktok/refresh')
    return response.data
  },

  // Refresh TikTok videos for a specific tag
  refreshTag: async (tag: string) => {
    const response = await api.post(`/tiktok/refresh/${tag}`)
    return response.data
  },
}

// Directories API - for organizing tasks
export const directoryApi = {
  // List all directories
  list: async (): Promise<import('../types').DirectoryListResponse> => {
    const response = await api.get('/directories')
    return response.data
  },

  // Create a new directory
  create: async (name: string, description?: string): Promise<import('../types').Directory> => {
    const response = await api.post('/directories', { name, description })
    return response.data
  },

  // Check if a directory name exists
  exists: async (name: string): Promise<{ exists: boolean }> => {
    const response = await api.get(`/directories/${encodeURIComponent(name)}/exists`)
    return response.data
  },

  // Delete a directory (only if empty)
  delete: async (directoryId: number): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/directories/${directoryId}`)
    return response.data
  },
}

// Glossary API - for user custom terminology
export const glossaryApi = {
  // Get all glossary entries
  getAll: async (category?: string, search?: string): Promise<import('../types').GlossaryListResponse> => {
    const params: Record<string, string> = {}
    if (category) params.category = category
    if (search) params.search = search
    const response = await api.get('/glossary', { params })
    return response.data
  },

  // Add a new entry
  add: async (entry: { source: string; target: string; note?: string; category?: string }): Promise<import('../types').GlossaryEntry> => {
    const response = await api.post('/glossary', entry)
    return response.data
  },

  // Update an entry
  update: async (source: string, entry: { source: string; target: string; note?: string; category?: string }): Promise<import('../types').GlossaryEntry> => {
    const response = await api.put(`/glossary/${encodeURIComponent(source)}`, entry)
    return response.data
  },

  // Delete an entry
  delete: async (source: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/glossary/${encodeURIComponent(source)}`)
    return response.data
  },

  // Import entries
  import: async (format: 'json' | 'csv', data: string): Promise<{ success: boolean; imported_count: number; message: string }> => {
    const response = await api.post('/glossary/import', { format, data })
    return response.data
  },

  // Export entries
  exportJson: async (): Promise<{ entries: import('../types').GlossaryEntry[]; name: string }> => {
    const response = await api.get('/glossary/export', { params: { format: 'json' } })
    return response.data
  },

  // Clear all entries
  clear: async (): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete('/glossary')
    return response.data
  },
}

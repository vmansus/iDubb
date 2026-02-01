export interface StepResult {
  step_name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  started_at?: string
  completed_at?: string
  duration_seconds?: number
  duration_formatted?: string
  error?: string
  output_files: Record<string, string | null>
  metadata: Record<string, unknown>
}

export interface TaskFiles {
  video?: string | null
  audio?: string | null
  original_subtitle?: string | null
  translated_subtitle?: string | null
  tts_audio?: string | null
  final_video?: string | null
  thumbnail?: string | null
  ai_thumbnail?: string | null
}

export interface FileInfo {
  available: boolean
  path?: string | null
  size: number
  download_url?: string | null
}

export interface StepTiming {
  step_name: string
  duration_seconds?: number
  duration_formatted?: string
}

export interface Task {
  task_id: string
  status: 'pending' | 'queued' | 'downloading' | 'transcribing' | 'translating' | 'generating_tts' | 'processing_video' | 'pending_review' | 'pending_upload' | 'uploading' | 'uploaded' | 'completed' | 'failed' | 'paused'
  progress: number
  message: string
  created_at: string
  updated_at: string
  current_step?: string
  steps: Record<string, StepResult>
  // Queue position (0 = processing, >0 = waiting in queue, -1 = not queued)
  queue_position?: number
  video_info?: VideoInfo
  files: TaskFiles
  upload_results: Record<string, UploadResult>
  error?: string
  // Timing and thumbnail fields
  thumbnail_url?: string
  step_timings?: Record<string, StepTiming>
  total_processing_time?: number
  total_time_formatted?: string
  // Task folder for file storage
  task_folder?: string
  // Directory for organizing tasks (immutable after creation)
  directory?: string
  // Metadata approval
  generated_metadata?: {
    title: string
    description: string
    keywords: string[]
    generated_at?: string
  }
  metadata_approved?: boolean
  metadata_approved_at?: string
  // AI thumbnail
  ai_thumbnail_title?: string
  use_ai_thumbnail?: boolean
  // Proofreading results
  proofreading_result?: ProofreadingResult
  // Optimization results
  optimization_result?: OptimizationResult
}

export interface OptimizationResult {
  success: boolean
  optimized_count: number
  total_segments: number
  changes: Array<{
    index: number
    start_time: number
    end_time: number
    original_text: string
    translated_text: string
    optimized_text: string
    suggestions: string[]
    issues: Array<{
      type?: string
      severity?: string
      message?: string
      suggestion?: string
    }>
  }>
}

export interface VideoInfo {
  title: string
  description: string
  duration: number
  thumbnail_url: string
  uploader: string
  platform: string
  tags: string[]
  // Video dimensions (added after download)
  width?: number
  height?: number
  is_vertical?: boolean
}

export interface UploadResult {
  success: boolean
  video_id?: string
  video_url?: string
  error?: string
}

export interface UploadResponse {
  success: boolean
  file_path: string
  original_filename: string
  size_bytes: number
  size_mb: number
}

export interface CreateTaskRequest {
  source_url: string
  source_platform: string
  local_file_path?: string  // Path to uploaded local video file
  video_quality: string
  format_id?: string
  video_quality_label?: string  // Human-readable quality label (e.g., "1080p (mp4) - 15.2MB")
  source_language: string
  target_language: string
  processing_mode?: 'full' | 'subtitle' | 'direct' | 'auto'  // Processing workflow mode
  skip_translation?: boolean  // deprecated, use processing_mode instead
  // Whisper transcription settings
  whisper_backend: string  // 'auto', 'faster', 'openai', 'whisperx'
  whisper_model: string    // 'auto', 'faster:tiny', 'faster:base', 'faster:small', etc.
  whisper_device: string   // 'auto', 'cpu', 'cuda', 'mps'
  // OCR settings
  use_ocr?: boolean
  ocr_engine?: string
  ocr_frame_interval?: number
  add_subtitles: boolean
  dual_subtitles: boolean
  use_existing_subtitles: boolean
  subtitle_language?: string
  subtitle_preset?: string
  add_tts: boolean
  tts_service: string
  tts_voice: string
  voice_cloning_mode?: VoiceCloningMode  // disabled, custom, video_audio
  tts_ref_audio?: string
  tts_ref_text?: string
  replace_original_audio: boolean
  original_audio_volume: number
  tts_audio_volume: number
  translation_engine: string
  upload_bilibili: boolean
  upload_douyin: boolean
  upload_xiaohongshu: boolean
  bilibili_account_uid?: string  // Specific Bilibili account UID (empty = default)
  douyin_account_id?: string  // Specific Douyin account ID (empty = default)
  xiaohongshu_account_id?: string  // Specific Xiaohongshu account ID (empty = default)
  custom_title?: string
  custom_description?: string
  custom_tags: string[]
  // Metadata preset
  metadata_preset_id?: string
  use_ai_preset_selection?: boolean
  use_global_settings: boolean
  // Directory for organizing tasks (immutable after creation)
  directory?: string
}

export interface Language {
  code: string
  name: string
}

export interface Voice {
  name: string
  display_name: string
  gender: string
  locale: string
}

export interface PlatformStatus {
  configured: boolean
  authenticated: boolean
}

export interface AllPlatformStatus {
  bilibili: PlatformStatus
  douyin: PlatformStatus
  xiaohongshu: PlatformStatus
}

// Video format information
export interface VideoFormat {
  format_id: string
  ext: string
  resolution: string
  width: number
  height: number
  fps: number | null
  vcodec: string
  acodec: string
  filesize: number | null
  filesize_mb: number | null
  tbr: number | null
  quality_label: string
  has_audio: boolean
  has_video: boolean
}

export interface AudioTrack {
  format_id: string
  ext: string
  acodec: string
  abr: number | null
  asr: number | null
  language: string | null
  language_name: string | null
  filesize: number | null
  filesize_mb: number | null
  is_original: boolean
}

export interface SubtitleTrack {
  language: string
  language_name: string
  ext: string
  is_auto_generated: boolean
  is_translatable: boolean
}

export interface DetailedVideoInfo {
  video_id: string
  title: string
  description: string
  duration: number
  thumbnail_url: string
  uploader: string
  platform: string
  original_url: string
  formats: VideoFormat[]
  audio_tracks: AudioTrack[]
  subtitles: SubtitleTrack[]
  recommended_format: string | null
  best_quality: string | null
  view_count: number
  like_count: number
  tags: string[]
  cookies_refreshed?: boolean  // True if cookies were auto-refreshed during fetch
}

// Global settings types
export interface ProcessingSettings {
  max_concurrent_tasks: number
  use_gpu_lock: boolean
  translation_timeout: number
  translation_retry_count: number
  timezone: string
}

export interface VideoSettings {
  default_quality: string
  preferred_format: string
  max_duration: number
  download_subtitles: boolean
  prefer_existing_subtitles: boolean
  // Whisper transcription settings
  whisper_backend: string  // 'auto', 'faster', 'openai', 'whisperx'
  whisper_model: string    // 'auto', 'faster:tiny', 'faster:small', etc.
  whisper_device: string   // 'auto', 'cpu', 'cuda', 'mps'
}

export interface TranslationSettings {
  engine: string
  api_key: string | null
  api_keys?: {
    openai: string
    anthropic: string
    deepseek: string
    deepl: string
  }
  model: string
  preserve_formatting: boolean
  batch_size: number
  use_optimized?: boolean
  fast_mode?: boolean
  use_two_step_mode?: boolean  // VideoLingo-style two-step vs three-step mode
  enable_alignment?: boolean   // Enable subtitle alignment
  // Post-processing settings
  enable_length_control?: boolean  // Auto-split long subtitles
  max_chars_per_line?: number      // Max chars per line (default 42)
  max_lines?: number               // Max lines per subtitle (default 2)
  enable_localization?: boolean    // Localize numbers/units
  use_custom_glossary?: boolean    // Use custom terminology
}

// Glossary types
export interface GlossaryEntry {
  source: string
  target: string
  note: string
  category: string
  created_at: string
  updated_at: string
}

export interface GlossaryListResponse {
  entries: GlossaryEntry[]
  total: number
  categories: string[]
}

export type VoiceCloningMode = 'disabled' | 'custom' | 'video_audio'

export interface TTSSettings {
  engine: string
  voice: string
  rate: string
  volume: string
  pitch: string
  api_key: string | null
  // Voice cloning settings
  voice_cloning_mode?: VoiceCloningMode  // disabled, custom, video_audio
  ref_audio_path?: string    // Path to uploaded reference audio (for custom mode)
  ref_audio_text?: string    // Transcription of reference audio (for custom mode)
}

export interface SubtitleStyle {
  // Basic text settings
  font_name?: string
  font_size: number
  color: string
  bold: boolean
  italic: boolean

  // Outline settings
  outline_color: string
  outline_width: number

  // Shadow settings
  shadow: number  // 0-4
  shadow_color: string

  // Position settings
  alignment: 'top' | 'middle' | 'bottom'
  margin_h: number
  margin_v: number
  max_width?: number  // 50-100, percentage of video width

  // Background settings
  back_color: string
  back_opacity: number  // 0-100

  // Advanced text settings
  spacing: number  // -10 to 20
  scale_x: number  // 50-150
  scale_y: number  // 50-150
}

export type SubtitleMode = 'dual' | 'original_only' | 'translated_only'

export interface SubtitlePreset {
  id: string
  name: string
  description: string
  is_builtin: boolean
  is_vertical?: boolean  // True for vertical video presets (9:16)
  subtitle_mode: SubtitleMode  // Which subtitles are enabled
  source_language?: string     // Original subtitle language
  target_language?: string     // Translated subtitle language
  original_style?: SubtitleStyle   // Only needed if mode includes original
  translated_style?: SubtitleStyle // Only needed if mode includes translated
}

export interface FontInfo {
  name: string
  path: string
  is_custom: boolean
  is_recommended?: boolean
  supported_scripts?: string[]  // 'latin', 'cjk', 'arabic', 'hebrew', 'thai', 'devanagari', 'cyrillic', 'greek'
}

export interface SubtitleSettings {
  enabled: boolean
  dual_subtitles: boolean
  font_name: string
  font_size: number
  position: string
  style: string
  chinese_on_top?: boolean
  translated_style?: SubtitleStyle
  original_style?: SubtitleStyle
  source_language?: string  // Original subtitle language (e.g., 'en')
  target_language?: string  // Translated subtitle language (e.g., 'zh-CN')
  default_preset?: string   // Default subtitle preset ID
}

export interface AudioSettings {
  generate_tts: boolean
  replace_original: boolean
  original_volume: number
  tts_volume: number
}

export interface MetadataSettings {
  enabled: boolean
  auto_generate: boolean
  require_review: boolean
  include_source_url: boolean
  max_keywords: number
  default_use_ai_preset_selection: boolean
  // Note: title_prefix and custom_signature are now managed via MetadataPresets
}

export interface BilibiliSettings {
  is_original: boolean  // true = 自制, false = 转载
  default_tid: number   // 0 = AI auto-match, otherwise specific partition ID
  auto_match_partition: boolean  // Use AI to match partition
  source_url_required: boolean   // Require source URL for 转载
}

export interface ThumbnailSettings {
  enabled: boolean
  auto_generate: boolean
  default_use_ai: boolean
  style: string  // 'gradient_bar', 'top_banner', 'corner_tag'
  font_name: string
  font_size: number
  text_color: string
  gradient_color: string
  gradient_opacity: number
}

// Proofreading settings for AI subtitle validation
export interface ProofreadingSettings {
  enabled: boolean
  auto_pause: boolean
  min_confidence: number  // 0.0 - 1.0
  check_grammar: boolean
  check_terminology: boolean
  check_timing: boolean
  check_formatting: boolean
  use_ai_validation: boolean
  auto_optimize: boolean  // Auto-optimize subtitles after proofreading
  optimization_level: 'minimal' | 'moderate' | 'aggressive'  // AI optimization level
}

// Proofreading issue types
export type IssueSeverity = 'info' | 'warning' | 'error' | 'critical'
export type IssueType =
  | 'grammar_error'
  | 'unnatural_phrasing'
  | 'mistranslation'
  | 'inconsistent_term'
  | 'missing_translation'
  | 'speech_too_fast'
  | 'speech_too_slow'
  | 'overlap_detected'
  | 'gap_too_long'
  | 'line_too_long'
  | 'encoding_error'
  | 'empty_segment'

export interface ProofreadingIssue {
  segment_index: number
  issue_type: IssueType
  severity: IssueSeverity
  message: string
  original_text?: string
  translated_text?: string
  suggestion?: string
  start_time?: number
  end_time?: number
  auto_fixable: boolean
}

export interface SegmentProofreadResult {
  index: number
  original_text: string
  translated_text: string
  start_time: number
  end_time: number
  issues: ProofreadingIssue[]
  confidence: number
}

export interface ProofreadingResult {
  segments: SegmentProofreadResult[]
  total_issues: number
  issues_by_severity: Record<IssueSeverity, number>
  issues_by_type: Record<string, number>
  overall_confidence: number
  avg_chars_per_second: number
  terminology_consistency_score: number
  should_pause: boolean
  pause_reason?: string
}

export interface MetadataResult {
  success: boolean
  title: string
  title_translated: string
  description: string
  keywords: string[]
  error?: string
}

// Per-platform metadata for new format
export interface PlatformMetadata {
  title: string
  description: string
  keywords: string[]
}

// Per-platform metadata container
export interface PlatformMetadataMap {
  douyin?: PlatformMetadata
  bilibili?: PlatformMetadata
  xiaohongshu?: PlatformMetadata
  generic?: PlatformMetadata
}

export interface GlobalSettings {
  processing?: ProcessingSettings
  video: VideoSettings
  translation: TranslationSettings
  tts: TTSSettings
  subtitle: SubtitleSettings
  audio: AudioSettings
  metadata?: MetadataSettings
  thumbnail?: ThumbnailSettings
  bilibili?: BilibiliSettings
  proofreading?: ProofreadingSettings
  trending?: TrendingSettings
  tiktok?: TikTokSettings
  auto_upload_bilibili: boolean
  auto_upload_douyin: boolean
  auto_upload_xiaohongshu: boolean
}

export interface TranslationEngine {
  id: string
  name: string
  description: string
  requires_api_key: boolean
  free: boolean
}

export interface TTSEngine {
  id: string
  name: string
  description: string
  requires_api_key: boolean
  requires_local_server: boolean
  supports_voice_cloning: boolean
  server_port?: number
  modes?: string[]
  free: boolean
}

export interface TTSHealthStatus {
  engine: string
  available: boolean
  host?: string
  port?: number
  message: string
}

// Metadata Preset Types
export interface MetadataPreset {
  id: string
  name: string
  description?: string
  title_prefix: string
  custom_signature: string
  tags: string[]
  is_default: boolean
  is_builtin: boolean
  sort_order: number
  created_at?: string
  updated_at?: string
}

export interface MetadataPresetsListResponse {
  presets: MetadataPreset[]
  builtin_count: number
  custom_count: number
}

export interface CreateMetadataPresetRequest {
  name: string
  description?: string
  title_prefix: string
  custom_signature: string
  tags: string[]
}

export interface AIPresetSelectResponse {
  success: boolean
  preset_id: string
  preset_name: string
  confidence: number
  reason: string
  all_matches: Array<{
    preset_id: string
    confidence: number
    reason: string
  }>
}

// Subscription types
export interface Subscription {
  id: string
  platform: 'youtube' | 'tiktok' | 'instagram'
  channel_id: string
  channel_name: string
  channel_url?: string
  channel_avatar?: string
  // Directory for organizing tasks from this subscription
  directory?: string
  last_video_id?: string
  last_video_title?: string
  last_video_published_at?: string
  check_interval: number
  next_check_at?: string
  last_checked_at?: string
  auto_process: boolean
  process_options?: ProcessOptions
  enabled: boolean
  error_count: number
  last_error?: string
  created_at?: string
  updated_at?: string
}

export interface ProcessOptions {
  processing_mode?: 'full' | 'subtitle' | 'direct' | 'auto'
  source_language?: string
  target_language?: string
  // Whisper transcription settings
  whisper_backend?: string
  whisper_model?: string
  whisper_device?: string
  // Translation settings
  translation_engine?: string
  // Video settings
  video_quality?: string
  // Subtitle settings
  add_subtitles?: boolean
  dual_subtitles?: boolean
  use_existing_subtitles?: boolean
  subtitle_preset?: string
  // TTS settings
  add_tts?: boolean
  tts_service?: string
  tts_voice?: string
  voice_cloning_mode?: VoiceCloningMode
  replace_original_audio?: boolean
  original_audio_volume?: number
  tts_audio_volume?: number
  // Metadata settings
  metadata_preset_id?: string
  use_ai_preset_selection?: boolean
  // Upload settings
  upload_bilibili?: boolean
  upload_douyin?: boolean
  upload_xiaohongshu?: boolean
  bilibili_account_uid?: string
  douyin_account_id?: string
  xiaohongshu_account_id?: string
}

export interface CreateSubscriptionRequest {
  platform: string
  channel_url: string
  directory: string  // Required: directory for organizing tasks
  check_interval?: number
  auto_process?: boolean
  process_options?: ProcessOptions
}

export interface UpdateSubscriptionRequest {
  directory?: string
  check_interval?: number
  auto_process?: boolean
  process_options?: ProcessOptions
  enabled?: boolean
}

export interface ChannelLookupResponse {
  success: boolean
  platform?: string
  channel_id?: string
  channel_name?: string
  channel_url?: string
  channel_avatar?: string
  error?: string
}

export interface SubscriptionPlatform {
  id: string
  name: string
  description: string
  url_patterns: string[]
}

export interface NewVideosResponse {
  videos: Array<{
    video_id: string
    title: string
    url: string
    published_at?: string
    thumbnail_url?: string
  }>
  count: number
}

// Historical video fetching
export interface FetchHistoricalVideosRequest {
  start_date: string  // YYYY-MM-DD
  end_date: string    // YYYY-MM-DD
  max_videos?: number // default 50, max 100
}

export interface VideoItem {
  video_id: string
  title: string
  url: string
  published_at?: string
  thumbnail_url?: string
  duration?: number
}

export interface FetchHistoricalVideosResponse {
  success: boolean
  videos: VideoItem[]
  count: number
  start_date: string
  end_date: string
  error?: string
}

// Batch task creation
export interface BatchCreateTasksRequest {
  videos: VideoItem[]
  process_options?: ProcessOptions
}

export interface BatchCreateTasksResponse {
  success: boolean
  created_count: number
  failed_count: number
  task_ids: string[]
  errors: string[]
}

// Directory types for task organization
export interface Directory {
  id: number
  name: string
  description?: string
  task_count: number
  created_at?: string
  updated_at?: string
}

export interface DirectoryListResponse {
  directories: Directory[]
  count: number
}

export interface CreateDirectoryRequest {
  name: string
  description?: string
}

// Trending Video Types

export interface TrendingVideo {
  id: number
  video_id: string
  title: string
  channel_name: string
  channel_url?: string
  thumbnail_url?: string
  duration: number
  view_count: number
  category: string
  platform: string
  video_url: string
  published_at?: string
  fetched_at?: string
}

export interface TrendingCategory {
  id: string
  name: string
  youtube_category_id: number
  description: string
}

export interface TrendingSettings {
  enabled: boolean
  update_interval: number
  last_updated?: string
  enabled_categories: string[]
  max_videos_per_category: number
  time_filter: 'hour' | 'today' | 'week' | 'month' | 'year'
  sort_by: 'relevance' | 'upload_date' | 'view_count' | 'rating'
  min_view_count: number
  max_duration: number
  exclude_shorts: boolean  // Exclude YouTube Shorts (videos <= 60 seconds)
  has_youtube_api_key?: boolean
  youtube_api_key?: string  // Only used for updates
}

export interface TikTokSettings {
  enabled: boolean
  update_interval: number
  last_updated?: string
  region_code: string
  enabled_tags: string[]
  max_videos_per_tag: number
  min_view_count: number
  min_like_count: number
  max_duration: number
}

// YouTube Search
export interface YouTubeSearchResult {
  video_id: string
  title: string
  channel_name: string
  channel_url?: string
  thumbnail_url?: string
  duration: number
  view_count: number
  published_at?: string
  video_url: string
}

export interface YouTubeSearchResponse {
  success: boolean
  results: YouTubeSearchResult[]
  count: number
  error?: string
}

export interface TrendingVideosListResponse {
  videos: TrendingVideo[]
  count: number
  category?: string
}

export interface TrendingCategoriesResponse {
  categories: TrendingCategory[]
}

export interface TrendingRefreshResponse {
  success: boolean
  message: string
  categories_updated: string[]
  total_videos: number
}

export interface TrendingBatchCreateResponse {
  success: boolean
  created_count: number
  failed_count: number
  task_ids: string[]
  errors: string[]
}

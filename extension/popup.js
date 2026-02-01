/**
 * iDubb Browser Extension - Popup Script
 * Handles settings UI and custom publish options
 */

// DOM elements
const elements = {
  // 服务器
  apiUrl: document.getElementById('apiUrl'),
  // 视频设置
  videoQuality: document.getElementById('videoQuality'),
  processingMode: document.getElementById('processingMode'),
  // 翻译设置
  sourceLanguage: document.getElementById('sourceLanguage'),
  targetLanguage: document.getElementById('targetLanguage'),
  translationEngine: document.getElementById('translationEngine'),
  // 字幕设置
  addSubtitles: document.getElementById('addSubtitles'),
  dualSubtitles: document.getElementById('dualSubtitles'),
  subtitlePreset: document.getElementById('subtitlePreset'),
  // 配音设置
  addTts: document.getElementById('addTts'),
  ttsService: document.getElementById('ttsService'),
  ttsVoice: document.getElementById('ttsVoice'),
  replaceOriginalAudio: document.getElementById('replaceOriginalAudio'),
  originalVolume: document.getElementById('originalVolume'),
  originalVolumeValue: document.getElementById('originalVolumeValue'),
  // 发布设置
  uploadDouyin: document.getElementById('uploadDouyin'),
  uploadXiaohongshu: document.getElementById('uploadXiaohongshu'),
  uploadBilibili: document.getElementById('uploadBilibili'),
  // UI
  saveBtn: document.getElementById('saveBtn'),
  status: document.getElementById('status'),
  pendingVideoSection: document.getElementById('pendingVideoSection'),
  pendingVideoUrl: document.getElementById('pendingVideoUrl'),
  publishBtn: document.getElementById('publishBtn'),
  cancelBtn: document.getElementById('cancelBtn')
};

// Current pending video URL
let pendingVideoUrl = null;

// Default settings
const DEFAULT_SETTINGS = {
  apiUrl: 'http://localhost:8888',
  videoQuality: '1080p',
  processingMode: 'smart',
  sourceLanguage: 'auto',
  targetLanguage: 'zh-CN',
  translationEngine: 'deepseek',
  addSubtitles: true,
  dualSubtitles: true,
  subtitlePreset: '',
  addTts: true,
  ttsService: 'edge',
  ttsVoice: 'zh-CN-XiaoxiaoNeural',
  replaceOriginalAudio: false,
  originalVolume: 30,
  uploadDouyin: true,
  uploadXiaohongshu: false,
  uploadBilibili: false
};

// Load settings on popup open
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  await checkPendingVideo();
  setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
  // Volume slider
  elements.originalVolume.addEventListener('input', () => {
    elements.originalVolumeValue.textContent = elements.originalVolume.value + '%';
  });
  
  // Processing mode changes
  elements.processingMode.addEventListener('change', () => {
    const mode = elements.processingMode.value;
    // 根据模式自动设置相关选项
    switch (mode) {
      case 'full_translation':
        elements.addSubtitles.checked = true;
        elements.addTts.checked = true;
        break;
      case 'subtitles_only':
        elements.addSubtitles.checked = true;
        elements.addTts.checked = false;
        break;
      case 'direct_transfer':
        elements.addSubtitles.checked = false;
        elements.addTts.checked = false;
        break;
    }
  });
}

// Load saved settings
async function loadSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get('settings', (data) => {
      const settings = { ...DEFAULT_SETTINGS, ...data.settings };
      
      // 服务器
      elements.apiUrl.value = settings.apiUrl;
      // 视频设置
      elements.videoQuality.value = settings.videoQuality;
      elements.processingMode.value = settings.processingMode;
      // 翻译设置
      elements.sourceLanguage.value = settings.sourceLanguage;
      elements.targetLanguage.value = settings.targetLanguage;
      elements.translationEngine.value = settings.translationEngine;
      // 字幕设置
      elements.addSubtitles.checked = settings.addSubtitles;
      elements.dualSubtitles.checked = settings.dualSubtitles;
      elements.subtitlePreset.value = settings.subtitlePreset;
      // 配音设置
      elements.addTts.checked = settings.addTts;
      elements.ttsService.value = settings.ttsService;
      elements.ttsVoice.value = settings.ttsVoice;
      elements.replaceOriginalAudio.checked = settings.replaceOriginalAudio;
      elements.originalVolume.value = settings.originalVolume;
      elements.originalVolumeValue.textContent = settings.originalVolume + '%';
      // 发布设置
      elements.uploadDouyin.checked = settings.uploadDouyin;
      elements.uploadXiaohongshu.checked = settings.uploadXiaohongshu;
      elements.uploadBilibili.checked = settings.uploadBilibili;
      
      resolve();
    });
  });
}

// Check for pending video from context menu
async function checkPendingVideo() {
  return new Promise((resolve) => {
    chrome.storage.local.get('pendingVideoUrl', (data) => {
      if (data.pendingVideoUrl) {
        pendingVideoUrl = data.pendingVideoUrl;
        elements.pendingVideoSection.style.display = 'block';
        elements.pendingVideoUrl.textContent = pendingVideoUrl;
        
        // Clear the pending URL
        chrome.storage.local.remove('pendingVideoUrl');
      }
      resolve();
    });
  });
}

// Get current settings from form
function getFormSettings() {
  return {
    apiUrl: elements.apiUrl.value.trim() || DEFAULT_SETTINGS.apiUrl,
    videoQuality: elements.videoQuality.value,
    processingMode: elements.processingMode.value,
    sourceLanguage: elements.sourceLanguage.value,
    targetLanguage: elements.targetLanguage.value,
    translationEngine: elements.translationEngine.value,
    addSubtitles: elements.addSubtitles.checked,
    dualSubtitles: elements.dualSubtitles.checked,
    subtitlePreset: elements.subtitlePreset.value,
    addTts: elements.addTts.checked,
    ttsService: elements.ttsService.value,
    ttsVoice: elements.ttsVoice.value,
    replaceOriginalAudio: elements.replaceOriginalAudio.checked,
    originalVolume: parseInt(elements.originalVolume.value),
    uploadDouyin: elements.uploadDouyin.checked,
    uploadXiaohongshu: elements.uploadXiaohongshu.checked,
    uploadBilibili: elements.uploadBilibili.checked
  };
}

// Save settings
elements.saveBtn.addEventListener('click', async () => {
  const settings = getFormSettings();
  chrome.storage.sync.set({ settings }, () => {
    showStatus('✅ 设置已保存', 'success');
  });
});

// Publish pending video
elements.publishBtn.addEventListener('click', async () => {
  if (!pendingVideoUrl) return;

  const settings = getFormSettings();

  elements.publishBtn.disabled = true;
  elements.publishBtn.textContent = '处理中...';

  try {
    // 构建任务参数
    const taskPayload = {
      source_url: pendingVideoUrl,
      source_platform: 'tiktok',
      video_quality: settings.videoQuality,
      source_language: settings.sourceLanguage,
      target_language: settings.targetLanguage,
      translation_engine: settings.translationEngine,
      add_subtitles: settings.addSubtitles,
      dual_subtitles: settings.dualSubtitles,
      subtitle_preset: settings.subtitlePreset || undefined,
      add_tts: settings.addTts,
      tts_service: settings.ttsService,
      tts_voice: settings.ttsVoice,
      replace_original_audio: settings.replaceOriginalAudio,
      original_audio_volume: settings.originalVolume / 100,
      upload_douyin: settings.uploadDouyin,
      upload_xiaohongshu: settings.uploadXiaohongshu,
      upload_bilibili: settings.uploadBilibili
    };
    
    // 根据处理模式调整参数
    if (settings.processingMode === 'direct_transfer') {
      taskPayload.skip_translation = true;
      taskPayload.add_subtitles = false;
      taskPayload.add_tts = false;
    }

    const response = await fetch(`${settings.apiUrl}/api/tasks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(taskPayload)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || '创建任务失败');
    }

    const result = await response.json();
    showStatus(`✅ 任务已创建: ${result.task_id}`, 'success');
    
    // Hide pending section
    elements.pendingVideoSection.style.display = 'none';
    pendingVideoUrl = null;
  } catch (error) {
    showStatus(`❌ 错误: ${error.message}`, 'error');
  } finally {
    elements.publishBtn.disabled = false;
    elements.publishBtn.textContent = '🚀 发布';
  }
});

// Cancel pending video
elements.cancelBtn.addEventListener('click', () => {
  elements.pendingVideoSection.style.display = 'none';
  pendingVideoUrl = null;
});

// Show status message
function showStatus(message, type = '') {
  elements.status.textContent = message;
  elements.status.className = `status ${type}`;
  
  setTimeout(() => {
    elements.status.textContent = '';
    elements.status.className = 'status';
  }, 3000);
}

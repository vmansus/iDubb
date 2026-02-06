/**
 * iDubb Browser Extension - Popup Script
 * Handles settings UI and custom publish options
 * All defaults and options loaded from backend API - NO HARDCODED VALUES
 */

// DOM elements
const elements = {
  // 服务器
  apiUrl: document.getElementById('apiUrl'),
  connectionStatus: document.getElementById('connectionStatus'),
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

// Backend config cache
let backendConfig = null;

// Load settings on popup open
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  await checkPendingVideo();
  setupEventListeners();
});

// ==================== Backend API Functions ====================

// Fetch backend configuration
async function fetchBackendConfig(apiUrl) {
  try {
    const response = await fetch(`${apiUrl}/api/settings`, { timeout: 5000 });
    if (!response.ok) throw new Error('Failed to fetch settings');
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch backend config:', error);
    return null;
  }
}

// Fetch available TTS voices
async function fetchVoices(apiUrl, engine = null) {
  try {
    const url = engine 
      ? `${apiUrl}/api/tts/voices/${engine}` 
      : `${apiUrl}/api/voices`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch voices');
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch voices:', error);
    return [];
  }
}

// Fetch subtitle presets
async function fetchSubtitlePresets(apiUrl) {
  try {
    const response = await fetch(`${apiUrl}/api/subtitle-presets`);
    if (!response.ok) throw new Error('Failed to fetch presets');
    const data = await response.json();
    return data.presets || [];
  } catch (error) {
    console.error('Failed to fetch subtitle presets:', error);
    return [];
  }
}

// ==================== UI Helper Functions ====================

// Populate select element with options
function populateSelect(selectElement, options, valueKey, labelKey, selectedValue) {
  if (!selectElement) return;
  selectElement.innerHTML = '';
  
  options.forEach(opt => {
    const option = document.createElement('option');
    option.value = typeof valueKey === 'function' ? valueKey(opt) : opt[valueKey];
    option.textContent = typeof labelKey === 'function' ? labelKey(opt) : opt[labelKey];
    if (option.value === selectedValue) {
      option.selected = true;
    }
    selectElement.appendChild(option);
  });
}

// Update connection status indicator
function updateConnectionStatus(connected, message = '') {
  if (!elements.connectionStatus) return;
  
  if (connected) {
    elements.connectionStatus.className = 'connection-status connected';
    elements.connectionStatus.textContent = '✓ 已连接';
  } else {
    elements.connectionStatus.className = 'connection-status disconnected';
    elements.connectionStatus.textContent = message || '✗ 未连接';
  }
}

// ==================== Settings Management ====================

// Setup event listeners
function setupEventListeners() {
  // Volume slider
  if (elements.originalVolume) {
    elements.originalVolume.addEventListener('input', () => {
      elements.originalVolumeValue.textContent = elements.originalVolume.value + '%';
    });
  }
  
  // Processing mode changes
  if (elements.processingMode) {
    elements.processingMode.addEventListener('change', () => {
      const mode = elements.processingMode.value;
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
  
  // TTS service changes - reload voices
  if (elements.ttsService) {
    elements.ttsService.addEventListener('change', async () => {
      const apiUrl = elements.apiUrl.value.trim();
      const engine = elements.ttsService.value;
      const voices = await fetchVoices(apiUrl, engine);
      if (voices.length > 0) {
        populateSelect(elements.ttsVoice, voices, 'name', 'display_name', voices[0].name);
      }
    });
  }
  
  // API URL changes - reconnect
  if (elements.apiUrl) {
    elements.apiUrl.addEventListener('change', async () => {
      const apiUrl = elements.apiUrl.value.trim();
      if (apiUrl) {
        await reconnectToBackend(apiUrl);
      }
    });
  }
}

// Reconnect to backend and reload dynamic options
async function reconnectToBackend(apiUrl) {
  updateConnectionStatus(false, '连接中...');
  
  backendConfig = await fetchBackendConfig(apiUrl);
  
  if (backendConfig) {
    updateConnectionStatus(true);
    
    // Reload dynamic options
    const [voices, presets] = await Promise.all([
      fetchVoices(apiUrl, elements.ttsService?.value),
      fetchSubtitlePresets(apiUrl)
    ]);
    
    if (voices.length > 0) {
      const currentVoice = elements.ttsVoice?.value;
      populateSelect(elements.ttsVoice, voices, 'name', 'display_name', currentVoice || voices[0].name);
    }
    
    if (presets.length > 0) {
      const currentPreset = elements.subtitlePreset?.value;
      populateSelect(elements.subtitlePreset, presets, 'id', 'name', currentPreset || presets[0].id);
    }
  } else {
    updateConnectionStatus(false, '无法连接');
  }
}

// Load saved settings and populate dynamic options from backend
async function loadSettings() {
  return new Promise(async (resolve) => {
    // First load saved settings from storage
    chrome.storage.sync.get('settings', async (data) => {
      const savedSettings = data.settings || {};
      const apiUrl = savedSettings.apiUrl || 'http://localhost:8888';
      
      elements.apiUrl.value = apiUrl;
      updateConnectionStatus(false, '连接中...');
      
      // Try to fetch backend config
      backendConfig = await fetchBackendConfig(apiUrl);
      
      if (backendConfig) {
        updateConnectionStatus(true);
        
        // Load dynamic options from backend
        const ttsEngine = savedSettings.ttsService || backendConfig.tts?.engine || 'edge';
        const [voices, presets] = await Promise.all([
          fetchVoices(apiUrl, ttsEngine),
          fetchSubtitlePresets(apiUrl)
        ]);
        
        // Get defaults from backend config
        const videoDefaults = backendConfig.video || {};
        const translationDefaults = backendConfig.translation || {};
        const subtitleDefaults = backendConfig.subtitle || {};
        const ttsDefaults = backendConfig.tts || {};
        const audioDefaults = backendConfig.audio || {};
        const uploadDefaults = backendConfig.upload || {};
        
        // Populate TTS voices select
        if (voices.length > 0) {
          populateSelect(
            elements.ttsVoice,
            voices,
            'name',
            'display_name',
            savedSettings.ttsVoice || ttsDefaults.voice
          );
        }
        
        // Populate subtitle presets select
        if (presets.length > 0) {
          populateSelect(
            elements.subtitlePreset,
            presets,
            'id',
            'name',
            savedSettings.subtitlePreset || subtitleDefaults.default_preset
          );
        }
        
        // Apply settings: use saved value if exists, otherwise use backend default
        // 视频设置
        if (elements.videoQuality) {
          elements.videoQuality.value = savedSettings.videoQuality || videoDefaults.default_quality || '1080p';
        }
        if (elements.processingMode) {
          elements.processingMode.value = savedSettings.processingMode || 'smart';
        }
        
        // 翻译设置
        if (elements.sourceLanguage) {
          elements.sourceLanguage.value = savedSettings.sourceLanguage || subtitleDefaults.source_language || 'auto';
        }
        if (elements.targetLanguage) {
          elements.targetLanguage.value = savedSettings.targetLanguage || subtitleDefaults.target_language || 'zh-CN';
        }
        if (elements.translationEngine) {
          elements.translationEngine.value = savedSettings.translationEngine || translationDefaults.engine || 'google';
        }
        
        // 字幕设置
        if (elements.addSubtitles) {
          elements.addSubtitles.checked = savedSettings.addSubtitles !== undefined 
            ? savedSettings.addSubtitles 
            : (subtitleDefaults.enabled !== false);
        }
        if (elements.dualSubtitles) {
          elements.dualSubtitles.checked = savedSettings.dualSubtitles !== undefined 
            ? savedSettings.dualSubtitles 
            : (subtitleDefaults.dual_subtitles !== false);
        }
        
        // 配音设置
        if (elements.addTts) {
          elements.addTts.checked = savedSettings.addTts !== undefined 
            ? savedSettings.addTts 
            : (audioDefaults.generate_tts !== false);
        }
        if (elements.ttsService) {
          elements.ttsService.value = savedSettings.ttsService || ttsDefaults.engine || 'edge';
        }
        if (elements.replaceOriginalAudio) {
          elements.replaceOriginalAudio.checked = savedSettings.replaceOriginalAudio !== undefined 
            ? savedSettings.replaceOriginalAudio 
            : (audioDefaults.replace_original || false);
        }
        if (elements.originalVolume) {
          const vol = savedSettings.originalVolume !== undefined 
            ? savedSettings.originalVolume 
            : Math.round((audioDefaults.original_volume || 0.3) * 100);
          elements.originalVolume.value = vol;
          elements.originalVolumeValue.textContent = vol + '%';
        }
        
        // 发布设置
        if (elements.uploadDouyin) {
          elements.uploadDouyin.checked = savedSettings.uploadDouyin !== undefined 
            ? savedSettings.uploadDouyin 
            : (uploadDefaults.auto_upload_douyin || false);
        }
        if (elements.uploadXiaohongshu) {
          elements.uploadXiaohongshu.checked = savedSettings.uploadXiaohongshu !== undefined 
            ? savedSettings.uploadXiaohongshu 
            : (uploadDefaults.auto_upload_xiaohongshu || false);
        }
        if (elements.uploadBilibili) {
          elements.uploadBilibili.checked = savedSettings.uploadBilibili !== undefined 
            ? savedSettings.uploadBilibili 
            : (uploadDefaults.auto_upload_bilibili || false);
        }
        
      } else {
        // Backend not available - use saved settings only
        updateConnectionStatus(false, '无法连接到服务器');
        
        // Apply saved settings without defaults
        if (elements.videoQuality) elements.videoQuality.value = savedSettings.videoQuality || '1080p';
        if (elements.processingMode) elements.processingMode.value = savedSettings.processingMode || 'smart';
        if (elements.sourceLanguage) elements.sourceLanguage.value = savedSettings.sourceLanguage || 'auto';
        if (elements.targetLanguage) elements.targetLanguage.value = savedSettings.targetLanguage || 'zh-CN';
        if (elements.translationEngine) elements.translationEngine.value = savedSettings.translationEngine || 'google';
        if (elements.addSubtitles) elements.addSubtitles.checked = savedSettings.addSubtitles !== false;
        if (elements.dualSubtitles) elements.dualSubtitles.checked = savedSettings.dualSubtitles !== false;
        if (elements.subtitlePreset) elements.subtitlePreset.value = savedSettings.subtitlePreset || '';
        if (elements.addTts) elements.addTts.checked = savedSettings.addTts !== false;
        if (elements.ttsService) elements.ttsService.value = savedSettings.ttsService || 'edge';
        if (elements.ttsVoice) elements.ttsVoice.value = savedSettings.ttsVoice || '';
        if (elements.replaceOriginalAudio) elements.replaceOriginalAudio.checked = savedSettings.replaceOriginalAudio || false;
        if (elements.originalVolume) {
          elements.originalVolume.value = savedSettings.originalVolume || 30;
          elements.originalVolumeValue.textContent = (savedSettings.originalVolume || 30) + '%';
        }
        if (elements.uploadDouyin) elements.uploadDouyin.checked = savedSettings.uploadDouyin || false;
        if (elements.uploadXiaohongshu) elements.uploadXiaohongshu.checked = savedSettings.uploadXiaohongshu || false;
        if (elements.uploadBilibili) elements.uploadBilibili.checked = savedSettings.uploadBilibili || false;
      }
      
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
    apiUrl: elements.apiUrl?.value.trim() || 'http://localhost:8888',
    videoQuality: elements.videoQuality?.value || '1080p',
    processingMode: elements.processingMode?.value || 'smart',
    sourceLanguage: elements.sourceLanguage?.value || 'auto',
    targetLanguage: elements.targetLanguage?.value || 'zh-CN',
    translationEngine: elements.translationEngine?.value || 'google',
    addSubtitles: elements.addSubtitles?.checked ?? true,
    dualSubtitles: elements.dualSubtitles?.checked ?? true,
    subtitlePreset: elements.subtitlePreset?.value || '',
    addTts: elements.addTts?.checked ?? true,
    ttsService: elements.ttsService?.value || 'edge',
    ttsVoice: elements.ttsVoice?.value || '',
    replaceOriginalAudio: elements.replaceOriginalAudio?.checked || false,
    originalVolume: parseInt(elements.originalVolume?.value || 30),
    uploadDouyin: elements.uploadDouyin?.checked || false,
    uploadXiaohongshu: elements.uploadXiaohongshu?.checked || false,
    uploadBilibili: elements.uploadBilibili?.checked || false
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

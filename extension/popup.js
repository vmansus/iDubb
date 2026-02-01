/**
 * iDubb Browser Extension - Popup Script
 * Handles settings UI and custom publish options
 * All defaults and options loaded from backend API
 */

// DOM elements
const elements = {
  apiUrl: document.getElementById('apiUrl'),
  uploadDouyin: document.getElementById('uploadDouyin'),
  uploadXiaohongshu: document.getElementById('uploadXiaohongshu'),
  targetLanguage: document.getElementById('targetLanguage'),
  addSubtitles: document.getElementById('addSubtitles'),
  addTts: document.getElementById('addTts'),
  ttsVoice: document.getElementById('ttsVoice'),
  subtitlePreset: document.getElementById('subtitlePreset'),
  saveBtn: document.getElementById('saveBtn'),
  status: document.getElementById('status'),
  pendingVideoSection: document.getElementById('pendingVideoSection'),
  pendingVideoUrl: document.getElementById('pendingVideoUrl'),
  publishBtn: document.getElementById('publishBtn'),
  cancelBtn: document.getElementById('cancelBtn'),
  connectionStatus: document.getElementById('connectionStatus')
};

// Current pending video URL
let pendingVideoUrl = null;

// Backend config cache
let backendConfig = null;

// Load settings on popup open
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  await checkPendingVideo();
});

// Get API URL (from storage or default)
async function getApiUrl() {
  return new Promise((resolve) => {
    chrome.storage.sync.get('settings', (data) => {
      const settings = data.settings || {};
      resolve(settings.apiUrl || 'http://localhost:8000');
    });
  });
}

// Fetch backend configuration
async function fetchBackendConfig(apiUrl) {
  try {
    const response = await fetch(`${apiUrl}/api/settings`);
    if (!response.ok) throw new Error('Failed to fetch settings');
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch backend config:', error);
    return null;
  }
}

// Fetch available TTS voices
async function fetchVoices(apiUrl, language = null) {
  try {
    const url = language ? `${apiUrl}/api/voices?language=${language}` : `${apiUrl}/api/voices`;
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

// Populate select element with options
function populateSelect(selectElement, options, valueKey, labelKey, selectedValue) {
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

// Load saved settings and populate dynamic options
async function loadSettings() {
  return new Promise(async (resolve) => {
    // First load saved settings from storage
    chrome.storage.sync.get('settings', async (data) => {
      const settings = data.settings || {};
      const apiUrl = settings.apiUrl || 'http://localhost:8000';
      
      elements.apiUrl.value = apiUrl;
      
      // Try to fetch backend config
      backendConfig = await fetchBackendConfig(apiUrl);
      
      if (backendConfig) {
        updateConnectionStatus(true);
        
        // Load dynamic options from backend
        const [voices, presets] = await Promise.all([
          fetchVoices(apiUrl),
          fetchSubtitlePresets(apiUrl)
        ]);
        
        // Get defaults from backend config
        const uploadDefaults = backendConfig.upload || {};
        const ttsDefaults = backendConfig.tts || {};
        const subtitleDefaults = backendConfig.subtitle || {};
        const audioDefaults = backendConfig.audio || {};
        
        // Populate TTS voices select
        if (voices.length > 0) {
          populateSelect(
            elements.ttsVoice,
            voices,
            'name',
            'display_name',
            settings.ttsVoice || ttsDefaults.voice || voices[0].name
          );
        }
        
        // Populate subtitle presets select
        if (presets.length > 0 && elements.subtitlePreset) {
          populateSelect(
            elements.subtitlePreset,
            presets,
            'id',
            'name',
            settings.subtitlePreset || subtitleDefaults.default_preset || presets[0].id
          );
        }
        
        // Apply settings: use saved value if exists, otherwise use backend default
        elements.uploadDouyin.checked = settings.uploadDouyin !== undefined 
          ? settings.uploadDouyin 
          : (uploadDefaults.auto_upload_douyin || false);
        
        elements.uploadXiaohongshu.checked = settings.uploadXiaohongshu !== undefined 
          ? settings.uploadXiaohongshu 
          : (uploadDefaults.auto_upload_xiaohongshu || false);
        
        elements.targetLanguage.value = settings.targetLanguage || subtitleDefaults.target_language || 'zh-CN';
        
        elements.addSubtitles.checked = settings.addSubtitles !== undefined 
          ? settings.addSubtitles 
          : (subtitleDefaults.enabled !== false);
        
        elements.addTts.checked = settings.addTts !== undefined 
          ? settings.addTts 
          : (audioDefaults.generate_tts !== false);
        
      } else {
        // Backend not available - use cached settings or show warning
        updateConnectionStatus(false, '无法连接到服务器');
        
        // Still apply saved settings
        elements.uploadDouyin.checked = settings.uploadDouyin || false;
        elements.uploadXiaohongshu.checked = settings.uploadXiaohongshu || false;
        elements.targetLanguage.value = settings.targetLanguage || 'zh-CN';
        elements.addSubtitles.checked = settings.addSubtitles !== false;
        elements.addTts.checked = settings.addTts !== false;
        elements.ttsVoice.value = settings.ttsVoice || '';
        if (elements.subtitlePreset) {
          elements.subtitlePreset.value = settings.subtitlePreset || '';
        }
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

// Save settings
elements.saveBtn.addEventListener('click', async () => {
  const settings = {
    apiUrl: elements.apiUrl.value.trim() || 'http://localhost:8000',
    uploadDouyin: elements.uploadDouyin.checked,
    uploadXiaohongshu: elements.uploadXiaohongshu.checked,
    targetLanguage: elements.targetLanguage.value,
    addSubtitles: elements.addSubtitles.checked,
    addTts: elements.addTts.checked,
    ttsVoice: elements.ttsVoice.value,
    subtitlePreset: elements.subtitlePreset ? elements.subtitlePreset.value : null
  };

  chrome.storage.sync.set({ settings }, () => {
    showStatus('设置已保存', 'success');
  });
});

// Refresh connection when API URL changes
elements.apiUrl.addEventListener('change', async () => {
  const apiUrl = elements.apiUrl.value.trim();
  if (apiUrl) {
    updateConnectionStatus(false, '连接中...');
    backendConfig = await fetchBackendConfig(apiUrl);
    if (backendConfig) {
      updateConnectionStatus(true);
      // Reload dynamic options
      const [voices, presets] = await Promise.all([
        fetchVoices(apiUrl),
        fetchSubtitlePresets(apiUrl)
      ]);
      
      if (voices.length > 0) {
        const currentVoice = elements.ttsVoice.value;
        populateSelect(elements.ttsVoice, voices, 'name', 'display_name', currentVoice);
      }
      
      if (presets.length > 0 && elements.subtitlePreset) {
        const currentPreset = elements.subtitlePreset.value;
        populateSelect(elements.subtitlePreset, presets, 'id', 'name', currentPreset);
      }
    } else {
      updateConnectionStatus(false, '无法连接');
    }
  }
});

// Publish pending video
elements.publishBtn.addEventListener('click', async () => {
  if (!pendingVideoUrl) return;

  const settings = {
    apiUrl: elements.apiUrl.value.trim() || 'http://localhost:8000',
    uploadDouyin: elements.uploadDouyin.checked,
    uploadXiaohongshu: elements.uploadXiaohongshu.checked,
    targetLanguage: elements.targetLanguage.value,
    addSubtitles: elements.addSubtitles.checked,
    addTts: elements.addTts.checked,
    ttsVoice: elements.ttsVoice.value,
    subtitlePreset: elements.subtitlePreset ? elements.subtitlePreset.value : null
  };

  elements.publishBtn.disabled = true;
  elements.publishBtn.textContent = '处理中...';

  try {
    const response = await fetch(`${settings.apiUrl}/api/tasks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        source_url: pendingVideoUrl,
        source_platform: 'tiktok',
        target_language: settings.targetLanguage,
        add_subtitles: settings.addSubtitles,
        add_tts: settings.addTts,
        tts_voice: settings.ttsVoice,
        subtitle_preset: settings.subtitlePreset,
        upload_douyin: settings.uploadDouyin,
        upload_xiaohongshu: settings.uploadXiaohongshu
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || '创建任务失败');
    }

    const result = await response.json();
    showStatus(`任务已创建: ${result.task_id}`, 'success');
    
    // Hide pending section
    elements.pendingVideoSection.style.display = 'none';
    pendingVideoUrl = null;
  } catch (error) {
    showStatus(`错误: ${error.message}`, 'error');
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

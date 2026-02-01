/**
 * iDubb Browser Extension - Popup Script
 * Handles settings UI and custom publish options
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
  saveBtn: document.getElementById('saveBtn'),
  status: document.getElementById('status'),
  pendingVideoSection: document.getElementById('pendingVideoSection'),
  pendingVideoUrl: document.getElementById('pendingVideoUrl'),
  publishBtn: document.getElementById('publishBtn'),
  cancelBtn: document.getElementById('cancelBtn')
};

// Current pending video URL
let pendingVideoUrl = null;

// Load settings on popup open
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  await checkPendingVideo();
});

// Load saved settings
async function loadSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get('settings', (data) => {
      const settings = data.settings || {};
      
      elements.apiUrl.value = settings.apiUrl || 'http://localhost:8000';
      elements.uploadDouyin.checked = settings.uploadDouyin || false;
      elements.uploadXiaohongshu.checked = settings.uploadXiaohongshu || false;
      elements.targetLanguage.value = settings.targetLanguage || 'zh-CN';
      elements.addSubtitles.checked = settings.addSubtitles !== false;
      elements.addTts.checked = settings.addTts !== false;
      elements.ttsVoice.value = settings.ttsVoice || 'zh-CN-XiaoxiaoNeural';
      
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
    ttsVoice: elements.ttsVoice.value
  };

  chrome.storage.sync.set({ settings }, () => {
    showStatus('设置已保存', 'success');
  });
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
    ttsVoice: elements.ttsVoice.value
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

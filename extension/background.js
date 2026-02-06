/**
 * iDubb Browser Extension - Background Service Worker
 * Handles context menu and API communication
 * Uses saved settings - NO HARDCODED DEFAULTS
 */

// Create context menu on install
chrome.runtime.onInstalled.addListener(() => {
  // Parent menu
  chrome.contextMenus.create({
    id: 'idubb-parent',
    title: 'iDubb 一键发布',
    contexts: ['page', 'video', 'link']
  });

  // Quick actions - use saved settings
  chrome.contextMenus.create({
    id: 'idubb-quick',
    parentId: 'idubb-parent',
    title: '🚀 使用默认配置发布',
    contexts: ['page', 'video', 'link']
  });

  chrome.contextMenus.create({
    id: 'idubb-separator1',
    parentId: 'idubb-parent',
    type: 'separator',
    contexts: ['page', 'video', 'link']
  });

  // Platform-specific quick actions
  chrome.contextMenus.create({
    id: 'idubb-douyin',
    parentId: 'idubb-parent',
    title: '📱 仅发布到抖音',
    contexts: ['page', 'video', 'link']
  });

  chrome.contextMenus.create({
    id: 'idubb-xiaohongshu',
    parentId: 'idubb-parent',
    title: '📕 仅发布到小红书',
    contexts: ['page', 'video', 'link']
  });

  chrome.contextMenus.create({
    id: 'idubb-bilibili',
    parentId: 'idubb-parent',
    title: '📺 仅发布到 B站',
    contexts: ['page', 'video', 'link']
  });

  chrome.contextMenus.create({
    id: 'idubb-separator2',
    parentId: 'idubb-parent',
    type: 'separator',
    contexts: ['page', 'video', 'link']
  });

  chrome.contextMenus.create({
    id: 'idubb-custom',
    parentId: 'idubb-parent',
    title: '⚙️ 自定义选项...',
    contexts: ['page', 'video', 'link']
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  console.log('[iDubb] 右键菜单点击:', info.menuItemId);
  
  const videoUrl = await getVideoUrl(info, tab);
  console.log('[iDubb] 获取到视频 URL:', videoUrl);
  
  if (!videoUrl) {
    showNotification('错误', '无法获取视频链接，请确保在 TikTok/YouTube 视频页面右键点击');
    return;
  }

  // Get saved settings
  const settings = await getSettings();
  
  // Handle different menu actions
  switch (info.menuItemId) {
    case 'idubb-quick':
      // Use all saved settings as-is
      await createTask(videoUrl, settings);
      break;
      
    case 'idubb-douyin':
      // Override: only Douyin
      await createTask(videoUrl, { 
        ...settings, 
        uploadDouyin: true, 
        uploadXiaohongshu: false, 
        uploadBilibili: false 
      });
      break;
      
    case 'idubb-xiaohongshu':
      // Override: only Xiaohongshu
      await createTask(videoUrl, { 
        ...settings, 
        uploadDouyin: false, 
        uploadXiaohongshu: true, 
        uploadBilibili: false 
      });
      break;
      
    case 'idubb-bilibili':
      // Override: only Bilibili
      await createTask(videoUrl, { 
        ...settings, 
        uploadDouyin: false, 
        uploadXiaohongshu: false, 
        uploadBilibili: true 
      });
      break;
      
    case 'idubb-custom':
      // Open popup for custom options
      chrome.storage.local.set({ pendingVideoUrl: videoUrl });
      chrome.action.openPopup();
      break;
      
    default:
      return;
  }
});

// Get video URL from context
async function getVideoUrl(info, tab) {
  // If clicked on a link
  if (info.linkUrl && isVideoUrl(info.linkUrl)) {
    return info.linkUrl;
  }

  // If on supported video page
  if (tab.url && isVideoUrl(tab.url)) {
    // Check if it's a video detail page
    if (tab.url.includes('/video/') || tab.url.includes('/watch')) {
      return tab.url;
    }

    // Try to get video URL from content script (for feed pages)
    try {
      const response = await chrome.tabs.sendMessage(tab.id, { action: 'getVideoUrl' });
      if (response && response.videoUrl) {
        return response.videoUrl;
      }
    } catch (e) {
      console.log('[iDubb] Content script error:', e.message);
    }
  }

  return null;
}

function isVideoUrl(url) {
  if (!url) return false;
  return (
    url.includes('tiktok.com') ||
    url.includes('vm.tiktok.com') ||
    url.includes('youtube.com') ||
    url.includes('youtu.be')
  );
}

// Get saved settings from storage
async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get('settings', (data) => {
      // Return saved settings, or empty object if none
      // popup.js will have loaded defaults from backend
      resolve(data.settings || {});
    });
  });
}

// Create task using saved settings
async function createTask(videoUrl, settings) {
  const apiUrl = settings.apiUrl || 'http://localhost:8888';
  
  console.log('[iDubb] Creating task with settings:', settings);

  // Build task payload - only include relevant settings
  const taskPayload = {
    source_url: videoUrl,
    source_platform: detectPlatform(videoUrl),
    // Video quality
    video_quality: settings.videoQuality,
    // Language
    source_language: settings.sourceLanguage,
    target_language: settings.targetLanguage,
    // Upload targets
    upload_douyin: settings.uploadDouyin,
    upload_xiaohongshu: settings.uploadXiaohongshu,
    upload_bilibili: settings.uploadBilibili
  };

  // Handle processing mode
  if (settings.processingMode === 'direct_transfer') {
    // 直接搬运：不需要翻译、字幕、配音
    taskPayload.skip_translation = true;
    taskPayload.add_subtitles = false;
    taskPayload.add_tts = false;
  } else if (settings.processingMode === 'subtitles_only') {
    // 仅字幕：需要翻译和字幕，不需要配音
    taskPayload.add_subtitles = true;
    taskPayload.add_tts = false;
    taskPayload.translation_engine = settings.translationEngine;
    // 字幕相关配置
    taskPayload.dual_subtitles = settings.dualSubtitles;
    if (settings.subtitlePreset) {
      taskPayload.subtitle_preset = settings.subtitlePreset;
    }
  } else {
    // full_translation 或 smart：根据实际配置决定
    taskPayload.add_subtitles = settings.addSubtitles;
    taskPayload.add_tts = settings.addTts;
    
    // 只在需要翻译时传翻译配置
    if (settings.addSubtitles || settings.addTts) {
      taskPayload.translation_engine = settings.translationEngine;
    }
    
    // 只在需要字幕时传字幕配置
    if (settings.addSubtitles) {
      taskPayload.dual_subtitles = settings.dualSubtitles;
      if (settings.subtitlePreset) {
        taskPayload.subtitle_preset = settings.subtitlePreset;
      }
    }
    
    // 只在需要配音时传 TTS 配置
    if (settings.addTts) {
      taskPayload.tts_service = settings.ttsService;
      taskPayload.tts_voice = settings.ttsVoice;
      taskPayload.replace_original_audio = settings.replaceOriginalAudio;
      taskPayload.original_audio_volume = (settings.originalVolume || 30) / 100;
    }
  }

  // Remove undefined values
  Object.keys(taskPayload).forEach(key => {
    if (taskPayload[key] === undefined) {
      delete taskPayload[key];
    }
  });

  // Show processing notification
  const platforms = [];
  if (settings.uploadDouyin) platforms.push('抖音');
  if (settings.uploadXiaohongshu) platforms.push('小红书');
  if (settings.uploadBilibili) platforms.push('B站');
  const platformText = platforms.length > 0 ? platforms.join('+') : '不上传';
  
  showNotification('处理中', `正在创建任务...\n目标: ${platformText}`);

  try {
    const response = await fetch(`${apiUrl}/api/tasks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(taskPayload)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || '创建任务失败');
    }

    const result = await response.json();
    showNotification('成功 ✓', `任务已创建: ${result.task_id}\n视频将自动处理`);
    
    return result;
  } catch (error) {
    console.error('[iDubb] Create task error:', error);
    showNotification('错误', `创建失败: ${error.message}`);
    throw error;
  }
}

function detectPlatform(url) {
  if (url.includes('tiktok.com') || url.includes('vm.tiktok.com')) {
    return 'tiktok';
  }
  if (url.includes('youtube.com') || url.includes('youtu.be')) {
    return 'youtube';
  }
  return 'auto';
}

function showNotification(title, message) {
  if (chrome.notifications) {
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: `iDubb - ${title}`,
      message: message
    });
  }
}

// Listen for messages from content script or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'createTask') {
    createTask(request.videoUrl, request.options)
      .then(result => sendResponse({ success: true, result }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
  
  if (request.action === 'getSettings') {
    getSettings().then(settings => sendResponse(settings));
    return true;
  }
});

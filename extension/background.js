/**
 * iDubb Browser Extension - Background Service Worker
 * Handles context menu and API communication
 */

// Default settings
const DEFAULT_SETTINGS = {
  apiUrl: 'http://localhost:8000',
  uploadDouyin: true,
  uploadXiaohongshu: false,
  targetLanguage: 'zh-CN',
  addSubtitles: true,
  addTts: true,
  ttsVoice: 'zh-CN-XiaoxiaoNeural'
};

// Create context menu on install
chrome.runtime.onInstalled.addListener(() => {
  // Parent menu
  chrome.contextMenus.create({
    id: 'idubb-parent',
    title: 'iDubb 一键发布',
    contexts: ['page', 'video', 'link']
  });

  // Submenu items
  chrome.contextMenus.create({
    id: 'idubb-douyin',
    parentId: 'idubb-parent',
    title: '📱 发布到抖音',
    contexts: ['page', 'video', 'link']
  });

  chrome.contextMenus.create({
    id: 'idubb-xiaohongshu',
    parentId: 'idubb-parent',
    title: '📕 发布到小红书',
    contexts: ['page', 'video', 'link']
  });

  chrome.contextMenus.create({
    id: 'idubb-both',
    parentId: 'idubb-parent',
    title: '🚀 发布到全部平台',
    contexts: ['page', 'video', 'link']
  });

  chrome.contextMenus.create({
    id: 'idubb-separator',
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

  // Initialize default settings
  chrome.storage.sync.get('settings', (data) => {
    if (!data.settings) {
      chrome.storage.sync.set({ settings: DEFAULT_SETTINGS });
    }
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  console.log('[iDubb] 右键菜单点击:', info.menuItemId);
  console.log('[iDubb] 页面 URL:', tab?.url);
  console.log('[iDubb] 链接 URL:', info.linkUrl);
  
  // 先弹个 alert 确认事件触发了
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (msg) => alert(msg),
      args: ['iDubb: 正在处理...']
    });
  } catch (e) {
    console.log('[iDubb] 无法弹窗:', e.message);
  }
  
  const videoUrl = await getVideoUrl(info, tab);
  console.log('[iDubb] 获取到视频 URL:', videoUrl);
  
  if (!videoUrl) {
    console.log('[iDubb] 错误: 无法获取视频链接');
    showNotification('错误', '无法获取视频链接，请确保在 TikTok 视频页面右键点击');
    // 也用 alert 提示
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => alert('iDubb 错误: 无法获取视频链接，请确保在 TikTok 视频页面')
      });
    } catch (e) {}
    return;
  }

  const settings = await getSettings();
  
  let uploadDouyin = false;
  let uploadXiaohongshu = false;

  switch (info.menuItemId) {
    case 'idubb-douyin':
      uploadDouyin = true;
      break;
    case 'idubb-xiaohongshu':
      uploadXiaohongshu = true;
      break;
    case 'idubb-both':
      uploadDouyin = true;
      uploadXiaohongshu = true;
      break;
    case 'idubb-custom':
      // Open popup for custom options
      chrome.action.openPopup();
      // Store the URL for the popup to use
      chrome.storage.local.set({ pendingVideoUrl: videoUrl });
      return;
    default:
      return;
  }

  await createTask(videoUrl, { ...settings, uploadDouyin, uploadXiaohongshu });
});

// Get video URL from context
async function getVideoUrl(info, tab) {
  // If clicked on a link
  if (info.linkUrl) {
    if (isTikTokUrl(info.linkUrl)) {
      return info.linkUrl;
    }
  }

  // If on TikTok page, try to get current video URL
  if (tab.url && isTikTokUrl(tab.url)) {
    console.log('[iDubb] 在 TikTok 页面，URL:', tab.url);
    
    // Check if it's a video page (直接返回)
    if (tab.url.includes('/video/')) {
      console.log('[iDubb] 是视频详情页，直接使用 URL');
      return tab.url;
    }

    // Try to get video URL from content script (首页 feed 等情况)
    console.log('[iDubb] 不是视频详情页，尝试从 content script 获取');
    try {
      const response = await chrome.tabs.sendMessage(tab.id, { action: 'getVideoUrl' });
      console.log('[iDubb] Content script 响应:', response);
      if (response && response.videoUrl) {
        return response.videoUrl;
      }
    } catch (e) {
      console.log('[iDubb] Content script 错误:', e.message);
    }
    
    // 如果还是没有，提示用户点击具体视频
    console.log('[iDubb] 无法获取视频 URL，可能需要点击进入具体视频页面');
  }

  return null;
}

function isTikTokUrl(url) {
  return url && (
    url.includes('tiktok.com') ||
    url.includes('vm.tiktok.com')
  );
}

async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get('settings', (data) => {
      resolve(data.settings || DEFAULT_SETTINGS);
    });
  });
}

async function createTask(videoUrl, options) {
  const settings = await getSettings();
  const apiUrl = settings.apiUrl || DEFAULT_SETTINGS.apiUrl;

  showNotification('处理中', `正在创建任务: ${videoUrl.substring(0, 50)}...`);

  try {
    const response = await fetch(`${apiUrl}/api/tasks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        source_url: videoUrl,
        source_platform: 'tiktok',
        target_language: options.targetLanguage || 'zh-CN',
        add_subtitles: options.addSubtitles !== false,
        add_tts: options.addTts !== false,
        tts_voice: options.ttsVoice || 'zh-CN-XiaoxiaoNeural',
        upload_douyin: options.uploadDouyin || false,
        upload_xiaohongshu: options.uploadXiaohongshu || false,
        // Use transfer mode if no translation needed
        transfer_only: false
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || '创建任务失败');
    }

    const result = await response.json();
    showNotification('成功', `任务已创建: ${result.task_id}\n视频将自动处理并发布`);
    
    return result;
  } catch (error) {
    console.error('Create task error:', error);
    showNotification('错误', `创建任务失败: ${error.message}`);
    throw error;
  }
}

function showNotification(title, message) {
  // Use chrome notifications if available
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
    return true; // Keep channel open for async response
  }
  
  if (request.action === 'getSettings') {
    getSettings().then(settings => sendResponse(settings));
    return true;
  }
});

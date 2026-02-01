/**
 * iDubb Browser Extension - Content Script
 * Runs on TikTok pages to detect videos and enable quick actions
 * Uses saved settings from popup - NO HARDCODED DEFAULTS
 */

// Global saved settings (loaded from storage)
let savedSettings = null;

// Load saved settings from storage
async function loadSavedSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get('settings', (data) => {
      savedSettings = data.settings || {};
      console.log('[iDubb] Loaded saved settings:', savedSettings);
      resolve(savedSettings);
    });
  });
}

// Listen for settings changes (when user updates popup config)
chrome.storage.onChanged.addListener((changes, namespace) => {
  if (namespace === 'sync' && changes.settings) {
    savedSettings = changes.settings.newValue || {};
    console.log('[iDubb] Settings updated:', savedSettings);
  }
});

// Get default mode from saved settings
function getDefaultMode() {
  if (!savedSettings) return 'smart';
  
  // Map popup's processingMode to content script's mode names
  const modeMap = {
    'full_translation': 'full_translation',
    'subtitles_only': 'subtitles_only',
    'direct_transfer': 'direct_transfer',
    'smart': 'smart'
  };
  return modeMap[savedSettings.processingMode] || 'smart';
}

// Get default platforms from saved settings
function getDefaultPlatforms() {
  if (!savedSettings) return { douyin: true, xiaohongshu: false, bilibili: false };
  
  return {
    douyin: savedSettings.uploadDouyin !== undefined ? savedSettings.uploadDouyin : true,
    xiaohongshu: savedSettings.uploadXiaohongshu !== undefined ? savedSettings.uploadXiaohongshu : false,
    bilibili: savedSettings.uploadBilibili !== undefined ? savedSettings.uploadBilibili : false
  };
}

// Inject styles for iDubb menu
const idubbStyles = document.createElement('style');
idubbStyles.textContent = `
  .idubb-menu {
    position: absolute;
    top: 100%;
    right: 0;
    background: rgba(22, 24, 35, 0.95);
    border-radius: 8px;
    padding: 8px 0;
    min-width: 160px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    backdrop-filter: blur(10px);
    z-index: 1000;
  }
  .idubb-menu-section {
    padding: 4px 12px;
    color: #888;
    font-size: 11px;
    text-transform: uppercase;
  }
  .idubb-menu-divider {
    height: 1px;
    background: rgba(255,255,255,0.1);
    margin: 6px 0;
  }
  .idubb-menu-item {
    padding: 10px 12px;
    color: white;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .idubb-menu-item:hover {
    background: rgba(255,255,255,0.1);
  }
  .idubb-menu-item.selected {
    background: rgba(254, 44, 85, 0.2);
    color: #fe2c55;
  }
  .idubb-menu-item.selected::after {
    content: '✓';
    margin-left: auto;
    font-size: 12px;
  }
  .idubb-menu-submit {
    background: #fe2c55 !important;
    margin: 8px;
    border-radius: 4px;
    justify-content: center;
    font-weight: bold;
  }
  .idubb-menu-submit:hover {
    background: #ff4466 !important;
  }
  .idubb-btn {
    width: 36px;
    height: 36px;
    background: rgba(254, 44, 85, 0.9);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    cursor: pointer;
    transition: transform 0.2s, background 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }
  .idubb-btn:hover {
    transform: scale(1.1);
    background: #fe2c55;
  }
  @keyframes idubb-fadeIn {
    from { opacity: 0; transform: translateX(-50%) translateY(10px); }
    to { opacity: 1; transform: translateX(-50%) translateY(0); }
  }
  @keyframes idubb-fadeOut {
    from { opacity: 1; }
    to { opacity: 0; }
  }
`;
document.head.appendChild(idubbStyles);

// Track current video URL
let currentVideoUrl = null;

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getVideoUrl') {
    const videoUrl = getCurrentVideoUrl();
    sendResponse({ videoUrl });
  }
  return true;
});

// Get current video URL from page
function getCurrentVideoUrl() {
  // Method 1: Check URL directly
  const url = window.location.href;
  if (url.includes('/video/')) {
    return url;
  }

  // Method 2: Find video element and get its container's link
  const videoContainers = document.querySelectorAll('[data-e2e="recommend-list-item-container"]');
  for (const container of videoContainers) {
    const link = container.querySelector('a[href*="/video/"]');
    if (link) {
      return link.href;
    }
  }

  // Method 3: For single video view
  const canonicalLink = document.querySelector('link[rel="canonical"]');
  if (canonicalLink && canonicalLink.href.includes('/video/')) {
    return canonicalLink.href;
  }

  // Method 4: Check meta og:url
  const ogUrl = document.querySelector('meta[property="og:url"]');
  if (ogUrl && ogUrl.content.includes('/video/')) {
    return ogUrl.content;
  }

  return null;
}

// Add floating action button on video hover
function addQuickActionButton() {
  // Find all video containers
  const videoItems = document.querySelectorAll('[data-e2e="recommend-list-item-container"], [class*="DivItemContainer"]');
  
  videoItems.forEach(item => {
    // Skip if already has our button
    if (item.querySelector('.idubb-quick-action')) {
      return;
    }

    // Get default values from saved settings
    const defaultMode = getDefaultMode();
    const defaultPlatforms = getDefaultPlatforms();

    // Create button
    const button = document.createElement('div');
    button.className = 'idubb-quick-action';
    button.innerHTML = `
      <div class="idubb-btn" title="iDubb 一键发布">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
      </div>
      <div class="idubb-menu" style="display: none;">
        <div class="idubb-menu-section">处理模式</div>
        <div class="idubb-menu-item${defaultMode === 'full_translation' ? ' selected' : ''}" data-mode="full_translation">🎙️ 完整翻译</div>
        <div class="idubb-menu-item${defaultMode === 'subtitles_only' ? ' selected' : ''}" data-mode="subtitles_only">📝 仅字幕</div>
        <div class="idubb-menu-item${defaultMode === 'direct_transfer' ? ' selected' : ''}" data-mode="direct_transfer">📦 直接搬运</div>
        <div class="idubb-menu-item${defaultMode === 'smart' ? ' selected' : ''}" data-mode="smart">🤖 智能判断</div>
        <div class="idubb-menu-divider"></div>
        <div class="idubb-menu-section">发布平台</div>
        <div class="idubb-menu-item${defaultPlatforms.douyin ? ' selected' : ''}" data-action="douyin">📱 抖音</div>
        <div class="idubb-menu-item${defaultPlatforms.xiaohongshu ? ' selected' : ''}" data-action="xiaohongshu">📕 小红书</div>
        <div class="idubb-menu-item${defaultPlatforms.bilibili ? ' selected' : ''}" data-action="bilibili">📺 哔哩哔哩</div>
        <div class="idubb-menu-item idubb-menu-submit" data-action="submit">🚀 开始处理</div>
      </div>
    `;

    // Find video link
    const videoLink = item.querySelector('a[href*="/video/"]');
    if (!videoLink) return;

    // Position button
    button.style.cssText = `
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 999;
      opacity: 0;
      transition: opacity 0.2s;
    `;

    // Ensure container has position relative
    if (getComputedStyle(item).position === 'static') {
      item.style.position = 'relative';
    }

    // Show on hover
    item.addEventListener('mouseenter', () => {
      button.style.opacity = '1';
    });
    item.addEventListener('mouseleave', () => {
      button.style.opacity = '0';
      button.querySelector('.idubb-menu').style.display = 'none';
    });

    // Toggle menu on button click
    button.querySelector('.idubb-btn').addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const menu = button.querySelector('.idubb-menu');
      menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
    });

    // Track selected options (initialized from saved settings)
    let selectedMode = defaultMode;
    let selectedPlatforms = { ...defaultPlatforms };

    // Handle menu item clicks
    button.querySelectorAll('.idubb-menu-item').forEach(menuItem => {
      menuItem.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        const mode = menuItem.dataset.mode;
        const action = menuItem.dataset.action;
        
        // 处理模式选择
        if (mode) {
          selectedMode = mode;
          // 更新 UI 显示选中状态
          button.querySelectorAll('[data-mode]').forEach(el => el.classList.remove('selected'));
          menuItem.classList.add('selected');
          return; // 不关闭菜单
        }
        
        // 平台选择 (toggle)
        if (action === 'douyin') {
          selectedPlatforms.douyin = !selectedPlatforms.douyin;
          menuItem.classList.toggle('selected', selectedPlatforms.douyin);
          return;
        }
        if (action === 'xiaohongshu') {
          selectedPlatforms.xiaohongshu = !selectedPlatforms.xiaohongshu;
          menuItem.classList.toggle('selected', selectedPlatforms.xiaohongshu);
          return;
        }
        if (action === 'bilibili') {
          selectedPlatforms.bilibili = !selectedPlatforms.bilibili;
          menuItem.classList.toggle('selected', selectedPlatforms.bilibili);
          return;
        }
        
        // 提交按钮
        if (action === 'submit') {
          const videoUrl = videoLink.href;
          
          if (!selectedPlatforms.douyin && !selectedPlatforms.xiaohongshu && !selectedPlatforms.bilibili) {
            showToast('⚠️ 请至少选择一个发布平台');
            return;
          }
          
          // Build options - only include relevant settings based on mode
          const options = {
            uploadDouyin: selectedPlatforms.douyin,
            uploadXiaohongshu: selectedPlatforms.xiaohongshu,
            uploadBilibili: selectedPlatforms.bilibili,
            processingMode: selectedMode,
            // 基础设置
            apiUrl: savedSettings?.apiUrl,
            videoQuality: savedSettings?.videoQuality,
            sourceLanguage: savedSettings?.sourceLanguage,
            targetLanguage: savedSettings?.targetLanguage
          };
          
          // 根据模式设置具体参数，只传必要的配置
          switch (selectedMode) {
            case 'full_translation':
              options.add_subtitles = true;
              options.add_tts = true;
              options.skip_translation = false;
              // 字幕配置
              options.dualSubtitles = savedSettings?.dualSubtitles;
              options.subtitlePreset = savedSettings?.subtitlePreset;
              // TTS 配置
              options.ttsService = savedSettings?.ttsService;
              options.ttsVoice = savedSettings?.ttsVoice;
              options.replaceOriginalAudio = savedSettings?.replaceOriginalAudio;
              options.originalVolume = savedSettings?.originalVolume;
              // 翻译配置
              options.translationEngine = savedSettings?.translationEngine;
              break;
            case 'subtitles_only':
              options.add_subtitles = true;
              options.add_tts = false;  // 不要配音
              options.skip_translation = false;
              // 字幕配置
              options.dualSubtitles = savedSettings?.dualSubtitles;
              options.subtitlePreset = savedSettings?.subtitlePreset;
              // 翻译配置
              options.translationEngine = savedSettings?.translationEngine;
              // 不传 TTS 配置
              break;
            case 'direct_transfer':
              options.add_subtitles = false;  // 不要字幕
              options.add_tts = false;  // 不要配音
              options.skip_translation = true;
              // 不传字幕和 TTS 配置
              break;
            case 'smart':
              // 让后端自动判断
              options.auto_detect_mode = true;
              break;
          }

          // Send to background script
          chrome.runtime.sendMessage({
            action: 'createTask',
            videoUrl: videoUrl,
            options: options
          }, (response) => {
            if (response.success) {
              const modeNames = {
                'full_translation': '完整翻译',
                'subtitles_only': '仅字幕',
                'direct_transfer': '直接搬运',
                'smart': '智能判断'
              };
              showToast(`✅ 任务已创建 [${modeNames[selectedMode]}]`);
            } else {
              showToast('❌ ' + response.error);
            }
          });

          // Hide menu
          button.querySelector('.idubb-menu').style.display = 'none';
        }
      });
    });

    item.appendChild(button);
  });
}

// Show toast notification
function showToast(message) {
  const existing = document.querySelector('.idubb-toast');
  if (existing) {
    existing.remove();
  }

  const toast = document.createElement('div');
  toast.className = 'idubb-toast';
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 14px;
    z-index: 10000;
    animation: idubb-fadeIn 0.3s ease;
  `;

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'idubb-fadeOut 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// Inject our option into TikTok's context menu
function injectIntoTikTokMenu() {
  // Watch for TikTok's context menu to appear
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        
        // TikTok 菜单通常包含这些选项
        const menuContainer = node.querySelector ? 
          (node.querySelector('[data-e2e="video-share-container"]') || 
           node.querySelector('[class*="DivContextMenu"]') ||
           node.querySelector('[class*="ContextMenu"]')) : null;
        
        // 或者检查是否是菜单本身
        const isMenu = node.textContent && 
          (node.textContent.includes('下载视频') || 
           node.textContent.includes('Download video') ||
           node.textContent.includes('复制链接') ||
           node.textContent.includes('Copy link'));
        
        if (menuContainer || isMenu) {
          const menu = menuContainer || node;
          addIdubbOptionToMenu(menu);
        }
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Add iDubb option to TikTok's menu
function addIdubbOptionToMenu(menuContainer) {
  // 如果 iDubb 菜单已存在，直接返回不重新创建
  if (document.querySelector('.idubb-tiktok-option')) {
    return;
  }
  
  // 找到菜单项列表
  const menuItems = menuContainer.querySelectorAll('button, [role="button"], [class*="MenuItem"], [class*="Item"]');
  if (menuItems.length === 0) return;
  
  // 找到最后一个菜单项作为参考
  const lastItem = menuItems[menuItems.length - 1];
  if (!lastItem) return;
  
  // 等待 TikTok 菜单渲染完成后再获取位置
  requestAnimationFrame(() => {
    setTimeout(() => {
      createIdubbMenu(menuContainer);
    }, 50);
  });
}

// 创建 iDubb 菜单
function createIdubbMenu(menuContainer) {
  // 再次检查避免重复 - 如果已存在则不重复创建
  if (document.querySelector('.idubb-tiktok-option')) {
    return;
  }
  
  // 获取 TikTok 菜单的位置
  const menuRect = menuContainer.getBoundingClientRect();
  
  // Get default values from saved settings
  const defaultMode = getDefaultMode();
  const defaultPlatforms = getDefaultPlatforms();
  
  // 创建我们的菜单项容器 - 紧贴 TikTok 菜单右边，上边缘对齐
  const idubbContainer = document.createElement('div');
  idubbContainer.className = 'idubb-tiktok-option';
  
  // 放在 TikTok 菜单右边，紧贴着 (间距 2px)
  let leftPos = menuRect.right + 2;
  
  // 如果右边放不下，放左边
  const idubbWidth = 185;
  if (leftPos + idubbWidth > window.innerWidth - 10) {
    leftPos = menuRect.left - idubbWidth - 2;
  }
  
  idubbContainer.style.cssText = `
    position: fixed;
    top: ${menuRect.top}px;
    left: ${leftPos}px;
    background: rgba(22, 24, 35, 0.98);
    border-radius: 8px;
    padding: 8px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 10001;
    min-width: 180px;
  `;
  
  // Build HTML with correct default selections
  const modes = [
    { id: 'full_translation', icon: '🎙️', label: '完整翻译' },
    { id: 'subtitles_only', icon: '📝', label: '仅字幕' },
    { id: 'direct_transfer', icon: '📦', label: '直接搬运' },
    { id: 'smart', icon: '🤖', label: '智能判断' }
  ];
  
  const modeButtonsHtml = modes.map(m => {
    const isSelected = m.id === defaultMode;
    return `<button class="idubb-tiktok-btn idubb-mode-btn${isSelected ? ' selected' : ''}" data-mode="${m.id}" style="display: flex; align-items: center; gap: 12px; width: 100%; padding: 10px 16px; background: ${isSelected ? 'rgba(254,44,85,0.2)' : 'none'}; border: none; color: white; cursor: pointer; font-size: 13px; text-align: left;">
      <span>${m.icon}</span> ${m.label}${isSelected ? ' ✓' : ''}
    </button>`;
  }).join('');
  
  idubbContainer.innerHTML = `
    <div class="idubb-menu-header" style="padding: 8px 16px; color: #fe2c55; font-weight: bold; font-size: 12px;">
      🚀 iDubb 一键发布
    </div>
    <div style="padding: 4px 16px; color: #888; font-size: 11px;">处理模式</div>
    ${modeButtonsHtml}
    <div style="height: 1px; background: rgba(255,255,255,0.1); margin: 8px 0;"></div>
    <div style="padding: 4px 16px; color: #888; font-size: 11px;">发布平台</div>
    <button class="idubb-tiktok-btn idubb-platform-btn${defaultPlatforms.douyin ? ' selected' : ''}" data-platform="douyin" style="display: flex; align-items: center; gap: 12px; width: 100%; padding: 10px 16px; background: ${defaultPlatforms.douyin ? 'rgba(254,44,85,0.2)' : 'none'}; border: none; color: white; cursor: pointer; font-size: 13px; text-align: left;">
      <span>📱</span> 抖音${defaultPlatforms.douyin ? ' ✓' : ''}
    </button>
    <button class="idubb-tiktok-btn idubb-platform-btn${defaultPlatforms.xiaohongshu ? ' selected' : ''}" data-platform="xiaohongshu" style="display: flex; align-items: center; gap: 12px; width: 100%; padding: 10px 16px; background: ${defaultPlatforms.xiaohongshu ? 'rgba(254,44,85,0.2)' : 'none'}; border: none; color: white; cursor: pointer; font-size: 13px; text-align: left;">
      <span>📕</span> 小红书${defaultPlatforms.xiaohongshu ? ' ✓' : ''}
    </button>
    <button class="idubb-tiktok-btn idubb-platform-btn${defaultPlatforms.bilibili ? ' selected' : ''}" data-platform="bilibili" style="display: flex; align-items: center; gap: 12px; width: 100%; padding: 10px 16px; background: ${defaultPlatforms.bilibili ? 'rgba(254,44,85,0.2)' : 'none'}; border: none; color: white; cursor: pointer; font-size: 13px; text-align: left;">
      <span>📺</span> 哔哩哔哩${defaultPlatforms.bilibili ? ' ✓' : ''}
    </button>
    <div style="height: 1px; background: rgba(255,255,255,0.1); margin: 8px 0;"></div>
    <button class="idubb-tiktok-btn idubb-submit-btn" data-action="submit" style="display: flex; align-items: center; justify-content: center; gap: 8px; width: calc(100% - 16px); margin: 8px; padding: 12px 16px; background: #fe2c55; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 14px; font-weight: bold;">
      🚀 开始处理
    </button>
  `;
  
  // 状态追踪 (initialized from saved settings)
  let tiktokMenuState = {
    mode: defaultMode,
    platforms: { ...defaultPlatforms }
  };

  // 添加 hover 效果
  idubbContainer.querySelectorAll('.idubb-tiktok-btn').forEach(btn => {
    if (!btn.classList.contains('idubb-submit-btn')) {
      btn.addEventListener('mouseenter', () => {
        if (!btn.classList.contains('selected')) {
          btn.style.background = 'rgba(255,255,255,0.1)';
        }
      });
      btn.addEventListener('mouseleave', () => {
        if (!btn.classList.contains('selected')) {
          btn.style.background = 'none';
        }
      });
    }
    
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      const mode = btn.dataset.mode;
      const platform = btn.dataset.platform;
      const action = btn.dataset.action;
      
      // 处理模式选择
      if (mode) {
        tiktokMenuState.mode = mode;
        idubbContainer.querySelectorAll('.idubb-mode-btn').forEach(el => {
          el.classList.remove('selected');
          el.style.background = 'none';
          el.innerHTML = el.innerHTML.replace(' ✓', '');
        });
        btn.classList.add('selected');
        btn.style.background = 'rgba(254,44,85,0.2)';
        btn.innerHTML = btn.innerHTML + ' ✓';
        return;
      }
      
      // 平台选择 (toggle)
      if (platform) {
        tiktokMenuState.platforms[platform] = !tiktokMenuState.platforms[platform];
        btn.classList.toggle('selected', tiktokMenuState.platforms[platform]);
        btn.style.background = tiktokMenuState.platforms[platform] ? 'rgba(254,44,85,0.2)' : 'none';
        if (tiktokMenuState.platforms[platform]) {
          if (!btn.innerHTML.includes('✓')) btn.innerHTML = btn.innerHTML + ' ✓';
        } else {
          btn.innerHTML = btn.innerHTML.replace(' ✓', '');
        }
        return;
      }
      
      // 提交
      if (action === 'submit') {
        const videoUrl = getCurrentVideoUrl() || window.location.href;
        
        if (!tiktokMenuState.platforms.douyin && !tiktokMenuState.platforms.xiaohongshu && !tiktokMenuState.platforms.bilibili) {
          showToast('⚠️ 请至少选择一个发布平台');
          return;
        }
        
        // Build options - only include relevant settings based on mode
        const options = {
          uploadDouyin: tiktokMenuState.platforms.douyin,
          uploadXiaohongshu: tiktokMenuState.platforms.xiaohongshu,
          uploadBilibili: tiktokMenuState.platforms.bilibili,
          processingMode: tiktokMenuState.mode,
          // 基础设置
          apiUrl: savedSettings?.apiUrl,
          videoQuality: savedSettings?.videoQuality,
          sourceLanguage: savedSettings?.sourceLanguage,
          targetLanguage: savedSettings?.targetLanguage
        };
        
        // 根据模式设置具体参数，只传必要的配置
        switch (tiktokMenuState.mode) {
          case 'full_translation':
            options.add_subtitles = true;
            options.add_tts = true;
            options.skip_translation = false;
            // 字幕配置
            options.dualSubtitles = savedSettings?.dualSubtitles;
            options.subtitlePreset = savedSettings?.subtitlePreset;
            // TTS 配置
            options.ttsService = savedSettings?.ttsService;
            options.ttsVoice = savedSettings?.ttsVoice;
            options.replaceOriginalAudio = savedSettings?.replaceOriginalAudio;
            options.originalVolume = savedSettings?.originalVolume;
            // 翻译配置
            options.translationEngine = savedSettings?.translationEngine;
            break;
          case 'subtitles_only':
            options.add_subtitles = true;
            options.add_tts = false;  // 不要配音
            options.skip_translation = false;
            // 字幕配置
            options.dualSubtitles = savedSettings?.dualSubtitles;
            options.subtitlePreset = savedSettings?.subtitlePreset;
            // 翻译配置
            options.translationEngine = savedSettings?.translationEngine;
            // 不传 TTS 配置
            break;
          case 'direct_transfer':
            options.add_subtitles = false;  // 不要字幕
            options.add_tts = false;  // 不要配音
            options.skip_translation = true;
            // 不传字幕和 TTS 配置
            break;
          case 'smart':
            // 让后端自动判断
            options.auto_detect_mode = true;
            break;
        }
        
        chrome.runtime.sendMessage({
          action: 'createTask',
          videoUrl: videoUrl,
          options: options
        }, (response) => {
          const modeNames = {
            'full_translation': '完整翻译',
            'subtitles_only': '仅字幕',
            'direct_transfer': '直接搬运',
            'smart': '智能判断'
          };
          if (response && response.success) {
            showToast(`✅ 任务已创建 [${modeNames[tiktokMenuState.mode]}]`);
          } else {
            showToast('❌ ' + (response?.error || '发送失败'));
          }
        });
        
        // 关闭菜单
        document.body.click();
      }
    });
  });
  
  // 插入到 body（独立定位）
  document.body.appendChild(idubbContainer);
  
  // 监听 TikTok 菜单关闭，同时关闭我们的菜单
  const closeObserver = new MutationObserver((mutations) => {
    // 检查 TikTok 菜单是否还在
    if (!document.body.contains(menuContainer)) {
      idubbContainer.remove();
      closeObserver.disconnect();
    }
  });
  closeObserver.observe(document.body, { childList: true, subtree: true });
  
  // 点击其他地方关闭
  const clickHandler = (e) => {
    if (!idubbContainer.contains(e.target) && !menuContainer.contains(e.target)) {
      idubbContainer.remove();
      document.removeEventListener('click', clickHandler);
    }
  };
  setTimeout(() => document.addEventListener('click', clickHandler), 100);
  
  console.log('[iDubb] 已注入到 TikTok 菜单旁边');
}

// Initialize
async function init() {
  // Load saved settings first
  await loadSavedSettings();
  
  // Add buttons to existing videos
  addQuickActionButton();
  
  // Inject into TikTok's context menu
  injectIntoTikTokMenu();

  // Watch for new videos loaded (infinite scroll)
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.addedNodes.length > 0) {
        addQuickActionButton();
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Run when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

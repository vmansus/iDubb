/**
 * iDubb Browser Extension - Content Script
 * Runs on TikTok pages to detect videos and enable quick actions
 */

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
        <div class="idubb-menu-item" data-action="douyin">📱 发布到抖音</div>
        <div class="idubb-menu-item" data-action="xiaohongshu">📕 发布到小红书</div>
        <div class="idubb-menu-item" data-action="both">🚀 全部平台</div>
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

    // Handle menu item clicks
    button.querySelectorAll('.idubb-menu-item').forEach(menuItem => {
      menuItem.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        const action = menuItem.dataset.action;
        const videoUrl = videoLink.href;
        
        let uploadDouyin = false;
        let uploadXiaohongshu = false;
        
        if (action === 'douyin' || action === 'both') {
          uploadDouyin = true;
        }
        if (action === 'xiaohongshu' || action === 'both') {
          uploadXiaohongshu = true;
        }

        // Send to background script
        chrome.runtime.sendMessage({
          action: 'createTask',
          videoUrl: videoUrl,
          options: {
            uploadDouyin,
            uploadXiaohongshu
          }
        }, (response) => {
          if (response.success) {
            showToast('✅ 任务已创建，视频将自动处理并发布');
          } else {
            showToast('❌ ' + response.error);
          }
        });

        // Hide menu
        button.querySelector('.idubb-menu').style.display = 'none';
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
  // 避免重复添加
  if (menuContainer.querySelector('.idubb-tiktok-option')) return;
  
  // 找到菜单项列表
  const menuItems = menuContainer.querySelectorAll('button, [role="button"], [class*="MenuItem"], [class*="Item"]');
  if (menuItems.length === 0) return;
  
  // 找到最后一个菜单项作为参考
  const lastItem = menuItems[menuItems.length - 1];
  if (!lastItem) return;
  
  // 创建分割线
  const divider = document.createElement('div');
  divider.style.cssText = 'height: 1px; background: rgba(255,255,255,0.1); margin: 8px 0;';
  
  // 创建我们的菜单项容器
  const idubbContainer = document.createElement('div');
  idubbContainer.className = 'idubb-tiktok-option';
  idubbContainer.innerHTML = `
    <div class="idubb-menu-header" style="padding: 8px 16px; color: #fe2c55; font-weight: bold; font-size: 12px;">
      🚀 iDubb 一键发布
    </div>
    <button class="idubb-tiktok-btn" data-action="douyin" style="display: flex; align-items: center; gap: 12px; width: 100%; padding: 12px 16px; background: none; border: none; color: white; cursor: pointer; font-size: 14px; text-align: left;">
      <span>📱</span> 发布到抖音
    </button>
    <button class="idubb-tiktok-btn" data-action="xiaohongshu" style="display: flex; align-items: center; gap: 12px; width: 100%; padding: 12px 16px; background: none; border: none; color: white; cursor: pointer; font-size: 14px; text-align: left;">
      <span>📕</span> 发布到小红书
    </button>
    <button class="idubb-tiktok-btn" data-action="both" style="display: flex; align-items: center; gap: 12px; width: 100%; padding: 12px 16px; background: none; border: none; color: white; cursor: pointer; font-size: 14px; text-align: left;">
      <span>🎯</span> 全部平台
    </button>
  `;
  
  // 添加 hover 效果
  idubbContainer.querySelectorAll('.idubb-tiktok-btn').forEach(btn => {
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(255,255,255,0.1)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'none';
    });
    
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      const action = btn.dataset.action;
      const videoUrl = getCurrentVideoUrl() || window.location.href;
      
      let uploadDouyin = action === 'douyin' || action === 'both';
      let uploadXiaohongshu = action === 'xiaohongshu' || action === 'both';
      
      chrome.runtime.sendMessage({
        action: 'createTask',
        videoUrl: videoUrl,
        options: { uploadDouyin, uploadXiaohongshu }
      }, (response) => {
        if (response && response.success) {
          showToast('✅ 任务已创建: ' + videoUrl.substring(0, 50) + '...');
        } else {
          showToast('❌ ' + (response?.error || '发送失败'));
        }
      });
      
      // 关闭菜单 - 模拟点击其他地方
      document.body.click();
    });
  });
  
  // 插入到菜单末尾
  const parent = lastItem.parentElement || menuContainer;
  parent.appendChild(divider);
  parent.appendChild(idubbContainer);
  
  console.log('[iDubb] 已注入到 TikTok 菜单');
}

// Initialize
function init() {
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

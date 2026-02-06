# iDubb 浏览器插件

一键将 TikTok 视频发布到抖音、小红书等平台。

## 功能特性

- 🖱️ **右键菜单** - 在 TikTok 视频页面右键，选择发布目标
- 🎯 **快捷按钮** - 视频悬停时显示快速发布按钮
- ⚙️ **自定义选项** - 设置翻译语言、配音声音等
- 🚀 **一键发布** - 自动下载、翻译、配音、发布

## 安装方法

### Chrome / Edge

1. 打开 `chrome://extensions/`（Chrome）或 `edge://extensions/`（Edge）
2. 开启「开发者模式」
3. 点击「加载已解压的扩展程序」
4. 选择 `extension` 文件夹

### Firefox

1. 打开 `about:debugging#/runtime/this-firefox`
2. 点击「加载临时附加组件」
3. 选择 `extension/manifest.json` 文件

## 使用方法

### 方式一：右键菜单

1. 访问 TikTok 视频页面
2. 右键点击视频或页面
3. 选择「iDubb 一键发布」
4. 选择发布目标（抖音/小红书/全部）

### 方式二：快捷按钮

1. 鼠标悬停在 TikTok 视频上
2. 点击右上角的 iDubb 按钮
3. 选择发布目标

### 方式三：插件弹窗

1. 点击浏览器工具栏的 iDubb 图标
2. 配置设置
3. 从右键菜单选择「自定义选项」会打开弹窗

## 配置说明

在插件弹窗中可以配置：

| 选项 | 说明 |
|------|------|
| API 地址 | iDubb 后端服务地址，默认 `http://localhost:8000` |
| 发布目标 | 选择要发布的平台 |
| 目标语言 | 翻译的目标语言 |
| 添加字幕 | 是否在视频上添加字幕 |
| 添加配音 | 是否添加 TTS 配音 |
| 配音声音 | 选择 TTS 声音 |

## 技术架构

```
extension/
├── manifest.json      # 扩展配置（Manifest V3）
├── background.js      # Service Worker，处理右键菜单和 API 通信
├── content.js         # 内容脚本，注入 TikTok 页面
├── content.css        # 内容脚本样式
├── popup.html         # 弹窗界面
├── popup.js           # 弹窗逻辑
└── icons/             # 图标资源
```

## 开发说明

### 本地开发

1. 修改代码后，在扩展管理页面点击「刷新」按钮
2. 如果修改了 `manifest.json`，需要重新加载扩展

### 调试

- **Service Worker 日志**: `chrome://extensions/` → 点击 "Service Worker" 链接
- **Content Script 日志**: 在 TikTok 页面打开开发者工具
- **Popup 日志**: 右键点击扩展图标 → "检查弹出式窗口"

## 依赖

- iDubb 后端服务运行在 `http://localhost:8000`
- 已登录的抖音/小红书账号（在 iDubb 后端配置）

## 许可证

MIT License

# iDubb 🎬

中文 | [English](./README.md)

> ⚠️ **开发状态**：本项目正在积极开发中，功能可能变动，存在已知 Bug。请谨慎使用。

> 🤖 **AI 生成代码**：本项目代码完全由 AI 编程助手编写，包括 [Claude Code](https://claude.ai)、[Clawdbot](https://github.com/clawdbot/clawdbot) 等工具。人类仅参与产品方向指导和测试。

一站式视频翻译配音工具。下载、转录、翻译、配音、上传 - 全自动完成。

## ✨ 功能特性

### 🎯 核心流程
- **视频下载** - 支持 YouTube、TikTok、Bilibili 等平台（基于 yt-dlp）
- **语音转录** - 多种后端支持：
  - Whisper (OpenAI)
  - Faster Whisper (4-8倍速度提升，GPU优化)
  - WhisperX (词级对齐，说话人分离)
- **智能翻译** - 多引擎支持：
  - Google 翻译（免费）
  - OpenAI GPT-4/GPT-4o
  - Anthropic Claude
  - DeepSeek
- **AI 配音** - 多种语音合成引擎：
  - Edge TTS (微软，免费，400+语音)
  - CosyVoice (声音克隆)
  - Index TTS (声音克隆)
  - Qwen3 TTS (声音克隆)
- **字幕处理** - 双语字幕、自定义样式、ASS/SRT导出、硬字幕烧录
- **一键上传** - B站、抖音、小红书，支持多账号

### 🔄 处理模式
| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **完整翻译** | 转录 → 翻译 → 配音 → 合成 | YouTube → B站 翻译搬运 |
| **仅字幕** | 转录 → 翻译 → 嵌入字幕 | 有对白但不需要配音 |
| **直接搬运** | 下载 → 直接上传 | 搞笑视频、无对白内容 |
| **智能判断** | AI 分析内容自动决定 | 不确定的内容类型 |

### 📡 订阅系统
- 订阅 YouTube/TikTok 频道
- 自动检测新视频（可配置检查间隔）
- 自动处理流水线（下载 → 处理 → 上传）
- 批量导入历史视频

### 🤖 AI 增强
- **AI 校对**：语法检查、术语一致性、时间轴优化
- **AI 元数据**：自动生成平台专属的标题、描述、标签
  - 抖音风格（短平快、网感）、B站风格（详细专业）、小红书风格（种草向）
- **自定义术语表**：保证翻译术语一致性

### 🎨 其他功能
- 现代 React UI，支持暗色主题
- 多语言界面（中文/英文）
- 处理预设，快速复用配置
- 热门视频发现（YouTube）
- 任务管理与进度追踪
- 所有上传平台支持多账号

## 🛠️ 技术栈

**后端**
- Python 3.10+ / FastAPI / SQLAlchemy + SQLite
- yt-dlp（视频下载）
- FFmpeg（视频处理）
- Whisper/Faster-Whisper/WhisperX（语音转录）
- Playwright（浏览器自动化，用于 TikTok、抖音）

**前端**
- React 18 + TypeScript
- Tailwind CSS
- Vite
- React Query
- i18next

## 📦 安装

### 环境要求
- Python 3.10+
- Node.js 18+
- FFmpeg
- GPU（可选，推荐用于 Whisper 加速）

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/vmansus/iDubb.git
cd iDubb

# 后端
cd backend
pip install -r requirements.txt
playwright install chromium  # TikTok/抖音支持

# 启动后端
uvicorn api.main:app --host 0.0.0.0 --port 8888

# 前端（新终端）
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173

## ⚙️ 配置

### API Keys

在设置页面配置：

| 服务 | 用途 | 说明 |
|------|------|------|
| OpenAI API Key | GPT 翻译、AI 校对 | 使用 Google 翻译则可选 |
| Anthropic API Key | Claude 翻译 | 可选 |
| DeepSeek API Key | DeepSeek 翻译 | 可选，性价比高 |
| YouTube Data API | 热门视频 | 可选 |

### 平台凭证

在设置页面配置上传平台登录信息：
- **B站**：扫码登录
- **抖音**：扫码或 Cookie 登录
- **小红书**：扫码或 Cookie 登录

### 环境变量

```bash
# .env（可选）
WHISPER_MODEL=small          # tiny, base, small, medium, large-v3
WHISPER_DEVICE=auto          # auto, cpu, cuda, mps
```

## 📖 使用指南

### 基本流程

1. **新建任务** - 粘贴视频 URL 或上传本地文件
2. **选择模式** - 完整翻译、仅字幕、直接搬运
3. **配置选项** - 选择语言、语音、字幕样式
4. **开始处理** - 一键启动，实时查看进度
5. **审核结果** - 预览成品，按需编辑字幕
6. **上传发布** - 一键上传多平台

### 订阅频道

1. 进入「订阅管理」页面
2. 点击「添加订阅」
3. 粘贴 YouTube/TikTok 频道 URL
4. 配置检查间隔和处理选项
5. 开启自动处理，实现全自动运行

## 📁 项目结构

```
iDubb/
├── backend/
│   ├── api/              # FastAPI 路由
│   ├── database/         # SQLAlchemy 模型
│   ├── downloaders/      # 视频下载 (yt-dlp)
│   ├── transcription/    # Whisper 转录后端
│   ├── translation/      # 翻译引擎
│   ├── tts/              # 语音合成引擎
│   ├── dubbing/          # 音视频合成
│   ├── subtitles/        # 字幕处理
│   ├── uploaders/        # 平台上传器
│   ├── subscriptions/    # 订阅调度器
│   ├── metadata/         # AI 元数据生成
│   ├── proofreading/     # AI 校对
│   └── pipeline.py       # 主处理流水线
├── frontend/
│   ├── src/
│   │   ├── components/   # React 组件
│   │   ├── pages/        # 页面组件
│   │   ├── services/     # API 客户端
│   │   └── locales/      # i18n 翻译
│   └── public/
└── docs/                 # 文档
```

## 🔧 声音克隆配置

### CosyVoice
```bash
# 安装 CosyVoice 到 external/cosyvoice
git clone https://github.com/FunAudioLLM/CosyVoice external/cosyvoice
cd external/cosyvoice && pip install -r requirements.txt
```

### Index TTS
```bash
# 安装 IndexTTS 到 external/indextts
git clone https://github.com/indexteam/IndexTTS external/indextts
cd external/indextts && pip install -r requirements.txt
```

## 🚧 开发计划

- [ ] 更多视频平台（Instagram Reels、Twitter/X）
- [ ] 更多 TTS 引擎（Fish Audio、ChatTTS）
- [ ] 视频编辑器（裁剪、拼接、水印）
- [ ] Docker 部署
- [ ] 浏览器扩展，快速搬运

## 🤝 贡献

欢迎贡献代码！请先阅读现有代码风格，然后提交 PR。

## 📄 License

MIT

---

**Made with ❤️ by vmansus & Chad 🐕**

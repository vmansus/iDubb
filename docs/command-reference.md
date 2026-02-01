# iDubb 开发命令参考

> 由 Chad 维护，记录开发过程中常用的命令和技巧

## Git 分支管理

### 创建新分支
```bash
git checkout -b feature/xxx
```

### 切换分支
```bash
git checkout main
git checkout feature/20260202s2
```

### 查看所有分支
```bash
git branch        # 本地分支
git branch -a     # 包括远程分支
```

## Git Worktree（多分支同时工作）

### 创建 worktree
```bash
git worktree add ../iDubb-s2 feature/20260202s2
```

### 列出所有 worktree
```bash
git worktree list
```

### 删除 worktree
```bash
git worktree remove ../iDubb-s2
```

### 使用场景
- 两个目录，两个分支，同时开发
- 不用来回 checkout，互不干扰
- 共享 .git 目录，省空间

## Git 常用操作

### 拉取最新代码
```bash
git fetch origin
git pull
```

### 暂存改动
```bash
git stash          # 暂存
git stash pop      # 恢复
git stash list     # 查看暂存列表
```

### 查看提交历史
```bash
git log --oneline -10
```

### 查看文件改动
```bash
git diff HEAD~5..HEAD
```

## 浏览器插件测试

### 加载插件（Chrome/Edge）
1. 打开 `chrome://extensions/`
2. 开启「开发者模式」
3. 点击「加载已解压的扩展程序」
4. 选择 `extension` 文件夹

### 调试
- Service Worker: 扩展页面点击 "Service Worker" 链接
- Content Script: TikTok 页面打开 DevTools
- Popup: 右键扩展图标 → "检查弹出式窗口"

---

*最后更新: 2026-02-01*

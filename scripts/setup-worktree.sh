#!/bin/bash
#
# iDubb Worktree 环境初始化脚本
# 用法: ./scripts/setup-worktree.sh [主目录路径]
#
# 功能:
#   1. 创建 Python 虚拟环境并安装依赖
#   2. 安装前端 npm 依赖
#   3. 软链接 data 目录到主目录（可选）
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN_DATA_DIR="$1"

echo "========================================"
echo "iDubb Worktree 环境初始化"
echo "========================================"
echo "当前目录: $WORKTREE_DIR"
echo ""

# 检测 Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ 未找到 Python，请先安装 Python 3.10+"
    exit 1
fi

echo "✓ Python: $($PYTHON --version)"

# 1. 设置 Python 虚拟环境
echo ""
echo "▶ 设置 Python 虚拟环境..."
cd "$WORKTREE_DIR/backend"

if [ ! -d "venv" ]; then
    echo "  创建 venv..."
    $PYTHON -m venv venv
fi

echo "  激活 venv..."
source venv/bin/activate

echo "  安装依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "  安装 PaddleOCR..."
pip install -q paddlepaddle paddleocr

echo "✓ Python 环境就绪"

# 2. 安装前端依赖
echo ""
echo "▶ 安装前端依赖..."
cd "$WORKTREE_DIR/frontend"

if command -v npm &> /dev/null; then
    npm install --silent
    echo "✓ 前端依赖就绪"
else
    echo "⚠ npm 未找到，跳过前端依赖安装"
fi

# 3. 软链接 data 目录（可选）
if [ -n "$MAIN_DATA_DIR" ]; then
    echo ""
    echo "▶ 设置 data 目录软链接..."
    cd "$WORKTREE_DIR"
    
    if [ -L "data" ]; then
        echo "  data 已经是软链接，跳过"
    elif [ -d "data" ]; then
        echo "  删除现有 data 目录..."
        rm -rf data
        ln -s "$MAIN_DATA_DIR" data
        echo "✓ data -> $MAIN_DATA_DIR"
    else
        ln -s "$MAIN_DATA_DIR" data
        echo "✓ data -> $MAIN_DATA_DIR"
    fi
fi

echo ""
echo "========================================"
echo "✅ 初始化完成！"
echo "========================================"
echo ""
echo "启动服务:"
echo "  cd $WORKTREE_DIR"
echo "  ./scripts/start_local.sh"
echo ""

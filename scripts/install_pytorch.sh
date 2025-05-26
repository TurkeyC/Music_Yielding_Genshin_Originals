#!/bin/bash
# PyTorch 2.6.0 + CUDA 12.4 安装脚本

echo "🔥 HoyoMusic生成器 - PyTorch环境安装"
echo "====================================="

# 检测操作系统
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "🖥️  检测到Windows系统"
    PYTHON_CMD="python"
else
    echo "🐧 检测到Unix/Linux系统"
    PYTHON_CMD="python3"
fi

# 检查Python版本
echo "🐍 检查Python版本..."
$PYTHON_CMD --version

# 检查CUDA版本
echo "🔧 检查CUDA版本..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "⚠️  未找到CUDA，将使用CPU版本"
fi

# 升级pip
echo "📦 升级pip..."
$PYTHON_CMD -m pip install --upgrade pip

# 安装PyTorch 2.6.0 with CUDA 12.4
echo "🔥 安装PyTorch 2.6.0 (CUDA 12.4)..."
$PYTHON_CMD -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
echo "📚 安装其他依赖包..."
$PYTHON_CMD -m pip install numpy pandas matplotlib requests tqdm datasets huggingface_hub

# 安装音乐处理相关包
echo "🎵 安装音乐处理包..."
$PYTHON_CMD -m pip install music21 mido pretty_midi

# 尝试安装pyfluidsynth（可选）
echo "🎼 尝试安装pyfluidsynth..."
$PYTHON_CMD -m pip install pyfluidsynth || echo "⚠️  pyfluidsynth安装失败，将跳过"

# 验证安装
echo "✅ 验证PyTorch安装..."
$PYTHON_CMD -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU设备: {torch.cuda.get_device_name()}')
else:
    print('将使用CPU运行')
"

echo ""
echo "🎉 安装完成！"
echo "📝 现在可以运行以下命令开始训练:"
echo "   python train.py --use-hoyomusic"
echo ""
echo "🎼 生成音乐:"
echo "   python generate.py --region Mondstadt"

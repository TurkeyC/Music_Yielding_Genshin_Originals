@echo off
REM PyTorch 2.6.0 + CUDA 12.4 安装脚本 (Windows)

echo 🔥 HoyoMusic生成器 - PyTorch环境安装
echo =====================================

REM 检查Python版本
echo 🐍 检查Python版本...
python --version

REM 检查CUDA版本
echo 🔧 检查CUDA版本...
where nvcc >nul 2>nul
if %ERRORLEVEL% == 0 (
    nvcc --version | findstr "release"
) else (
    echo ⚠️  未找到CUDA，将使用CPU版本
)

REM 升级pip
echo 📦 升级pip...
python -m pip install --upgrade pip

REM 安装PyTorch 2.6.0 with CUDA 12.4
echo 🔥 安装PyTorch 2.6.0 (CUDA 12.4)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM 安装其他依赖
echo 📚 安装其他依赖包...
python -m pip install numpy pandas matplotlib requests tqdm datasets huggingface_hub

REM 安装音乐处理相关包
echo 🎵 安装音乐处理包...
python -m pip install music21 mido pretty_midi

REM 尝试安装pyfluidsynth（可选）
echo 🎼 尝试安装pyfluidsynth...
python -m pip install pyfluidsynth
if %ERRORLEVEL% neq 0 (
    echo ⚠️  pyfluidsynth安装失败，将跳过
)

REM 验证安装
echo ✅ 验证PyTorch安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}' if torch.cuda.is_available() else '将使用CPU运行'); print(f'GPU设备: {torch.cuda.get_device_name()}' if torch.cuda.is_available() else '')"

echo.
echo 🎉 安装完成！
echo 📝 现在可以运行以下命令开始训练:
echo    python train.py --use-hoyomusic
echo.
echo 🎼 生成音乐:
echo    python generate.py --region Mondstadt

pause

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoyoMusic AI Generator - 启动脚本
自动检查依赖并启动Web UI
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        sys.exit(1)
    print(f"✅ Python版本: {sys.version}")

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def install_requirements():
    """安装依赖包"""
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        print("📦 安装依赖包...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("✅ 依赖包安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖包安装失败: {e}")
            return False
    return True

def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU可用: {gpu_name} (共{gpu_count}个设备)")
            print(f"   CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("⚠️  GPU不可用，将使用CPU模式")
            return False
    except ImportError:
        print("⚠️  PyTorch未安装，无法检查GPU")
        return False

def check_directories():
    """检查并创建必要的目录"""
    directories = [
        "models",
        "generated_music", 
        "data/abc_files",
        "logs",
        "temp"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 目录已准备: {dir_path}")

def check_model_files():
    """检查模型文件"""
    model_files = [
        "models/hoyomusic_generator.pth",
        "models/hoyomusic_mappings.pkl"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ 模型文件存在: {file_path}")
    
    if missing_files:
        print("⚠️  以下模型文件缺失:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("   请先训练模型或下载预训练模型")

def launch_ui():
    """启动Streamlit UI"""
    print("\n🚀 启动HoyoMusic AI Generator Web UI...")
    print("📍 访问地址: http://localhost:8501")
    print("⏹️  按Ctrl+C停止服务")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 服务已停止")

def main():
    """主函数"""
    print("🎵 HoyoMusic AI Generator - 启动程序")
    print("=" * 50)
    
    # 检查Python版本
    check_python_version()
    
    # 检查核心依赖
    print("\n📋 检查依赖包...")
    required_packages = [
        ("streamlit", "streamlit"),
        ("torch", "torch"), 
        ("numpy", "numpy"),
        ("plotly", "plotly")
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        if check_package(package_name, import_name):
            print(f"✅ {package_name}")
        else:
            print(f"❌ {package_name} (缺失)")
            missing_packages.append(package_name)
    
    # 如果有缺失的包，尝试安装
    if missing_packages:
        print(f"\n📦 发现{len(missing_packages)}个缺失的依赖包")
        if input("是否自动安装? (y/N): ").lower() == 'y':
            if not install_requirements():
                print("❌ 依赖安装失败，请手动安装")
                return
        else:
            print("❌ 请手动安装依赖包: pip install -r requirements.txt")
            return
    
    # 检查GPU
    print("\n🔍 检查硬件...")
    check_gpu()
    
    # 检查目录结构
    print("\n📁 检查目录结构...")
    check_directories()
    
    # 检查模型文件
    print("\n🧠 检查模型文件...")
    check_model_files()
    
    # 启动UI
    print("\n" + "=" * 50)
    launch_ui()

if __name__ == "__main__":
    main()

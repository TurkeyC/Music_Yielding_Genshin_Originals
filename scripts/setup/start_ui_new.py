#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoyoMusic AI Generator - UI启动脚本 (重构版)
自动检查依赖并启动Streamlit界面
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def check_dependencies():
    """检查必要的依赖"""
    required_packages = [
        'streamlit',
        'torch',
        'numpy',
        'pandas',
        'plotly',
        'streamlit_option_menu'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖已安装")
    return True

def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            print(f"🎮 GPU可用: {gpu_name}")
        else:
            print("💻 使用CPU模式")
    except ImportError:
        print("⚠️ PyTorch未安装")

def validate_project_structure():
    """验证项目结构"""
    required_dirs = [
        "src/core",
        "src/ui", 
        "src/tools",
        "models",
        "output/generated",
        "data"
    ]
    
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if not full_path.exists():
            print(f"❌ 缺少目录: {dir_path}")
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 已创建目录: {dir_path}")
    
    print("✅ 项目结构验证完成")

def start_ui():
    """启动UI界面"""
    app_path = PROJECT_ROOT / "src" / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"❌ 找不到应用文件: {app_path}")
        return False
    
    print("🚀 启动HoyoMusic AI Generator...")
    print("🌐 访问地址: http://localhost:8501")
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        
    except KeyboardInterrupt:
        print("\n👋 感谢使用HoyoMusic AI Generator!")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🎵 HoyoMusic AI Generator - 启动检查 (重构版)")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查GPU
    check_gpu()
    
    # 验证项目结构
    validate_project_structure()
    
    # 启动UI
    print("\n" + "=" * 50)
    start_ui()

if __name__ == "__main__":
    main()

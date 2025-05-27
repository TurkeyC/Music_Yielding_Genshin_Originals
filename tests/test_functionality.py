#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoyoMusic AI 音乐生成器 - 功能测试脚本
测试所有核心功能是否正常工作
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        from src.core.model import HoyoMusicGenerator
        print("✅ 核心模型导入成功")
    except ImportError as e:
        print(f"❌ 核心模型导入失败: {e}")
        return False
    
    try:
        from src.core.data_processor import HoyoMusicDataProcessor
        print("✅ 数据处理器导入成功")
    except ImportError as e:
        print(f"❌ 数据处理器导入失败: {e}")
        return False
        
    try:
        from src.tools.abc_to_midi import ABCToMIDIConverter
        print("✅ MIDI转换器导入成功")
    except ImportError as e:
        print(f"❌ MIDI转换器导入失败: {e}")
        return False
        
    try:
        from src.tools.abc_cleaner import fix_abc_structure
        print("✅ ABC清理工具导入成功")
    except ImportError as e:
        print(f"❌ ABC清理工具导入失败: {e}")
        return False
        
    return True

def test_model_files():
    """测试模型文件完整性"""
    print("\n📁 测试模型文件...")
    
    models_dir = project_root / "models"
    required_files = [
        "hoyomusic_generator.pth",
        "hoyomusic_mappings.pkl", 
        "training_config.json",
        "training_history.json"
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = models_dir / file_name
        if file_path.exists():
            print(f"✅ 找到文件: {file_name}")
        else:
            print(f"❌ 缺少文件: {file_name}")
            all_exist = False
            
    return all_exist

def test_directories():
    """测试目录结构"""
    print("\n📂 测试目录结构...")
    
    required_dirs = [
        "src/core",
        "src/ui", 
        "src/tools",
        "src/utils",
        "models",
        "data",
        "output/generated",
        "docs",
        "tests"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ 目录存在: {dir_path}")
        else:
            print(f"❌ 目录缺失: {dir_path}")
            all_exist = False
            
    return all_exist

def test_dependencies():
    """测试依赖包"""
    print("\n📦 测试依赖包...")
    
    required_packages = [
        "torch",
        "numpy", 
        "pandas",
        "streamlit",
        "plotly",
        "matplotlib"
    ]
    
    all_available = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ 包可用: {package}")
        except ImportError:
            print(f"❌ 包缺失: {package}")
            all_available = False
            
    return all_available

def test_pytorch_functionality():
    """测试PyTorch基本功能"""
    print("\n🔥 测试PyTorch功能...")
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ PyTorch设备: {device}")
        
        # 创建简单测试张量
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        print(f"✅ 张量运算正常: {z.shape}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.get_device_name()}")
        else:
            print("ℹ️ 使用CPU模式")
            
        return True
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎵 HoyoMusic AI 音乐生成器 - 功能测试")
    print("=" * 50)
    
    # 执行测试
    tests = [
        ("依赖包", test_dependencies),
        ("PyTorch功能", test_pytorch_functionality),
        ("目录结构", test_directories),
        ("模型文件", test_model_files),
        ("模块导入", test_imports)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🧪 开始测试: {test_name}")
        results[test_name] = test_func()
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！HoyoMusic AI 音乐生成器已准备就绪")
        print("🚀 可以使用以下命令启动WebUI:")
        print("   python start_app.py")
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息")
        
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

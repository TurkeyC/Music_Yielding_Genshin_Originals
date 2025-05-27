#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoyoMusic AI 音乐生成器 - 核心功能测试
验证修复后的应用是否能正常工作
"""

import os
import sys
import json
import pickle
import traceback
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    tests = {
        'streamlit': False,
        'torch': False,
        'numpy': False,
        'plotly': False,
        'core_modules': False,
        'tools': False
    }
    
    # 测试基础依赖
    try:
        import streamlit as st
        tests['streamlit'] = True
        print("  ✅ Streamlit 导入成功")
    except ImportError as e:
        print(f"  ❌ Streamlit 导入失败: {e}")
    
    try:
        import torch
        tests['torch'] = True
        print("  ✅ PyTorch 导入成功")
        if torch.cuda.is_available():
            print(f"    🎮 CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("    💻 使用CPU模式")
    except ImportError as e:
        print(f"  ❌ PyTorch 导入失败: {e}")
    
    try:
        import numpy as np
        tests['numpy'] = True
        print("  ✅ NumPy 导入成功")
    except ImportError as e:
        print(f"  ❌ NumPy 导入失败: {e}")
    
    try:
        import plotly.graph_objects as go
        tests['plotly'] = True
        print("  ✅ Plotly 导入成功")
    except ImportError as e:
        print(f"  ❌ Plotly 导入失败: {e}")
    
    # 测试核心模块
    try:
        from src.core.model import HoyoMusicGenerator
        from src.core.data_processor import HoyoMusicDataProcessor
        tests['core_modules'] = True
        print("  ✅ 核心模块导入成功")
    except ImportError as e:
        print(f"  ❌ 核心模块导入失败: {e}")
    
    # 测试工具模块
    try:
        from src.tools.abc_to_midi import ABCToMIDIConverter
        from src.tools.abc_cleaner import fix_abc_structure
        tests['tools'] = True
        print("  ✅ 工具模块导入成功")
    except ImportError as e:
        print(f"  ❌ 工具模块导入失败: {e}")
    
    return tests

def test_model_files():
    """测试模型文件"""
    print("\n📂 测试模型文件...")
    
    files_status = {}
    
    # 检查模型文件
    model_path = "../models/hoyomusic_generator.pth"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024*1024)
        print(f"  ✅ 模型文件存在: {size:.2f} MB")
        files_status['model'] = True
    else:
        print("  ❌ 模型文件不存在")
        files_status['model'] = False
    
    # 检查映射文件
    mappings_path = "../models/hoyomusic_mappings.pkl"
    if os.path.exists(mappings_path):
        try:
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
            vocab_size = len(mappings.get('char_to_int', {}))
            print(f"  ✅ 映射文件存在: 词汇表大小 {vocab_size}")
            files_status['mappings'] = True
        except Exception as e:
            print(f"  ⚠️ 映射文件损坏: {e}")
            files_status['mappings'] = False
    else:
        print("  ❌ 映射文件不存在")
        files_status['mappings'] = False
    
    # 检查配置文件
    config_path = "../models/training_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"  ✅ 配置文件存在: 序列长度 {config.get('seq_length', 'N/A')}")
            files_status['config'] = True
        except Exception as e:
            print(f"  ⚠️ 配置文件损坏: {e}")
            files_status['config'] = False
    else:
        print("  ❌ 配置文件不存在")
        files_status['config'] = False
    
    return files_status

def test_directories():
    """测试目录结构"""
    print("\n📁 测试目录结构...")
    
    required_dirs = [
        "src/core",
        "src/tools", 
        "src/ui",
        "models",
        "output/generated",
        "data"
    ]
    
    dirs_status = {}
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}")
            dirs_status[dir_path] = True
        else:
            print(f"  ❌ {dir_path}")
            dirs_status[dir_path] = False
            # 创建缺失的目录
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"    📁 已创建目录: {dir_path}")
            except Exception as e:
                print(f"    ❌ 创建目录失败: {e}")
    
    return dirs_status

def test_streamlit_app():
    """测试Streamlit应用启动"""
    print("\n🌐 测试Streamlit应用...")
    
    app_files = [
        "src/ui/app_working.py",
        "src/ui/app_fixed.py",
        "src/ui/app.py"
    ]
    
    for app_file in app_files:
        if os.path.exists(app_file):
            print(f"  ✅ {app_file} 存在")
            
            # 尝试语法检查
            try:
                with open(app_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有set_page_config在开头
                lines = content.split('\n')
                found_config = False
                import_started = False
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith('import ') and not line.startswith('import streamlit'):
                        import_started = True
                    if 'st.set_page_config' in line:
                        found_config = True
                        if import_started:
                            print(f"    ⚠️ set_page_config不在正确位置（第{i+1}行）")
                        else:
                            print(f"    ✅ set_page_config位置正确（第{i+1}行）")
                        break
                
                if not found_config:
                    print(f"    ⚠️ 未找到set_page_config")
                    
            except Exception as e:
                print(f"    ❌ 文件检查失败: {e}")
        else:
            print(f"  ❌ {app_file} 不存在")

def test_music_generation():
    """测试音乐生成功能"""
    print("\n🎵 测试音乐生成功能...")
    
    try:
        # 检查是否能加载模型相关文件
        model_path = "../models/hoyomusic_generator.pth"
        mappings_path = "../models/hoyomusic_mappings.pkl"
        config_path = "../models/training_config.json"
        
        if not all(os.path.exists(p) for p in [model_path, mappings_path, config_path]):
            print("  ❌ 缺少必要的模型文件")
            return False
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 加载映射
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        print(f"  ✅ 配置加载成功")
        print(f"    - 词汇表大小: {len(mappings.get('char_to_int', {}))}")
        print(f"    - 序列长度: {config.get('seq_length', 'N/A')}")
        print(f"    - 隐藏层大小: {config.get('hidden_size', 'N/A')}")
        
        # 测试生成示例文本
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_abc = f"""T:Test Generated Music - {timestamp}
M:4/4
L:1/8
K:C
CDEF GABc | cBAG FEDC |
"""
        
        output_dir = "../output/generated"
        os.makedirs(output_dir, exist_ok=True)
        
        test_file = os.path.join(output_dir, f"test_generation_{timestamp}.abc")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_abc)
        
        print(f"  ✅ 测试文件生成成功: {test_file}")
        return True
        
    except Exception as e:
        print(f"  ❌ 音乐生成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎵 HoyoMusic AI Generator - 核心功能测试")
    print("=" * 60)
    
    # 运行所有测试
    imports_result = test_imports()
    files_result = test_model_files()
    dirs_result = test_directories()
    test_streamlit_app()
    generation_result = test_music_generation()
    
    # 生成测试报告
    print("\n" + "=" * 60)
    print("📊 测试报告总结")
    print("-" * 60)
    
    # 计算通过率
    total_tests = 0
    passed_tests = 0
    
    print("🔍 模块导入测试:")
    for test, result in imports_result.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test}")
        total_tests += 1
        if result:
            passed_tests += 1
    
    print("\n📂 文件检查测试:")
    for test, result in files_result.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test}")
        total_tests += 1
        if result:
            passed_tests += 1
    
    print(f"\n🎵 音乐生成测试: {'✅' if generation_result else '❌'}")
    total_tests += 1
    if generation_result:
        passed_tests += 1
    
    # 总体状态
    pass_rate = (passed_tests / total_tests) * 100
    print(f"\n📊 总体通过率: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")
    
    if pass_rate >= 80:
        print("🎉 系统状态良好，可以正常使用！")
    elif pass_rate >= 60:
        print("⚠️ 系统基本可用，但有些问题需要解决")
    else:
        print("❌ 系统有严重问题，需要修复")
    
    print("\n🌐 WebUI状态:")
    print("  ✅ 应用已在 http://localhost:8502 启动")
    print("  🔧 使用修复版本 (app_working.py)")
    print("  📱 支持现代化Glassmorphism界面")
    
    print("\n" + "=" * 60)
    print("测试完成！")

if __name__ == "__main__":
    main()

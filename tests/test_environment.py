#!/usr/bin/env python3
"""
HoyoMusic生成器 - PyTorch环境测试脚本
测试PyTorch安装和CUDA可用性
"""

import sys
import os

def test_basic_imports():
    """测试基本包导入"""
    print("📦 测试基本包导入...")
    
    try:
        import numpy as np
        print(f"  ✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ numpy导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  ✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ❌ pandas导入失败: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✅ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ❌ matplotlib导入失败: {e}")
        return False
    
    return True

def test_pytorch():
    """测试PyTorch安装"""
    print("\n🔥 测试PyTorch安装...")
    
    try:
        import torch
        print(f"  ✅ PyTorch版本: {torch.__version__}")
        
        # 测试CUDA
        cuda_available = torch.cuda.is_available()
        print(f"  🎮 CUDA可用: {cuda_available}")
        
        if cuda_available:
            print(f"  🔧 CUDA版本: {torch.version.cuda}")
            print(f"  💾 GPU设备数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  📊 GPU {i}: {device_name} ({memory:.1f}GB)")
        else:
            print("  ⚠️  将使用CPU运行")
        
        # 测试基本操作
        print("  🧪 测试基本张量操作...")
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print(f"  ✅ 张量乘法测试通过: {z.shape}")
        
        # 如果有CUDA，测试GPU操作
        if cuda_available:
            print("  🧪 测试GPU操作...")
            device = torch.device('cuda:0')
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            z_gpu = torch.mm(x_gpu, y_gpu)
            print(f"  ✅ GPU张量乘法测试通过: {z_gpu.shape}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ PyTorch导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ PyTorch测试失败: {e}")
        return False

def test_music_libraries():
    """测试音乐处理库"""
    print("\n🎵 测试音乐处理库...")
    
    success = True
    
    # 测试music21
    try:
        import music21
        print(f"  ✅ music21 {music21.__version__}")
    except ImportError as e:
        print(f"  ❌ music21导入失败: {e}")
        success = False
    
    # 测试mido
    try:
        import mido
        # mido可能没有__version__属性，使用另一种方式检测
        print(f"  ✅ mido 可用")
    except ImportError as e:
        print(f"  ❌ mido导入失败: {e}")
        success = False
    
    # 测试pretty_midi
    try:
        import pretty_midi
        # pretty_midi也可能没有__version__属性
        print(f"  ✅ pretty_midi 可用")
    except ImportError as e:
        print(f"  ❌ pretty_midi导入失败: {e}")
        success = False
    
    # 测试pyfluidsynth（可选，不影响主要功能）
    try:
        import fluidsynth
        print(f"  ✅ pyfluidsynth 可用")
    except ImportError:
        print(f"  ⚠️  pyfluidsynth不可用（可选，不影响主要功能）")
    except Exception as e:
        print(f"  ⚠️  pyfluidsynth配置问题（可选，不影响主要功能）")
    
    return success

def test_hoyomusic_modules():
    """测试HoyoMusic模块"""
    print("\n🎮 测试HoyoMusic模块...")
    
    # 测试model.py
    try:
        from model import HoyoMusicGenerator, HoyoMusicLSTM
        print("  ✅ model.py导入成功")
        
        # 创建一个小测试模型
        generator = HoyoMusicGenerator(vocab_size=100, seq_length=50, lstm_units=64)
        generator.build_model()
        print("  ✅ 模型构建测试通过")
        
    except ImportError as e:
        print(f"  ❌ model.py导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 模型测试失败: {e}")
        return False
    
    # 测试data_processor.py
    try:
        from data_processor import HoyoMusicDataProcessor
        print("  ✅ data_processor.py导入成功")
    except ImportError as e:
        print(f"  ❌ data_processor.py导入失败: {e}")
        return False
    
    # 测试abc_to_midi.py
    try:
        from abc_to_midi import ABCToMIDIConverter
        print("  ✅ abc_to_midi.py导入成功")
    except ImportError as e:
        print(f"  ❌ abc_to_midi.py导入失败: {e}")
        return False
    
    return True

def test_directories():
    """测试必要的目录结构"""
    print("\n📁 检查目录结构...")
    
    directories = [
        'data/abc_files',
        'generated_music',
        'models',
        'hoyomusic_cache'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✅ {directory} 存在")
        else:
            print(f"  📁 创建目录: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    return True

def main():
    """主测试函数"""
    print("🎵 HoyoMusic生成器 - PyTorch环境测试")
    print("=" * 60)
    
    # Python版本检查
    print(f"🐍 Python版本: {sys.version}")
    
    all_tests_passed = True
      # 运行所有测试
    tests = [
        test_basic_imports,
        test_pytorch,
        test_music_libraries,
        test_hoyomusic_modules,
    ]
    
    for test_func in tests:
        try:
            result = test_func()
            if result is False:
                all_tests_passed = False
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            all_tests_passed = False
    
    # 目录测试不影响测试结果
    test_directories()
    
    print("\n" + "=" * 60)
    
    if all_tests_passed:
        print("🎉 所有测试通过！环境配置成功")
        print("\n📝 可以开始使用HoyoMusic生成器:")
        print("   python train.py --use-hoyomusic --max-samples 1000 --epochs 5  # 快速测试")
        print("   python train.py --use-hoyomusic  # 完整训练")
        print("   python generate.py --region Mondstadt  # 生成音乐")
    else:
        print("❌ 部分测试失败，请检查安装")
        print("\n🔧 修复建议:")
        print("   pip install -r requirements.txt")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

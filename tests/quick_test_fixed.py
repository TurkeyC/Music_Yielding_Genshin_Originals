#!/usr/bin/env python3
"""
HoyoMusic生成器 - 快速功能测试脚本
测试训练和生成功能的基本可用性
"""

import os
import sys
# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from model import HoyoMusicGenerator
from data_processor import HoyoMusicDataProcessor

def test_model_creation():
    """测试模型创建"""
    print("🔧 测试模型创建...")
    
    try:
        generator = HoyoMusicGenerator(
            vocab_size=100,
            seq_length=50,
            embedding_dim=128,
            lstm_units=64
        )
        generator.build_model()
        
        print(f"  ✅ 模型创建成功，参数数量: {generator.get_model_size():,}")
        return generator
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        return None

def test_data_processing():
    """测试数据处理"""
    print("\n📊 测试数据处理...")
    
    try:
        processor = HoyoMusicDataProcessor()
        
        # 创建一些测试数据
        test_abc = """X:1
T:Test Melody
M:4/4
L:1/8
K:C
|:C2 D2 E2 F2|G2 A2 B2 c2:|

X:2
T:Another Melody  
M:3/4
L:1/4
K:G
|:G A B|c d e|d c B|A G F|G3|G3:|
"""
        
        # 测试ABC文本清理
        cleaned = processor.clean_abc_text(test_abc)
        print(f"  ✅ ABC清理成功，长度: {len(cleaned)}")
        
        # 模拟数据准备过程（使用更多数据）
        processor.raw_text = cleaned * 5  # 重复几次以有足够的数据
        chars = sorted(list(set(processor.raw_text)))
        processor.vocab_size = len(chars)
        processor.char_to_int = {ch: i for i, ch in enumerate(chars)}
        processor.int_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"  ✅ 词汇表创建成功，大小: {processor.vocab_size}")
        
        # 测试序列创建
        if len(processor.raw_text) > processor.seq_length:
            X, y = processor.create_sequences()
            print(f"  ✅ 序列创建成功，X: {X.shape}, y: {y.shape}")
            return processor, processor.char_to_int, [X, y]
        else:
            print(f"  ⚠️  数据太短，无法创建序列（需要>{processor.seq_length}字符）")
            return processor, processor.char_to_int, []
            
    except Exception as e:
        print(f"  ❌ 数据处理失败: {e}")
        return None, None, None

def test_training():
    """测试训练功能"""
    print("\n🏋️ 测试训练功能...")
    
    generator = test_model_creation()
    if generator is None:
        return False
    
    processor, char_to_int, data = test_data_processing()
    if processor is None:
        return False
    
    try:
        # 创建一些虚拟训练数据
        seq_length = 20
        vocab_size = max(100, processor.vocab_size if processor.vocab_size else 100)
        
        # 生成虚拟数据
        num_samples = 100
        X = np.random.randint(0, vocab_size, (num_samples, seq_length))
        y = np.random.randint(0, vocab_size, (num_samples,))
        
        print(f"  📊 创建训练数据: {X.shape}, 目标: {y.shape}")
        
        # 重新构建模型以匹配词汇表大小
        generator = HoyoMusicGenerator(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=64,
            lstm_units=32
        )
        generator.build_model()
        
        # 进行少量训练步骤测试
        loss = generator.train_step(X[:10], y[:10])
        print(f"  ✅ 训练步骤测试成功，损失: {loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 训练测试失败: {e}")
        return False

def test_generation():
    """测试生成功能"""
    print("\n🎵 测试生成功能...")
    
    try:
        # 创建一个简单的模型用于生成测试
        vocab_size = 100
        seq_length = 20
        
        generator = HoyoMusicGenerator(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=64,
            lstm_units=32
        )
        generator.build_model()
        
        # 创建起始序列
        seed_sequence = np.random.randint(0, vocab_size, seq_length)
        
        # 生成序列
        generated = generator.generate_sequence(
            seed_sequence=seed_sequence,
            length=50,
            temperature=1.0
        )
        
        print(f"  ✅ 序列生成成功，长度: {len(generated)}")
        
        # 测试不同温度
        for temp in [0.5, 1.0, 1.5]:
            generated_temp = generator.generate_sequence(
                seed_sequence=seed_sequence,
                length=20,
                temperature=temp
            )
            print(f"  ✅ 温度 {temp} 生成成功，长度: {len(generated_temp)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 生成测试失败: {e}")
        return False

def test_model_save_load():
    """测试模型保存和加载"""
    print("\n💾 测试模型保存和加载...")
    
    try:
        # 创建模型
        generator = HoyoMusicGenerator(
            vocab_size=100,
            seq_length=20,
            embedding_dim=64,
            lstm_units=32
        )
        generator.build_model()
        
        # 保存模型
        test_model_path = "models/test_model.pth"
        generator.save_model(test_model_path)
        print(f"  ✅ 模型保存成功: {test_model_path}")
        
        # 创建新的生成器并加载模型
        new_generator = HoyoMusicGenerator(
            vocab_size=100,
            seq_length=20,
            embedding_dim=64,
            lstm_units=32
        )
        new_generator.build_model()
        new_generator.load_model(test_model_path)
        print(f"  ✅ 模型加载成功")
        
        # 清理测试文件
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
            print(f"  🗑️ 清理测试文件")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 保存/加载测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎵 HoyoMusic生成器 - 功能测试")
    print("=" * 60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"💾 GPU: {torch.cuda.get_device_name()}")
    
    # 运行所有测试
    tests = [
        ("模型创建", lambda: test_model_creation() is not None),
        ("数据处理", lambda: test_data_processing()[0] is not None),
        ("训练功能", test_training),
        ("生成功能", test_generation),
        ("保存/加载", test_model_save_load),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有功能测试通过！")
        print("\n📝 可以开始使用:")
        print("   python train.py --use-hoyomusic --max-samples 100 --epochs 2  # 快速测试")
        print("   python generate.py --region Mondstadt --length 100  # 生成音乐")
    else:
        print("⚠️  部分功能测试失败，请检查实现")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

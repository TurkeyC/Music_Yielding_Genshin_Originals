#!/usr/bin/env python3
"""
HoyoMusic生成器 - 性能基准测试
对比不同配置下的训练和生成性能
"""

import os
import sys
# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import numpy as np
from model import HoyoMusicGenerator
from data_processor import HoyoMusicDataProcessor
import matplotlib.pyplot as plt
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

def benchmark_model_sizes():
    """测试不同模型大小的性能"""
    print("🔬 模型大小性能基准测试")
    print("=" * 60)
    
    configs = [
        {"name": "小型", "embedding": 64, "lstm": 128, "seq_len": 50},
        {"name": "中型", "embedding": 128, "lstm": 256, "seq_len": 100},
        {"name": "大型", "embedding": 256, "lstm": 512, "seq_len": 120},
        {"name": "超大", "embedding": 512, "lstm": 1024, "seq_len": 150},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 测试 {config['name']} 模型...")
        
        try:
            # 创建模型
            start_time = time.time()
            generator = HoyoMusicGenerator(
                vocab_size=100,
                seq_length=config['seq_len'],
                embedding_dim=config['embedding'],
                lstm_units=config['lstm']
            )
            generator.build_model()
            build_time = time.time() - start_time
            
            # 模型大小
            param_count = generator.get_model_size()
            
            # 训练性能测试
            batch_size = 16
            X = np.random.randint(0, 100, (batch_size, config['seq_len']))
            y = np.random.randint(0, 100, (batch_size,))
            
            # 预热
            for _ in range(3):
                generator.train_step(X, y)
            
            # 测试训练速度
            start_time = time.time()
            for _ in range(10):
                loss = generator.train_step(X, y)
            train_time = (time.time() - start_time) / 10
            
            # 测试生成速度
            seed = np.random.randint(0, 100, config['seq_len'])
            start_time = time.time()
            generated = generator.generate_sequence(seed, length=50)
            gen_time = time.time() - start_time
            
            result = {
                'name': config['name'],
                'params': param_count,
                'build_time': build_time,
                'train_time': train_time,
                'gen_time': gen_time,
                'config': config
            }
            results.append(result)
            
            print(f"  ✅ 参数数量: {param_count:,}")
            print(f"  ⏱️ 构建时间: {build_time:.3f}s")
            print(f"  🏃 训练时间: {train_time:.3f}s/batch")
            print(f"  🎵 生成时间: {gen_time:.3f}s")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
    
    return results

def benchmark_device_performance():
    """测试GPU vs CPU性能"""
    print("\n⚡ 设备性能基准测试")
    print("=" * 60)
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    results = {}
    
    for device_name in devices:
        print(f"\n🎮 测试设备: {device_name.upper()}")
        
        device = torch.device(device_name)
        
        # 创建模型
        generator = HoyoMusicGenerator(
            vocab_size=100,
            seq_length=100,
            embedding_dim=128,
            lstm_units=256
        )
        generator.build_model()
        generator.model.to(device)
        
        # 测试数据
        batch_size = 32
        X = np.random.randint(0, 100, (batch_size, 100))
        y = np.random.randint(0, 100, (batch_size,))
        
        # --- 修复关键：将X和y转为当前device上的tensor ---
        X_tensor = torch.LongTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        # 预热
        for _ in range(5):
            generator.model.train()
            outputs = generator.model(X_tensor)
            loss = generator.criterion(outputs, y_tensor)
            generator.optimizer.zero_grad()
            loss.backward()
            generator.optimizer.step()
        
        # 训练性能测试
        start_time = time.time()
        for _ in range(20):
            generator.model.train()
            outputs = generator.model(X_tensor)
            loss = generator.criterion(outputs, y_tensor)
            generator.optimizer.zero_grad()
            loss.backward()
            generator.optimizer.step()
        train_time = (time.time() - start_time) / 20
        
        # 生成性能测试
        seed = np.random.randint(0, 100, 100)
        start_time = time.time()
        for _ in range(10):
            # 生成时模型已在device上，seed输入转为device
            current_seq = list(seed[-generator.seq_length:])
            x = torch.LongTensor([current_seq]).to(device)
            with torch.no_grad():
                outputs = generator.model(x)
                predictions = torch.softmax(outputs / 1.0, dim=1).cpu().numpy()[0]
                _ = np.random.choice(len(predictions), p=predictions)
        gen_time = (time.time() - start_time) / 10
        
        results[device_name] = {
            'train_time': train_time,
            'gen_time': gen_time
        }
        
        print(f"  🏃 训练时间: {train_time:.3f}s/batch")
        print(f"  🎵 生成时间: {gen_time:.3f}s")
    
    # 显示性能对比
    if 'cuda' in results and 'cpu' in results:
        speedup_train = results['cpu']['train_time'] / results['cuda']['train_time']
        speedup_gen = results['cpu']['gen_time'] / results['cuda']['gen_time']
        print(f"\n📈 GPU加速效果:")
        print(f"  - 训练加速: {speedup_train:.1f}x")
        print(f"  - 生成加速: {speedup_gen:.1f}x")
    
    return results

def benchmark_generation_quality():
    """测试不同温度参数的生成质量"""
    print("\n🎨 生成质量基准测试")
    print("=" * 60)
    
    # 创建小型模型用于测试
    generator = HoyoMusicGenerator(
        vocab_size=50,
        seq_length=50,
        embedding_dim=64,
        lstm_units=128
    )
    generator.build_model()
    
    temperatures = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = {}
    
    seed = np.random.randint(0, 50, 50)
    
    for temp in temperatures:
        print(f"🌡️ 测试温度: {temp}")
        
        sequences = []
        for _ in range(5):
            seq = generator.generate_sequence(seed, length=100, temperature=temp)
            sequences.append(seq)
        
        # 计算多样性指标
        unique_sequences = len(set(map(tuple, sequences)))
        diversity_score = unique_sequences / len(sequences)
        
        # 计算平均序列长度
        avg_length = np.mean([len(seq) for seq in sequences])
        
        results[temp] = {
            'diversity': diversity_score,
            'avg_length': avg_length,
            'sequences': sequences
        }
        
        print(f"  📊 多样性: {diversity_score:.2f}")
        print(f"  📏 平均长度: {avg_length:.1f}")
    
    return results

def save_benchmark_report(model_results, device_results, quality_results):
    """保存基准测试报告"""
    
    # 创建可视化报告
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('HoyoMusic PyTorch性能基准测试报告', fontsize=16)
    
    # 1. 模型大小 vs 参数数量
    if model_results:
        names = [r['name'] for r in model_results]
        params = [r['params'] for r in model_results]
        train_times = [r['train_time'] for r in model_results]
        
        ax1.bar(names, params, color='skyblue')
        ax1.set_title('模型参数数量对比')
        ax1.set_ylabel('参数数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 训练时间对比
        ax2.bar(names, train_times, color='lightgreen')
        ax2.set_title('训练时间对比')
        ax2.set_ylabel('时间(秒/batch)')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. 设备性能对比
    if device_results:
        devices = list(device_results.keys())
        train_times = [device_results[d]['train_time'] for d in devices]
        gen_times = [device_results[d]['gen_time'] for d in devices]
        
        x = np.arange(len(devices))
        width = 0.35
        
        ax3.bar(x - width/2, train_times, width, label='训练时间', color='orange')
        ax3.bar(x + width/2, gen_times, width, label='生成时间', color='purple')
        ax3.set_title('设备性能对比')
        ax3.set_ylabel('时间(秒)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([d.upper() for d in devices])
        ax3.legend()
    
    # 4. 温度 vs 多样性
    if quality_results:
        temps = list(quality_results.keys())
        diversity = [quality_results[t]['diversity'] for t in temps]
        
        ax4.plot(temps, diversity, 'ro-', linewidth=2, markersize=8)
        ax4.set_title('温度参数 vs 生成多样性')
        ax4.set_xlabel('温度')
        ax4.set_ylabel('多样性分数')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'tests/report/pytorch_benchmark_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存文本报告
    with open(r'tests/report/pytorch_benchmark_report.txt', 'w', encoding='utf-8') as f:
        f.write("HoyoMusic PyTorch性能基准测试报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("模型大小测试结果:\n")
        f.write("-" * 30 + "\n")
        for result in model_results:
            f.write(f"{result['name']}模型:\n")
            f.write(f"  参数数量: {result['params']:,}\n")
            f.write(f"  训练时间: {result['train_time']:.3f}s/batch\n")
            f.write(f"  生成时间: {result['gen_time']:.3f}s\n\n")
        
        f.write("设备性能测试结果:\n")
        f.write("-" * 30 + "\n")
        for device, perf in device_results.items():
            f.write(f"{device.upper()}:\n")
            f.write(f"  训练时间: {perf['train_time']:.3f}s/batch\n")
            f.write(f"  生成时间: {perf['gen_time']:.3f}s\n\n")
        
        f.write("生成质量测试结果:\n")
        f.write("-" * 30 + "\n")
        for temp, quality in quality_results.items():
            f.write(f"温度 {temp}:\n")
            f.write(f"  多样性: {quality['diversity']:.2f}\n")
            f.write(f"  平均长度: {quality['avg_length']:.1f}\n\n")

def main():
    """主基准测试函数"""
    print("🚀 HoyoMusic PyTorch性能基准测试")
    print("=" * 60)
    
    # 显示系统信息
    print(f"🖥️ 系统信息:")
    print(f"  - PyTorch版本: {torch.__version__}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name()}")
        print(f"  - CUDA版本: {torch.version.cuda}")
    
    try:
        # 运行基准测试
        print("\n🔬 开始基准测试...")
        
        model_results = benchmark_model_sizes()
        device_results = benchmark_device_performance()
        quality_results = benchmark_generation_quality()
        
        # 保存报告
        save_benchmark_report(model_results, device_results, quality_results)
        
        print("\n✅ 基准测试完成！")
        print("📁 报告已保存:")
        print("  - 图表报告: pytorch_benchmark_report.png")
        print("  - 文本报告: pytorch_benchmark_report.txt")
        
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

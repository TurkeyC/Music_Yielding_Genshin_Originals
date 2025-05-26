import os
import numpy as np
from data_processor import HoyoMusicDataProcessor
from model import HoyoMusicGenerator
import matplotlib.pyplot as plt
import argparse

def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='训练损失')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    if 'sparse_top_k_categorical_accuracy' in history.history:
        plt.plot(history.history['sparse_top_k_categorical_accuracy'], label='Top-5准确率')
        plt.title('Top-5准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('hoyomusic_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='训练HoyoMusic生成器')
    parser.add_argument('--use-hoyomusic', action='store_true', default=True, 
                       help='使用HoyoMusic数据集')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='限制使用的样本数量（用于测试）')
    parser.add_argument('--seq-length', type=int, default=120,
                       help='序列长度')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lstm-units', type=int, default=512,
                       help='LSTM单元数')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs('data/abc_files', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('generated_music', exist_ok=True)
    
    print("🎵 开始训练HoyoMusic风格生成器...")
    print(f"配置: 序列长度={args.seq_length}, 批次大小={args.batch_size}, 训练轮数={args.epochs}")
    
    # 1. 数据处理
    print("\n=== 步骤1: 数据处理 ===")
    processor = HoyoMusicDataProcessor(seq_length=args.seq_length)
    
    X, y = processor.prepare_data(
        use_hoyomusic=args.use_hoyomusic,
        data_dir='data/abc_files',
        max_samples=args.max_samples
    )
    
    # 保存字符映射
    processor.save_mappings('models/hoyomusic_mappings.pkl')
    
    print(f"训练数据形状: X={X.shape}, y={y.shape}")
    
    # 2. 构建和训练模型
    print("\n=== 步骤2: 模型训练 ===")
    generator = HoyoMusicGenerator(
        vocab_size=processor.vocab_size,
        seq_length=args.seq_length,
        embedding_dim=256,
        lstm_units=args.lstm_units
    )
    
    # 训练模型
    history = generator.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        model_save_path='models/hoyomusic_generator.h5'
    )
    
    # 3. 保存训练历史
    print("\n=== 步骤3: 保存结果 ===")
    plot_training_history(history)
    
    # 保存训练配置
    config = {
        'seq_length': args.seq_length,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lstm_units': args.lstm_units,
        'vocab_size': processor.vocab_size,
        'final_loss': history.history['loss'][-1],
        'final_accuracy': history.history['accuracy'][-1]
    }
    
    import json
    with open('models/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("🎉 训练完成！")
    print("📁 文件保存位置:")
    print("  - 模型: models/hoyomusic_generator.h5")
    print("  - 字符映射: models/hoyomusic_mappings.pkl")
    print("  - 训练历史: hoyomusic_training_history.png")
    print("  - 训练配置: models/training_config.json")
    
    print("\n🎼 现在可以使用 generate.py 生成原神风格的音乐了！")

if __name__ == "__main__":
    main()
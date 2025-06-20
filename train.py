import os
import numpy as np
import torch
from data_processor import HoyoMusicDataProcessor
from model import HoyoMusicGenerator
from tools.training_visualizer import TrainingVisualizer
import matplotlib.pyplot as plt
import argparse
import json
import threading
import time

def plot_training_history(history, save_path='hoyomusic_training_history.png'):
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    """绘制训练历史"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='训练损失', color='red', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='验证损失', color='orange', linewidth=2)
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='训练准确率', color='green', linewidth=2)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='验证准确率', color='cyan', linewidth=2)
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 移除第三个子图，因为PyTorch版本不需要Top-K准确率
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def start_real_time_monitoring():
    """启动实时监控（在单独线程中）"""
    def monitor():
        visualizer = TrainingVisualizer()
        visualizer.start_monitoring()
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread

def main():
    parser = argparse.ArgumentParser(description='训练HoyoMusic生成器（支持增量训练）')
    
    # 数据相关参数
    parser.add_argument('--use-hoyomusic', action='store_true', default=True, 
                       help='使用HoyoMusic数据集')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='限制使用的样本数量（用于测试）')
    parser.add_argument('--additional-data-dir', type=str, default=None,
                       help='额外的ABC数据目录（用于增量训练）')
      # 模型相关参数
    parser.add_argument('--seq-length', type=int, default=120,
                       help='序列长度')
    parser.add_argument('--lstm-units', type=int, default=512,
                       help='LSTM单元数')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    
    # 增量训练参数
    parser.add_argument('--incremental', action='store_true',
                       help='进行增量训练（基于现有模型）')
    parser.add_argument('--base-model', type=str, default='models/hoyomusic_generator.pth',
                       help='基础模型路径（用于增量训练）')
    parser.add_argument('--incremental-lr', type=float, default=0.0005,
                       help='增量训练的学习率')
    
    # 断点续连参数
    parser.add_argument('--resume', action='store_true',
                       help='自动从最新checkpoint恢复训练')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='从指定checkpoint文件恢复训练')
    parser.add_argument('--enable-checkpoints', action='store_true', default=True,
                       help='启用自动checkpoint保存')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='checkpoint保存间隔（epoch数）')
    parser.add_argument('--disable-checkpoints', action='store_true',
                       help='禁用checkpoint保存')
    
    # 数据管理参数
    parser.add_argument('--merge-datasets', action='store_true',
                       help='合并多个数据集进行训练')
    parser.add_argument('--dataset-weights', type=str, default=None,
                       help='数据集权重，格式: "old:0.7,new:0.3"')
      # 监控相关参数
    parser.add_argument('--real-time-monitor', action='store_true',
                       help='启用实时训练监控')
    parser.add_argument('--no-visualization', action='store_true',
                       help='禁用可视化')
    
    args = parser.parse_args()
    
    # 处理checkpoint相关参数
    enable_checkpoints = args.enable_checkpoints and not args.disable_checkpoints
    
    # 创建必要的目录
    os.makedirs('data/abc_files', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('generated_music', exist_ok=True)
    
    print("🎵 HoyoMusic风格生成器训练程序")
    print("=" * 60)
    
    # 确定训练模式
    training_mode = "新训练"
    if args.resume or args.resume_from:
        training_mode = "断点续连"
    elif args.incremental:
        training_mode = "增量训练"
    
    print(f"🎯 训练模式: {training_mode}")
    
    if args.incremental:
        print(f"📁 基础模型: {args.base_model}")
    
    if args.resume_from:
        print(f"📂 恢复点: {args.resume_from}")
    elif args.resume:
        print("🔍 将自动查找最新checkpoint")
    
    if enable_checkpoints:
        print(f"💾 Checkpoint: 启用 (间隔: {args.checkpoint_interval} epochs)")
    else:
        print("💾 Checkpoint: 禁用")
    
    print(f"📊 配置参数:")
    print(f"  - 序列长度: {args.seq_length}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - LSTM单元: {args.lstm_units}")
    if args.incremental:
        print(f"  - 增量学习率: {args.incremental_lr}")
    
    # 启动实时监控（如果需要）
    monitor_thread = None
    if args.real_time_monitor and not args.no_visualization:
        print("📊 启动实时训练监控...")
        monitor_thread = start_real_time_monitoring()
        time.sleep(2)  # 等待监控窗口启动
    
    # 1. 数据处理
    print("\n=== 步骤1: 数据处理 ===")
    processor = HoyoMusicDataProcessor(seq_length=args.seq_length)
    
    # 准备数据
    data_sources = []
    
    # HoyoMusic数据集
    if args.use_hoyomusic:
        data_sources.append("HoyoMusic数据集")
    
    # 额外的本地数据
    if args.additional_data_dir and os.path.exists(args.additional_data_dir):
        data_sources.append(f"本地数据: {args.additional_data_dir}")
    
    print(f"📚 数据来源: {', '.join(data_sources)}")
    
    X, y = processor.prepare_data(
        use_hoyomusic=args.use_hoyomusic,
        data_dir=args.additional_data_dir or 'data/abc_files',
        max_samples=args.max_samples
    )
    
    # 保存字符映射
    processor.save_mappings('models/hoyomusic_mappings.pkl')
    
    print(f"✅ 训练数据准备完成: X={X.shape}, y={y.shape}")
      # 2. 构建和训练模型
    print("\n=== 步骤2: 模型训练 ===")
    generator = HoyoMusicGenerator(
        vocab_size=processor.vocab_size,
        seq_length=args.seq_length,
        embedding_dim=256,
        lstm_units=args.lstm_units
    )
    
    # 处理不同的训练模式
    training_resumed = False
    
    # 1. 断点续连模式
    if args.resume_from:
        print(f"🔄 从指定checkpoint恢复: {args.resume_from}")
        if os.path.exists(args.resume_from):
            training_resumed = generator.load_checkpoint(args.resume_from, resume_training=True)
            if training_resumed:
                print("✅ 断点续连恢复成功")
            else:
                print("❌ 断点续连失败，将进行全新训练")
        else:
            print(f"❌ Checkpoint文件不存在: {args.resume_from}")
    
    elif args.resume:
        print("🔍 查找最新checkpoint进行恢复...")
        latest_checkpoint = generator.find_latest_checkpoint()
        if latest_checkpoint:
            print(f"📂 找到最新checkpoint: {os.path.basename(latest_checkpoint)}")
            training_resumed = generator.load_checkpoint(latest_checkpoint, resume_training=True)
            if training_resumed:
                print("✅ 自动断点续连成功")
            else:
                print("❌ 自动断点续连失败，将进行全新训练")
        else:
            print("📄 未找到可用的checkpoint")
    
    # 2. 增量训练模式
    elif args.incremental:
        if os.path.exists(args.base_model):            success = generator.load_model_for_incremental_training(
                args.base_model, 
                learning_rate=args.incremental_lr
            )
        if not success:
                print("⚠️ 加载基础模型失败，将进行全新训练")
                args.incremental = False
        else:
            print(f"❌ 基础模型文件不存在: {args.base_model}")
            print("🔧 将进行全新训练")
            args.incremental = False
    
    # 训练模型
    training_type = "增量" if (args.incremental or training_resumed) else ""
    print(f"🚀 开始{training_type}训练...")
    
    history = generator.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        model_save_path='models/hoyomusic_generator.pth',
        is_incremental=(args.incremental or training_resumed),
        enable_checkpoints=enable_checkpoints,
        checkpoint_interval=args.checkpoint_interval,
        auto_resume=False  # 已经在上面处理过了
    )
    
    # 3. 保存训练结果
    print("\n=== 步骤3: 保存结果 ===")
      # 绘制训练历史
    if not args.no_visualization:
        plot_training_history(history)
    
    # 保存训练配置
    config = {
        # 基本训练信息
        'training_type': 'resumed' if training_resumed else ('incremental' if args.incremental else 'new'),
        'seq_length': args.seq_length,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lstm_units': args.lstm_units,
        'vocab_size': processor.vocab_size,
        'total_samples': len(X),
        'final_loss': history.history['loss'][-1],
        'final_accuracy': history.history['accuracy'][-1],
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_sources': data_sources,
        
        # 新增：checkpoint和恢复信息
        'checkpoint_enabled': enable_checkpoints,
        'checkpoint_interval': args.checkpoint_interval if enable_checkpoints else None,
        'training_resumed': training_resumed,
        'resume_from_checkpoint': args.resume_from if args.resume_from else None,
        'auto_resume_used': args.resume,
        
        # 新增：数据集管理信息
        'merge_datasets': args.merge_datasets,
        'dataset_weights': args.dataset_weights,
        'additional_data_dir': args.additional_data_dir,
        
        # 新增：模型状态信息
        'current_epoch': getattr(generator, 'current_epoch', args.epochs),
        'best_val_loss': getattr(generator, 'best_val_loss', float('inf')),
        'total_epochs_trained': len(history.history['loss']),
        
        # PyTorch版本信息
        'pytorch_version': torch.__version__,
        'config_version': '2.0'
    }
    
    # 增量训练特定信息
    if args.incremental:
        config['base_model'] = args.base_model
        config['incremental_lr'] = args.incremental_lr
    
    # 断点续连特定信息
    if training_resumed:
        config['resumed_from_epoch'] = getattr(generator, 'current_epoch', 0)
        config['previous_training_sessions'] = getattr(generator, 'training_history', {})
    
    with open('models/training_config.json', 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 保存最终监控报告
    if args.real_time_monitor and not args.no_visualization:
        try:
            visualizer = TrainingVisualizer()
            visualizer.save_final_report('final_training_report.png')
        except Exception as e:
            print(f"⚠️ 保存监控报告失败: {e}")
    
    print("🎉 训练完成！")
    print("=" * 60)
    print("📁 文件保存位置:")
    print("  - 模型: models/hoyomusic_generator.pth")
    print("  - 字符映射: models/hoyomusic_mappings.pkl")
    print("  - 训练历史: models/training_history.json")
    print("  - 训练配置: models/training_config.json")
    if not args.no_visualization:
        print("  - 训练历史图: hoyomusic_training_history.png")
        if args.real_time_monitor:
            print("  - 监控报告: final_training_report.png")
    
    print(f"\n📊 训练摘要:")
    print(f"  - 训练类型: {'增量训练' if args.incremental else '全新训练'}")
    print(f"  - 最终损失: {history.history['loss'][-1]:.4f}")
    print(f"  - 最终准确率: {history.history['accuracy'][-1]:.4f}")
    if 'val_loss' in history.history:
        print(f"  - 最终验证损失: {history.history['val_loss'][-1]:.4f}")
        print(f"  - 最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    
    print(f"\n🎼 现在可以使用以下命令生成音乐:")
    print(f"python generate.py --region Mondstadt")
    
    # 如果有监控线程，等待用户关闭
    if monitor_thread and monitor_thread.is_alive():
        print(f"\n📊 实时监控正在运行，关闭图表窗口以结束程序")
        try:
            monitor_thread.join()
        except KeyboardInterrupt:
            print("✋ 用户中断，程序结束")

if __name__ == "__main__":
    main()
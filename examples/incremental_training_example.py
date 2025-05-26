#!/usr/bin/env python3
"""
增量训练示例脚本
演示如何基于已有模型进行增量训练
"""

import subprocess
import os
import json

def check_model_exists(model_path):
    """检查模型是否存在"""
    return os.path.exists(model_path)

def get_model_info(config_path):
    """获取模型训练信息"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        return None

def main():
    print("🔄 HoyoMusic增量训练示例")
    print("=" * 50)
    
    base_model = "models/hoyomusic_generator.h5"
    config_file = "models/training_config.json"
    
    # 检查是否有现有模型
    if check_model_exists(base_model):
        print(f"✅ 找到现有模型: {base_model}")
        
        # 显示模型信息
        config = get_model_info(config_file)
        if config:
            print(f"📊 模型信息:")
            print(f"  - 训练类型: {config.get('training_type', '未知')}")
            print(f"  - 已训练轮数: {config.get('epochs', '未知')}")
            print(f"  - 最终准确率: {config.get('final_accuracy', 'N/A'):.4f}")
            print(f"  - 训练日期: {config.get('training_date', '未知')}")
            print(f"  - 数据来源: {', '.join(config.get('data_sources', ['未知']))}")
        
        print(f"\n🚀 开始增量训练...")
        
        # 执行增量训练
        cmd = [
            "python", "train.py",
            "--incremental",
            "--base-model", base_model,
            "--epochs", "50",  # 增量训练轮数
            "--incremental-lr", "0.0003",  # 较低的学习率
            "--real-time-monitor",  # 启用实时监控
            "--use-hoyomusic"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    else:
        print(f"❌ 未找到现有模型: {base_model}")
        print(f"🆕 将进行全新训练...")
        
        # 执行全新训练
        cmd = [
            "python", "train.py",
            "--use-hoyomusic",
            "--epochs", "100",
            "--real-time-monitor"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
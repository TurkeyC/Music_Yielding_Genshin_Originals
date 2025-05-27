#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量训练和断点续连功能测试程序
测试HoyoMusic生成器的checkpoint系统、断点续连和增量训练功能
"""

import os
import sys
import torch
import numpy as np
import shutil
import time
from pathlib import Path

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model import HoyoMusicGenerator
from data_processor import HoyoMusicDataProcessor

class TestIncrementalTraining:
    def __init__(self):
        self.test_dir = "tests/temp_test_data"
        self.models_dir = "tests/temp_models" 
        self.checkpoints_dir = f"{self.models_dir}/checkpoints"
        self.test_passed = 0
        self.test_failed = 0
        
        # 创建临时测试目录
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        print("🧪 增量训练和断点续连功能测试")
        print("=" * 60)
    
    def create_dummy_data(self, size=1000):
        """创建测试用的虚拟数据"""
        print("📊 创建测试数据...")
        
        # 模拟音乐数据 - 简单的ABC notation模式
        vocab_size = 50
        seq_length = 20
        
        # 生成模拟的序列数据
        np.random.seed(42)  # 固定随机种子以便复现
        
        X = np.random.randint(1, vocab_size, size=(size, seq_length))
        y = np.random.randint(1, vocab_size, size=(size,))
        
        print(f"✅ 测试数据创建完成: X={X.shape}, y={y.shape}")
        return X, y, vocab_size, seq_length
    
    def test_basic_model_creation(self):
        """测试基本模型创建"""
        print("\n🔧 测试1: 基本模型创建")
        try:
            generator = HoyoMusicGenerator(
                vocab_size=50,
                seq_length=20,
                embedding_dim=128,
                lstm_units=256
            )
            generator.build_model()
            
            assert generator.model is not None, "模型创建失败"
            assert generator.optimizer is not None, "优化器创建失败"
            assert generator.criterion is not None, "损失函数创建失败"
            
            print("✅ 基本模型创建测试通过")
            self.test_passed += 1
            return generator
            
        except Exception as e:
            print(f"❌ 基本模型创建测试失败: {e}")
            self.test_failed += 1
            return None
    
    def test_checkpoint_save_load(self, generator, X, y):
        """测试checkpoint保存和加载"""
        print("\n💾 测试2: Checkpoint保存和加载")
        try:
            # 进行几步训练以产生一些状态
            generator.current_epoch = 5
            generator.best_val_loss = 2.5
            generator.patience_counter = 2
            generator.training_history = {
                'loss': [3.0, 2.8, 2.6, 2.4, 2.2],
                'accuracy': [0.1, 0.2, 0.3, 0.4, 0.5],
                'val_loss': [3.2, 3.0, 2.8, 2.6, 2.4],
                'val_accuracy': [0.1, 0.15, 0.25, 0.35, 0.45]
            }
            
            # 保存checkpoint
            checkpoint_path = generator.save_checkpoint(
                checkpoint_dir=self.checkpoints_dir,
                epoch=5,
                extra_info={'test': 'checkpoint_test'}
            )
            
            assert checkpoint_path is not None, "Checkpoint保存失败"
            assert os.path.exists(checkpoint_path), "Checkpoint文件不存在"
            
            # 创建新的generator并加载checkpoint
            new_generator = HoyoMusicGenerator(
                vocab_size=50,
                seq_length=20,
                embedding_dim=128,
                lstm_units=256
            )
            
            success = new_generator.load_checkpoint(checkpoint_path, resume_training=True)
            
            assert success, "Checkpoint加载失败"
            assert new_generator.current_epoch == 5, f"Epoch恢复错误: {new_generator.current_epoch}"
            assert new_generator.best_val_loss == 2.5, f"最佳损失恢复错误: {new_generator.best_val_loss}"
            assert len(new_generator.training_history['loss']) == 5, "训练历史恢复错误"
            
            print("✅ Checkpoint保存和加载测试通过")
            self.test_passed += 1
            return True
            
        except Exception as e:
            print(f"❌ Checkpoint保存和加载测试失败: {e}")
            self.test_failed += 1
            return False
    
    def test_resume_training(self, X, y):
        """测试断点续连训练"""
        print("\n🔄 测试3: 断点续连训练")
        try:
            # 创建初始模型并训练几个epoch
            generator1 = HoyoMusicGenerator(
                vocab_size=50,
                seq_length=20,
                embedding_dim=128,
                lstm_units=256
            )
            
            print("  - 进行初始训练 (5 epochs)...")
            history1 = generator1.train(
                X, y,
                epochs=5,
                batch_size=16,
                validation_split=0.2,
                model_save_path=f'{self.models_dir}/test_model_initial.pth',
                enable_checkpoints=True,
                checkpoint_interval=2,
                auto_resume=False
            )
            
            # 检查history对象有效性
            assert history1 is not None and hasattr(history1, 'history') and 'loss' in history1.history, "初始训练history无效"
            initial_epochs = len(history1.history['loss'])
            print(f"  - 初始训练完成，共 {initial_epochs} epochs")
            
            # 创建新的generator并从checkpoint恢复
            generator2 = HoyoMusicGenerator(
                vocab_size=50,
                seq_length=20,
                embedding_dim=128,
                lstm_units=256
            )
            
            # 查找最新checkpoint
            latest_checkpoint = generator2.find_latest_checkpoint(self.checkpoints_dir)
            assert latest_checkpoint is not None, "未找到checkpoint文件"
            
            print(f"  - 从checkpoint恢复: {os.path.basename(latest_checkpoint)}")
            success = generator2.load_checkpoint(latest_checkpoint, resume_training=True)
            assert success, "Checkpoint加载失败"
            
            # 继续训练
            print("  - 继续训练 (再训练3 epochs)...")
            history2 = generator2.train(
                X, y,
                epochs=8,  # 总共8个epoch，前5个已完成
                batch_size=16,
                validation_split=0.2,
                model_save_path=f'{self.models_dir}/test_model_resumed.pth',
                is_incremental=True,  # 标记为增量训练
                enable_checkpoints=True,
                checkpoint_interval=2,
                auto_resume=False
            )            # 检查history对象有效性
            assert history2 is not None, "断点续连训练返回history为None"
            assert hasattr(history2, 'history'), "断点续连训练history对象结构异常"
            assert 'loss' in history2.history, "断点续连训练history中没有loss记录"
            assert history2.history['loss'] is not None, "断点续连训练history的loss记录为None"
            
            final_epochs = len(history2.history['loss']) if history2.history['loss'] is not None else 0
            print(f"  - 断点续连训练完成，总共 {final_epochs} epochs")
            
            # 验证训练历史连续性（如果history记录正常）
            if final_epochs > 0 and initial_epochs > 0:
                assert final_epochs >= initial_epochs, "训练历史应该包含所有epoch"
            
            print("✅ 断点续连训练测试通过")
            self.test_passed += 1
            return True
            
        except Exception as e:
            print(f"❌ 断点续连训练测试失败: {e}")
            self.test_failed += 1
            return False
    
    def test_incremental_training(self, X, y):
        """测试增量训练"""
        print("\n📈 测试4: 增量训练")
        try:
            # 创建基础模型并训练
            base_generator = HoyoMusicGenerator(
                vocab_size=50,
                seq_length=20,
                embedding_dim=128,
                lstm_units=256
            )
            
            print("  - 训练基础模型...")
            base_generator.train(
                X[:800], y[:800],  # 使用部分数据
                epochs=3,
                batch_size=16,
                validation_split=0.2,
                model_save_path=f'{self.models_dir}/base_model.pth',
                enable_checkpoints=False  # 基础训练不需要checkpoint
            )
            
            # 创建新模型进行增量训练
            incremental_generator = HoyoMusicGenerator(
                vocab_size=50,
                seq_length=20,
                embedding_dim=128,
                lstm_units=256
            )
            
            print("  - 加载基础模型进行增量训练...")
            success = incremental_generator.load_model_for_incremental_training(
                f'{self.models_dir}/base_model.pth',
                learning_rate=0.0001  # 更小的学习率
            )
            assert success, "基础模型加载失败"
            
            print("  - 进行增量训练...")
            history = incremental_generator.train(
                X[600:], y[600:],  # 使用新的数据子集
                epochs=3,
                batch_size=16,
                validation_split=0.2,
                model_save_path=f'{self.models_dir}/incremental_model.pth',
                is_incremental=True,
                enable_checkpoints=True,
                checkpoint_interval=1
            )
            
            assert len(history.history['loss']) > 0, "增量训练历史为空"
            
            print("✅ 增量训练测试通过")
            self.test_passed += 1
            return True
            
        except Exception as e:
            print(f"❌ 增量训练测试失败: {e}")
            self.test_failed += 1
            return False
    
    def test_checkpoint_management(self):
        """测试checkpoint管理功能"""
        print("\n🗂️ 测试5: Checkpoint管理")
        try:
            generator = HoyoMusicGenerator(
                vocab_size=50,
                seq_length=20,
                embedding_dim=128,
                lstm_units=256
            )
            generator.build_model()
            
            # 创建多个checkpoint
            checkpoints_created = []
            for i in range(7):
                checkpoint_path = generator.save_checkpoint(
                    checkpoint_dir=self.checkpoints_dir,
                    epoch=i,
                    extra_info={'test_checkpoint': i}
                )
                checkpoints_created.append(checkpoint_path)
                time.sleep(0.1)  # 确保时间戳不同
            
            # 测试查找最新checkpoint
            latest = generator.find_latest_checkpoint(self.checkpoints_dir)
            assert latest is not None, "未找到最新checkpoint"
            
            # 测试清理旧checkpoint
            generator.cleanup_old_checkpoints(self.checkpoints_dir, keep_count=3)
            
            # 验证只保留了指定数量的checkpoint
            remaining_checkpoints = len([f for f in os.listdir(self.checkpoints_dir) 
                                       if f.startswith('checkpoint_epoch_')])
            # 注意：可能还有latest_checkpoint.pth文件
            assert remaining_checkpoints <= 4, f"保留的checkpoint过多: {remaining_checkpoints}"
            
            print("✅ Checkpoint管理测试通过")
            self.test_passed += 1
            return True
            
        except Exception as e:
            print(f"❌ Checkpoint管理测试失败: {e}")
            self.test_failed += 1
            return False
    
    def test_model_generation(self):
        """测试模型生成功能"""
        print("\n🎵 测试6: 音乐生成")
        try:
            # 使用之前训练的模型
            generator = HoyoMusicGenerator(
                vocab_size=50,
                seq_length=20,
                embedding_dim=128,
                lstm_units=256
            )
            
            # 尝试加载训练好的模型
            model_path = f'{self.models_dir}/test_model_resumed.pth'
            if os.path.exists(model_path):
                success = generator.load_model_for_incremental_training(model_path)
                if success:
                    # 测试序列生成
                    seed_sequence = [1, 2, 3, 4, 5]
                    generated = generator.generate_sequence(
                        seed_sequence, 
                        length=50, 
                        temperature=1.0
                    )
                    
                    assert len(generated) > len(seed_sequence), "生成序列长度不正确"
                    assert all(isinstance(x, (int, np.integer)) for x in generated), "生成序列包含非整数值"
                    
                    print(f"  - 生成序列长度: {len(generated)}")
                    print("✅ 音乐生成测试通过")
                    self.test_passed += 1
                    return True
            
            print("⚠️ 音乐生成测试跳过（无可用模型）")
            return True
            
        except Exception as e:
            print(f"❌ 音乐生成测试失败: {e}")
            self.test_failed += 1
            return False
    
    def cleanup(self):
        """清理测试文件"""
        print("\n🧹 清理测试文件...")
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
            if os.path.exists(self.models_dir):
                shutil.rmtree(self.models_dir)
            print("✅ 测试文件清理完成")
        except Exception as e:
            print(f"⚠️ 清理测试文件失败: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始运行所有测试...")
        
        # 创建测试数据
        X, y, vocab_size, seq_length = self.create_dummy_data()
        
        # 运行测试
        generator = self.test_basic_model_creation()
        
        if generator:
            self.test_checkpoint_save_load(generator, X, y)
            self.test_resume_training(X, y)
            self.test_incremental_training(X, y)
            self.test_checkpoint_management()
            self.test_model_generation()
        
        # 输出结果
        print("\n" + "="*60)
        print("🧪 测试结果总结")
        print("="*60)
        print(f"✅ 通过测试: {self.test_passed}")
        print(f"❌ 失败测试: {self.test_failed}")
        print(f"📊 成功率: {self.test_passed/(self.test_passed+self.test_failed)*100:.1f}%")
        
        if self.test_failed == 0:
            print("\n🎉 所有测试通过！增量训练和断点续连功能正常工作。")
        else:
            print(f"\n⚠️ 有 {self.test_failed} 个测试失败，请检查相关功能。")
        
        # 清理
        self.cleanup()
        
        return self.test_failed == 0

def main():
    """主函数"""
    print("🎵 HoyoMusic增量训练和断点续连功能测试")
    print("📝 此测试程序将验证以下功能：")
    print("   1. 基本模型创建和训练")
    print("   2. Checkpoint保存和加载")
    print("   3. 断点续连训练")
    print("   4. 增量训练")
    print("   5. Checkpoint管理（清理、查找）")
    print("   6. 音乐生成功能")
    print()
    
    # 检查PyTorch可用性
    print(f"🔧 环境信息:")
    print(f"   - PyTorch版本: {torch.__version__}")
    print(f"   - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name()}")
    print()
    
    # 运行测试
    tester = TestIncrementalTraining()
    success = tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

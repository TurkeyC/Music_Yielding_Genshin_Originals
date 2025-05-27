import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import threading

# 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA版本: {torch.version.cuda}")

class HoyoMusicLSTM(nn.Module):
    """基于PyTorch的HoyoMusic LSTM模型"""
    def __init__(self, vocab_size, seq_length, embedding_dim=256, lstm_units=512):
        super(HoyoMusicLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
          # LSTM层 - 修复dropout警告
        self.lstm1 = nn.LSTM(embedding_dim, lstm_units, batch_first=True, num_layers=2, dropout=0.3)
        self.bn1 = nn.BatchNorm1d(lstm_units)
        
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=2, dropout=0.3)
        self.bn2 = nn.BatchNorm1d(lstm_units)
        
        self.lstm3 = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=2, dropout=0.3)
        self.bn3 = nn.BatchNorm1d(lstm_units)
        
        # 全连接层
        self.fc1 = nn.Linear(lstm_units, lstm_units)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(lstm_units, lstm_units // 2)
        self.dropout2 = nn.Dropout(0.3)
        
        # 输出层
        self.output = nn.Linear(lstm_units // 2, vocab_size)
        
    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # 第一层LSTM
        lstm_out, _ = self.lstm1(embedded)
        # 批量归一化需要调整维度: (batch, seq_len, features) -> (batch, features, seq_len)
        if lstm_out.size(1) > 1:  # 避免单个时间步的情况
            lstm_out = lstm_out.transpose(1, 2)
            lstm_out = self.bn1(lstm_out)
            lstm_out = lstm_out.transpose(1, 2)
        
        # 第二层LSTM
        lstm_out, _ = self.lstm2(lstm_out)
        if lstm_out.size(1) > 1:
            lstm_out = lstm_out.transpose(1, 2)
            lstm_out = self.bn2(lstm_out)
            lstm_out = lstm_out.transpose(1, 2)
        
        # 第三层LSTM
        lstm_out, (hidden, _) = self.lstm3(lstm_out)
        if lstm_out.size(1) > 1:
            lstm_out = lstm_out.transpose(1, 2)
            lstm_out = self.bn3(lstm_out)
            lstm_out = lstm_out.transpose(1, 2)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch, lstm_units)
        
        # 全连接层
        out = F.relu(self.fc1(last_output))
        out = self.dropout1(out)
        
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        
        # 输出层
        out = self.output(out)  # (batch, vocab_size)
        
        return out

class HoyoMusicGenerator:
    def __init__(self, vocab_size, seq_length, embedding_dim=256, lstm_units=512):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.training_start_time = None
        self.estimated_time_remaining = None
        # 新增：checkpoint和断点续连相关属性
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.checkpoint_interval = 5  # 每5个epoch保存一次checkpoint
        self.auto_save_enabled = True
        self.resume_from_checkpoint = False
        self.last_checkpoint_path = None
    
    def build_model(self):
        """构建针对HoyoMusic优化的LSTM模型"""
        self.model = HoyoMusicLSTM(
            self.vocab_size, 
            self.seq_length, 
            self.embedding_dim, 
            self.lstm_units
        ).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8, 
            min_lr=0.0001,
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        
        param_count = sum(p.numel() for p in self.model.parameters())        
        print(f"✅ HoyoMusic模型已构建，参数数量: {param_count:,}")
        return self.model
    
    def load_model_for_incremental_training(self, model_path, learning_rate=0.0005):
        """加载现有模型进行增量训练"""
        print(f"🔄 加载现有模型进行增量训练: {model_path}")
        
        try:
            # 首先构建模型结构
            if self.model is None:
                self.build_model()
            
            # 加载模型权重 - PyTorch 2.6+ 安全性更新
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 模型权重加载成功")
                
                # 如果有优化器状态，也加载
                if 'optimizer_state_dict' in checkpoint and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"✅ 优化器状态加载成功")
            else:
                # 兼容旧格式
                self.model.load_state_dict(checkpoint)
                print(f"✅ 模型权重加载成功（兼容模式）")
            
            # 重新设置学习率进行增量训练
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
            
            print(f"🎯 增量训练学习率设置为: {learning_rate}")
            
            # 加载历史训练信息（如果存在）
            self.load_training_history()
            
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            print("🔧 将创建新模型...")
            self.build_model()
            return False
    
    def save_model(self, model_path, optimizer_path=None, include_optimizer=True):
        """保存模型"""
        try:
            if include_optimizer and self.optimizer:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'vocab_size': self.vocab_size,
                    'seq_length': self.seq_length,
                    'embedding_dim': self.embedding_dim,
                    'lstm_units': self.lstm_units
                }
            else:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'vocab_size': self.vocab_size,
                    'seq_length': self.seq_length,
                    'embedding_dim': self.embedding_dim,
                    'lstm_units': self.lstm_units
                }
            
            torch.save(checkpoint, model_path)
            print(f"✅ 模型已保存到: {model_path}")
            
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
    
    def save_training_history(self, history_path='models/training_history.json'):
        """保存训练历史"""
        try:
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"📊 训练历史已保存到: {history_path}")
        except Exception as e:
            print(f"⚠️ 保存训练历史失败: {e}")
    
    def load_training_history(self, history_path='models/training_history.json'):
        """加载训练历史"""
        try:
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            print(f"📈 训练历史已加载，包含 {len(self.training_history.get('loss', []))} 个epoch")
        except FileNotFoundError:
            print("📝 未找到训练历史文件，将创建新的历史记录")
            self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        except Exception as e:
            print(f"⚠️ 加载训练历史失败: {e}")
            self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    def estimate_training_time(self, total_samples, batch_size, epochs, validation_split=0.2):
        """预估训练时间"""
        # 估算每个batch的训练时间（基于经验值）
        samples_per_epoch = int(total_samples * (1 - validation_split))
        batches_per_epoch = (samples_per_epoch + batch_size - 1) // batch_size
        
        # 基于模型复杂度估算时间（秒/batch）
        base_time_per_batch = 0.15 if device.type == 'cuda' else 0.5  # GPU vs CPU
        
        # 根据模型参数调整
        complexity_factor = (self.lstm_units / 512) * (self.seq_length / 100)
        time_per_batch = base_time_per_batch * complexity_factor
        
        # 验证时间（通常比训练快）
        validation_samples = int(total_samples * validation_split)
        validation_batches = (validation_samples + batch_size - 1) // batch_size
        validation_time_per_epoch = validation_batches * time_per_batch * 0.3
        
        # 总时间估算
        training_time_per_epoch = batches_per_epoch * time_per_batch
        total_time_per_epoch = training_time_per_epoch + validation_time_per_epoch
        total_estimated_time = total_time_per_epoch * epochs
        
        return {
            'total_estimated_seconds': total_estimated_time,
            'time_per_epoch': total_time_per_epoch,
            'batches_per_epoch': batches_per_epoch,
            'time_per_batch': time_per_batch
        }
    def format_time(self, seconds):
        """格式化时间显示"""
        if seconds is None:
            return "未知时间"
        
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes:.0f}分{remaining_seconds:.0f}秒"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{hours:.0f}小时{remaining_minutes:.0f}分钟"
    
    def validate_model(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
          # 防御性编程：处理空数据加载器或除零情况
        if len(val_loader) > 0:
            avg_loss = total_loss / len(val_loader)
        else:
            avg_loss = 0.0
            
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0.0
            
        return avg_loss, accuracy    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2, 
              model_save_path='models/hoyomusic_generator.pth', is_incremental=False,
              enable_checkpoints=True, checkpoint_interval=5, auto_resume=False, 
              checkpoint_dir='models/checkpoints'):
        """训练模型（支持增量训练和断点续连）"""
        
        # 尝试自动恢复训练（如果启用）
        if auto_resume and not is_incremental:
            if self.auto_resume_training():
                is_incremental = True  # 如果恢复成功，标记为增量训练
        
        print(f"🚀 开始{'增量' if is_incremental else ''}训练...")
        
        # 如果不是增量训练且模型不存在，则构建新模型
        if not is_incremental and self.model is None:
            self.build_model()
        
        # 设置checkpoint参数
        self.checkpoint_interval = checkpoint_interval
        self.auto_save_enabled = enable_checkpoints
        
        # 转换数据到PyTorch张量
        X_tensor = torch.LongTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # 分割数据
        n_samples = len(X)
        n_train = int(n_samples * (1 - validation_split))
        
        # 创建数据集
        train_dataset = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
        val_dataset = TensorDataset(X_tensor[n_train:], y_tensor[n_train:])
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"🎵 HoyoMusic生成器模型摘要:")
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"  - 参数数量: {param_count:,}")
        print(f"  - 训练样本: {len(train_dataset):,}")
        print(f"  - 验证样本: {len(val_dataset):,}")        # 计算实际需要训练的epoch数（如果是恢复训练）
        # 确保current_epoch非None并且是整数类型
        if self.resume_from_checkpoint and isinstance(self.current_epoch, (int, float)):
            start_epoch = self.current_epoch
        else:
            start_epoch = 0
            
        remaining_epochs = epochs - start_epoch
        
        if self.resume_from_checkpoint and start_epoch > 0:
            print(f"🔄 从epoch {start_epoch} 恢复训练，还需训练 {remaining_epochs} 个epoch")
        
        # 预估训练时间
        time_estimates = self.estimate_training_time(len(X), batch_size, remaining_epochs, validation_split)
        
        print(f"\n⏱️  训练时间预估:")
        print(f"  - 每个epoch预估: {self.format_time(time_estimates['time_per_epoch'])}")
        print(f"  - 剩余训练时间预估: {self.format_time(time_estimates['total_estimated_seconds'])}")
        print(f"  - 每批次时间: {time_estimates['time_per_batch']:.2f}秒")
        print(f"  - 每个epoch批次数: {time_estimates['batches_per_epoch']}")
        
        completion_time = datetime.now() + timedelta(seconds=time_estimates['total_estimated_seconds'])
        print(f"  - 预计完成时间: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Checkpoint设置信息
        if enable_checkpoints:
            print(f"\n💾 Checkpoint设置:")
            print(f"  - 自动保存间隔: 每 {checkpoint_interval} 个epoch")
            print(f"  - Checkpoint目录: models/checkpoints/")
          # 记录训练开始时间（如果不是恢复训练）
        if not self.resume_from_checkpoint:
            self.training_start_time = time.time()
          # 初始化历史记录（如果不是增量训练）
        if not is_incremental or not self.training_history:
            self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        else:
            # 确保训练历史包含所有必需的键
            for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                if key not in self.training_history:
                    self.training_history[key] = []
        
        # 使用checkpoint中的最佳损失和耐心计数器（如果是恢复训练）
        if not self.resume_from_checkpoint:
            self.best_val_loss = float('inf')
            self.patience_counter = 0
        
        max_patience = 15
        
        # 训练循环 - 从正确的epoch开始
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch  # 更新当前epoch
            
            # 训练阶段
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # 计算训练指标
            avg_train_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            train_accuracy = correct / total if total > 0 else 0.0
            
            # 验证阶段
            val_loss, val_accuracy = self.validate_model(val_loader)
            # 防御性修正：val_loss为None时赋值为float('inf')
            if val_loss is None:
                val_loss = float('inf')
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.training_history['loss'].append(avg_train_loss)
            self.training_history['accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
              # 时间估算
            if self.training_start_time:
                elapsed_time = time.time() - self.training_start_time
                
                # 更多防御性检查，确保所有值都是有效数字
                if epoch is None:
                    epoch = 0
                if start_epoch is None:
                    start_epoch = 0
                if epochs is None:
                    epochs = 1
                    
                # 计算已完成和剩余的epochs
                try:
                    epochs_completed = max(0, epoch - start_epoch + 1)
                    remaining_epochs = max(0, epochs - epoch - 1)
                except TypeError:  # 捕获任何类型错误
                    epochs_completed = 0
                    remaining_epochs = 0                
                try:
                    if epochs_completed > 0:
                        avg_time_per_epoch = elapsed_time / epochs_completed
                        self.estimated_time_remaining = avg_time_per_epoch * remaining_epochs
                        completion_time = datetime.now() + timedelta(seconds=self.estimated_time_remaining)
                except (TypeError, ValueError, ZeroDivisionError):
                    # 如果出现任何计算错误，将估计时间设为None
                    self.estimated_time_remaining = None
                    completion_time = None
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练损失: {avg_train_loss:.4f} | 训练准确率: {train_accuracy:.4f}")
            print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_accuracy:.4f}")
            if hasattr(self, 'estimated_time_remaining'):
                print(f"  剩余时间: {self.format_time(self.estimated_time_remaining)}")
            
            # 早停和模型保存
            is_best_model = False
            # 防御性修正：self.best_val_loss为None或非法类型时赋值为float('inf')
            if self.best_val_loss is None or not isinstance(self.best_val_loss, (int, float)) or (isinstance(self.best_val_loss, float) and (self.best_val_loss != self.best_val_loss)):
                self.best_val_loss = float('inf')
            if val_loss is None or not isinstance(val_loss, (int, float)) or (isinstance(val_loss, float) and (val_loss != val_loss)):
                val_loss = float('inf')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best_model = True
                self.save_model(model_save_path)
                print(f"  ✅ 保存最佳模型 (验证损失: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                
            # Checkpoint保存
            if enable_checkpoints:
                # 定期保存checkpoint
                if (epoch + 1) % checkpoint_interval == 0:                    checkpoint_info = {
                        'epoch_performance': {
                            'train_loss': avg_train_loss,
                            'train_acc': train_accuracy,
                            'val_loss': val_loss,
                            'val_acc': val_accuracy
                        }
                    }
                self.save_checkpoint(checkpoint_dir=checkpoint_dir, epoch=epoch, extra_info=checkpoint_info)
                
                # 保存最佳模型checkpoint
                if is_best_model:                    checkpoint_info = {
                        'best_model': True,
                        'epoch_performance': {
                            'train_loss': avg_train_loss,
                            'train_acc': train_accuracy,
                            'val_loss': val_loss,
                            'val_acc': val_accuracy
                        }
                    }
                self.save_checkpoint(checkpoint_dir=checkpoint_dir, epoch=epoch, is_best=True, extra_info=checkpoint_info)
            # 早停检查
            if self.patience_counter >= max_patience:
                print(f"  ⏹️ 早停：验证损失在 {max_patience} 个epoch内未改善")
                break
        
        # 保存最终checkpoint
        if enable_checkpoints:
            final_checkpoint_info = {
                'training_completed': True,
                'final_epoch': epoch,
                'final_performance': {
                    'train_loss': avg_train_loss,
                    'train_acc': train_accuracy,
                    'val_loss': val_loss,
                    'val_acc': val_accuracy
                }
            }
            self.save_checkpoint(checkpoint_dir=checkpoint_dir, epoch=epoch, extra_info=final_checkpoint_info)
              # 清理旧的checkpoint
            self.cleanup_old_checkpoints(checkpoint_dir=checkpoint_dir)
        
        # 保存训练历史
        self.save_training_history()
          # 计算实际训练时间
        actual_training_time = time.time() - self.training_start_time
        print(f"\n✅ 训练完成！")
        print(f"🕐 实际训练时间: {self.format_time(actual_training_time)}")
        
        # 防御性检查：确保历史记录不为空且包含所有必需的键
        for key in ['val_loss', 'val_accuracy', 'loss', 'accuracy']:
            if key not in self.training_history or not self.training_history[key]:
                self.training_history[key] = [0.0]  # 提供默认值
        
        print(f"📈 最终验证损失: {self.training_history['val_loss'][-1]:.4f}")
        print(f"🎯 最终验证准确率: {self.training_history['val_accuracy'][-1]:.4f}")
        
        # 创建兼容的历史对象
        class HistoryCompat:
            def __init__(self, history_dict):
                self.history = history_dict if history_dict else {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        return HistoryCompat(self.training_history)
    
    def save_checkpoint(self, checkpoint_dir='models/checkpoints', epoch=None, is_best=False, extra_info=None):
        """保存训练checkpoint"""
        import os
        import time
        from datetime import datetime
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if epoch is None:
            epoch = self.current_epoch
            
        # 构建checkpoint文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if is_best:
            checkpoint_name = f'best_model_epoch_{epoch}_{timestamp}.pth'
        else:
            checkpoint_name = f'checkpoint_epoch_{epoch}_{timestamp}.pth'
            
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # 准备checkpoint数据
        checkpoint_data = {
            # 模型和优化器状态
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            
            # 模型配置
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            
            # 训练状态
            'current_epoch': epoch,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            
            # 时间信息
            'training_start_time': self.training_start_time,
            'checkpoint_time': time.time(),
            'timestamp': timestamp,
            
            # 额外信息
            'extra_info': extra_info or {},
            
            # 版本信息
            'pytorch_version': torch.__version__,
            'checkpoint_version': '2.0'
        }
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.last_checkpoint_path = checkpoint_path
            
            if is_best:
                print(f"💎 保存最佳模型checkpoint: {checkpoint_name}")
            else:
                print(f"💾 保存训练checkpoint: {checkpoint_name}")
                
            # 创建符号链接到最新checkpoint（便于恢复）
            latest_link = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_link):
                os.remove(latest_link)
            
            # 在Windows上创建副本而不是符号链接
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copy2(checkpoint_path, latest_link)
            else:
                os.symlink(os.path.basename(checkpoint_path), latest_link)
                
            return checkpoint_path
            
        except Exception as e:
            print(f"❌ 保存checkpoint失败: {e}")
            return None
    def load_checkpoint(self, checkpoint_path, resume_training=True):
        """加载checkpoint并恢复训练状态"""
        try:
            print(f"📂 加载checkpoint: {checkpoint_path}")
            
            # PyTorch 2.6+ 安全性更新，需要设置weights_only=False
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # 验证checkpoint版本
            checkpoint_version = checkpoint.get('checkpoint_version', '1.0')
            print(f"📋 Checkpoint版本: {checkpoint_version}")
            
            # 恢复模型配置
            self.vocab_size = checkpoint.get('vocab_size', self.vocab_size)
            self.seq_length = checkpoint.get('seq_length', self.seq_length)
            self.embedding_dim = checkpoint.get('embedding_dim', self.embedding_dim)
            self.lstm_units = checkpoint.get('lstm_units', self.lstm_units)
            
            # 构建模型（如果还没有）
            if self.model is None:
                self.build_model()
            
            # 加载模型状态
            if 'model_state_dict' in checkpoint and checkpoint['model_state_dict']:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ 模型权重已恢复")
            
            if resume_training:                # 恢复训练状态
                self.current_epoch = checkpoint.get('current_epoch', 0)
                
                # 获取并验证训练历史
                checkpoint_history = checkpoint.get('training_history', {})
                default_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
                
                # 如果checkpoint没有历史记录，使用默认空历史
                if not checkpoint_history:
                    self.training_history = default_history
                else:
                    # 确保所有必需的键都存在
                    self.training_history = checkpoint_history
                    for key in default_history:
                        if key not in self.training_history:
                            self.training_history[key] = []
                        # 确保值是有效的列表
                        if not isinstance(self.training_history[key], list):
                            self.training_history[key] = []
                
                # 修复 best_val_loss 可能为 None 或非法类型的问题
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                if best_val_loss is None or not isinstance(best_val_loss, (int, float)) or (isinstance(best_val_loss, float) and (best_val_loss != best_val_loss)):
                    self.best_val_loss = float('inf')
                else:
                    self.best_val_loss = float(best_val_loss)
                self.patience_counter = checkpoint.get('patience_counter', 0)
                self.training_start_time = checkpoint.get('training_start_time', None)
                
                # 恢复优化器状态
                if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ 优化器状态已恢复")
                
                # 恢复学习率调度器状态
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✅ 学习率调度器状态已恢复")
                
                self.resume_from_checkpoint = True
                
                print(f"🔄 训练状态已恢复:")
                print(f"  - 当前epoch: {self.current_epoch}")
                print(f"  - 最佳验证损失: {self.best_val_loss:.4f}")
                print(f"  - 训练历史长度: {len(self.training_history.get('loss', []))}")
                
            else:
                print("📖 仅加载模型权重（不恢复训练状态）")
            
            # 显示额外信息
            extra_info = checkpoint.get('extra_info', {})
            if extra_info:
                print(f"📝 额外信息: {extra_info}")
                
            return True
            
        except Exception as e:
            print(f"❌ 加载checkpoint失败: {e}")
            return False
    def find_latest_checkpoint(self, checkpoint_dir='models/checkpoints'):
        """查找最新的checkpoint文件"""
        import os
        import glob
        
        if not os.path.exists(checkpoint_dir):
            return None
            
        # 首先尝试latest_checkpoint.pth
        latest_link = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        if os.path.exists(latest_link):
            return latest_link
            
        # 如果没有，查找最新的checkpoint文件
        checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None
            
        # 过滤掉无效文件并按修改时间排序，返回最新的
        valid_files = []
        for file_path in checkpoint_files:
            try:
                if os.path.exists(file_path):
                    mtime = os.path.getmtime(file_path)
                    if mtime is not None:
                        valid_files.append(file_path)
            except (OSError, TypeError):
                continue
                
        if not valid_files:
            return None
            
        latest_checkpoint = max(valid_files, key=os.path.getmtime)
        return latest_checkpoint
    def cleanup_old_checkpoints(self, checkpoint_dir='models/checkpoints', keep_count=5):
        """清理旧的checkpoint文件，只保留最新的几个"""
        import os
        import glob
        
        checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        # 过滤掉无效文件
        valid_files = []
        for file_path in checkpoint_files:
            try:
                if os.path.exists(file_path):
                    mtime = os.path.getmtime(file_path)
                    if mtime is not None:
                        valid_files.append(file_path)
            except (OSError, TypeError):
                continue
        
        if len(valid_files) <= keep_count:
            return
            
        # 按修改时间排序
        valid_files.sort(key=os.path.getmtime, reverse=True)
        
        # 删除旧的checkpoint
        files_to_delete = valid_files[keep_count:]
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"🗑️ 清理旧checkpoint: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"⚠️ 清理checkpoint失败: {e}")
    
    def auto_resume_training(self, checkpoint_dir='models/checkpoints'):
        """自动从最新checkpoint恢复训练"""
        latest_checkpoint = self.find_latest_checkpoint(checkpoint_dir)
        
        if latest_checkpoint:
            print(f"🔍 发现最新checkpoint: {os.path.basename(latest_checkpoint)}")
            response = input("是否从此checkpoint恢复训练？(y/n): ").lower().strip()
            
            if response in ['y', 'yes', '是']:
                return self.load_checkpoint(latest_checkpoint, resume_training=True)
        
        print("📄 未找到可恢复的checkpoint，将开始新的训练")
        return False
    
    def generate_music(self, seed_text, char_to_int, int_to_char, length=800, temperature=0.8):
        """生成HoyoMusic风格的音乐"""
        if self.model is None:
            raise ValueError("模型未加载或未训练")
        
        self.model.eval()
        
        # 准备种子序列
        seed_sequence = [char_to_int.get(char, 0) for char in seed_text[-self.seq_length:]]
        
        # 如果种子序列太短，用空格填充
        while len(seed_sequence) < self.seq_length:
            seed_sequence.insert(0, char_to_int.get(' ', 0))
        
        generated_text = seed_text
        
        # 生成音乐
        with torch.no_grad():
            for i in range(length):
                # 预测下一个字符
                x = torch.LongTensor([seed_sequence]).to(device)
                outputs = self.model(x)
                predictions = F.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # 应用温度参数进行采样
                predictions = np.log(predictions + 1e-8) / temperature
                exp_preds = np.exp(predictions)
                predictions = exp_preds / np.sum(exp_preds)
                
                # 采样下一个字符
                next_char_idx = np.random.choice(len(predictions), p=predictions)
                next_char = int_to_char[next_char_idx]
                
                generated_text += next_char
                
                # 更新种子序列
                seed_sequence = seed_sequence[1:] + [next_char_idx]
                
                # 如果生成了完整的曲子标记，可以选择停止
                if i > 100 and generated_text.count('X:') > 3:  # 生成了多首曲子
                    break
        
        return generated_text
    
    def generate_hoyomusic_style(self, region="Mondstadt", length=600, temperature=0.8, char_to_int=None, int_to_char=None):
        """生成特定地区风格的原神音乐"""
        
        # 不同地区的种子模板
        region_seeds = {
            "Mondstadt": "X:1\nT:Mondstadt Breeze\nC:Generated by AI\nM:4/4\nL:1/8\nK:C major\n",
            "Liyue": "X:1\nT:Liyue Harbor\nC:Generated by AI\nM:4/4\nL:1/8\nK:A minor\n",
            "Inazuma": "X:1\nT:Inazuma Thunder\nC:Generated by AI\nM:4/4\nL:1/8\nK:D major\n",
            "Sumeru": "X:1\nT:Sumeru Forest\nC:Generated by AI\nM:6/8\nL:1/8\nK:G major\n",
            "Fontaine": "X:1\nT:Fontaine Waters\nC:Generated by AI\nM:3/4\nL:1/4\nK:F major\n"
        }
        
        seed = region_seeds.get(region, region_seeds["Mondstadt"])
        
        return self.generate_music(seed, char_to_int, int_to_char, length, temperature)
    
    def get_model_size(self):
        """获取模型参数数量"""
        if self.model is None:
            return 0
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return trainable_params
    
    def train_step(self, X, y):
        """执行单步训练并返回损失值"""
        if self.model is None:
            raise ValueError("模型未构建，请先调用build_model()")
        
        self.model.train()
        
        # 转换数据到tensor
        X_tensor = torch.LongTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        # 前向传播
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def generate_sequence(self, seed_sequence, length=100, temperature=1.0):
        """生成音符序列（用于测试）"""
        if self.model is None:
            raise ValueError("模型未加载或未训练")
        
        self.model.eval()
        
        # 确保种子序列长度正确
        if len(seed_sequence) < self.seq_length:
            # 用0填充到所需长度
            seed_sequence = [0] * (self.seq_length - len(seed_sequence)) + list(seed_sequence)
        elif len(seed_sequence) > self.seq_length:
            # 截取最后seq_length个元素
            seed_sequence = seed_sequence[-self.seq_length:]
        
        generated_sequence = list(seed_sequence)
        
        # 生成新的音符
        with torch.no_grad():
            for _ in range(length):
                # 预测下一个字符
                current_seq = generated_sequence[-self.seq_length:]
                x = torch.LongTensor([current_seq]).to(device)
                outputs = self.model(x)
                predictions = F.softmax(outputs / temperature, dim=1).cpu().numpy()[0]
                
                # 采样下一个字符
                next_idx = np.random.choice(len(predictions), p=predictions)
                generated_sequence.append(next_idx)
        
        return generated_sequence

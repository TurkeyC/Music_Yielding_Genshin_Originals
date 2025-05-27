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
        self.training_history = {}
        self.training_start_time = None
        self.estimated_time_remaining = None
        
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
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=device)
            
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
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2, 
              model_save_path='models/hoyomusic_generator.pth', is_incremental=False):
        """训练模型（支持增量训练）"""
        
        print(f"🚀 开始{'增量' if is_incremental else ''}训练...")
        
        # 如果不是增量训练且模型不存在，则构建新模型
        if not is_incremental and self.model is None:
            self.build_model()
        
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
        print(f"  - 验证样本: {len(val_dataset):,}")
        
        # 预估训练时间
        time_estimates = self.estimate_training_time(len(X), batch_size, epochs, validation_split)
        
        print(f"\n⏱️  训练时间预估:")
        print(f"  - 每个epoch预估: {self.format_time(time_estimates['time_per_epoch'])}")
        print(f"  - 总训练时间预估: {self.format_time(time_estimates['total_estimated_seconds'])}")
        print(f"  - 每批次时间: {time_estimates['time_per_batch']:.2f}秒")
        print(f"  - 每个epoch批次数: {time_estimates['batches_per_epoch']}")
        
        completion_time = datetime.now() + timedelta(seconds=time_estimates['total_estimated_seconds'])
        print(f"  - 预计完成时间: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 记录训练开始时间
        self.training_start_time = time.time()
        
        # 初始化历史记录
        if not is_incremental or not self.training_history:
            self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        # 训练循环
        for epoch in range(epochs):
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
            avg_train_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total
            
            # 验证阶段
            val_loss, val_accuracy = self.validate_model(val_loader)
            
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
                epochs_completed = epoch + 1
                
                if epochs_completed > 0:
                    avg_time_per_epoch = elapsed_time / epochs_completed
                    remaining_epochs = epochs - epochs_completed
                    self.estimated_time_remaining = avg_time_per_epoch * remaining_epochs
                    
                    completion_time = datetime.now() + timedelta(seconds=self.estimated_time_remaining)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练损失: {avg_train_loss:.4f} | 训练准确率: {train_accuracy:.4f}")
            print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_accuracy:.4f}")
            if hasattr(self, 'estimated_time_remaining'):
                print(f"  剩余时间: {self.format_time(self.estimated_time_remaining)}")
            
            # 早停和模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(model_save_path)
                print(f"  ✅ 保存最佳模型 (验证损失: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"  ⏹️ 早停：验证损失在 {max_patience} 个epoch内未改善")
                    break
        
        # 保存训练历史
        self.save_training_history()
        
        # 计算实际训练时间
        actual_training_time = time.time() - self.training_start_time
        print(f"\n✅ 训练完成！")
        print(f"🕐 实际训练时间: {self.format_time(actual_training_time)}")
        print(f"📈 最终验证损失: {self.training_history['val_loss'][-1]:.4f}")
        print(f"🎯 最终验证准确率: {self.training_history['val_accuracy'][-1]:.4f}")
        
        # 创建兼容的历史对象
        class HistoryCompat:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return HistoryCompat(self.training_history)
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # 根据检查点内容构建模型
            if 'vocab_size' in checkpoint:
                self.vocab_size = checkpoint['vocab_size']
                self.seq_length = checkpoint['seq_length']
                self.embedding_dim = checkpoint.get('embedding_dim', 256)
                self.lstm_units = checkpoint.get('lstm_units', 512)
            
            if self.model is None:
                self.build_model()
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"✅ HoyoMusic模型已从 {model_path} 加载")
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise
    
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

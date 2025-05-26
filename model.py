import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
import numpy as np
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

class HoyoMusicGenerator:
    def __init__(self, vocab_size, seq_length, embedding_dim=256, lstm_units=512):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.training_history = {}
        self.training_start_time = None
        self.estimated_time_remaining = None
        
    def build_model(self):
        """构建针对HoyoMusic优化的LSTM模型"""
        model = Sequential([
            # 嵌入层 - 为ABC记谱优化
            Embedding(
                self.vocab_size, 
                self.embedding_dim, 
                input_length=self.seq_length,
                mask_zero=True,
                name='embedding'
            ),
            
            # 第一层LSTM - 捕获短期模式
            LSTM(
                self.lstm_units, 
                return_sequences=True, 
                dropout=0.3, 
                recurrent_dropout=0.3,
                name='lstm_1'
            ),
            BatchNormalization(name='bn_1'),
            
            # 第二层LSTM - 捕获中期模式
            LSTM(
                self.lstm_units, 
                return_sequences=True, 
                dropout=0.3, 
                recurrent_dropout=0.3,
                name='lstm_2'
            ),
            BatchNormalization(name='bn_2'),
            
            # 第三层LSTM - 捕获长期依赖
            LSTM(
                self.lstm_units, 
                dropout=0.3, 
                recurrent_dropout=0.3,
                name='lstm_3'
            ),
            BatchNormalization(name='bn_3'),
            
            # 全连接层
            Dense(self.lstm_units, activation='relu', name='dense_1'),
            Dropout(0.5),
            
            Dense(self.lstm_units // 2, activation='relu', name='dense_2'),
            Dropout(0.3),
            
            # 输出层
            Dense(self.vocab_size, activation='softmax', name='output')
        ])
        
        # 使用自适应学习率优化器
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # 编译模型
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def load_model_for_incremental_training(self, model_path, learning_rate=0.0005):
        """加载现有模型进行增量训练"""
        print(f"🔄 加载现有模型进行增量训练: {model_path}")
        
        try:
            # 加载现有模型
            self.model = load_model(model_path)
            print(f"✅ 模型加载成功")
            
            # 降低学习率以进行增量训练
            new_optimizer = Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )
            
            # 重新编译模型
            self.model.compile(
                optimizer=new_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
            )
            
            print(f"🎯 增量训练学习率设置为: {learning_rate}")
            
            # 加载历史训练信息（如果存在）
            self.load_training_history()
            
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            print("🔧 将创建新模型...")
            self.build_model()
            return False
    
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
        base_time_per_batch = 0.1  # RTX4060的基准时间
        
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
    
    def create_time_estimation_callback(self, time_estimates):
        """创建时间估算回调函数"""
        def on_epoch_end(epoch, logs):
            if self.training_start_time:
                elapsed_time = time.time() - self.training_start_time
                epochs_completed = epoch + 1
                
                if epochs_completed > 0:
                    avg_time_per_epoch = elapsed_time / epochs_completed
                    remaining_epochs = time_estimates['total_epochs'] - epochs_completed
                    self.estimated_time_remaining = avg_time_per_epoch * remaining_epochs
                    
                    # 更新预估完成时间
                    completion_time = datetime.now() + timedelta(seconds=self.estimated_time_remaining)
                    
                    print(f"⏱️  Epoch {epochs_completed}: 剩余时间 {self.format_time(self.estimated_time_remaining)}")
                    print(f"📅 预计完成时间: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return LambdaCallback(on_epoch_end=on_epoch_end)
    
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
    
    def get_callbacks(self, model_save_path, time_estimates=None):
        """获取训练回调函数"""
        callbacks = [
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.0001,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # 添加时间估算回调
        if time_estimates:
            callbacks.append(self.create_time_estimation_callback(time_estimates))
        
        return callbacks
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2, 
              model_save_path='models/hoyomusic_generator.h5', is_incremental=False):
        """训练模型（支持增量训练）"""
        
        print(f"🚀 开始{'增量' if is_incremental else ''}训练...")
        
        # 如果不是增量训练且模型不存在，则构建新模型
        if not is_incremental and self.model is None:
            self.build_model()
        
        print(f"🎵 HoyoMusic生成器模型摘要:")
        self.model.summary()
        
        # 预估训练时间
        time_estimates = self.estimate_training_time(len(X), batch_size, epochs, validation_split)
        time_estimates['total_epochs'] = epochs
        
        print(f"\n⏱️  训练时间预估:")
        print(f"  - 每个epoch预估: {self.format_time(time_estimates['time_per_epoch'])}")
        print(f"  - 总训练时间预估: {self.format_time(time_estimates['total_estimated_seconds'])}")
        print(f"  - 每批次时间: {time_estimates['time_per_batch']:.2f}秒")
        print(f"  - 每个epoch批次数: {time_estimates['batches_per_epoch']}")
        
        completion_time = datetime.now() + timedelta(seconds=time_estimates['total_estimated_seconds'])
        print(f"  - 预计完成时间: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 记录训练开始时间
        self.training_start_time = time.time()
        
        # 获取回调函数
        callbacks = self.get_callbacks(model_save_path, time_estimates)
        
        # 训练模型
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        # 更新训练历史
        if is_incremental and self.training_history:
            # 合并历史记录
            for key in history.history:
                if key in self.training_history:
                    self.training_history[key].extend(history.history[key])
                else:
                    self.training_history[key] = history.history[key]
        else:
            self.training_history = history.history
        
        # 保存训练历史
        self.save_training_history()
        
        # 计算实际训练时间
        actual_training_time = time.time() - self.training_start_time
        print(f"\n✅ 训练完成！")
        print(f"🕐 实际训练时间: {self.format_time(actual_training_time)}")
        print(f"📈 最终验证损失: {history.history['val_loss'][-1]:.4f}")
        print(f"🎯 最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
        
        return history
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        self.model = load_model(model_path)
        print(f"HoyoMusic模型已从 {model_path} 加载")
    
    def generate_music(self, seed_text, char_to_int, int_to_char, length=800, temperature=0.8):
        """生成HoyoMusic风格的音乐"""
        if self.model is None:
            raise ValueError("模型未加载或未训练")
        
        # 准备种子序列
        seed_sequence = [char_to_int.get(char, 0) for char in seed_text[-self.seq_length:]]
        
        # 如果种子序列太短，用空格填充
        while len(seed_sequence) < self.seq_length:
            seed_sequence.insert(0, char_to_int.get(' ', 0))
        
        generated_text = seed_text
        
        # 生成音乐
        for i in range(length):
            # 预测下一个字符
            x = np.array([seed_sequence])
            predictions = self.model.predict(x, verbose=0)[0]
            
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
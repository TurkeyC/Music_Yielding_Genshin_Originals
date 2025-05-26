import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np

class HoyoMusicGenerator:
    def __init__(self, vocab_size, seq_length, embedding_dim=256, lstm_units=512):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        
    def build_model(self):
        """构建针对HoyoMusic优化的LSTM模型"""
        model = Sequential([
            # 嵌入层 - 为ABC记谱优化
            Embedding(
                self.vocab_size, 
                self.embedding_dim, 
                input_length=self.seq_length,
                mask_zero=True  # 支持变长序列
            ),
            
            # 第一层LSTM - 捕获短期模式
            LSTM(
                self.lstm_units, 
                return_sequences=True, 
                dropout=0.3, 
                recurrent_dropout=0.3,
                name='lstm_1'
            ),
            BatchNormalization(),
            
            # 第二层LSTM - 捕获中期模式
            LSTM(
                self.lstm_units, 
                return_sequences=True, 
                dropout=0.3, 
                recurrent_dropout=0.3,
                name='lstm_2'
            ),
            BatchNormalization(),
            
            # 第三层LSTM - 捕获长期依赖
            LSTM(
                self.lstm_units, 
                dropout=0.3, 
                recurrent_dropout=0.3,
                name='lstm_3'
            ),
            BatchNormalization(),
            
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
    
    def get_callbacks(self, model_save_path):
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
        
        return callbacks
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2, model_save_path='models/hoyomusic_generator.h5'):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        print(f"HoyoMusic生成器模型摘要:")
        self.model.summary()
        
        # 获取回调函数
        callbacks = self.get_callbacks(model_save_path)
        
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
        
        return history
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        self.model = tf.keras.models.load_model(model_path)
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
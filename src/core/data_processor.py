import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import pickle
from datasets import load_dataset
from tqdm import tqdm

class HoyoMusicDataProcessor:
    def __init__(self, seq_length=100):
        self.seq_length = seq_length
        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0
        self.raw_text = ""
        self.dataset_info = {}
        
    def download_hoyomusic_dataset(self, cache_dir="./hoyomusic_cache"):
        """下载HoyoMusic数据集"""
        print("正在下载HoyoMusic数据集...")
        print("这是一个包含305,264个原神音乐ABC记谱的大型数据集")
        
        try:
            # 加载数据集
            dataset = load_dataset("Genius-Society/hoyoMusic", cache_dir=cache_dir)
            
            print(f"数据集加载成功！")
            print(f"训练集大小: {len(dataset['train'])}")
            
            # 显示数据集信息
            if len(dataset['train']) > 0:
                sample = dataset['train'][0]
                print("\n数据集字段:")
                for key in sample.keys():
                    print(f"  - {key}: {type(sample[key])}")
                
                print(f"\n示例ABC记谱片段:")
                abc_field = self.find_abc_field(sample)
                if abc_field:
                    print(sample[abc_field][:200] + "...")
            
            return dataset
            
        except Exception as e:
            print(f"下载HoyoMusic数据集失败: {e}")
            print("将使用示例数据...")
            return None
    
    def find_abc_field(self, sample):
        """查找包含ABC记谱的字段"""
        # 常见的ABC记谱字段名
        possible_fields = ['abc', 'notation', 'score', 'music', 'text', 'abc_notation']
        
        for field in possible_fields:
            if field in sample and isinstance(sample[field], str):
                # 检查是否包含ABC记谱特征
                content = sample[field]
                if any(marker in content for marker in ['X:', 'T:', 'M:', 'K:', 'L:']):
                    return field
        
        # 如果没有找到明确的字段，尝试所有字符串字段
        for key, value in sample.items():
            if isinstance(value, str) and any(marker in value for marker in ['X:', 'T:', 'M:', 'K:']):
                return key
        
        return None
    
    def load_abc_files(self, data_dir):
        """加载本地ABC文件（备用方案）"""
        abc_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.abc') or file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            abc_files.append(content)
                    except:
                        try:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                content = f.read()
                                abc_files.append(content)
                        except Exception as e:
                            print(f"无法读取文件 {file_path}: {e}")
        
        return abc_files
    
    def create_sample_data(self, data_dir):
        """创建示例数据（原神风格）"""
        os.makedirs(data_dir, exist_ok=True)
        
        # 原神风格的示例ABC记谱
        sample_abc = """X:1
T:Mondstadt Theme
C:miHoYo
M:4/4
L:1/8
K:C major
|: G2 A2 B2 c2 | d2 e2 f2 g2 | e2 d2 c2 B2 | A2 G2 F2 E2 |
   D2 E2 F2 G2 | A2 B2 c2 d2 | B2 A2 G2 F2 | G4 G4 :|

X:2
T:Liyue Harbor
C:miHoYo
M:3/4
L:1/4
K:A minor
|: A B c | d e f | e d c | B A G |
   F G A | B c d | c B A | A3 :|

X:3
T:Inazuma Melody
C:miHoYo
M:4/4
L:1/8
K:D major
|: D2 F2 A2 d2 | c2 B2 A2 G2 | F2 E2 D2 C2 | B,2 A,2 G,2 F,2 |
   G,2 A,2 B,2 C2 | D2 E2 F2 G2 | A2 B2 c2 d2 | d4 d4 :|

X:4
T:Sumeru Forest
C:miHoYo
M:6/8
L:1/8
K:G major
|: G2 A B2 c | d2 e f2 g | e2 d c2 B | A2 G F2 E |
   D2 E F2 G | A2 B c2 d | B2 A G2 F | G3 G3 :|

X:5
T:Fontaine Waltz
C:miHoYo
M:3/4
L:1/4
K:F major
|: F G A | B c d | c B A | G F E |
   D E F | G A B | A G F | F3 :|
"""
        
        sample_file = os.path.join(data_dir, 'hoyoverse_sample.abc')
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_abc)
        
        print(f"原神风格示例数据已保存到: {sample_file}")
        return [sample_abc]
    
    def clean_abc_text(self, text):
        """清理ABC文本，保留HoyoMusic特定格式"""
        # 移除注释但保留作曲家信息
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 保留重要的ABC标头信息
            if line.strip().startswith(('X:', 'T:', 'C:', 'M:', 'L:', 'K:', 'Q:')):
                cleaned_lines.append(line.strip())
            elif line.strip() and not line.strip().startswith('%'):
                # 保留音乐内容行
                # 清理特殊字符但保留ABC记谱符号
                allowed_chars = set('ABCDEFGabcdefg|:[](){}1234567890.,/\\-+#^=_~*><zZxX \t')
                cleaned_line = ''.join(char for char in line if char in allowed_chars)
                if cleaned_line.strip():
                    cleaned_lines.append(cleaned_line.strip())
        
        return '\n'.join(cleaned_lines)
    
    def prepare_data(self, use_hoyomusic=True, data_dir='data/abc_files', max_samples=None):
        """准备训练数据"""
        abc_texts = []
        
        if use_hoyomusic:
            # 尝试加载HoyoMusic数据集
            dataset = self.download_hoyomusic_dataset()
            
            if dataset:
                print("正在处理HoyoMusic数据集...")
                
                # 获取训练数据
                train_data = dataset['train']
                
                # 限制样本数量（可选）
                if max_samples and len(train_data) > max_samples:
                    print(f"限制样本数量为: {max_samples}")
                    train_data = train_data.select(range(max_samples))
                
                # 查找ABC记谱字段
                if len(train_data) > 0:
                    abc_field = self.find_abc_field(train_data[0])
                    
                    if abc_field:
                        print(f"找到ABC记谱字段: {abc_field}")
                        
                        # 提取所有ABC记谱
                        for i, sample in enumerate(tqdm(train_data, desc="处理ABC记谱")):
                            abc_text = sample[abc_field]
                            if abc_text and len(abc_text.strip()) > 50:  # 过滤太短的记谱
                                abc_texts.append(abc_text)
                                
                                # 保存数据集信息
                                if i == 0:
                                    self.dataset_info = {k: v for k, v in sample.items() if k != abc_field}
                    else:
                        print("未找到ABC记谱字段，使用示例数据")
                        abc_texts = self.create_sample_data(data_dir)
                else:
                    print("数据集为空，使用示例数据")
                    abc_texts = self.create_sample_data(data_dir)
            else:
                print("使用示例数据...")
                abc_texts = self.create_sample_data(data_dir)
        else:
            # 加载本地文件
            print("加载本地ABC文件...")
            abc_texts = self.load_abc_files(data_dir)
            
            if not abc_texts:
                print("未找到本地ABC文件，使用示例数据")
                abc_texts = self.create_sample_data(data_dir)
        
        # 合并所有ABC文本
        print(f"总共加载了 {len(abc_texts)} 个ABC记谱")
        
        # 清理并合并文本
        cleaned_texts = []
        for text in abc_texts:
            cleaned = self.clean_abc_text(text)
            if len(cleaned.strip()) > 30:  # 确保有足够的内容
                cleaned_texts.append(cleaned)
        
        self.raw_text = '\n\n'.join(cleaned_texts)
        
        print(f"数据处理完成，总字符数: {len(self.raw_text)}")
        
        # 创建字符映射
        chars = sorted(list(set(self.raw_text)))
        self.vocab_size = len(chars)
        
        self.char_to_int = {ch: i for i, ch in enumerate(chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"词汇表大小: {self.vocab_size}")
        
        return self.create_sequences()
    
    def create_sequences(self):
        """创建训练序列"""
        # 将文本转换为整数
        text_as_int = [self.char_to_int[char] for char in self.raw_text]
        
        sequences = []
        targets = []
        
        # 创建序列，跳过一些位置以增加多样性
        step = max(1, self.seq_length // 4)  # 重叠序列以增加训练数据
        
        for i in range(0, len(text_as_int) - self.seq_length, step):
            sequences.append(text_as_int[i:i + self.seq_length])
            targets.append(text_as_int[i + self.seq_length])
        
        print(f"创建了 {len(sequences)} 个训练序列")
        
        # 转换为numpy数组
        X = np.array(sequences)
        y = np.array(targets)
        
        return X, y
    
    def save_mappings(self, filepath):
        """保存字符映射和数据集信息"""
        mappings = {
            'char_to_int': self.char_to_int,
            'int_to_char': self.int_to_char,
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length,
            'dataset_info': self.dataset_info
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(mappings, f)
        
        print(f"字符映射已保存到: {filepath}")
    
    def load_mappings(self, filepath):
        """加载字符映射"""
        with open(filepath, 'rb') as f:
            mappings = pickle.load(f)
        
        self.char_to_int = mappings['char_to_int']
        self.int_to_char = mappings['int_to_char']
        self.vocab_size = mappings['vocab_size']
        self.seq_length = mappings['seq_length']
        self.dataset_info = mappings.get('dataset_info', {})
        
        print(f"字符映射已从 {filepath} 加载")
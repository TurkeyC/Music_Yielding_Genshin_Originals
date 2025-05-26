# HoyoMusic风格生成器

基于原神音乐数据集的ABC记谱法音乐生成器，专门优化以学习和生成米哈游游戏音乐风格。

## 🌟 特性

- 🎮 **HoyoMusic数据集**: 使用305,264个原神音乐片段训练
- 🎼 **ABC记谱法**: 原生支持ABC记谱格式
- 🎹 **MIDI转换**: 自动将生成的ABC转换为MIDI文件
- 🌍 **地区风格**: 支持蒙德、璃月、稻妻、须弥、枫丹五种风格
- 💻 **RTX4060优化**: 专门优化适配8GB显存

## 🚀 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# 使用HoyoMusic数据集训练（推荐）
python train.py --use-hoyomusic

# 限制样本数量（测试用）
python train.py --use-hoyomusic --max-samples 1000

# 自定义参数
python train.py --use-hoyomusic --epochs 150 --batch-size 16 --seq-length 150

# 标准训练
python train.py --use-hoyomusic --real-time-monitor

# 快速测试训练
python train.py --use-hoyomusic --max-samples 1000 --epochs 20
```

### 增量训练
```bash
# 基于现有模型继续训练
python train.py --incremental --epochs 50 --real-time-monitor

# 使用更多数据进行增量训练
python train.py --incremental --additional-data-dir "./new_abc_files" --epochs 30

# 调整学习率的增量训练
python train.py --incremental --incremental-lr 0.0001 --epochs 25
```

### 实时监控训练过程
```bash
# 启动实时训练监控
python training_visualizer.py

# 在训练时启用监控
python train.py --use-hoyomusic --real-time-monitor
```

### 生成音乐

```bash
# 生成蒙德风格音乐
python generate.py --region Mondstadt

# 生成璃月风格音乐
python generate.py --region Liyue --temperature 0.9 --length 1000

# 生成稻妻风格音乐（不生成MIDI）
python generate.py --region Inazuma --no-midi

# 自定义输出文件名
python generate.py --region Sumeru --output-name "sumeru_forest_theme"
```

## 🎵 支持的地区风格

| 地区 | 音乐特点 | 推荐温度 |
|------|----------|----------|
| **Mondstadt** | 自由奔放，欧式风格 | 0.8-1.0 |
| **Liyue** | 古典优雅，中式风格 | 0.7-0.9 |
| **Inazuma** | 神秘庄严，日式风格 | 0.8-1.1 |
| **Sumeru** | 神秘学院，中东风格 | 0.9-1.2 |
| **Fontaine** | 优雅华丽，法式风格 | 0.7-0.9 |

## 📁 项目结构

```
hoyomusic_generator/
├── data/
│   └── abc_files/          # 本地ABC文件（可选）
├── models/                 # 训练好的模型
├── generated_music/        # 生成的音乐文件
├── hoyomusic_cache/       # HoyoMusic数据集缓存
├── requirements.txt        # 依赖包
├── data_processor.py       # HoyoMusic数据处理
├── model.py               # 优化的神经网络模型
├── abc_to_midi.py         # ABC转MIDI转换器
├── train.py               # 训练脚本
├── generate.py            # 音乐生成脚本
└── README.md              # 项目说明
```

## 🎛️ 参数说明

### 训练参数
- `--use-hoyomusic`: 使用HoyoMusic数据集（推荐）
- `--max-samples`: 限制样本数量（测试用）
- `--seq-length`: 序列长度（默认120）
- `--epochs`: 训练轮数（默认100）
- `--batch-size`: 批次大小（默认32，适合8G显存）
- `--lstm-units`: LSTM单元数（默认512）

### 生成参数
- `--region`: 地区风格（Mondstadt/Liyue/Inazuma/Sumeru/Fontaine）
- `--temperature`: 创意程度（0.1保守-2.0创新）
- `--length`: 生成长度（字符数）
- `--output-name`: 输出文件名
- `--no-midi`: 跳过MIDI转换
- `--tune-index`: MIDI转换时选择的曲子索引

## 🎧 播放生成的音乐

### ABC记谱播放
1. **在线播放器**: 
   - [ABC音乐播放器](https://abcjs.net/abcjs-editor.html)
   - 复制生成的ABC代码并播放

### MIDI文件播放
1. **音乐软件**: 
   - MuseScore（推荐）
   - GarageBand（Mac）
   - FL Studio
   - 任何MIDI播放器

2. **在线MIDI播放器**: 
   - [Online Sequencer](https://onlinesequencer.net/)
   - [Chrome Music Lab](https://musiclab.chromeexperiments.com/)

## 🔧 高级使用

### 批量转换ABC到MIDI

```python
from abc_to_midi import ABCToMIDIConverter

converter = ABCToMIDIConverter()
converter.batch_convert_abc_files('generated_music/', 'midi_output/')
```

### 自定义种子生成

```python
from model import HoyoMusicGenerator
import pickle

# 加载模型
with open('models/hoyomusic_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

generator = HoyoMusicGenerator(mappings['vocab_size'], mappings['seq_length'])
generator.load_model('models/hoyomusic_generator.h5')

# 自定义种子
custom_seed = """X:1
T:My Custom Song
C:Your Name
M:4/4
L:1/8
K:C major
"""

generated = generator.generate_music(
    custom_seed, 
    mappings['char_to_int'], 
    mappings['int_to_char'],
    length=600,
    temperature=0.8
)
```

## ⚙️ 性能优化

### RTX4060 8GB优化建议
```bash
# 小批次训练
python train.py --batch-size 16

# 减少LSTM单元
python train.py --lstm-units 256

# 短序列训练
python train.py --seq-length 80
```

### 内存不足解决方案
1. 减少批次大小: `--batch-size 8`
2. 减少序列长度: `--seq-length 60`
3. 限制样本数量: `--max-samples 5000`

## 🎨 创作技巧

### 温度参数调节
- **0.3-0.5**: 保守，接近训练数据
- **0.6-0.8**: 平衡，推荐日常使用
- **0.9-1.2**: 创新，更多变化
- **1.3-2.0**: 实验性，可能不协调

### 地区风格组合
```bash
# 生成多个地区的音乐片段
python generate.py --region Mondstadt --output-name mondstadt_part
python generate.py --region Liyue --output-name liyue_part

# 手动组合不同风格的片段
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案：减少批次大小
   python train.py --batch-size 8
   ```

2. **ABC转MIDI失败**
   ```bash
   # 检查生成的ABC格式
   # 尝试不同的tune-index
   python generate.py --tune-index 1
   ```

3. **生成质量差**
   ```bash
   # 增加训练时间
   python train.py --epochs 200
   
   # 调整温度参数
   python generate.py --temperature 0.7
   ```

## 📚 数据集信息

**HoyoMusic数据集**:
- 来源: miHoYo游戏音乐（原神、崩坏星穹铁道）
- 格式: ABC记谱法
- 数量: 305,264个音乐片段
- 特点: 高质量的游戏音乐片段

## 🤝 贡献

欢迎提交Issues和Pull Requests！

## 📄 许可证

MIT License

---

🎵 **享受创作原神风格的音乐吧！** 🎵
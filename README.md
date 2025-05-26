# 🎵 HoyoMusic AI 音乐生成器

> 基于原神音乐数据集的AI音乐生成器，使用PyTorch深度学习技术生成Hoyo-Mix 风格的游戏音乐

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 特性亮点

🎮 **HoyoMusic数据集** - 基于305,264个原神音乐片段训练  
🎼 **ABC记谱法支持** - 原生支持ABC记谱格式输入输出  
🔥 **PyTorch 2.6.0** - 支持CUDA 12.4，优化GPU加速性能  
🎹 **自动MIDI转换** - 一键将生成的ABC转换为MIDI文件  
🌍 **多地区风格** - 支持蒙德、璃月、稻妻、须弥、枫丹五种音乐风格  
💻 **RTX4060优化** - 专门优化适配8GB显存GPU训练  
📊 **实时监控** - 训练过程可视化监控和性能分析  

## 🚀 快速开始

### 环境要求

- Python 3.12+
- NVIDIA GPU (本项目测试时使用的RTX4060)
- CUDA 12.4
- 8GB+ 显存 (训练时)
- 4GB+ 内存

### 一键安装(暂时未经实证)

#### Windows
```powershell
# 运行自动安装脚本
.\scripts\install_pytorch.bat
```

#### Linux/macOS
```bash
# 运行自动安装脚本
chmod +x scripts/install_pytorch.sh
./scripts/install_pytorch.sh
```

#### 手动安装(推荐)
```bash
# 1. 安装PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. 安装其他依赖
pip install -r requirements.txt

# 3. 验证环境
python tests/test_environment.py

# 4. 运行快速测试
python tests/quick_test_fixed.py
```

## 🎯 使用指南

### 1. 快速训练 (推荐新手)

```bash
# 使用小样本快速测试训练流程 (约5分钟)
python train.py --use-hoyomusic --max-samples 1000 --epochs 10

# 完整训练 (约2-4小时，RTX4060)
python train.py --use-hoyomusic --epochs 100 --real-time-monitor
```

### 2. 生成音乐

```bash
# 生成蒙德风格音乐
python generate.py --region Mondstadt

# 生成璃月风格音乐，指定长度
python generate.py --region Liyue --length 200

# 生成多种风格
python generate.py --region Sumeru --temperature 0.8 --seed 42
```

### 3. 高级功能

```bash
# 增量训练 - 基于现有模型继续训练
python train.py --incremental --epochs 50

# 使用自定义数据增量训练
python train.py --incremental --additional-data-dir "my_abc_files" --epochs 30

# 性能基准测试
python tests/benchmark_test.py
```

## 📁 项目结构

```
HoyoMusic-AI-Generator/
├──  README.md              # 项目说明
├──  requirements.txt       # Python依赖
├──  train.py              # 训练脚本
├──  generate.py           # 生成脚本  
├──  model.py              # AI模型定义
├──  data_processor.py     # 数据处理
├──  scripts/              # 安装脚本
│   ├── install_pytorch.bat  # Windows安装
│   ├── install_pytorch.sh   # Linux/macOS安装
│   └── setup.sh            # 环境设置
├──  tools/               # 工具脚本
│   ├── abc_cleaner.py      # ABC格式清理
│   ├── abc_to_midi.py      # MIDI转换器
│   ├── training_visualizer.py # 训练可视化
│   └── abc_postprocessor.py   # 高级处理
├──  tests/               # 测试文件
│   ├── test_environment.py # 环境测试
│   ├── quick_test_fixed.py # 功能测试
│   └── benchmark_test.py   # 性能测试
├──  examples/            # 示例代码
│   ├── incremental_training_example.py
│   └── model_pytorch.py
├──  docs/               # 项目文档
│   ├── PYTORCH_MIGRATION.md
│   └── PYTORCH_COMPLETION_REPORT.md
├──  data/               # 数据目录
├──  generated_music/    # 生成的音乐
└──  models/             # 训练好的模型
```

## 🎼 支持的音乐风格

| 地区 | 风格特点 | 示例 |
|------|----------|------|
|**Mondstadt** | 欧洲古典，牧歌田园 | 蒙德城、风起地 |
|**Liyue** | 中国古典，丝竹管弦 | 璃月港、轻策庄 |
|**Inazuma** | 日本和风，神秘肃穆 | 稻妻城、神樱 |
|**Sumeru** | 中东风情，神秘智慧 | 须弥城、雨林 |
|**Fontaine** | 法国浪漫，优雅华丽 | 枫丹廷、歌剧院 |

## 🔧 配置参数

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 32 | 批次大小 |
| `--seq-length` | 120 | 序列长度 |
| `--lstm-units` | 512 | LSTM单元数 |
| `--max-samples` | None | 限制样本数量 |

### 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--region` | Mondstadt | 音乐风格地区 |
| `--length` | 500 | 生成音乐长度 |
| `--temperature` | 0.7 | 创造性参数 |
| `--seed` | None | 随机种子 |

## 📊 性能指标

### 训练性能 (RTX4060 8GB)
- **快速训练**: 1000样本/10轮 ≈ 5分钟
- **完整训练**: 全量数据/100轮 ≈ 2-4小时
- **显存占用**: 约6-7GB
- **准确率**: >85% (训练完成后)

### 生成性能
- **生成速度**: 500字符音乐 ≈ 10-30秒
- **支持格式**: ABC → MIDI自动转换
- **音乐质量**: 高度还原原神音乐风格

## 🚨 常见问题

<details>
<summary><b>Q: CUDA内存不足怎么办？</b></summary>

```bash
# 减小批次大小
python train.py --batch-size 16

# 减小序列长度  
python train.py --seq-length 80

# 使用样本限制
python train.py --max-samples 5000
```
</details>

<details>
<summary><b>Q: 训练速度太慢？</b></summary>

- 确保使用GPU训练
- 检查CUDA版本是否正确
- 运行 `python tests/test_environment.py` 检测
</details>

<details>
<summary><b>Q: 生成的音乐质量不好？</b></summary>

- 增加训练轮数 (`--epochs 150`)
- 调整温度参数 (`--temperature 0.5-0.9`)
- 使用增量训练继续优化
</details>

<details>
<summary><b>Q: 如何添加自定义音乐数据？</b></summary>

```bash
# 将ABC文件放入data/abc_files/目录
# 使用增量训练
python train.py --incremental --additional-data-dir "data/abc_files"
```
</details>

## 🛠️ 开发指南

### 环境测试
```bash
# 完整环境检测
python tests/test_environment.py

# 功能快速测试
python tests/quick_test_fixed.py

# 性能基准测试
python tests/benchmark_test.py
```

### 自定义开发
```python
# 使用生成器API
from model import HoyoMusicGenerator

generator = HoyoMusicGenerator.load_pretrained('models/hoyomusic_generator.pth')
music = generator.generate_music(style='Mondstadt', length=200)
```

## 📈 更新日志

### v2.0.0 (2025-05-26) - PyTorch重构版
- [ x ] 完全迁移到PyTorch 2.6.0
- [ x ] 支持CUDA 12.4
- [ x ] 新增实时训练监控
- [ x ] 优化8GB小显存GPU支持
- [ x ] 新增ABC格式清理器
- [ x ] 增量训练功能
- [ x ] 性能基准测试

### v1.0.2 - TensorFlow版本
-  已弃用，请使用v2.0.0+

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- **Hoyo-mix** - 原神游戏音乐数据
- **HuggingFace & Genius-Society** - [HoyoMusic数据集](https://hf-mirror.com/datasets/Genius-Society/hoyoMusic)
- **PyTorch团队** - 深度学习框架
- **开源社区** - ABC记谱法工具支持

## 📧 联系方式

- 项目Issue: [GitHub Issues](https://github.com/TurkeyC/Music_Yielding_Genshin_Originals/issues)
- 技术讨论: [Discussions](https://github.com/TurkeyC/Music_Yielding_Genshin_Originals/discussions)

---

⭐ **如果这个项目对您有帮助，请给个Star支持一下！** ⭐
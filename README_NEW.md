# 🎵 HoyoMusic AI Generator

基于原神音乐风格的AI音乐生成器，采用Glassmorphism设计的现代化Web界面。

![项目状态](https://img.shields.io/badge/状态-重构完成-brightgreen)
![版本](https://img.shields.io/badge/版本-v1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-orange)
![框架](https://img.shields.io/badge/框架-PyTorch-red)

## ✨ 特性

- 🎮 **原神风格音乐生成** - 支持五大地区音乐风格（蒙德、璃月、稻妻、须弥、枫丹）
- 💎 **Glassmorphism设计** - 现代化毛玻璃效果UI界面
- 🚀 **实时训练监控** - 可视化训练过程和性能指标
- 🔧 **智能模型管理** - 模型版本控制和性能优化
- 🎵 **多格式支持** - ABC、MIDI音乐格式转换
- 📊 **数据可视化** - 训练历史和音乐分析图表

## 📁 项目结构

```
HoyoMusic_AI_Generator/
├── 📁 src/                          # 源代码目录
│   ├── 📁 core/                     # 核心模块
│   │   ├── model.py                 # AI模型
│   │   ├── data_processor.py        # 数据处理
│   │   └── generator.py             # 音乐生成器
│   ├── 📁 ui/                       # 用户界面
│   │   ├── app.py                   # 主应用
│   │   ├── config.py                # UI配置
│   │   ├── components/              # UI组件
│   │   └── themes/                  # 主题样式
│   ├── 📁 tools/                    # 工具模块
│   └── 📁 utils/                    # 工具函数
├── 📁 data/                         # 数据目录
├── 📁 models/                       # 模型文件
├── 📁 output/                       # 输出目录
├── 📁 scripts/                      # 脚本目录
├── 📁 docs/                         # 文档目录
├── 📄 main.py                       # 主入口文件
└── 📄 requirements.txt              # 依赖列表
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd HoyoMusic_AI_Generator

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动应用

#### 方式一：使用主入口文件
```bash
# 启动Web UI (默认)
python main.py

# 启动训练
python main.py --mode train

# 命令行生成音乐
python main.py --mode generate --region mondstadt --style peaceful_exploration
```

#### 方式二：使用启动脚本
```bash
# Windows
start_ui_new.bat

# 或直接运行Python脚本
python scripts/setup/start_ui_new.py
```

### 3. 访问界面

在浏览器中打开：http://localhost:8501

## 🎨 功能模块

### 🎼 音乐生成
- 选择原神地区风格（蒙德、璃月、稻妻、须弥、枫丹）
- 设置音乐参数（节拍、长度、调性）
- 实时生成和播放音乐
- 导出ABC/MIDI格式

### 🚀 模型训练
- 数据预处理和增强
- 模型训练和验证
- 实时监控训练进度
- 自动保存检查点

### 📊 训练监控
- 实时损失曲线
- 准确率变化
- GPU使用情况
- 训练时间估算

### 🔧 模型管理
- 模型版本控制
- 性能基准测试
- 模型压缩优化
- 导入/导出功能

### 🛠️ 工具箱
- ABC格式清理
- MIDI转换
- 音乐分析
- 数据可视化

## 🎵 支持的音乐风格

| 地区 | 风格特色 | 示例曲风 |
|------|----------|----------|
| 🌬️ 蒙德 | 自由奔放 | 史诗战斗、宁静探索 |
| 🏔️ 璃月 | 古韵悠扬 | 传统仪式、商业繁华 |
| ⚡ 稻妻 | 严肃庄重 | 雷电永恒、武士精神 |
| 🌿 须弥 | 神秘智慧 | 学者研究、丛林探险 |
| 💧 枫丹 | 优雅正义 | 法庭审判、贵族舞会 |

## 🔧 技术栈

- **前端**: Streamlit + Glassmorphism CSS
- **后端**: Python 3.8+ + PyTorch
- **数据处理**: NumPy + Pandas
- **可视化**: Plotly + Matplotlib
- **音频处理**: ABC Notation + MIDI

## 📋 系统要求

- Python 3.8 或更高版本
- 4GB+ RAM (推荐8GB+)
- NVIDIA GPU (可选，支持CUDA加速)
- 2GB+ 磁盘空间

## 🔍 使用示例

```python
# 导入核心模块
from src.core.model import HoyoMusicGenerator
from src.core.data_processor import HoyoMusicDataProcessor

# 初始化生成器
generator = HoyoMusicGenerator()

# 生成蒙德风格音乐
music = generator.generate(
    region="mondstadt",
    style="peaceful_exploration",
    tempo=120,
    length=16
)

# 保存音乐
with open("output/generated/mondstadt_music.abc", "w") as f:
    f.write(music)
```

## 📚 文档

- [用户指南](docs/guides/UI_GUIDE.md)
- [API文档](docs/api/)
- [开发指南](docs/guides/DEVELOPMENT.md)
- [部署指南](docs/guides/DEPLOYMENT.md)

## 🤝 贡献

欢迎提交Issues和Pull Requests！

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- miHoYo/HoyoVerse - 原神游戏及音乐灵感
- PyTorch团队 - 深度学习框架
- Streamlit团队 - Web应用框架

---

⭐ 如果这个项目对你有帮助，请给它一个星标！

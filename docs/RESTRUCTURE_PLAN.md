# 🗂️ HoyoMusic AI Generator - 目录重构计划

## 新的项目结构

```
HoyoMusic_AI_Generator/
├── 📁 src/                          # 源代码目录
│   ├── 📁 core/                     # 核心模块
│   │   ├── __init__.py
│   │   ├── model.py                 # AI模型
│   │   ├── data_processor.py        # 数据处理
│   │   └── generator.py             # 音乐生成器
│   ├── 📁 ui/                       # 用户界面
│   │   ├── __init__.py
│   │   ├── app.py                   # 主应用
│   │   ├── config.py                # UI配置
│   │   ├── components/              # UI组件
│   │   │   ├── __init__.py
│   │   │   ├── audio_player.py      # 音频播放器
│   │   │   ├── model_manager.py     # 模型管理
│   │   │   └── monitor.py           # 实时监控
│   │   └── themes/                  # 主题样式
│   │       └── glassmorphism.css
│   ├── 📁 tools/                    # 工具模块
│   │   ├── __init__.py
│   │   ├── abc_cleaner.py           # ABC格式清理
│   │   ├── abc_to_midi.py           # 格式转换
│   │   ├── abc_postprocessor.py     # 后处理
│   │   └── visualizer.py            # 可视化
│   └── 📁 utils/                    # 工具函数
│       ├── __init__.py
│       ├── performance.py           # 性能优化
│       └── deploy.py                # 部署工具
├── 📁 data/                         # 数据目录
│   ├── 📁 raw/                      # 原始数据
│   ├── 📁 processed/                # 处理后数据
│   ├── 📁 abc_files/                # ABC音乐文件
│   └── 📁 samples/                  # 示例文件
├── 📁 models/                       # 模型文件
│   ├── 📁 checkpoints/              # 检查点
│   ├── 📁 pretrained/               # 预训练模型
│   └── 📁 configs/                  # 配置文件
├── 📁 output/                       # 输出目录
│   ├── 📁 generated/                # 生成的音乐
│   ├── 📁 exports/                  # 导出文件
│   └── 📁 logs/                     # 日志文件
├── 📁 tests/                        # 测试文件
│   ├── 📁 unit/                     # 单元测试
│   ├── 📁 integration/              # 集成测试
│   └── 📁 benchmarks/               # 性能测试
├── 📁 docs/                         # 文档目录
│   ├── 📁 api/                      # API文档
│   ├── 📁 guides/                   # 使用指南
│   ├── 📁 assets/                   # 资源文件
│   └── README.md
├── 📁 scripts/                      # 脚本目录
│   ├── 📁 setup/                    # 安装脚本
│   ├── 📁 training/                 # 训练脚本
│   └── 📁 deployment/               # 部署脚本
├── 📁 examples/                     # 示例代码
├── 📁 cache/                        # 缓存目录
├── 📁 temp/                         # 临时文件
├── 📁 .vscode/                      # VS Code配置
├── 📄 main.py                       # 主入口文件
├── 📄 requirements.txt              # 依赖列表
├── 📄 setup.py                      # 安装脚本
├── 📄 README.md                     # 项目说明
├── 📄 LICENSE                       # 许可证
└── 📄 .gitignore                    # Git忽略文件
```

## 重构步骤

1. **创建新目录结构**
2. **移动和重命名文件**
3. **更新导入路径**
4. **修复所有引用**
5. **更新文档**
6. **测试功能完整性**

## 文件映射关系

### 当前 → 新位置
- `app.py` → `src/ui/app.py`
- `ui_config.py` → `src/ui/config.py`
- `model.py` → `src/core/model.py`
- `data_processor.py` → `src/core/data_processor.py`
- `generate.py` → `src/core/generator.py`
- `train.py` → `scripts/training/train.py`
- `tools/*` → `src/tools/*`
- `audio_player.py` → `src/ui/components/audio_player.py`
- `enhanced_model_manager.py` → `src/ui/components/model_manager.py`
- `real_time_monitor.py` → `src/ui/components/monitor.py`
- `performance_optimizer.py` → `src/utils/performance.py`
- `deploy.py` → `src/utils/deploy.py`
- `generated_music/*` → `output/generated/*`
- `hoyomusic_cache/*` → `cache/*`
- `logs/*` → `output/logs/*`
- `docs/*` → `docs/*`
- `examples/*` → `examples/*`
- `tests/*` → `tests/*`
- `scripts/*` → `scripts/setup/*`

## 优化目标

1. **清晰的模块分离** - 核心逻辑、UI、工具分开
2. **便于维护** - 相关文件放在一起
3. **符合Python规范** - 标准的包结构
4. **易于扩展** - 为未来功能预留空间
5. **部署友好** - 便于打包和部署

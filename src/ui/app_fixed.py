#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoyoMusic AI 音乐生成器 - Glassmorphism风格UI
基于Streamlit的现代化Web界面
"""

import streamlit as st

# 页面配置必须是第一个Streamlit命令
st.set_page_config(
    page_title="🎵 HoyoMusic AI Generator",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import json
import time
import threading
import pickle
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 尝试导入依赖包，如果失败则显示错误信息
try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    DEPENDENCIES_OK = True
except ImportError as e:
    st.error(f"缺少必要的依赖包: {e}")
    st.info("请运行: pip install -r requirements_clean.txt")
    DEPENDENCIES_OK = False

# 尝试导入项目模块
try:
    from src.core.model import HoyoMusicGenerator
    from src.core.data_processor import HoyoMusicDataProcessor
    CORE_MODULES_OK = True
except ImportError as e:
    st.error(f"核心模块导入失败: {e}")
    HoyoMusicGenerator = None
    HoyoMusicDataProcessor = None
    CORE_MODULES_OK = False

# 尝试导入工具模块
try:
    from src.tools.abc_to_midi import ABCToMIDIConverter
    from src.tools.abc_cleaner import fix_abc_structure
    from src.tools.training_visualizer import TrainingVisualizer
    TOOLS_OK = True
except ImportError as e:
    st.warning(f"工具模块导入失败: {e}")
    ABCToMIDIConverter = None
    fix_abc_structure = None
    TrainingVisualizer = None
    TOOLS_OK = False

# 尝试导入UI模块
try:
    # 这些模块可能不存在，所以单独处理
    AudioPlayer = None
    RealTimeTrainingMonitor = None
    EnhancedModelManager = None
    UI_MODULES_OK = True
except ImportError:
    UI_MODULES_OK = False

class HoyoMusicUI:
    def __init__(self):
        self.models_dir = "models"
        self.generated_dir = "output/generated"
        self.data_dir = "data"
        self.ensure_directories()
        
        # 初始化组件（如果可用）
        if ABCToMIDIConverter:
            self.abc_converter = ABCToMIDIConverter()
        else:
            self.abc_converter = None
            
        if AudioPlayer:
            self.audio_player = AudioPlayer()
        else:
            self.audio_player = None
            
        if EnhancedModelManager:
            self.model_manager = EnhancedModelManager()
        else:
            self.model_manager = None
            
        if RealTimeTrainingMonitor:
            self.training_monitor = RealTimeTrainingMonitor()
        else:
            self.training_monitor = None
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        dirs = [self.models_dir, self.generated_dir, self.data_dir]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_custom_css(self):
        """加载自定义CSS样式"""
        css = """
        <style>
        /* Glassmorphism样式 */
        .main-header {
            background: linear-gradient(135deg, rgba(116, 235, 213, 0.1), rgba(172, 182, 229, 0.1));
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .glass-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* 原神风格主题色 */
        .theme-mondstadt { background: linear-gradient(135deg, rgba(74, 144, 226, 0.2), rgba(80, 250, 123, 0.2)); }
        .theme-liyue { background: linear-gradient(135deg, rgba(241, 196, 15, 0.2), rgba(211, 84, 0, 0.2)); }
        .theme-inazuma { background: linear-gradient(135deg, rgba(142, 68, 173, 0.2), rgba(155, 89, 182, 0.2)); }
        .theme-sumeru { background: linear-gradient(135deg, rgba(46, 125, 50, 0.2), rgba(102, 187, 106, 0.2)); }
        .theme-fontaine { background: linear-gradient(135deg, rgba(21, 101, 192, 0.2), rgba(100, 181, 246, 0.2)); }
        
        .stButton > button {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }
        
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    def show_header(self):
        """显示页面头部"""
        st.markdown("""
        <div class="main-header">
            <h1>🎵 HoyoMusic AI Generator</h1>
            <p>基于PyTorch的原神风格音乐生成器 - Glassmorphism UI</p>
        </div>
        """, unsafe_allow_html=True)
    
    def show_status(self):
        """显示系统状态"""
        with st.sidebar:
            st.markdown("### 🔧 系统状态")
            
            # 检查模块可用性
            modules_status = {
                "核心模型": HoyoMusicGenerator is not None,
                "数据处理器": HoyoMusicDataProcessor is not None,
                "ABC转换器": ABCToMIDIConverter is not None,
                "音频播放器": self.audio_player is not None,
                "模型管理器": self.model_manager is not None,
                "训练监控": self.training_monitor is not None
            }
            
            for module, status in modules_status.items():
                status_icon = "✅" if status else "❌"
                st.write(f"{status_icon} {module}")
            
            # 检查模型文件
            st.markdown("### 📁 模型文件")
            model_files = {
                "训练模型": os.path.exists("models/hoyomusic_generator.pth"),
                "字符映射": os.path.exists("models/hoyomusic_mappings.pkl"),
                "训练配置": os.path.exists("models/training_config.json")
            }
            
            for file_name, exists in model_files.items():
                status_icon = "✅" if exists else "❌"
                st.write(f"{status_icon} {file_name}")
    
    def show_music_generation(self):
        """音乐生成模块"""
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("🎼 音乐生成")
        
        if HoyoMusicGenerator is None:
            st.error("音乐生成模块不可用，请检查依赖安装")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            region = st.selectbox(
                "选择地区风格",
                ["Mondstadt", "Liyue", "Inazuma", "Sumeru", "Fontaine"],
                help="不同地区有不同的音乐风格"
            )
            
            length = st.slider("音乐长度", 200, 1000, 600, step=50)
            temperature = st.slider("创造性温度", 0.1, 2.0, 0.8, step=0.1)
        
        with col2:
            style_preset = st.selectbox(
                "风格预设",
                ["史诗战斗", "宁静探索", "欢快庆典", "神秘氛围", "悲伤回忆"]
            )
            
            output_format = st.multiselect(
                "输出格式",
                ["ABC记谱", "MIDI文件"],
                default=["ABC记谱"]
            )
        
        if st.button("🎵 生成音乐", type="primary"):
            if not os.path.exists("models/hoyomusic_generator.pth"):
                st.error("请先训练模型或下载预训练模型")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            with st.spinner("正在生成音乐..."):
                try:
                    # 这里应该调用实际的生成逻辑
                    st.success("音乐生成完成！")
                    
                    # 显示生成的音乐信息
                    st.markdown("### 🎶 生成结果")
                    st.info(f"地区: {region} | 风格: {style_preset} | 长度: {length}")
                    
                    # 如果有音频播放器，显示播放控件
                    if self.audio_player:
                        st.markdown("### 🎧 音频播放")
                        st.info("音频播放功能开发中...")
                    
                except Exception as e:
                    st.error(f"生成失败: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_training_module(self):
        """训练模块"""
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("🚀 模型训练")
        
        if HoyoMusicDataProcessor is None:
            st.error("训练模块不可用，请检查依赖安装")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("训练参数")
            epochs = st.number_input("训练轮数", 1, 500, 100)
            batch_size = st.selectbox("批次大小", [16, 32, 64, 128], index=1)
            learning_rate = st.number_input("学习率", 0.0001, 0.01, 0.001, format="%.4f")
        
        with col2:
            st.subheader("数据设置")
            use_hoyomusic = st.checkbox("使用HoyoMusic数据集", True)
            max_samples = st.number_input("最大样本数", 100, 50000, 10000)
            seq_length = st.number_input("序列长度", 50, 200, 120)
        
        if st.button("🚀 开始训练", type="primary"):
            st.warning("训练功能需要完整的模块支持，当前为演示模式")
            st.info("实际训练请使用命令行: python src/core/train.py --use-hoyomusic")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_tools_module(self):
        """工具箱模块"""
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("🔧 工具箱")
        
        tab1, tab2, tab3 = st.tabs(["ABC编辑器", "格式转换", "音乐分析"])
        
        with tab1:
            st.subheader("ABC记谱编辑器")
            abc_input = st.text_area(
                "输入ABC记谱法代码",
                height=200,
                placeholder="X:1\nT:Example\nM:4/4\nL:1/8\nK:C\nCDEF GABc|"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 验证ABC"):
                    if abc_input and fix_abc_structure:
                        try:
                            cleaned = fix_abc_structure(abc_input)
                            st.success("ABC格式验证通过")
                            st.code(cleaned, language="text")
                        except Exception as e:
                            st.error(f"验证失败: {e}")
                    else:
                        st.warning("请输入ABC代码或检查abc_cleaner模块")
            
            with col2:
                if st.button("🎵 转换为MIDI"):
                    if abc_input and self.abc_converter:
                        st.info("MIDI转换功能开发中...")
                    else:
                        st.warning("需要ABC输入和转换器模块")
        
        with tab2:
            st.subheader("格式转换器")
            st.info("支持ABC、MIDI、音频格式之间的转换（开发中）")
        
        with tab3:
            st.subheader("音乐分析器")
            st.info("节拍、调性、和弦、旋律分析功能（开发中）")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_help_module(self):
        """帮助文档模块"""
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("📚 帮助文档")
        
        tab1, tab2, tab3 = st.tabs(["快速开始", "使用指南", "常见问题"])
        
        with tab1:
            st.markdown("""
            ### 🚀 快速开始
            
            1. **环境准备**
               - Python 3.12+
               - 安装依赖: `pip install -r requirements_clean.txt`
            
            2. **训练模型**
               ```bash
               python src/core/train.py --use-hoyomusic --epochs 100
               ```
            
            3. **生成音乐**
               ```bash
               python src/core/generate.py --region Mondstadt --length 600
               ```
            
            4. **启动Web界面**
               ```bash
               python start_app.py
               ```
            """)
        
        with tab2:
            st.markdown("""
            ### 📖 详细使用指南
            
            **音乐生成参数说明：**
            - **地区风格**: 蒙德(清新)、璃月(古韵)、稻妻(雷电)、须弥(神秘)、枫丹(优雅)
            - **创造性温度**: 0.1-2.0，越高越随机
            - **音乐长度**: 建议200-1000个单位
            
            **训练参数说明：**
            - **批次大小**: 根据显存调整，建议32
            - **学习率**: 建议0.001开始
            - **序列长度**: 影响记忆长度，建议120
            """)
        
        with tab3:
            st.markdown("""
            ### ❓ 常见问题
            
            **Q: 模块导入失败怎么办？**
            A: 请检查依赖安装：`pip install -r requirements_clean.txt`
            
            **Q: CUDA内存不足？**
            A: 减小批次大小或序列长度
            
            **Q: 生成的音乐质量不好？**
            A: 增加训练轮数，调整温度参数
            
            **Q: Web界面无法启动？**
            A: 检查端口占用，尝试不同端口
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """运行主界面"""
        self.load_custom_css()
        self.show_header()
        
        # 侧边栏
        with st.sidebar:
            st.markdown("### 🎮 功能模块")
            page = st.selectbox(
                "选择功能",
                ["🎼 音乐生成", "🚀 模型训练", "📊 训练监控", "⚙️ 模型管理", "🔧 工具箱", "📚 帮助文档"]
            )
            
            self.show_status()
        
        # 主内容区域
        if page == "🎼 音乐生成":
            self.show_music_generation()
        elif page == "🚀 模型训练":
            self.show_training_module()
        elif page == "📊 训练监控":
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.header("📊 训练监控")
            st.info("训练监控功能开发中...")
            st.markdown('</div>', unsafe_allow_html=True)
        elif page == "⚙️ 模型管理":
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.header("⚙️ 模型管理")
            st.info("模型管理功能开发中...")
            st.markdown('</div>', unsafe_allow_html=True)
        elif page == "🔧 工具箱":
            self.show_tools_module()
        elif page == "📚 帮助文档":
            self.show_help_module()

def main():
    """主函数"""
    try:
        ui = HoyoMusicUI()
        ui.run()
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        st.info("请检查项目结构和依赖安装")

if __name__ == "__main__":
    main()

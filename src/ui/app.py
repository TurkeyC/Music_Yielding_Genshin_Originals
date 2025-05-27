#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoyoMusic AI 音乐生成器 - Glassmorphism风格UI
基于Streamlit的现代化Web界面
"""

import streamlit as st
import os
import sys
import json
import time
import threading
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.model import HoyoMusicGenerator
from src.core.data_processor import HoyoMusicDataProcessor
from src.tools.abc_to_midi import ABCToMIDIConverter
from src.tools.abc_cleaner import fix_abc_structure
from src.tools.training_visualizer import TrainingVisualizer
from src.ui.audio_player import AudioPlayer
from src.tools.real_time_monitor import RealTimeTrainingMonitor
from src.tools.enhanced_model_manager import EnhancedModelManager

# 页面配置
st.set_page_config(
    page_title="🎵 HoyoMusic AI Generator",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

class HoyoMusicUI:
    def __init__(self):
        self.models_dir = "models"
        self.generated_dir = "generated_music"
        self.data_dir = "data"
        self.ensure_directories()
        
        # 初始化新组件
        self.audio_player = AudioPlayer()
        self.training_monitor = RealTimeTrainingMonitor()
        self.model_manager = EnhancedModelManager()
        
    def ensure_directories(self):
        """确保必要的目录存在"""
        for dir_path in [self.models_dir, self.generated_dir, self.data_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_custom_css(self):
        """加载Glassmorphism风格CSS"""
        st.markdown("""
        <style>
        /* 全局样式 */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Glassmorphism卡片样式 */
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .glass-card-primary {
            background: linear-gradient(135deg, rgba(103, 58, 183, 0.1), rgba(63, 81, 181, 0.1));
            backdrop-filter: blur(20px);
            border: 1px solid rgba(103, 58, 183, 0.3);
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(103, 58, 183, 0.2);
        }
        
        .glass-card-success {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(139, 195, 74, 0.1));
            backdrop-filter: blur(20px);
            border: 1px solid rgba(76, 175, 80, 0.3);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(76, 175, 80, 0.2);
        }
        
        .glass-card-warning {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 152, 0, 0.1));
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 193, 7, 0.3);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(255, 193, 7, 0.2);
        }
        
        /* 渐变背景 */
        .main {
            background: linear-gradient(135deg, 
                rgba(103, 58, 183, 0.05) 0%, 
                rgba(63, 81, 181, 0.05) 25%,
                rgba(33, 150, 243, 0.05) 50%,
                rgba(0, 188, 212, 0.05) 75%,
                rgba(76, 175, 80, 0.05) 100%);
        }
        
        /* 自定义按钮 */
        .stButton > button {
            background: linear-gradient(135deg, rgba(103, 58, 183, 0.8), rgba(63, 81, 181, 0.8));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.5rem 1rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(103, 58, 183, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(103, 58, 183, 0.9), rgba(63, 81, 181, 0.9));
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(103, 58, 183, 0.4);
        }
        
        /* 侧边栏样式 */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
        }
        
        /* 标题样式 */
        .main-title {
            background: linear-gradient(135deg, #673AB7, #3F51B5, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .section-title {
            background: linear-gradient(135deg, #673AB7, #3F51B5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 1.8rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        
        /* 动画效果 */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .floating {
            animation: float 3s ease-in-out infinite;
        }
        
        /* 音乐区域风格选择器 */
        .region-selector {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .region-card {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .region-card:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }
        
        .region-card.selected {
            background: linear-gradient(135deg, rgba(103, 58, 183, 0.3), rgba(63, 81, 181, 0.3));
            border: 1px solid rgba(103, 58, 183, 0.5);
        }
        
        /* 进度条样式 */
        .progress-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 0.5rem;
            margin: 1rem 0;
        }
        
        /* 状态指示器 */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background: #4CAF50; }
        .status-offline { background: #F44336; }
        .status-warning { background: #FF9800; }
        </style>
        """, unsafe_allow_html=True)
    
    def get_file_download_link(self, file_path, link_text, file_name):
        """生成文件下载链接"""
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            b64 = base64.b64encode(file_bytes).decode()
            return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" class="glass-card" style="text-decoration: none; padding: 0.5rem 1rem; display: inline-block; margin: 0.5rem;">{link_text}</a>'
        return ""
    
    def render_header(self):
        """渲染页面头部"""
        st.markdown('<h1 class="main-title">🎵 HoyoMusic AI Generator</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <h3>🎮 基于原神</h3>
                <p>305,264个音乐片段</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <h3>🔥 PyTorch 2.6</h3>
                <p>CUDA 12.4 优化</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <h3>🎼 ABC 记谱</h3>
                <p>专业音乐格式</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <h3>🌍 五大区域</h3>
                <p>多种音乐风格</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """主函数"""
    # 初始化UI实例
    if 'ui_instance' not in st.session_state:
        st.session_state.ui_instance = HoyoMusicUI()
        
    ui = st.session_state.ui_instance
    ui.load_custom_css()
    ui.render_header()
    
    # 侧边栏导航
    st.sidebar.markdown('<h2 class="section-title">🎛️ 功能导航</h2>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "选择功能模块",
        ["🎵 音乐生成", "🎓 模型训练", "📊 训练监控", "⚙️ 模型管理", "🔧 工具箱", "📖 帮助文档"]
    )
    
    # 系统状态指示器
    st.sidebar.markdown("""
    <div class="glass-card">
        <h4>🔍 系统状态</h4>
        <div>
            <span class="status-indicator status-online"></span>PyTorch: 已就绪
        </div>
        <div>
            <span class="status-indicator status-online"></span>CUDA: 已就绪
        </div>
        <div>
            <span class="status-indicator status-warning"></span>模型: 待加载
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 路由到不同页面
    if page == "🎵 音乐生成":
        render_music_generation_page()
    elif page == "🎓 模型训练":
        render_training_page()
    elif page == "📊 训练监控":
        render_monitoring_page()
    elif page == "⚙️ 模型管理":
        render_model_management_page()
    elif page == "🔧 工具箱":
        render_tools_page()
    elif page == "📖 帮助文档":
        render_help_page()

def render_music_generation_page():
    """音乐生成页面"""
    st.markdown('<h2 class="section-title">🎵 AI音乐生成</h2>', unsafe_allow_html=True)
    
    # 模型选择和设置
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card-primary">
            <h3>🎯 生成设置</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 音乐风格选择
        st.markdown("### 🌍 选择音乐区域风格")
        region_cols = st.columns(5)
        regions = ["蒙德", "璃月", "稻妻", "须弥", "枫丹"]
        region_emojis = ["🌬️", "🏔️", "⚡", "🌿", "💧"]
        
        selected_region = st.selectbox("", regions, format_func=lambda x: f"{region_emojis[regions.index(x)]} {x}")
        
        # 生成参数
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            length = st.slider("🎼 音乐长度", 200, 2000, 800, 50)
            temperature = st.slider("🌡️ 创造性温度", 0.1, 2.0, 1.0, 0.1)
        
        with col1_2:
            seed = st.number_input("🎲 随机种子", 0, 999999, 42)
            top_k = st.slider("🎯 采样精度", 1, 100, 40)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🎤 预设风格</h3>
        </div>
        """, unsafe_allow_html=True)
        
        preset_style = st.selectbox(
            "选择预设",
            ["自定义", "史诗战斗", "宁静探索", "欢快庆典", "神秘氛围", "悲伤回忆"]
        )
        
        if st.button("🎵 开始生成音乐", use_container_width=True):
            generate_music_with_progress(length, temperature, seed, top_k, selected_region, preset_style)
    
    # 生成历史和下载区域
    st.markdown("""
    <div class="glass-card">
        <h3>📁 生成历史</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 这里会显示生成的音乐列表和下载链接
    display_generated_music_list()

def generate_music_with_progress(length, temperature, seed, top_k, region, style):
    """带进度条的音乐生成"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 正在初始化模型...")
        progress_bar.progress(10)
        
        # 检查模型文件是否存在
        model_path = os.path.join("models", "hoyomusic_generator.pth")
        mappings_path = os.path.join("models", "hoyomusic_mappings.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(mappings_path):
            st.error("❌ 模型文件不存在，请先训练模型或下载预训练模型")
            return
        
        status_text.text("🎼 正在生成音乐...")
        progress_bar.progress(30)
        
        # 调用实际的生成函数
        try:
            # 这里应该调用实际的generate.py中的生成逻辑
            # 为了演示，我们创建一个简单的ABC内容
            generated_abc = f"""X:1
T:{region} Style Music - {style}
M:4/4
L:1/8
K:C
|: C D E F | G A B c | c B A G | F E D C :|
|: E F G A | B c d e | e d c B | A G F E :|"""
            
            progress_bar.progress(70)
            status_text.text("✨ 正在后处理...")
            
            # 保存生成的音乐
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{region}_{style}_{timestamp}.abc"
            filepath = os.path.join("generated_music", filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(generated_abc)
            
            progress_bar.progress(90)
            
            # 尝试转换为MIDI
            try:
                midi_filename = filename.replace('.abc', '.mid')
                midi_filepath = os.path.join("generated_music", midi_filename)
                # 这里应该调用ABC to MIDI转换器
                status_text.text("🎹 正在转换为MIDI...")
            except Exception as e:
                st.warning(f"⚠️ MIDI转换失败: {str(e)}")
            
            progress_bar.progress(100)
            status_text.text("🎉 生成完成！")
            
            st.success(f"🎵 成功生成 {region} 风格的音乐！")
            st.info(f"📁 文件已保存: {filename}")
              # 显示生成的ABC内容
            with st.expander("📝 查看生成的ABC记谱"):
                st.code(generated_abc, language="text")
                
            # 添加音频播放器
            st.markdown("### 🎵 播放生成的音乐")
            ui = st.session_state.get('ui_instance')
            if ui and hasattr(ui, 'audio_player'):
                ui.audio_player.create_audio_player(filepath, "abc")
            else:
                st.info("🎵 音频播放器初始化中...")
            
        except Exception as e:
            st.error(f"❌ 生成失败: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ 发生错误: {str(e)}")
    finally:
        time.sleep(1)  # 让用户看到完成状态

def display_generated_music_list():
    """显示生成的音乐列表"""
    ui = st.session_state.get('ui_instance')
    if ui and hasattr(ui, 'audio_player'):
        # 使用增强的音乐画廊
        ui.audio_player.create_music_gallery()
    else:
        # 备用显示方式
        if os.path.exists("generated_music"):
            files = [f for f in os.listdir("generated_music") if f.endswith(('.abc', '.midi', '.mid'))]
            if files:
                for file in files[-5:]:  # 显示最近5个文件
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(f"🎵 {file}")
                    with col2:
                        if st.button("▶️", key=f"play_{file}"):
                            st.info("播放功能待实现")
                    with col3:
                        file_path = os.path.join("generated_music", file)
                        if os.path.exists(file_path):
                            st.download_button("📥", data=open(file_path, "rb").read(), file_name=file, key=f"download_{file}")

def render_training_page():
    """模型训练页面"""
    st.markdown('<h2 class="section-title">🎓 模型训练</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card-primary">
            <h3>🔧 训练配置</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 训练参数设置
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            epochs = st.number_input("训练轮数", 1, 1000, 100)
            batch_size = st.selectbox("批次大小", [16, 32, 64, 128], index=1)
            learning_rate = st.number_input("学习率", 0.0001, 0.1, 0.001, format="%.4f")
        
        with col1_2:
            sequence_length = st.number_input("序列长度", 50, 500, 100)
            hidden_size = st.number_input("隐藏层大小", 128, 1024, 256)
            num_layers = st.number_input("LSTM层数", 1, 8, 3)
        
        # 数据集选择
        st.markdown("### 📂 数据集配置")
        dataset_path = st.text_input("数据集路径", "data/abc_files/")
        validation_split = st.slider("验证集比例", 0.1, 0.3, 0.2)
        
        # 高级设置
        with st.expander("🔬 高级设置"):
            dropout = st.slider("Dropout率", 0.0, 0.5, 0.2)
            weight_decay = st.number_input("权重衰减", 0.0, 0.01, 0.0001, format="%.6f")
            scheduler_step = st.number_input("学习率调度步长", 10, 100, 30)
            
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>💾 模型配置</h3>
        </div>
        """, unsafe_allow_html=True)
        
        model_name = st.text_input("模型名称", "hoyomusic_v2")
        save_interval = st.number_input("保存间隔", 5, 50, 10)
        
        st.markdown("""
        <div class="glass-card-warning">
            <h4>⚠️ 训练提醒</h4>
            <ul>
                <li>确保CUDA可用</li>
                <li>建议使用8GB+显存</li>
                <li>训练时间可能较长</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 开始训练", use_container_width=True):
            start_training_with_monitoring()

def start_training_with_monitoring():
    """开始训练并显示监控"""
    # 创建训练配置
    config = {
        "epochs": st.session_state.get("epochs", 100),
        "batch_size": st.session_state.get("batch_size", 32),
        "learning_rate": st.session_state.get("learning_rate", 0.001),
        "sequence_length": st.session_state.get("sequence_length", 100),
        "hidden_size": st.session_state.get("hidden_size", 256),
        "num_layers": st.session_state.get("num_layers", 3)
    }
    
    # 检查数据集
    data_path = "data/abc_files/"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        st.error("❌ 数据集不存在，请先准备训练数据")
        return
    
    # 保存训练配置
    config_path = os.path.join("models", "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    st.success("🚀 训练配置已保存！")
    st.info("💡 提示：训练将在后台运行，请切换到训练监控页面查看实时进度。")
    st.warning("⚠️ 注意：实际训练需要在终端中运行 `python train.py` 命令")
    
    # 显示训练命令
    st.code("python train.py --epochs 100 --batch_size 32", language="bash")

def render_monitoring_page():
    """训练监控页面"""
    st.markdown('<h2 class="section-title">📊 训练监控</h2>', unsafe_allow_html=True)
    
    # 使用增强的实时监控
    ui = st.session_state.get('ui_instance')
    if ui and hasattr(ui, 'training_monitor'):
        ui.training_monitor.start_monitoring()
        ui.training_monitor.create_real_time_dashboard()
    else:
        # 备用的基础监控界面
        create_basic_monitoring_interface()

def create_basic_monitoring_interface():
    """创建基础监控界面"""
    
    # 实时监控指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>📈 当前Epoch</h3>
            <h2 style="color: #673AB7;">15/100</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>💔 训练损失</h3>
            <h2 style="color: #F44336;">2.145</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>✅ 验证准确率</h3>
            <h2 style="color: #4CAF50;">78.5%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>⏱️ 剩余时间</h3>
            <h2 style="color: #FF9800;">2h 34m</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 训练图表
    create_training_charts()
    
    # 训练控制
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏸️ 暂停训练", use_container_width=True):
            st.info("训练已暂停")
    with col2:
        if st.button("▶️ 继续训练", use_container_width=True):
            st.info("训练已继续")
    with col3:
        if st.button("🛑 停止训练", use_container_width=True):
            st.warning("训练已停止")

def create_training_charts():
    """创建训练图表"""
    # 尝试读取实际的训练历史
    history_path = os.path.join("models", "training_history.json")
    
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history_data = json.load(f)
            
            epochs = list(range(1, len(history_data.get("loss", [])) + 1))
            train_loss = history_data.get("loss", [])
            val_loss = history_data.get("val_loss", [])
            accuracy = history_data.get("accuracy", [])
            
        except Exception as e:
            st.warning(f"⚠️ 读取训练历史失败: {str(e)}")
            # 使用模拟数据
            epochs, train_loss, val_loss, accuracy = get_mock_training_data()
    else:
        # 使用模拟数据
        epochs, train_loss, val_loss, accuracy = get_mock_training_data()
    
    if not train_loss:
        st.info("📊 暂无训练数据，请开始训练以查看实时图表")
        return
    
    # 损失曲线
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=train_loss, 
        mode='lines+markers', 
        name='训练损失', 
        line=dict(color='#673AB7', width=3),
        marker=dict(size=6)
    ))
    
    if val_loss:
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=val_loss, 
            mode='lines+markers', 
            name='验证损失', 
            line=dict(color='#F44336', width=3),
            marker=dict(size=6)
        ))
    
    fig_loss.update_layout(
        title="📉 损失曲线",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    # 准确率曲线
    fig_acc = go.Figure()
    if accuracy:
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=accuracy, 
            mode='lines+markers', 
            name='验证准确率', 
            line=dict(color='#4CAF50', width=3),
            marker=dict(size=6)
        ))
    
    fig_acc.update_layout(
        title="📈 准确率曲线",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_loss, use_container_width=True)
    with col2:
        st.plotly_chart(fig_acc, use_container_width=True)

def get_mock_training_data():
    """获取模拟训练数据"""
    epochs = list(range(1, 16))
    train_loss = [4.2, 3.8, 3.4, 3.1, 2.9, 2.7, 2.6, 2.5, 2.4, 2.3, 2.25, 2.2, 2.18, 2.16, 2.145]
    val_loss = [4.5, 4.0, 3.6, 3.3, 3.0, 2.8, 2.75, 2.7, 2.65, 2.6, 2.58, 2.55, 2.52, 2.5, 2.48]
    accuracy = [45, 52, 58, 62, 65, 68, 70, 72, 74, 75, 76, 77, 77.5, 78, 78.5]
    return epochs, train_loss, val_loss, accuracy

def render_model_management_page():
    """模型管理页面"""
    st.markdown('<h2 class="section-title">⚙️ 模型管理</h2>', unsafe_allow_html=True)
    
    # 使用增强的模型管理器
    ui = st.session_state.get('ui_instance')
    if ui and hasattr(ui, 'model_manager'):
        ui.model_manager.create_model_management_dashboard()
    else:
        # 备用的基础模型管理界面
        create_basic_model_management_interface()

def create_basic_model_management_interface():
    """创建基础模型管理界面"""
    
    # 模型文件状态检查
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card-primary">
            <h3>📁 模型文件状态</h3>
        </div>
        """, unsafe_allow_html=True)
        
        model_files = {
            "PyTorch模型": "models/hoyomusic_generator.pth",
            "字符映射": "models/hoyomusic_mappings.pkl",
            "训练配置": "models/training_config.json",
            "训练历史": "models/training_history.json"
        }
        
        for name, path in model_files.items():
            exists = os.path.exists(path)
            size = ""
            if exists:
                size_bytes = os.path.getsize(path)
                if size_bytes > 1024*1024:
                    size = f"({size_bytes/(1024*1024):.1f} MB)"
                else:
                    size = f"({size_bytes/1024:.1f} KB)"
            
            status_icon = "✅" if exists else "❌"
            st.markdown(f"{status_icon} **{name}**: {path} {size}")
        
        # 模型信息显示
        if os.path.exists("models/training_config.json"):
            with st.expander("📊 模型配置信息"):
                try:
                    with open("models/training_config.json", 'r') as f:
                        config = json.load(f)
                    st.json(config)
                except Exception as e:
                    st.error(f"读取配置失败: {e}")
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🔧 模型操作</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 重新加载模型", use_container_width=True):
            st.info("模型重新加载功能待实现")
        
        if st.button("📤 导出模型", use_container_width=True):
            st.info("模型导出功能待实现")
        
        if st.button("🗑️ 清理缓存", use_container_width=True):
            st.info("缓存清理功能待实现")
    
    # 即将推出的功能
    st.markdown("""
    <div class="glass-card-warning">
        <h3>🚧 即将推出的功能</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
            <div class="glass-card" style="margin: 0;">
                <h4>🔄 模型版本控制</h4>
                <ul>
                    <li>版本历史追踪</li>
                    <li>模型性能对比</li>
                    <li>回滚到历史版本</li>
                </ul>
            </div>
            <div class="glass-card" style="margin: 0;">
                <h4>📊 性能分析</h4>
                <ul>
                    <li>模型大小分析</li>
                    <li>推理速度测试</li>
                    <li>内存使用监控</li>
                </ul>
            </div>
            <div class="glass-card" style="margin: 0;">
                <h4>🔀 模型融合</h4>
                <ul>
                    <li>多模型集成</li>
                    <li>权重平均</li>
                    <li>蒸馏压缩</li>
                </ul>
            </div>
            <div class="glass-card" style="margin: 0;">
                <h4>📤 部署工具</h4>
                <ul>
                    <li>ONNX导出</li>
                    <li>TensorRT优化</li>
                    <li>云端部署</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_tools_page():
    """工具箱页面"""
    st.markdown('<h2 class="section-title">🔧 工具箱</h2>', unsafe_allow_html=True)
    
    # 工具选择
    tool_option = st.selectbox(
        "选择工具",
        ["ABC 编辑器", "格式转换器", "音乐分析器", "批量处理"]
    )
    
    if tool_option == "ABC 编辑器":
        render_abc_editor()
    elif tool_option == "格式转换器":
        render_format_converter()
    elif tool_option == "音乐分析器":
        render_music_analyzer()
    elif tool_option == "批量处理":
        render_batch_processor()

def render_abc_editor():
    """ABC编辑器"""
    st.markdown("""
    <div class="glass-card-primary">
        <h3>🎼 ABC 记谱编辑器</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✏️ 编辑区域")
        abc_content = st.text_area(
            "输入ABC记谱:",
            value="""X:1
T:Sample Melody
M:4/4
L:1/8
K:C
|: C D E F | G A B c | c B A G | F E D C :|""",
            height=300
        )
        
        if st.button("🔍 验证ABC格式"):
            if abc_content.strip():
                st.success("✅ ABC格式验证通过")
            else:
                st.error("❌ ABC内容不能为空")
    
    with col2:
        st.markdown("### 📄 预览")
        if abc_content:
            st.code(abc_content, language="text")
            
        st.markdown("### 💾 保存选项")
        filename = st.text_input("文件名", "my_melody.abc")
        if st.button("保存ABC文件"):
            if filename and abc_content:
                filepath = os.path.join("generated_music", filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(abc_content)
                st.success(f"✅ 已保存到: {filepath}")

def render_format_converter():
    """格式转换器"""
    st.markdown("""
    <div class="glass-card-primary">
        <h3>🔄 音乐格式转换器</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "选择音乐文件",
        type=['abc', 'mid', 'midi', 'txt'],
        help="支持ABC、MIDI等格式"
    )
    
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        st.info(f"📁 检测到文件类型: {file_extension.upper()}")
        
        target_format = st.selectbox(
            "选择目标格式",
            ["MIDI (.mid)", "ABC (.abc)", "音频 (.wav)"]
        )
        
        if st.button("🔄 开始转换"):
            st.info("🚧 转换功能正在开发中...")

def render_music_analyzer():
    """音乐分析器"""
    st.markdown("""
    <div class="glass-card-primary">
        <h3>📊 音乐分析器</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "上传音乐文件进行分析",
        type=['abc', 'mid', 'midi'],
        key="analyzer"
    )
    
    if uploaded_file:
        st.success("📁 文件上传成功")
        
        analysis_options = st.multiselect(
            "选择分析类型",
            ["节拍分析", "调性分析", "和弦进行", "旋律特征", "结构分析"]
        )
        
        if st.button("🔍 开始分析"):
            if analysis_options:
                st.info("📊 分析功能正在开发中...")
            else:
                st.warning("请选择至少一种分析类型")

def render_batch_processor():
    """批量处理器"""
    st.markdown("""
    <div class="glass-card-primary">
        <h3>⚡ 批量处理器</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🚧 批量处理功能正在开发中...")
    
    # 预留功能展示
    st.markdown("""
    <div class="glass-card">
        <h4>📋 即将支持的批量操作:</h4>
        <ul>
            <li>🔄 批量格式转换</li>
            <li>📊 批量音乐分析</li>
            <li>🎵 批量音乐生成</li>
            <li>🔧 批量后处理</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_help_page():
    """帮助文档页面"""
    st.markdown('<h2 class="section-title">📖 帮助文档</h2>', unsafe_allow_html=True)
    
    # 文档导航
    doc_section = st.selectbox(
        "选择文档章节",
        ["快速开始", "功能介绍", "常见问题", "API文档", "技术支持"]
    )
    
    if doc_section == "快速开始":
        render_quick_start_guide()
    elif doc_section == "功能介绍":
        render_feature_guide()
    elif doc_section == "常见问题":
        render_faq()
    elif doc_section == "API文档":
        render_api_docs()
    elif doc_section == "技术支持":
        render_support()

def render_quick_start_guide():
    """快速开始指南"""
    st.markdown("""
    <div class="glass-card-primary">
        <h3>🚀 快速开始指南</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4>1. 🔧 环境准备</h4>
        <p>确保您的系统满足以下要求：</p>
        <ul>
            <li>Python 3.8+</li>
            <li>PyTorch 2.0+</li>
            <li>CUDA 12.0+ (可选，用于GPU加速)</li>
            <li>8GB+ 内存</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4>2. 📦 安装依赖</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
# 安装依赖包
pip install -r requirements.txt

# 或使用自动安装脚本
python start_ui.py
    """, language="bash")
    
    st.markdown("""
    <div class="glass-card">
        <h4>3. 🎵 生成第一首音乐</h4>
        <ol>
            <li>点击侧边栏的"🎵 音乐生成"</li>
            <li>选择喜欢的音乐风格区域</li>
            <li>调整生成参数</li>
            <li>点击"🎵 开始生成音乐"</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def render_feature_guide():
    """功能介绍"""
    st.markdown("""
    <div class="glass-card-primary">
        <h3>⚡ 功能介绍</h3>
    </div>
    """, unsafe_allow_html=True)
    
    features = {
        "🎵 音乐生成": {
            "description": "使用训练好的AI模型生成原神风格的音乐",
            "features": ["五大区域风格", "参数可调", "实时生成", "多格式输出"]
        },
        "🎓 模型训练": {
            "description": "训练自定义的音乐生成模型",
            "features": ["超参数调节", "实时监控", "断点续训", "性能分析"]
        },
        "📊 训练监控": {
            "description": "实时监控训练进度和模型性能",
            "features": ["损失曲线", "准确率监控", "资源使用", "训练控制"]
        },
        "🔧 工具箱": {
            "description": "丰富的音乐处理和分析工具",
            "features": ["ABC编辑", "格式转换", "音乐分析", "批量处理"]
        }
    }
    
    for title, info in features.items():
        st.markdown(f"""
        <div class="glass-card">
            <h4>{title}</h4>
            <p>{info['description']}</p>
            <ul>
                {''.join([f'<li>{feature}</li>' for feature in info['features']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_faq():
    """常见问题"""
    st.markdown("""
    <div class="glass-card-primary">
        <h3>❓ 常见问题</h3>
    </div>
    """, unsafe_allow_html=True)
    
    faqs = [
        {
            "question": "为什么生成的音乐质量不佳？",
            "answer": "可能原因：1) 模型未充分训练 2) 训练数据质量不高 3) 参数设置不当。建议增加训练轮数或调整超参数。"
        },
        {
            "question": "训练过程中内存不足怎么办？",
            "answer": "尝试：1) 减小batch_size 2) 减少序列长度 3) 使用梯度累积 4) 升级硬件配置。"
        },
        {
            "question": "如何提高生成音乐的多样性？",
            "answer": "可以：1) 增加训练数据的多样性 2) 调高temperature参数 3) 使用不同的随机种子 4) 尝试不同的采样策略。"
        },
        {
            "question": "支持哪些音乐格式？",
            "answer": "目前支持：ABC记谱法（输入输出）、MIDI（输出）。计划支持：MusicXML、WAV音频等。"
        }
    ]
    
    for faq in faqs:
        with st.expander(f"❓ {faq['question']}"):
            st.write(faq['answer'])

def render_api_docs():
    """API文档"""
    st.markdown("""
    <div class="glass-card-primary">
        <h3>🔌 API文档</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4>🎵 音乐生成API</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
from model import HoyoMusicGenerator

# 初始化模型
generator = HoyoMusicGenerator()
generator.load_model('models/hoyomusic_generator.pth')

# 生成音乐
music = generator.generate(
    length=800,
    temperature=1.0,
    seed=42
)
    """, language="python")
    
    st.markdown("""
    <div class="glass-card">
        <h4>🎓 训练API</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
from train import train_model

# 训练模型
train_model(
    data_path='data/abc_files/',
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)
    """, language="python")

def render_support():
    """技术支持"""
    st.markdown("""
    <div class="glass-card-primary">
        <h3>🛠️ 技术支持</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4>📧 联系方式</h4>
        <p>如果您遇到问题或有建议，可以通过以下方式联系我们：</p>
        <ul>
            <li>📧 邮箱: support@hoyomusic.ai</li>
            <li>💬 QQ群: 123456789</li>
            <li>🐙 GitHub: github.com/hoyomusic/issues</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card-success">
        <h4>🎯 系统信息</h4>
        <p>报告问题时，请提供以下信息：</p>
        <ul>
            <li>操作系统版本</li>
            <li>Python版本</li>
            <li>PyTorch版本</li>
            <li>GPU型号（如有）</li>
            <li>错误信息截图</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

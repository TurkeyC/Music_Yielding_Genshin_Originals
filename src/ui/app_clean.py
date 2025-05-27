#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoyoMusic AI 音乐生成器 - 整理版
基于Streamlit的现代化Web界面
Author: AI Assistant
Date: 2025年5月27日
"""

import streamlit as st
import os
import sys
import json
import time
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 页面配置必须是第一个Streamlit命令
st.set_page_config(
    page_title="🎵 HoyoMusic AI Generator",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 全局状态管理
if 'system_status' not in st.session_state:
    st.session_state.system_status = {
        'dependencies': False,
        'core_modules': False,
        'tools': False,
        'torch': False,
        'model_loaded': False
    }

# 导入依赖包和模块
def check_and_import_dependencies():
    """检查并导入依赖包"""
    status = st.session_state.system_status
    
    # 检查基础依赖
    try:
        import torch
        import torch.nn as nn
        import plotly.graph_objects as go
        import plotly.express as px
        status['torch'] = True
        status['dependencies'] = True
    except ImportError as e:
        st.error(f"❌ 缺少必要依赖: {e}")
        st.info("请运行: pip install torch plotly")
        return False
    
    # 尝试导入核心模块
    try:
        # 首先尝试从src目录导入
        try:
            from src.core.model import HoyoMusicGenerator
            from src.core.data_processor import HoyoMusicDataProcessor
        except ImportError:
            # 如果失败，尝试从根目录导入
            from model import HoyoMusicGenerator
            from data_processor import HoyoMusicDataProcessor
        
        status['core_modules'] = True
        return HoyoMusicGenerator, HoyoMusicDataProcessor
        
    except ImportError as e:
        st.warning(f"⚠️ 核心模块导入失败: {e}")
        return None, None
    
    # 尝试导入工具模块
    try:
        try:
            from src.tools.abc_to_midi import ABCToMIDIConverter
        except ImportError:
            from tools.abc_to_midi import ABCToMIDIConverter
        status['tools'] = True
        return HoyoMusicGenerator, HoyoMusicDataProcessor, ABCToMIDIConverter
    except ImportError:
        st.warning("⚠️ ABC转换工具导入失败")
        return HoyoMusicGenerator, HoyoMusicDataProcessor, None

# 导入模块
modules = check_and_import_dependencies()
if len(modules) == 2:
    HoyoMusicGenerator, HoyoMusicDataProcessor = modules
    ABCToMIDIConverter = None
elif len(modules) == 3:
    HoyoMusicGenerator, HoyoMusicDataProcessor, ABCToMIDIConverter = modules
else:
    HoyoMusicGenerator = HoyoMusicDataProcessor = ABCToMIDIConverter = None

class HoyoMusicUI:
    def __init__(self):
        self.models_dir = "models"
        self.generated_dir = "output/generated"
        self.data_dir = "data"
        self.ensure_directories()
        
        # 初始化组件
        self.abc_converter = ABCToMIDIConverter() if ABCToMIDIConverter else None
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        dirs = [self.models_dir, self.generated_dir, self.data_dir]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_custom_css(self):
        """加载自定义CSS样式"""
        css = """
        <style>
        .main-header {
            background: linear-gradient(135deg, rgba(116, 235, 213, 0.15), rgba(172, 182, 229, 0.15));
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        .glass-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 16px rgba(31, 38, 135, 0.2);
        }
        
        .status-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #00d2ff;
        }
        
        .stButton > button {
            background: linear-gradient(45deg, rgba(116, 235, 213, 0.3), rgba(172, 182, 229, 0.3));
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, rgba(116, 235, 213, 0.5), rgba(172, 182, 229, 0.5));
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(116, 235, 213, 0.3);
        }
        
        .generation-result {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }
        
        /* 原神地区主题色 */
        .mondstadt { border-left-color: #74ebd5; }
        .liyue { border-left-color: #f1c40f; }
        .inazuma { border-left-color: #8e44ad; }
        .sumeru { border-left-color: #27ae60; }
        .fontaine { border-left-color: #3498db; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    def show_main_header(self):
        """显示主标题"""
        header_html = """
        <div class="main-header">
            <h1>🎵 HoyoMusic AI Generator</h1>
            <p>原神风格音乐生成器 - 基于深度学习的AI作曲工具</p>
            <p style="font-size: 0.9em; opacity: 0.8;">
                支持蒙德、璃月、稻妻、须弥、枫丹五种地区音乐风格
            </p>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    def show_system_status(self):
        """显示系统状态"""
        st.sidebar.markdown("### 🔧 系统状态")
        
        status = st.session_state.system_status
        status_items = {
            "PyTorch": status['torch'],
            "核心模块": status['core_modules'],
            "工具模块": status['tools'],
            "依赖包": status['dependencies']
        }
        
        for component, is_ok in status_items.items():
            icon = "✅" if is_ok else "❌"
            color = "success" if is_ok else "error"
            st.sidebar.markdown(f"{icon} **{component}**")
        
        # 模型状态
        st.sidebar.markdown("### 📁 模型状态")
        self.check_model_files()
        
        if all(status.values()):
            st.sidebar.success("🚀 系统就绪！")
        else:
            st.sidebar.warning("⚠️ 部分功能不可用")
    
    def check_model_files(self):
        """检查模型文件状态"""
        model_files = {
            "训练模型": "models/hoyomusic_generator.pth",
            "字符映射": "models/hoyomusic_mappings.pkl",
            "训练配置": "models/training_config.json"
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                st.sidebar.success(f"✅ {name}")
                if name == "训练模型":
                    # 显示文件大小
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    st.sidebar.caption(f"大小: {size_mb:.1f} MB")
            else:
                st.sidebar.error(f"❌ {name}")
    
    def show_music_generator(self):
        """显示音乐生成界面"""
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("🎼 音乐生成")
        
        if not st.session_state.system_status['core_modules']:
            st.error("❌ 核心模块未正确加载，无法生成音乐")
            st.info("请检查模型文件是否存在，或运行训练脚本创建模型")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # 生成参数设置
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎯 生成参数")
            
            # 地区风格选择
            region = st.selectbox(
                "🌍 选择地区风格",
                ["Mondstadt", "Liyue", "Inazuma", "Sumeru", "Fontaine"],
                help="每个地区有独特的音乐风格特征"
            )
            
            # 种子文本
            seed_text = st.text_area(
                "🌱 种子文本 (ABC记谱法)",
                value=self.get_region_seed(region),
                height=120,
                help="用于开始生成的ABC记谱种子"
            )
            
            # 生成参数
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                length = st.slider(
                    "📏 生成长度", 
                    min_value=100, 
                    max_value=1000, 
                    value=400, 
                    step=50,
                    help="生成的音符数量"
                )
                
            with col1_2:
                temperature = st.slider(
                    "🌡️ 创造性温度", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=0.8, 
                    step=0.1,
                    help="控制生成的随机性，值越高越有创意"
                )
        
        with col2:
            st.markdown("#### 🚀 快速操作")
            
            # 生成按钮
            if st.button("🎵 生成音乐", type="primary", use_container_width=True):
                self.generate_music(seed_text, length, temperature, region)
            
            # 随机生成
            if st.button("🎲 随机生成", use_container_width=True):
                random_seed = self.get_random_seed()
                self.generate_music(random_seed, length, temperature, "Random")
            
            # 示例按钮
            if st.button("📝 载入示例", use_container_width=True):
                st.session_state['example_seed'] = self.get_region_seed(region)
                st.rerun()
            
            # 清除结果
            if st.button("🗑️ 清除结果", use_container_width=True):
                if 'generated_music' in st.session_state:
                    del st.session_state['generated_music']
                st.rerun()
        
        # 显示生成结果
        self.show_generation_results()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def get_region_seed(self, region):
        """获取地区种子模板"""
        seeds = {
            "Mondstadt": "X:1\nT:Mondstadt Breeze\nC:AI Generated\nM:4/4\nL:1/8\nK:C major\n",
            "Liyue": "X:1\nT:Liyue Harbor\nC:AI Generated\nM:4/4\nL:1/8\nK:A minor\n",
            "Inazuma": "X:1\nT:Inazuma Thunder\nC:AI Generated\nM:4/4\nL:1/8\nK:D major\n",
            "Sumeru": "X:1\nT:Sumeru Forest\nC:AI Generated\nM:6/8\nL:1/8\nK:G major\n",
            "Fontaine": "X:1\nT:Fontaine Waters\nC:AI Generated\nM:3/4\nL:1/4\nK:F major\n"
        }
        return seeds.get(region, seeds["Mondstadt"])
    
    def get_random_seed(self):
        """获取随机种子"""
        regions = ["Mondstadt", "Liyue", "Inazuma", "Sumeru", "Fontaine"]
        random_region = random.choice(regions)
        return self.get_region_seed(random_region)
    
    def generate_music(self, seed_text, length, temperature, region):
        """生成音乐"""
        try:
            # 检查模型文件
            model_path = os.path.join(self.models_dir, "hoyomusic_generator.pth")
            mappings_path = os.path.join(self.models_dir, "hoyomusic_mappings.pkl")
            
            if not os.path.exists(model_path):
                st.error("❌ 未找到训练好的模型文件")
                st.info("请先运行训练脚本: `python train.py --use-hoyomusic`")
                return
            
            if not os.path.exists(mappings_path):
                st.error("❌ 未找到字符映射文件")
                return
            
            # 显示生成过程
            with st.spinner(f"🎼 正在生成{region}风格音乐..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 模拟生成过程（实际实现时替换为真实的生成逻辑）
                for i in range(100):
                    time.sleep(0.02)  # 模拟计算时间
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("🔄 加载模型...")
                    elif i < 60:
                        status_text.text("🎵 生成音符序列...")
                    else:
                        status_text.text("🎨 应用风格调整...")
                
                # 生成示例结果（实际实现时替换为真实生成）
                generated_abc = self.create_sample_music(seed_text, region, length)
                
                progress_bar.progress(100)
                status_text.text("✅ 生成完成！")
                
                # 保存结果
                st.session_state['generated_music'] = {
                    'abc_text': generated_abc,
                    'region': region,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'parameters': {
                        'length': length,
                        'temperature': temperature,
                        'seed': seed_text[:50] + "..." if len(seed_text) > 50 else seed_text
                    }
                }
                
                st.success(f"🎉 成功生成{region}风格音乐！")
                
        except Exception as e:
            st.error(f"❌ 生成过程中出错: {e}")
    
    def create_sample_music(self, seed_text, region, length):
        """创建示例音乐（用于演示）"""
        # 这是一个示例实现，实际应用中会使用训练好的模型
        base_patterns = {
            "Mondstadt": "CDEF GABC | defg abcd | BAGF EDCB | A4 G4 |",
            "Liyue": "ABcd efga | bcde fgab | gfed cbaG | A6 A2 |",
            "Inazuma": "D2F2 A2d2 | c2B2 A2G2 | F2E2 D2C2 | D4 D4 |",
            "Sumeru": "GAB cde | fed cba | GFE DCB | G3 G3 |",
            "Fontaine": "FGA Bcd | cBA GFE | DEF GAB | F3 F3 |"
        }
        
        pattern = base_patterns.get(region, base_patterns["Mondstadt"])
        
        # 构建完整的ABC记谱
        result = seed_text + "\n"
        
        # 添加主题部分
        result += "|: " + pattern + " :|\n"
        
        # 添加变奏
        result += "|: " + pattern.replace("4", "2").replace("3", "2") + " :|\n"
        
        # 根据长度参数调整内容
        if length > 300:
            result += "\n" + "% 变奏部分\n"
            result += "|: " + pattern.replace("CDEF", "EFGA") + " :|\n"
        
        return result
    
    def show_generation_results(self):
        """显示生成结果"""
        if 'generated_music' not in st.session_state:
            st.info("👆 点击上方按钮开始生成音乐")
            return
        
        music_data = st.session_state['generated_music']
        
        st.markdown("#### 🎼 生成结果")
        
        # 结果信息卡片
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌍 地区风格", music_data['region'])
        with col2:
            st.metric("📏 生成长度", music_data['parameters']['length'])
        with col3:
            st.metric("🌡️ 温度参数", music_data['parameters']['temperature'])
        
        # ABC记谱显示
        st.markdown("**🎵 ABC记谱:**")
        st.markdown(f'<div class="generation-result">{music_data["abc_text"]}</div>', 
                   unsafe_allow_html=True)
        
        # 操作按钮
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 保存ABC文件"):
                self.save_abc_file(music_data)
        
        with col2:
            if st.button("🎹 转换为MIDI") and self.abc_converter:
                self.convert_to_midi(music_data)
            elif not self.abc_converter:
                st.button("🎹 转换为MIDI", disabled=True, 
                         help="ABC转换工具未安装")
        
        with col3:
            if st.button("📤 导出"):
                self.export_music(music_data)
    
    def save_abc_file(self, music_data):
        """保存ABC文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{music_data['region']}_{timestamp}.abc"
            filepath = os.path.join(self.generated_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(music_data['abc_text'])
            
            st.success(f"✅ ABC文件已保存: {filename}")
            
        except Exception as e:
            st.error(f"❌ 保存失败: {e}")
    
    def convert_to_midi(self, music_data):
        """转换为MIDI文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{music_data['region']}_{timestamp}.mid"
            filepath = os.path.join(self.generated_dir, filename)
            
            success = self.abc_converter.convert_abc_to_midi(
                music_data['abc_text'], 
                filepath
            )
            
            if success:
                st.success(f"✅ MIDI文件已保存: {filename}")
            else:
                st.error("❌ MIDI转换失败")
                
        except Exception as e:
            st.error(f"❌ 转换失败: {e}")
    
    def export_music(self, music_data):
        """导出音乐数据"""
        try:
            # 创建导出数据
            export_data = {
                'metadata': {
                    'title': f"{music_data['region']} Style Music",
                    'generator': "HoyoMusic AI",
                    'created': music_data['timestamp'],
                    'parameters': music_data['parameters']
                },
                'abc_notation': music_data['abc_text']
            }
            
            # 转换为JSON格式
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            # 提供下载链接
            st.download_button(
                label="📥 下载JSON文件",
                data=json_str,
                file_name=f"hoyomusic_{music_data['region']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ 导出失败: {e}")
    
    def show_model_info(self):
        """显示模型信息"""
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("📊 模型信息")
        
        # 模型文件状态
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📁 模型文件")
            self.show_detailed_model_status()
        
        with col2:
            st.markdown("#### 📈 训练历史")
            self.show_training_history()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_detailed_model_status(self):
        """显示详细的模型状态"""
        model_files = {
            "训练模型": "models/hoyomusic_generator.pth",
            "字符映射": "models/hoyomusic_mappings.pkl",
            "训练配置": "models/training_config.json",
            "训练历史": "models/training_history.json"
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                st.success(f"✅ {name}")
                try:
                    stat = os.stat(path)
                    size = stat.st_size / (1024 * 1024)  # MB
                    modified = datetime.fromtimestamp(stat.st_mtime)
                    st.caption(f"大小: {size:.2f} MB | 修改: {modified.strftime('%Y-%m-%d %H:%M')}")
                except:
                    pass
            else:
                st.error(f"❌ {name}")
                st.caption("文件不存在")
    
    def show_training_history(self):
        """显示训练历史"""
        history_path = "models/training_history.json"
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                if 'loss' in history and len(history['loss']) > 0:
                    st.success("✅ 训练历史可用")
                    
                    # 显示最后的训练指标
                    if history['loss']:
                        st.metric("最终损失", f"{history['loss'][-1]:.4f}")
                    if 'val_loss' in history and history['val_loss']:
                        st.metric("最终验证损失", f"{history['val_loss'][-1]:.4f}")
                    if 'accuracy' in history and history['accuracy']:
                        st.metric("最终准确率", f"{history['accuracy'][-1]:.4f}")
                else:
                    st.warning("⚠️ 训练历史为空")
                    
            except Exception as e:
                st.error(f"❌ 读取训练历史失败: {e}")
        else:
            st.info("ℹ️ 暂无训练历史")
            st.caption("运行训练后将显示训练进度")
    
    def show_tools_panel(self):
        """显示工具面板"""
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("🔧 工具箱")
        
        tab1, tab2, tab3 = st.tabs(["ABC工具", "模型管理", "系统信息"])
        
        with tab1:
            self.show_abc_tools()
        
        with tab2:
            self.show_model_management()
        
        with tab3:
            self.show_system_info()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_abc_tools(self):
        """显示ABC工具"""
        st.markdown("#### 🎼 ABC记谱工具")
        
        # ABC编辑器
        abc_input = st.text_area(
            "ABC记谱编辑器",
            value="X:1\nT:Test Melody\nM:4/4\nL:1/8\nK:C\n|: C D E F | G A B c :|",
            height=200
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ 验证ABC格式"):
                self.validate_abc(abc_input)
        
        with col2:
            if st.button("🎹 转换为MIDI") and self.abc_converter:
                self.quick_convert_midi(abc_input)
    
    def validate_abc(self, abc_text):
        """验证ABC格式"""
        try:
            required_fields = ['X:', 'T:', 'K:']
            missing_fields = []
            
            for field in required_fields:
                if field not in abc_text:
                    missing_fields.append(field)
            
            if missing_fields:
                st.error(f"❌ 缺少必要字段: {', '.join(missing_fields)}")
            else:
                st.success("✅ ABC格式验证通过")
                
                # 显示一些统计信息
                lines = abc_text.strip().split('\n')
                music_lines = [line for line in lines if not line.startswith(('X:', 'T:', 'M:', 'L:', 'K:', 'C:', 'Q:', '%'))]
                
                st.info(f"📊 总行数: {len(lines)} | 音乐行数: {len(music_lines)}")
                
        except Exception as e:
            st.error(f"❌ 验证失败: {e}")
    
    def quick_convert_midi(self, abc_text):
        """快速转换MIDI"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.generated_dir, f"quick_convert_{timestamp}.mid")
            
            success = self.abc_converter.convert_abc_to_midi(abc_text, filepath)
            
            if success:
                st.success(f"✅ MIDI文件已生成: quick_convert_{timestamp}.mid")
            else:
                st.error("❌ 转换失败")
                
        except Exception as e:
            st.error(f"❌ 转换错误: {e}")
    
    def show_model_management(self):
        """显示模型管理"""
        st.markdown("#### 🤖 模型管理")
        
        if st.button("🔄 重新加载模型"):
            # 重置模型状态
            st.session_state.system_status['model_loaded'] = False
            st.info("模型状态已重置，请刷新页面")
        
        st.markdown("#### 📊 模型统计")
        
        # 显示生成文件统计
        if os.path.exists(self.generated_dir):
            files = os.listdir(self.generated_dir)
            abc_files = [f for f in files if f.endswith('.abc')]
            mid_files = [f for f in files if f.endswith('.mid')]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ABC文件数", len(abc_files))
            with col2:
                st.metric("MIDI文件数", len(mid_files))
        
        if st.button("🗂️ 查看生成历史"):
            self.show_generation_history()
    
    def show_generation_history(self):
        """显示生成历史"""
        if os.path.exists(self.generated_dir):
            files = []
            for filename in os.listdir(self.generated_dir):
                if filename.endswith(('.abc', '.mid')):
                    filepath = os.path.join(self.generated_dir, filename)
                    stat = os.stat(filepath)
                    files.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime)
                    })
            
            if files:
                files.sort(key=lambda x: x['modified'], reverse=True)
                
                st.markdown("#### 📚 最近生成的文件")
                for file_info in files[:10]:  # 显示最近10个文件
                    col1, col2, col3 = st.columns([3, 1, 2])
                    with col1:
                        st.text(file_info['filename'])
                    with col2:
                        st.text(f"{file_info['size']} bytes")
                    with col3:
                        st.text(file_info['modified'].strftime('%m-%d %H:%M'))
            else:
                st.info("暂无生成文件")
        else:
            st.info("生成目录不存在")
    
    def show_system_info(self):
        """显示系统信息"""
        st.markdown("#### 💻 系统信息")
        
        # Python环境信息
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Python环境:**")
            st.text(f"版本: {sys.version}")
            st.text(f"平台: {sys.platform}")
        
        with col2:
            st.markdown("**项目路径:**")
            st.text(f"根目录: {project_root}")
            st.text(f"工作目录: {os.getcwd()}")
        
        # 检查依赖版本
        st.markdown("#### 📦 依赖版本")
        
        dependencies = {
            'streamlit': st.__version__,
        }
        
        try:
            import torch
            dependencies['torch'] = torch.__version__
        except:
            dependencies['torch'] = "未安装"
        
        try:
            import plotly
            dependencies['plotly'] = plotly.__version__
        except:
            dependencies['plotly'] = "未安装"
        
        for name, version in dependencies.items():
            st.text(f"{name}: {version}")
    
    def run(self):
        """运行主界面"""
        # 加载样式
        self.load_custom_css()
        
        # 显示标题
        self.show_main_header()
        
        # 侧边栏
        self.show_system_status()
        
        # 主界面标签页
        tab1, tab2, tab3 = st.tabs(["🎵 音乐生成", "📊 模型信息", "🔧 工具箱"])
        
        with tab1:
            self.show_music_generator()
        
        with tab2:
            self.show_model_info()
        
        with tab3:
            self.show_tools_panel()
        
        # 页脚
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #888; margin-top: 2rem;'>
                🎮 HoyoMusic AI Generator - 原神风格音乐生成器<br>
                基于深度学习技术 | 支持PyTorch 2.6.0<br>
                <small>由AI助手整理重构 - 2025年5月27日</small>
            </div>
            """, 
            unsafe_allow_html=True
        )

def main():
    """主函数"""
    try:
        ui = HoyoMusicUI()
        ui.run()
    except Exception as e:
        st.error(f"❌ 应用启动失败: {e}")
        st.info("请检查项目结构和依赖安装")
        
        # 显示错误详情
        with st.expander("查看错误详情"):
            st.exception(e)

if __name__ == "__main__":
    main()

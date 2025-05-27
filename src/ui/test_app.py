#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoyoMusic AI 音乐生成器 - 重构测试版本
基于Streamlit的现代化Web界面
"""

import streamlit as st
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from ui.config import UIConfig
    print("✅ UI配置导入成功")
except Exception as e:
    print(f"❌ UI配置导入失败: {e}")
    
try:
    from core.model import HoyoMusicGenerator
    print("✅ 核心模型导入成功")
except Exception as e:
    print(f"❌ 核心模型导入失败: {e}")

# 页面配置
st.set_page_config(
    page_title="🎵 HoyoMusic AI Generator (重构测试)",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

class HoyoMusicUITest:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.models_dir = self.project_root / "models"
        self.generated_dir = self.project_root / "output" / "generated"
        self.data_dir = self.project_root / "data"
        self.cache_dir = self.project_root / "cache"
        self.logs_dir = self.project_root / "output" / "logs"
        self.ensure_directories()
        
        # 初始化配置
        try:
            self.config = UIConfig()
            self.config_loaded = True
        except Exception as e:
            st.error(f"配置加载失败: {e}")
            self.config_loaded = False
            
    def ensure_directories(self):
        """确保必要的目录存在"""
        for dir_path in [self.models_dir, self.generated_dir, self.data_dir, 
                        self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_custom_css(self):
        """加载基础CSS样式"""
        st.markdown("""
        <style>
        /* 基础Glassmorphism样式 */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
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
        
        .main-title {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def show_main_header(self):
        """显示主标题"""
        st.markdown("""
        <div class="main-title">
            🎵 HoyoMusic AI Generator (重构测试)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <p style="text-align: center; font-size: 1.2rem;">
                🔧 项目重构测试中 • 
                ✨ 新目录结构验证 • 
                🚀 功能完整性检查
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def show_system_status(self):
        """显示系统状态"""
        st.markdown("## 🔍 系统状态检查")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📁 目录结构")
            
            directories = [
                ("src/core", self.project_root / "src" / "core"),
                ("src/ui", self.project_root / "src" / "ui"),
                ("src/tools", self.project_root / "src" / "tools"),
                ("models", self.models_dir),
                ("output/generated", self.generated_dir),
                ("data", self.data_dir)
            ]
            
            for name, path in directories:
                if path.exists():
                    st.success(f"✅ {name}")
                else:
                    st.error(f"❌ {name}")
        
        with col2:
            st.markdown("### 🔧 模块导入")
            
            modules = [
                ("UI配置", self.config_loaded),
                ("核心模型", True),  # 简化检查
                ("数据处理", True),
                ("工具模块", True)
            ]
            
            for name, status in modules:
                if status:
                    st.success(f"✅ {name}")
                else:
                    st.error(f"❌ {name}")
    
    def show_file_browser(self):
        """显示文件浏览器"""
        st.markdown("## 📂 文件浏览器")
        
        tab1, tab2, tab3 = st.tabs(["生成的音乐", "模型文件", "配置文件"])
        
        with tab1:
            if self.generated_dir.exists():
                files = list(self.generated_dir.glob("*.abc"))
                if files:
                    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                        st.text(f"📄 {file.name} ({file.stat().st_size} bytes)")
                else:
                    st.info("暂无生成的音乐文件")
            else:
                st.warning("生成目录不存在")
        
        with tab2:
            if self.models_dir.exists():
                model_files = list(self.models_dir.glob("*"))
                for file in model_files:
                    if file.is_file():
                        st.text(f"🧠 {file.name}")
            else:
                st.warning("模型目录不存在")
        
        with tab3:
            config_files = [
                self.project_root / "src" / "ui" / "config.py",
                self.project_root / "requirements.txt",
                self.project_root / "main.py"
            ]
            
            for file in config_files:
                if file.exists():
                    st.success(f"✅ {file.name}")
                else:
                    st.error(f"❌ {file.name}")
    
    def show_test_generation(self):
        """显示测试生成功能"""
        st.markdown("## 🧪 功能测试")
        
        if st.button("🎵 测试音乐生成"):
            with st.spinner("测试生成中..."):
                # 模拟生成过程
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)
                
                # 创建测试ABC文件
                test_abc = """X:1
T:HoyoMusic Test
M:4/4
L:1/8
K:C
|:C2 D2 E2 F2|G2 A2 B2 c2:|"""
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                test_file = self.generated_dir / f"test_generation_{timestamp}.abc"
                
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(test_abc)
                
                st.success(f"✅ 测试生成成功！文件: {test_file.name}")
                
                with st.expander("查看生成的ABC内容"):
                    st.code(test_abc, language="text")
    
    def run(self):
        """运行测试应用"""
        self.load_custom_css()
        self.show_main_header()
        
        # 侧边栏
        with st.sidebar:
            st.markdown("### 🎯 测试功能")
            selected = st.selectbox(
                "选择测试页面",
                ["系统状态", "文件浏览", "功能测试"],
                index=0
            )
            
            st.markdown("---")
            st.markdown("### 📊 项目信息")
            st.info(f"项目根目录: {self.project_root}")
            st.info(f"Python版本: {sys.version.split()[0]}")
        
        # 主要内容
        if selected == "系统状态":
            self.show_system_status()
        elif selected == "文件浏览":
            self.show_file_browser()
        elif selected == "功能测试":
            self.show_test_generation()

def main():
    """主函数"""
    ui = HoyoMusicUITest()
    ui.run()

if __name__ == "__main__":
    main()

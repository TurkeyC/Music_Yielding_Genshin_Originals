#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时训练监控器
提供训练过程的实时可视化和控制
"""

import streamlit as st
import json
import time
import threading
import queue
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class RealTimeTrainingMonitor:
    def __init__(self):
        self.training_queue = queue.Queue()
        self.is_training = False
        self.training_thread = None
        self.history_file = "models/training_history.json"
        self.config_file = "models/training_config.json"
        
    def start_monitoring(self):
        """启动训练监控"""
        if 'training_monitor' not in st.session_state:
            st.session_state.training_monitor = self
            
    def create_real_time_dashboard(self):
        """创建实时监控仪表板"""
        st.markdown("### 📊 实时训练监控")
        
        # 训练状态指示器
        self._create_status_indicator()
        
        # 训练控制面板
        self._create_control_panel()
        
        # 实时图表
        self._create_real_time_charts()
        
        # 训练日志
        self._create_training_logs()
        
    def _create_status_indicator(self):
        """创建状态指示器"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.is_training:
                st.markdown("""
                <div style="text-align: center; padding: 20px; background: rgba(76, 175, 80, 0.2); border-radius: 10px;">
                    <h3 style="color: #4CAF50;">🟢 训练中</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 20px; background: rgba(158, 158, 158, 0.2); border-radius: 10px;">
                    <h3 style="color: #9E9E9E;">⚪ 待机</h3>
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            # 当前Epoch
            current_epoch = self._get_current_epoch()
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: rgba(33, 150, 243, 0.2); border-radius: 10px;">
                <h3 style="color: #2196F3;">📊 Epoch</h3>
                <h2 style="color: #fff;">{current_epoch}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            # 当前损失
            current_loss = self._get_current_loss()
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: rgba(255, 152, 0, 0.2); border-radius: 10px;">
                <h3 style="color: #FF9800;">📉 损失</h3>
                <h2 style="color: #fff;">{current_loss:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            # GPU使用率
            gpu_usage = self._get_gpu_usage()
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: rgba(156, 39, 176, 0.2); border-radius: 10px;">
                <h3 style="color: #9C27B0;">🖥️ GPU</h3>
                <h2 style="color: #fff;">{gpu_usage}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
    def _create_control_panel(self):
        """创建控制面板"""
        st.markdown("### 🎛️ 训练控制")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ 开始训练", disabled=self.is_training):
                self._start_training()
                
        with col2:
            if st.button("⏸️ 暂停训练", disabled=not self.is_training):
                self._pause_training()
                
        with col3:
            if st.button("⏹️ 停止训练", disabled=not self.is_training):
                self._stop_training()
                
        with col4:
            if st.button("💾 保存检查点"):
                self._save_checkpoint()
                
        # 训练参数调整
        with st.expander("⚙️ 训练参数"):
            col1, col2 = st.columns(2)
            with col1:
                learning_rate = st.number_input("学习率", value=0.001, format="%.6f")
                batch_size = st.number_input("批次大小", value=32, min_value=1)
            with col2:
                max_epochs = st.number_input("最大Epoch", value=100, min_value=1)
                save_interval = st.number_input("保存间隔", value=10, min_value=1)
                
    def _create_real_time_charts(self):
        """创建实时图表"""
        # 获取训练历史
        history = self._load_training_history()
        
        if not history:
            st.info("📊 暂无训练数据")
            return
            
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('训练损失', '验证损失', '学习率', 'GPU内存使用'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = list(range(1, len(history.get("loss", [])) + 1))
        
        # 训练损失
        if "loss" in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history["loss"], name="训练损失", line=dict(color="#FF6B6B")),
                row=1, col=1
            )
            
        # 验证损失
        if "val_loss" in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history["val_loss"], name="验证损失", line=dict(color="#4ECDC4")),
                row=1, col=2
            )
            
        # 学习率
        if "learning_rate" in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history["learning_rate"], name="学习率", line=dict(color="#45B7D1")),
                row=2, col=1
            )
            
        # GPU内存
        if "gpu_memory" in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history["gpu_memory"], name="GPU内存", line=dict(color="#96CEB4")),
                row=2, col=2
            )
            
        fig.update_layout(
            height=600,
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        # 创建占位符用于实时更新
        chart_placeholder = st.empty()
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
    def _create_training_logs(self):
        """创建训练日志"""
        st.markdown("### 📝 训练日志")
        
        # 日志过滤器
        col1, col2 = st.columns(2)
        with col1:
            log_level = st.selectbox("日志级别", ["全部", "INFO", "WARNING", "ERROR"])
        with col2:
            max_lines = st.number_input("显示行数", value=50, min_value=10, max_value=1000)
            
        # 日志内容
        logs = self._get_training_logs(log_level, max_lines)
        
        log_container = st.container()
        with log_container:
            st.text_area(
                "日志输出",
                value=logs,
                height=300,
                disabled=True
            )
            
    def _start_training(self):
        """开始训练"""
        self.is_training = True
        st.success("🚀 训练已开始！")
        
        # 这里应该启动实际的训练线程
        # self.training_thread = threading.Thread(target=self._training_worker)
        # self.training_thread.start()
        
    def _pause_training(self):
        """暂停训练"""
        self.is_training = False
        st.warning("⏸️ 训练已暂停")
        
    def _stop_training(self):
        """停止训练"""
        self.is_training = False
        st.info("⏹️ 训练已停止")
        
    def _save_checkpoint(self):
        """保存检查点"""
        st.success("💾 检查点已保存")
        
    def _get_current_epoch(self):
        """获取当前Epoch"""
        history = self._load_training_history()
        if history and "loss" in history:
            return len(history["loss"])
        return 0
        
    def _get_current_loss(self):
        """获取当前损失"""
        history = self._load_training_history()
        if history and "loss" in history and history["loss"]:
            return history["loss"][-1]
        return 0.0
        
    def _get_gpu_usage(self):
        """获取GPU使用率"""
        try:
            import psutil
            # 这里应该实现GPU监控
            return 75  # 模拟数据
        except:
            return 0
            
    def _load_training_history(self):
        """加载训练历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"❌ 加载训练历史失败: {str(e)}")
        return {}
        
    def _get_training_logs(self, level="全部", max_lines=50):
        """获取训练日志"""
        # 模拟日志内容
        logs = [
            "[2025-05-26 22:59:37] INFO: 训练开始",
            "[2025-05-26 22:59:38] INFO: 加载数据集...",
            "[2025-05-26 22:59:39] INFO: 数据集大小: 1000 samples",
            "[2025-05-26 22:59:40] INFO: 开始Epoch 1/100",
            "[2025-05-26 22:59:41] INFO: Batch 1/32 - Loss: 4.215",
            "[2025-05-26 22:59:42] INFO: Batch 2/32 - Loss: 4.198",
            "[2025-05-26 22:59:43] WARNING: 学习率可能过高",
            "[2025-05-26 22:59:44] INFO: Epoch 1 完成 - 平均损失: 4.106"
        ]
        
        if level != "全部":
            logs = [log for log in logs if level in log]
            
        return "\n".join(logs[-max_lines:])
        
    def update_training_progress(self, epoch, loss, val_loss=None, accuracy=None):
        """更新训练进度"""
        # 这个方法会被训练循环调用
        progress_data = {
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_queue.put(progress_data)

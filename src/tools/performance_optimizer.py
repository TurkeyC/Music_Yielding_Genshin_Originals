#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化和缓存管理器
提供模型缓存、结果缓存和性能监控功能
"""

import streamlit as st
import os
import json
import pickle
import hashlib
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

class PerformanceOptimizer:
    def __init__(self):
        self.cache_dir = Path("hoyomusic_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.model_cache = {}
        self.result_cache = {}
        self.performance_metrics = {
            "generation_times": [],
            "memory_usage": [],
            "gpu_usage": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    def enable_caching(self):
        """启用缓存优化"""
        st.cache_data.clear()
        st.cache_resource.clear()
        
    @st.cache_resource
    def load_model_cached(self, model_path):
        """缓存模型加载"""
        try:
            start_time = time.time()
            
            # 检查模型是否已在内存中
            model_hash = self._get_file_hash(model_path)
            if model_hash in self.model_cache:
                self.performance_metrics["cache_hits"] += 1
                return self.model_cache[model_hash]
            
            # 加载模型
            from model import HoyoMusicGenerator
            model = HoyoMusicGenerator()
            model.load_model(model_path)
            
            # 缓存模型
            self.model_cache[model_hash] = model
            self.performance_metrics["cache_misses"] += 1
            
            load_time = time.time() - start_time
            self.performance_metrics["generation_times"].append({
                "type": "model_load",
                "time": load_time,
                "timestamp": datetime.now().isoformat()
            })
            
            return model
            
        except Exception as e:
            st.error(f"❌ 模型加载失败: {str(e)}")
            return None
            
    @st.cache_data(ttl=3600)  # 缓存1小时
    def generate_music_cached(self, region, style, length, temperature, seed, model_path):
        """缓存音乐生成结果"""
        try:
            start_time = time.time()
            
            # 生成缓存键
            cache_key = self._generate_cache_key(region, style, length, temperature, seed)
            
            # 检查缓存
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.performance_metrics["cache_hits"] += 1
                return cached_result
            
            # 加载模型
            model = self.load_model_cached(model_path)
            if not model:
                return None
                
            # 生成音乐
            result = model.generate(
                length=length,
                temperature=temperature,
                seed=seed,
                region=region,
                style=style
            )
            
            # 缓存结果
            self._cache_result(cache_key, result)
            self.performance_metrics["cache_misses"] += 1
            
            generation_time = time.time() - start_time
            self.performance_metrics["generation_times"].append({
                "type": "music_generation",
                "time": generation_time,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            st.error(f"❌ 音乐生成失败: {str(e)}")
            return None
            
    def monitor_system_resources(self):
        """监控系统资源使用"""
        try:
            # CPU和内存使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            }
            
            # GPU使用率（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_total = torch.cuda.max_memory_allocated() / (1024**3)
                    metrics.update({
                        "gpu_memory_gb": gpu_memory,
                        "gpu_memory_total_gb": gpu_memory_total,
                        "gpu_utilization": gpu_memory / gpu_memory_total * 100 if gpu_memory_total > 0 else 0
                    })
            except:
                pass
                
            self.performance_metrics["memory_usage"].append(metrics)
            
            # 保持最近100条记录
            if len(self.performance_metrics["memory_usage"]) > 100:
                self.performance_metrics["memory_usage"] = self.performance_metrics["memory_usage"][-100:]
                
            return metrics
            
        except Exception as e:
            st.error(f"❌ 资源监控失败: {str(e)}")
            return {}
            
    def create_performance_dashboard(self):
        """创建性能监控仪表板"""
        st.markdown("### ⚡ 性能监控")
        
        # 实时资源使用
        current_metrics = self.monitor_system_resources()
        
        if current_metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cpu_color = self._get_usage_color(current_metrics.get("cpu_percent", 0))
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: rgba({cpu_color}, 0.2); border-radius: 10px;">
                    <h3 style="color: rgb({cpu_color});">🖥️ CPU</h3>
                    <h2 style="color: #fff;">{current_metrics.get("cpu_percent", 0):.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                memory_color = self._get_usage_color(current_metrics.get("memory_percent", 0))
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: rgba({memory_color}, 0.2); border-radius: 10px;">
                    <h3 style="color: rgb({memory_color});">💾 内存</h3>
                    <h2 style="color: #fff;">{current_metrics.get("memory_percent", 0):.1f}%</h2>
                    <small>{current_metrics.get("memory_used_gb", 0):.1f}GB / {current_metrics.get("memory_total_gb", 0):.1f}GB</small>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                gpu_utilization = current_metrics.get("gpu_utilization", 0)
                gpu_color = self._get_usage_color(gpu_utilization)
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: rgba({gpu_color}, 0.2); border-radius: 10px;">
                    <h3 style="color: rgb({gpu_color});">🎮 GPU</h3>
                    <h2 style="color: #fff;">{gpu_utilization:.1f}%</h2>
                    <small>{current_metrics.get("gpu_memory_gb", 0):.1f}GB</small>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                cache_ratio = self._calculate_cache_hit_ratio()
                cache_color = self._get_cache_color(cache_ratio)
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: rgba({cache_color}, 0.2); border-radius: 10px;">
                    <h3 style="color: rgb({cache_color});">📦 缓存</h3>
                    <h2 style="color: #fff;">{cache_ratio:.1f}%</h2>
                    <small>命中率</small>
                </div>
                """, unsafe_allow_html=True)
                
        # 性能趋势图
        self._create_performance_charts()
        
        # 优化建议
        self._create_optimization_suggestions()
        
    def _create_performance_charts(self):
        """创建性能趋势图"""
        if not self.performance_metrics["memory_usage"]:
            return
            
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 获取最近数据
        recent_data = self.performance_metrics["memory_usage"][-20:]
        timestamps = [datetime.fromisoformat(d["timestamp"]) for d in recent_data]
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU使用率', '内存使用率', 'GPU使用率', '生成时间'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU使用率
        cpu_data = [d.get("cpu_percent", 0) for d in recent_data]
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_data, name="CPU", line=dict(color="#FF6B6B")),
            row=1, col=1
        )
        
        # 内存使用率
        memory_data = [d.get("memory_percent", 0) for d in recent_data]
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_data, name="内存", line=dict(color="#4ECDC4")),
            row=1, col=2
        )
        
        # GPU使用率
        gpu_data = [d.get("gpu_utilization", 0) for d in recent_data]
        fig.add_trace(
            go.Scatter(x=timestamps, y=gpu_data, name="GPU", line=dict(color="#45B7D1")),
            row=2, col=1
        )
        
        # 生成时间
        generation_times = [gt["time"] for gt in self.performance_metrics["generation_times"][-10:]]
        gen_timestamps = [datetime.fromisoformat(gt["timestamp"]) for gt in self.performance_metrics["generation_times"][-10:]]
        if generation_times:
            fig.add_trace(
                go.Scatter(x=gen_timestamps, y=generation_times, name="生成时间", line=dict(color="#96CEB4")),
                row=2, col=2
            )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _create_optimization_suggestions(self):
        """创建优化建议"""
        st.markdown("### 💡 性能优化建议")
        
        suggestions = []
        
        # 基于缓存命中率的建议
        cache_ratio = self._calculate_cache_hit_ratio()
        if cache_ratio < 50:
            suggestions.append({
                "type": "warning",
                "title": "缓存命中率较低",
                "suggestion": "考虑增加缓存大小或优化缓存策略"
            })
            
        # 基于内存使用的建议
        if self.performance_metrics["memory_usage"]:
            avg_memory = sum(d.get("memory_percent", 0) for d in self.performance_metrics["memory_usage"][-10:]) / 10
            if avg_memory > 80:
                suggestions.append({
                    "type": "error",
                    "title": "内存使用率过高",
                    "suggestion": "建议关闭其他应用程序或增加系统内存"
                })
            elif avg_memory > 60:
                suggestions.append({
                    "type": "warning",
                    "title": "内存使用率较高",
                    "suggestion": "注意监控内存使用，避免系统卡顿"
                })
                
        # 基于生成时间的建议
        if self.performance_metrics["generation_times"]:
            avg_time = sum(gt["time"] for gt in self.performance_metrics["generation_times"][-5:]) / 5
            if avg_time > 30:
                suggestions.append({
                    "type": "warning",
                    "title": "生成时间较长",
                    "suggestion": "考虑使用GPU加速或减少生成长度"
                })
                
        # 显示建议
        if suggestions:
            for suggestion in suggestions:
                if suggestion["type"] == "error":
                    st.error(f"🚨 {suggestion['title']}: {suggestion['suggestion']}")
                elif suggestion["type"] == "warning":
                    st.warning(f"⚠️ {suggestion['title']}: {suggestion['suggestion']}")
                else:
                    st.info(f"💡 {suggestion['title']}: {suggestion['suggestion']}")
        else:
            st.success("✅ 系统性能良好，无需优化建议")
            
    def cleanup_cache(self, max_age_hours=24):
        """清理过期缓存"""
        try:
            cleaned = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # 清理文件缓存
            for cache_file in self.cache_dir.glob("*.cache"):
                if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                    cache_file.unlink()
                    cleaned += 1
                    
            # 清理内存缓存
            self.result_cache.clear()
            
            st.success(f"🧹 已清理 {cleaned} 个过期缓存文件")
            
        except Exception as e:
            st.error(f"❌ 缓存清理失败: {str(e)}")
            
    def _get_file_hash(self, file_path):
        """获取文件哈希值"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
        
    def _generate_cache_key(self, *args):
        """生成缓存键"""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def _get_cached_result(self, cache_key):
        """获取缓存结果"""
        # 检查内存缓存
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
            
        # 检查文件缓存
        cache_file = self.cache_dir / f"{cache_key}.cache"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                # 同时加载到内存缓存
                self.result_cache[cache_key] = result
                return result
            except:
                # 缓存文件损坏，删除
                cache_file.unlink()
                
        return None
        
    def _cache_result(self, cache_key, result):
        """缓存结果"""
        # 内存缓存
        self.result_cache[cache_key] = result
        
        # 文件缓存
        cache_file = self.cache_dir / f"{cache_key}.cache"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except:
            pass  # 缓存失败不影响主要功能
            
    def _get_usage_color(self, usage_percent):
        """根据使用率获取颜色"""
        if usage_percent > 80:
            return "244, 67, 54"  # 红色
        elif usage_percent > 60:
            return "255, 152, 0"  # 橙色
        else:
            return "76, 175, 80"  # 绿色
            
    def _get_cache_color(self, hit_ratio):
        """根据缓存命中率获取颜色"""
        if hit_ratio > 70:
            return "76, 175, 80"  # 绿色
        elif hit_ratio > 40:
            return "255, 152, 0"  # 橙色
        else:
            return "244, 67, 54"  # 红色
            
    def _calculate_cache_hit_ratio(self):
        """计算缓存命中率"""
        total = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
        if total == 0:
            return 0
        return (self.performance_metrics["cache_hits"] / total) * 100
        
    def save_performance_report(self):
        """保存性能报告"""
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "metrics": self.performance_metrics,
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "platform": os.name
                }
            }
            
            # 添加GPU信息
            try:
                import torch
                if torch.cuda.is_available():
                    report["system_info"]["gpu_name"] = torch.cuda.get_device_name(0)
                    report["system_info"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                pass
                
            report_file = Path("logs") / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
            st.success(f"📊 性能报告已保存: {report_file}")
            
        except Exception as e:
            st.error(f"❌ 保存性能报告失败: {str(e)}")

# 全局性能优化器实例
performance_optimizer = PerformanceOptimizer()

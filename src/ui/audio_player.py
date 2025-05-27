#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频播放器组件
提供ABC和MIDI文件的播放功能
"""

import streamlit as st
import os
import base64
from pathlib import Path
import tempfile
from tools.abc_to_midi import ABCToMIDIConverter

class AudioPlayer:
    def __init__(self):
        self.converter = ABCToMIDIConverter()
        
    def create_audio_player(self, file_path, file_type="abc"):
        """创建音频播放器组件"""
        if not os.path.exists(file_path):
            st.error(f"❌ 文件不存在: {file_path}")
            return
            
        try:
            if file_type == "abc":
                # 转换ABC为MIDI
                midi_path = self._abc_to_midi(file_path)
                if midi_path:
                    self._create_midi_player(midi_path)
                else:
                    st.warning("⚠️ ABC转MIDI失败，显示文本内容")
                    self._display_abc_content(file_path)
            elif file_type in ["midi", "mid"]:
                self._create_midi_player(file_path)
            else:
                st.error(f"❌ 不支持的文件类型: {file_type}")
                
        except Exception as e:
            st.error(f"❌ 播放器创建失败: {str(e)}")
            
    def _abc_to_midi(self, abc_path):
        """将ABC文件转换为MIDI"""
        try:
            output_dir = os.path.dirname(abc_path)
            midi_filename = os.path.splitext(os.path.basename(abc_path))[0] + ".mid"
            midi_path = os.path.join(output_dir, midi_filename)
            
            # 如果MIDI文件已存在，直接返回
            if os.path.exists(midi_path):
                return midi_path
                
            # 转换ABC为MIDI
            if self.converter.convert_abc_to_midi(abc_path, midi_path):
                return midi_path
            else:
                return None
                
        except Exception as e:
            st.error(f"❌ ABC转MIDI失败: {str(e)}")
            return None
            
    def _create_midi_player(self, midi_path):
        """创建MIDI播放器"""
        try:
            # 读取MIDI文件
            with open(midi_path, "rb") as f:
                midi_data = f.read()
                
            # 转换为base64
            midi_b64 = base64.b64encode(midi_data).decode()
            
            # 创建HTML音频播放器
            audio_html = f"""
            <div style="margin: 10px 0;">
                <audio controls style="width: 100%; max-width: 400px;">
                    <source src="data:audio/midi;base64,{midi_b64}" type="audio/midi">
                    您的浏览器不支持音频播放。
                </audio>
            </div>
            """
            
            st.markdown(audio_html, unsafe_allow_html=True)
            
            # 添加下载按钮
            st.download_button(
                label="📥 下载MIDI文件",
                data=midi_data,
                file_name=os.path.basename(midi_path),
                mime="audio/midi"
            )
            
        except Exception as e:
            st.error(f"❌ MIDI播放器创建失败: {str(e)}")
            
    def _display_abc_content(self, abc_path):
        """显示ABC文件内容"""
        try:
            with open(abc_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            st.text_area(
                "📝 ABC记谱内容",
                content,
                height=200,
                disabled=True
            )
            
            # 添加下载按钮
            st.download_button(
                label="📥 下载ABC文件",
                data=content,
                file_name=os.path.basename(abc_path),
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"❌ 读取ABC文件失败: {str(e)}")
            
    def create_music_gallery(self, music_dir="generated_music"):
        """创建音乐画廊"""
        st.markdown("### 🎵 音乐作品集")
        
        if not os.path.exists(music_dir):
            st.info("📁 暂无生成的音乐文件")
            return
            
        # 获取所有音乐文件
        music_files = []
        for ext in ['.abc', '.mid', '.midi']:
            music_files.extend(Path(music_dir).glob(f"*{ext}"))
            
        if not music_files:
            st.info("🎵 暂无音乐文件")
            return
            
        # 按修改时间排序
        music_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 创建音乐卡片
        cols = st.columns(2)
        for i, file_path in enumerate(music_files[:10]):  # 显示最新10个
            with cols[i % 2]:
                self._create_music_card(file_path)
                
    def _create_music_card(self, file_path):
        """创建音乐卡片"""
        file_name = file_path.name
        file_size = file_path.stat().st_size
        file_time = file_path.stat().st_mtime
        
        # 解析文件名获取信息
        name_parts = file_name.split('_')
        region = name_parts[0] if len(name_parts) > 0 else "未知"
        style = name_parts[1] if len(name_parts) > 1 else "未知风格"
        
        # 创建卡片
        with st.container():
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 15px;
                margin: 10px 0;
                border: 1px solid rgba(255,255,255,0.2);
            ">
                <h4 style="margin: 0; color: #fff;">🎵 {region} - {style}</h4>
                <p style="margin: 5px 0; color: #ccc; font-size: 12px;">
                    📁 {file_name}<br>
                    📏 {file_size} bytes<br>
                    🕐 {self._format_time(file_time)}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 播放器
            self.create_audio_player(str(file_path), file_path.suffix[1:])
            
    def _format_time(self, timestamp):
        """格式化时间戳"""
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")

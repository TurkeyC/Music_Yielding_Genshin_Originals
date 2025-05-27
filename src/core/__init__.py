"""
HoyoMusic 核心模块
包含AI模型、数据处理和音乐生成的核心功能
"""
from .model import HoyoMusicGenerator
from .data_processor import HoyoMusicDataProcessor

__all__ = ['HoyoMusicGenerator', 'HoyoMusicDataProcessor']

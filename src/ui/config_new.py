# HoyoMusic AI Generator - UI Configuration
# Glassmorphism配色方案和主题设置

class UIConfig:
    """UI配置类"""
    
    # 主色调
    PRIMARY_COLORS = {
        "primary": "#673AB7",      # 深紫色
        "secondary": "#3F51B5",    # 靛蓝色
        "accent": "#2196F3",       # 蓝色
        "success": "#4CAF50",      # 绿色
        "warning": "#FF9800",      # 橙色
        "error": "#F44336",        # 红色
        "info": "#00BCD4",         # 青色
    }

    # 原神区域主题色
    REGION_THEMES = {
        "蒙德": {
            "primary": "#5DADE2",    # 天蓝色
            "secondary": "#85C1E9",  # 浅蓝色
            "accent": "#A9CCE3",     # 更浅蓝色
            "emoji": "🌬️",
            "description": "自由之风的故乡"
        },
        "璃月": {
            "primary": "#F7DC6F",    # 金黄色
            "secondary": "#F4D03F",  # 深金色
            "accent": "#F8C471",     # 橙金色
            "emoji": "🏔️",
            "description": "岩之神的契约之地"
        },
        "稻妻": {
            "primary": "#BB8FCE",    # 紫色
            "secondary": "#A569BD",  # 深紫色
            "accent": "#8E44AD",     # 更深紫色
            "emoji": "⚡",
            "description": "雷电将军的永恒之国"
        },
        "须弥": {
            "primary": "#82E0AA",    # 翠绿色
            "secondary": "#58D68D",  # 深绿色
            "accent": "#2ECC71",     # 更深绿色
            "emoji": "🌿",
            "description": "智慧之神的学者之国"
        },
        "枫丹": {
            "primary": "#85C1E9",    # 水蓝色
            "secondary": "#5DADE2",  # 深蓝色
            "accent": "#3498DB",     # 更深蓝色
            "emoji": "💧",
            "description": "正义之神的法庭之地"
        }
    }

    # 音乐风格预设
    MUSIC_STYLES = {
        "史诗战斗": {
            "emoji": "⚔️",
            "temperature": 1.2,
            "top_k": 35,
            "description": "激昂的战斗音乐，充满力量感",
            "tags": ["战斗", "激昂", "史诗"]
        },
        "宁静探索": {
            "emoji": "🌅",
            "temperature": 0.8,
            "top_k": 50,
            "description": "平和的探索音乐，营造轻松氛围",
            "tags": ["探索", "宁静", "放松"]
        },
        "欢快庆典": {
            "emoji": "🎉",
            "temperature": 1.0,
            "top_k": 40,
            "description": "欢乐的庆典音乐，节奏明快",
            "tags": ["庆典", "欢快", "节日"]
        },
        "神秘氛围": {
            "emoji": "🌙",
            "temperature": 0.9,
            "top_k": 30,
            "description": "神秘的环境音乐，营造悬疑感",
            "tags": ["神秘", "氛围", "悬疑"]
        },
        "悲伤回忆": {
            "emoji": "💧",
            "temperature": 0.7,
            "top_k": 45,
            "description": "感人的回忆音乐，触动心弦",
            "tags": ["悲伤", "回忆", "感人"]
        }
    }

    # UI布局配置
    UI_CONFIG = {
        "page_title": "🎵 HoyoMusic AI Generator",
        "page_icon": "🎮",
        "layout": "wide",
        "sidebar_state": "expanded",
        "theme_color": "#673AB7",
        "background_gradient": "linear-gradient(135deg, rgba(103, 58, 183, 0.05), rgba(63, 81, 181, 0.05), rgba(33, 150, 243, 0.05), rgba(0, 188, 212, 0.05), rgba(76, 175, 80, 0.05))"
    }

    # 功能模块配置
    MODULES = {
        "🎵 音乐生成": {
            "icon": "🎵",
            "description": "使用AI生成原神风格音乐",
            "status": "active",
            "features": ["区域风格选择", "参数调节", "实时生成", "格式转换"]
        },
        "🎓 模型训练": {
            "icon": "🎓", 
            "description": "训练自定义音乐生成模型",
            "status": "active",
            "features": ["数据预处理", "超参数调节", "分布式训练", "检查点保存"]
        },
        "📊 训练监控": {
            "icon": "📊",
            "description": "实时监控训练进度和性能",
            "status": "active",
            "features": ["实时图表", "性能指标", "资源监控", "日志查看"]
        },
        "⚙️ 模型管理": {
            "icon": "⚙️",
            "description": "管理和对比不同版本的模型",
            "status": "coming_soon",
            "features": ["版本控制", "性能对比", "模型融合", "导入导出"]
        },
        "🔧 工具箱": {
            "icon": "🔧",
            "description": "音乐处理和分析工具集合",
            "status": "coming_soon", 
            "features": ["ABC编辑器", "MIDI播放器", "格式转换", "音乐分析"]
        },
        "📖 帮助文档": {
            "icon": "📖",
            "description": "详细的使用指南和API文档",
            "status": "coming_soon",
            "features": ["使用教程", "API文档", "常见问题", "技术支持"]
        }
    }

    # 性能监控配置
    MONITORING_CONFIG = {
        "refresh_interval": 5,  # 秒
        "max_history_points": 100,
        "chart_height": 400,
        "auto_refresh": True
    }

    # 文件管理配置
    FILE_CONFIG = {
        "upload_max_size": 100,  # MB
        "supported_formats": [".abc", ".midi", ".mid", ".txt"],
        "download_formats": [".abc", ".midi", ".mid", ".wav"],
        "preview_enabled": True
    }

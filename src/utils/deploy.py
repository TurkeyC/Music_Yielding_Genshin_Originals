#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键部署脚本
自动配置项目环境并启动UI
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

class HoyoMusicDeployer:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.venv_dir = self.project_dir / ".venv"
        self.requirements_file = self.project_dir / "requirements.txt"
        self.config_file = self.project_dir / "deploy_config.json"
        
    def check_system_requirements(self):
        """检查系统要求"""
        print("🔍 检查系统要求...")
        
        # Python版本检查
        python_version = sys.version_info
        if python_version < (3, 8):
            print("❌ Python版本过低，需要3.8+")
            return False
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 操作系统检查
        os_name = platform.system()
        print(f"✅ 操作系统: {os_name}")
        
        # GPU检查
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print("⚠️ 未检测到CUDA GPU，将使用CPU模式")
        except ImportError:
            print("⚠️ PyTorch未安装，稍后自动安装")
            
        return True
        
    def setup_virtual_environment(self):
        """设置虚拟环境"""
        print("🐍 设置Python虚拟环境...")
        
        if not self.venv_dir.exists():
            print("创建新的虚拟环境...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)
        else:
            print("使用现有虚拟环境...")
            
        # 获取虚拟环境的Python路径
        if platform.system() == "Windows":
            venv_python = self.venv_dir / "Scripts" / "python.exe"
            venv_pip = self.venv_dir / "Scripts" / "pip.exe"
        else:
            venv_python = self.venv_dir / "bin" / "python"
            venv_pip = self.venv_dir / "bin" / "pip"
            
        # 升级pip
        print("升级pip...")
        subprocess.run([str(venv_pip), "install", "--upgrade", "pip"], check=True)
        
        return str(venv_python), str(venv_pip)
        
    def install_dependencies(self, pip_path):
        """安装依赖包"""
        print("📦 安装项目依赖...")
        
        if self.requirements_file.exists():
            subprocess.run([pip_path, "install", "-r", str(self.requirements_file)], check=True)
        else:
            # 基础依赖列表
            basic_deps = [
                "torch>=2.0.0",
                "streamlit>=1.28.0",
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "plotly>=5.0.0",
                "streamlit-option-menu>=0.3.0"
            ]
            
            for dep in basic_deps:
                print(f"安装 {dep}...")
                subprocess.run([pip_path, "install", dep], check=True)
                
    def setup_project_structure(self):
        """设置项目结构"""
        print("📁 设置项目目录结构...")
        
        dirs_to_create = [
            "models",
            "generated_music", 
            "data/abc_files",
            "logs",
            "temp",
            "models/backups",
            "models/versions"
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.project_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {dir_path}")
            
    def create_demo_data(self):
        """创建演示数据"""
        print("🎵 生成演示数据...")
        
        demo_abc = """X:1
T:Mondstadt Demo
M:4/4
L:1/8
K:C
|: C D E F | G A B c | c B A G | F E D C :|
|: E F G A | B c d e | e d c B | A G F E :|"""
        
        demo_file = self.project_dir / "generated_music" / "mondstadt_demo.abc"
        with open(demo_file, 'w', encoding='utf-8') as f:
            f.write(demo_abc)
            
        print("✅ 演示数据创建完成")
        
    def create_config_file(self):
        """创建配置文件"""
        print("⚙️ 创建配置文件...")
        
        config = {
            "deployment": {
                "host": "localhost",
                "port": 8501,
                "debug": True,
                "auto_reload": True
            },
            "model": {
                "default_model": "hoyomusic_generator.pth",
                "model_dir": "models",
                "cache_dir": "hoyomusic_cache"
            },
            "generation": {
                "default_length": 800,
                "default_temperature": 1.0,
                "max_length": 2000,
                "supported_regions": ["蒙德", "璃月", "稻妻", "须弥", "枫丹"]
            },
            "ui": {
                "theme": "glassmorphism",
                "language": "zh-cn",
                "enable_animations": True
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        print("✅ 配置文件创建完成")
        
    def create_startup_scripts(self, python_path):
        """创建启动脚本"""
        print("🚀 创建启动脚本...")
        
        # Windows批处理文件
        bat_content = f'''@echo off
echo 🎵 启动 HoyoMusic AI Generator...
cd /d "{self.project_dir}"
"{python_path}" -m streamlit run app.py --server.port 8501
pause
'''
        
        bat_file = self.project_dir / "start_enhanced_ui.bat"
        with open(bat_file, 'w', encoding='utf-8') as f:
            f.write(bat_content)
            
        # Python启动脚本
        py_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版UI启动脚本
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("🎵 启动 HoyoMusic AI Generator 增强版...")
    print("🌐 UI地址: http://localhost:8501")
    print("📚 使用帮助: 查看侧边栏的帮助文档")
    print("=" * 50)
    
    try:
        subprocess.run([
            "{python_path}", "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--theme.base", "dark"
        ])
    except KeyboardInterrupt:
        print("\\n👋 感谢使用 HoyoMusic AI Generator!")
    except Exception as e:
        print(f"❌ 启动失败: {{e}}")
        input("按任意键退出...")

if __name__ == "__main__":
    main()
'''
        
        py_file = self.project_dir / "start_enhanced_ui.py"
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(py_content)
            
        print("✅ 启动脚本创建完成")
        
    def run_deployment(self):
        """执行部署"""
        print("🚀 开始部署 HoyoMusic AI Generator 增强版...")
        print("=" * 60)
        
        try:
            # 1. 检查系统要求
            if not self.check_system_requirements():
                return False
                
            # 2. 设置虚拟环境
            python_path, pip_path = self.setup_virtual_environment()
            
            # 3. 安装依赖
            self.install_dependencies(pip_path)
            
            # 4. 设置项目结构
            self.setup_project_structure()
            
            # 5. 创建演示数据
            self.create_demo_data()
            
            # 6. 创建配置文件
            self.create_config_file()
            
            # 7. 创建启动脚本
            self.create_startup_scripts(python_path)
            
            print("=" * 60)
            print("🎉 部署完成!")
            print(f"📁 项目目录: {self.project_dir}")
            print("🚀 启动方式:")
            print("   - Windows: 双击 start_enhanced_ui.bat")
            print("   - Python:  python start_enhanced_ui.py")
            print("🌐 访问地址: http://localhost:8501")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ 部署失败: {e}")
            return False

def main():
    """主函数"""
    deployer = HoyoMusicDeployer()
    
    print("🎵 HoyoMusic AI Generator 增强版部署工具")
    print("=" * 60)
    
    if deployer.run_deployment():
        # 询问是否立即启动
        response = input("\\n🚀 是否立即启动UI? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是', '']:
            print("\\n🌟 启动UI中...")
            try:
                os.system(f'python "{deployer.project_dir}/start_enhanced_ui.py"')
            except:
                print("请手动运行启动脚本")
    else:
        print("\\n❌ 部署失败，请检查错误信息")
        input("按任意键退出...")

if __name__ == "__main__":
    main()

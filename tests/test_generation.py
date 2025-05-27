#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoyoMusic AI 音乐生成器 - 快速生成测试
验证音乐生成功能是否正常工作
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_music_generation():
    """测试音乐生成功能"""
    print("🎵 测试音乐生成功能...")
    
    try:
        from src.core.model import HoyoMusicGenerator
        
        # 初始化生成器
        print("📥 初始化生成器...")
        generator = HoyoMusicGenerator()
        
        # 加载模型
        model_path = project_root / "models" / "hoyomusic_generator.pth"
        mappings_path = project_root / "models" / "hoyomusic_mappings.pkl"
        
        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
            
        if not mappings_path.exists():
            print(f"❌ 映射文件不存在: {mappings_path}")
            return False
            
        print("📂 加载模型和映射...")
        generator.load_model(str(model_path), str(mappings_path))
        
        # 生成音乐
        print("🎼 生成测试音乐...")
        
        # 测试参数
        region = "蒙德"
        emotion = "欢快庆典"
        length = 50
        temperature = 0.8
        
        generated_abc = generator.generate_music(
            region=region,
            emotion=emotion, 
            length=length,
            temperature=temperature
        )
        
        if generated_abc and len(generated_abc.strip()) > 0:
            print("✅ 音乐生成成功!")
            print(f"📊 生成长度: {len(generated_abc)} 字符")
            
            # 保存测试结果
            output_dir = project_root / "output" / "generated"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            test_file = output_dir / f"test_generation_{region}_{emotion}.abc"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(generated_abc)
            
            print(f"💾 测试文件已保存: {test_file}")
            
            # 显示前几行预览
            lines = generated_abc.split('\n')[:10]
            print("\n🎼 生成预览:")
            for line in lines:
                if line.strip():
                    print(f"   {line}")
            
            return True
        else:
            print("❌ 音乐生成失败 - 返回空内容")
            return False
            
    except Exception as e:
        print(f"❌ 音乐生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎵 HoyoMusic AI 音乐生成器 - 生成测试")
    print("=" * 50)
    
    success = test_music_generation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 音乐生成测试通过！")
        print("🚀 HoyoMusic AI 音乐生成器完全可用")
    else:
        print("⚠️ 音乐生成测试失败")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

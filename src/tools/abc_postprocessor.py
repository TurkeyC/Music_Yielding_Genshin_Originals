#!/usr/bin/env python3
"""
ABC音乐后处理工具
修复生成的ABC记谱格式，确保符合标准
"""

import re
import os

class ABCPostProcessor:
    """ABC记谱后处理器"""
    
    def __init__(self):
        # ABC记谱的基本结构
        self.header_fields = ['X', 'T', 'C', 'M', 'L', 'K']
        self.valid_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'z']
        self.valid_accidentals = ['^', '=', '_']
        self.valid_octaves = [',', "'"]
        self.valid_durations = ['1', '2', '3', '4', '6', '8']
        
    def clean_abc_content(self, abc_text):
        """清理ABC内容，移除无效字符和结构"""
        lines = abc_text.split('\n')
        cleaned_lines = []
        in_header = True
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行
            if not line:
                continue
            
            # 处理头部信息
            if in_header and ':' in line:
                field, content = line.split(':', 1)
                if field.strip() in self.header_fields:
                    cleaned_lines.append(f"{field.strip()}:{content.strip()}")
                    if field.strip() == 'K':  # K字段是最后一个头部字段
                        in_header = False
                continue
            
            # 处理音乐内容
            if not in_header:
                cleaned_line = self.clean_music_line(line)
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def clean_music_line(self, line):
        """清理音乐行，移除无效字符"""
        # 移除无效字符，只保留合法的ABC记谱字符
        valid_chars = set('CDEFGABcdefgab^_=\'," |[]():123456789/z-')
        cleaned = ''.join(c for c in line if c in valid_chars or c.isspace())
        
        # 修复常见问题
        cleaned = re.sub(r'\s+', ' ', cleaned)  # 规范化空格
        cleaned = re.sub(r'\|+', '|', cleaned)  # 修复多重小节线
        cleaned = re.sub(r'(\d+)\s*(\d+)', r'\1\2', cleaned)  # 修复数字间的空格
        
        return cleaned.strip()
    
    def validate_abc_structure(self, abc_text):
        """验证ABC结构的完整性"""
        lines = abc_text.split('\n')
        
        # 检查必需的头部字段
        required_fields = ['X', 'T', 'M', 'L', 'K']
        found_fields = set()
        
        for line in lines:
            if ':' in line:
                field = line.split(':', 1)[0].strip()
                if field in required_fields:
                    found_fields.add(field)
        
        missing_fields = set(required_fields) - found_fields
        if missing_fields:
            return False, f"缺少必需字段: {', '.join(missing_fields)}"
        
        return True, "结构验证通过"
    
    def generate_standard_header(self, title="Generated Music", composer="AI", 
                                meter="4/4", length="1/8", key="C major"):
        """生成标准的ABC头部"""
        return f"""X:1
T:{title}
C:{composer}
M:{meter}
L:{length}
K:{key}"""
    
    def fix_music_content(self, content):
        """修复音乐内容的常见问题"""
        # 确保每行都有适当的小节分隔
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if line and not line.startswith(('X:', 'T:', 'C:', 'M:', 'L:', 'K:')):
                # 确保行以小节线结束
                if line and not line.endswith('|'):
                    line += '|'
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def process_generated_abc(self, abc_text, title="HoyoMusic Generated"):
        """处理生成的ABC记谱，修复格式问题"""
        try:
            # 1. 基本清理
            cleaned = self.clean_abc_content(abc_text)
            
            # 2. 验证结构
            is_valid, message = self.validate_abc_structure(cleaned)
            
            if not is_valid:
                # 如果结构无效，重新构建
                print(f"⚠️ ABC结构问题: {message}")
                print("🔧 重新构建ABC结构...")
                
                # 提取音乐内容（非头部行）
                lines = abc_text.split('\n')
                music_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not any(line.startswith(f"{field}:") for field in self.header_fields):
                        music_lines.append(line)
                
                # 清理音乐内容
                music_content = '\n'.join(music_lines)
                cleaned_music = self.fix_music_content(music_content)
                
                # 重新构建完整的ABC
                header = self.generate_standard_header(title=title)
                cleaned = f"{header}\n{cleaned_music}"
            
            # 3. 最终清理
            cleaned = self.clean_abc_content(cleaned)
            
            return cleaned, True
            
        except Exception as e:
            print(f"❌ ABC后处理失败: {e}")
            # 返回最小可用的ABC
            fallback = self.generate_standard_header(title=title) + "\nz4|z4|z4|z4|]"
            return fallback, False

def test_abc_postprocessor():
    """测试ABC后处理器"""
    print("🧪 测试ABC后处理器...")
    
    processor = ABCPostProcessor()
    
    # 测试用的坏格式ABC
    bad_abc = """X:1
T:Test Music
6][| -z1fGe |ec  e4| 4  2820|6>G]2 [[b: |]] A ge ec   g bd[4 22 fd|43[ 2x  2E  3e:x:
: Ez4G | 42  | |dD   2 ]L  |/c  G |(|3 [2| F[ d |4[g|^ -g   [4=3||/ d2BQc f|    z222/
/ |]|| B ]4|]e6  2gxg|e[  6/g-|3 |]"""
    
    print("📝 原始ABC:")
    print(bad_abc[:100] + "...")
    
    cleaned, success = processor.process_generated_abc(bad_abc, "Test Music")
    
    print(f"\n{'✅' if success else '⚠️'} 处理结果:")
    print(cleaned)
    
    # 验证结构
    is_valid, message = processor.validate_abc_structure(cleaned)
    print(f"\n📋 结构验证: {'✅' if is_valid else '❌'} {message}")

if __name__ == "__main__":
    test_abc_postprocessor()

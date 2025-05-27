from music21 import converter, midi, stream, meter, key, tempo
import os
import re

class ABCToMIDIConverter:
    def __init__(self):
        self.default_tempo = 120
        
    def clean_abc_for_conversion(self, abc_text):
        """清理ABC文本以便转换为MIDI"""
        lines = abc_text.split('\n')
        cleaned_lines = []
        current_tune = []
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行
            if not line:
                continue
                
            # 保留ABC头部信息
            if line.startswith(('X:', 'T:', 'M:', 'L:', 'K:', 'Q:', 'C:')):
                current_tune.append(line)
            # 保留音乐内容行
            elif line and not line.startswith('%'):
                # 清理音乐行，移除无效字符
                music_line = re.sub(r'[^\w\s|:\[\](){}.,/\\ -+#^=_~><]', '', line)
                if music_line.strip():
                    current_tune.append(music_line)
            
            # 如果遇到新的曲子开始，保存当前曲子
            if line.startswith('X:') and len(current_tune) > 1:
                if len(current_tune) > 4:  # 确保有足够的内容
                    cleaned_lines.extend(current_tune[:-1])  # 不包含新的X:行
                    cleaned_lines.append('')  # 添加分隔符
                current_tune = [line]  # 开始新曲子
        
        # 添加最后一个曲子
        if len(current_tune) > 4:
            cleaned_lines.extend(current_tune)
        
        return '\n'.join(cleaned_lines)
    
    def convert_abc_to_midi(self, abc_text, output_path, tune_index=0):
        """将ABC记谱转换为MIDI文件"""
        try:
            # 清理ABC文本
            cleaned_abc = self.clean_abc_for_conversion(abc_text)
            
            # 如果文本太短，返回错误
            if len(cleaned_abc.strip()) < 50:
                raise ValueError("ABC文本太短，无法转换")
            
            # 尝试解析ABC
            try:
                # 使用music21解析ABC
                parsed_stream = converter.parse(cleaned_abc, format='abc')
            except Exception as e:
                print(f"直接解析失败，尝试逐行解析: {e}")
                # 如果直接解析失败，尝试提取单个曲子
                tunes = self.extract_individual_tunes(cleaned_abc)
                if tunes and tune_index < len(tunes):
                    parsed_stream = converter.parse(tunes[tune_index], format='abc')
                else:
                    raise ValueError("无法解析ABC记谱")
            
            # 如果解析结果是列表，取第一个
            if isinstance(parsed_stream, list):
                if len(parsed_stream) > tune_index:
                    parsed_stream = parsed_stream[tune_index]
                else:
                    parsed_stream = parsed_stream[0]
            
            # 设置默认参数
            if not parsed_stream.getElementsByClass(tempo.TempoIndication):
                parsed_stream.insert(0, tempo.TempoIndication(number=self.default_tempo))
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 写入MIDI文件
            parsed_stream.write('midi', fp=output_path)
            
            print(f"MIDI文件已保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"转换ABC到MIDI失败: {e}")
            return False
    
    def extract_individual_tunes(self, abc_text):
        """从ABC文本中提取单个曲子"""
        tunes = []
        lines = abc_text.split('\n')
        current_tune = []
        
        for line in lines:
            line = line.strip()
            
            # 如果遇到新的曲子开始
            if line.startswith('X:'):
                # 保存之前的曲子
                if current_tune:
                    tune_text = '\n'.join(current_tune)
                    if self.is_valid_tune(tune_text):
                        tunes.append(tune_text)
                # 开始新曲子
                current_tune = [line]
            elif current_tune:  # 只有在已经开始一个曲子时才添加行
                current_tune.append(line)
        
        # 添加最后一个曲子
        if current_tune:
            tune_text = '\n'.join(current_tune)
            if self.is_valid_tune(tune_text):
                tunes.append(tune_text)
        
        return tunes
    
    def is_valid_tune(self, tune_text):
        """检查曲子是否有效"""
        # 检查必要的ABC标头
        has_x = 'X:' in tune_text
        has_k = 'K:' in tune_text
        has_music = any(char in tune_text for char in 'ABCDEFGabcdefg')
        
        return has_x and has_k and has_music and len(tune_text.strip()) > 30
    
    def batch_convert_abc_files(self, input_dir, output_dir):
        """批量转换ABC文件为MIDI"""
        if not os.path.exists(input_dir):
            print(f"输入目录不存在: {input_dir}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        converted_count = 0
        failed_count = 0
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.abc') or filename.endswith('.txt'):
                input_path = os.path.join(input_dir, filename)
                
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        abc_content = f.read()
                    
                    # 提取所有曲子
                    tunes = self.extract_individual_tunes(abc_content)
                    
                    for i, tune in enumerate(tunes):
                        base_name = os.path.splitext(filename)[0]
                        output_filename = f"{base_name}_tune_{i+1}.mid"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        if self.convert_abc_to_midi(tune, output_path):
                            converted_count += 1
                        else:
                            failed_count += 1
                
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
                    failed_count += 1
        
        print(f"转换完成！成功: {converted_count}, 失败: {failed_count}")

def convert_abc_string_to_midi(abc_string, output_path="generated_music/output.mid"):
    """便捷函数：将ABC字符串转换为MIDI文件"""
    converter_obj = ABCToMIDIConverter()
    return converter_obj.convert_abc_to_midi(abc_string, output_path)
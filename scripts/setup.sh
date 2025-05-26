#!/bin/bash

echo "正在设置ABC音乐生成器..."

# 创建项目目录
mkdir -p abc_music_generator
cd abc_music_generator

# 创建子目录
mkdir -p data/abc_files
mkdir -p models
mkdir -p generated_music

echo "项目目录创建完成！"
echo "请将提供的Python文件复制到此目录，然后运行："
echo "pip install -r requirements.txt"
echo "python train.py"
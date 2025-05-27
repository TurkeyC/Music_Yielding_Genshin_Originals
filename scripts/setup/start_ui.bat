@echo off
chcp 65001 >nul
title HoyoMusic AI Generator - Web UI

echo 🎵 HoyoMusic AI Generator
echo ========================
echo 正在启动Web界面...
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Python，请确保Python已正确安装
    pause
    exit /b 1
)

REM 启动UI
python start_ui.py

pause

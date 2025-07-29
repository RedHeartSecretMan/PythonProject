#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice应用程序配置文件
"""

import os
from pathlib import Path

# 应用程序基本信息
APP_NAME = "SenseVoice"
APP_VERSION = "1.0.0"
APP_AUTHOR = "SenseVoice Team"
APP_DESCRIPTION = "基于ONNX引擎的语音转录工具"

# 路径配置
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "stores" / "checkpoints"
SENSEVOICE_MODEL_PATH = MODEL_DIR / "SenseVoiceSmall"
VAD_MODEL_PATH = MODEL_DIR / "fsmn_vad"

# 支持的音频格式
SUPPORTED_AUDIO_FORMATS = [
    "*.wav", "*.mp3", "*.flac", "*.m4a", "*.aac", "*.ogg", "*.wma"
]

# 音频格式过滤器（用于文件对话框）
AUDIO_FILTER = "音频文件 (" + " ".join(SUPPORTED_AUDIO_FORMATS) + ");;所有文件 (*)"

# 模型配置
MODEL_CONFIG = {
    "batch_size": 1,
    "quantize": False,
    "language": "auto",  # auto, zh, en, yue, ja, ko, nospeech
    "textnorm": "withitn"
}

# UI配置
UI_CONFIG = {
    "window_width": 800,
    "window_height": 600,
    "window_min_width": 600,
    "window_min_height": 400,
    "font_size": 14,
    "title_font_size": 24
}

# 颜色主题
COLOR_THEME = {
    "primary": "#007AFF",
    "primary_hover": "#0056CC",
    "primary_pressed": "#004499",
    "background": "#f5f5f5",
    "card_background": "white",
    "border": "#e0e0e0",
    "text_primary": "#333333",
    "text_secondary": "#666666",
    "success": "#00aa00",
    "error": "#ff4444",
    "disabled": "#cccccc"
}

# 应用程序样式表
APP_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLOR_THEME['background']};
}}

QPushButton {{
    background-color: {COLOR_THEME['primary']};
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: {UI_CONFIG['font_size']}px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {COLOR_THEME['primary_hover']};
}}

QPushButton:pressed {{
    background-color: {COLOR_THEME['primary_pressed']};
}}

QPushButton:disabled {{
    background-color: {COLOR_THEME['disabled']};
    color: {COLOR_THEME['text_secondary']};
}}

QTextEdit {{
    border: 2px solid {COLOR_THEME['border']};
    border-radius: 8px;
    padding: 10px;
    font-size: {UI_CONFIG['font_size']}px;
    background-color: {COLOR_THEME['card_background']};
}}

QLabel {{
    font-size: {UI_CONFIG['font_size']}px;
    color: {COLOR_THEME['text_primary']};
}}

QProgressBar {{
    border: 2px solid {COLOR_THEME['border']};
    border-radius: 8px;
    text-align: center;
    font-size: 12px;
}}

QProgressBar::chunk {{
    background-color: {COLOR_THEME['primary']};
    border-radius: 6px;
}}

QFrame {{
    background-color: {COLOR_THEME['card_background']};
    border: 2px solid {COLOR_THEME['border']};
    border-radius: 12px;
    padding: 20px;
}}
"""

# 错误消息
ERROR_MESSAGES = {
    "model_not_found": f"未找到模型文件: {SENSEVOICE_MODEL_PATH}\n请确保模型文件已正确下载到指定目录。",
    "no_file_selected": "请先选择音频文件",
    "transcription_failed": "转录过程中发生错误",
    "empty_result": "转录结果为空",
    "file_not_found": "选择的音频文件不存在"
}

# 状态消息
STATUS_MESSAGES = {
    "ready": "就绪",
    "model_ready": "模型已就绪",
    "model_missing": "模型文件缺失",
    "loading_model": "正在加载模型...",
    "transcribing": "正在进行语音转录...",
    "transcription_complete": "转录完成",
    "transcription_failed": "转录失败",
    "copied_to_clipboard": "结果已复制到剪贴板"
}

def get_model_path():
    """获取模型路径"""
    return str(SENSEVOICE_MODEL_PATH)

def is_model_available():
    """检查模型是否可用"""
    return SENSEVOICE_MODEL_PATH.exists()

def get_supported_formats():
    """获取支持的音频格式"""
    return SUPPORTED_AUDIO_FORMATS

def get_audio_filter():
    """获取音频文件过滤器"""
    return AUDIO_FILTER
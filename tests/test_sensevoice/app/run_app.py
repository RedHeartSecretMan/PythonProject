#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice应用程序启动脚本
用于开发和测试阶段运行应用程序
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def check_dependencies():
    """检查必要的依赖是否已安装"""
    missing_deps = []
    
    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5")
    
    try:
        import funasr_onnx
    except ImportError:
        missing_deps.append("funasr-onnx")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    if missing_deps:
        print("❌ 缺少以下依赖包:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n请运行以下命令安装依赖:")
        print(f"pip install -r {current_dir / 'app_requirements.txt'}")
        return False
    
    return True

def check_model_files():
    """检查模型文件是否存在"""
    model_path = current_dir / "stores" / "checkpoints" / "SenseVoiceSmall"
    
    if not model_path.exists():
        print("⚠️  模型文件不存在:")
        print(f"   {model_path}")
        print("\n请确保已下载SenseVoice模型文件到指定目录")
        print("应用程序仍可启动，但转录功能将不可用")
        return False
    
    print("✅ 模型文件检查通过")
    return True

def main():
    """主函数"""
    print("SenseVoice 语音转录应用程序")
    print("=" * 30)
    
    # 检查依赖
    print("检查依赖包...")
    if not check_dependencies():
        return
    print("✅ 依赖检查通过")
    
    # 检查模型文件
    print("\n检查模型文件...")
    check_model_files()
    
    # 启动应用程序
    print("\n🚀 启动应用程序...")
    try:
        from .sensevoice_app import main as app_main
        app_main()
    except Exception as e:
        print(f"❌ 应用程序启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
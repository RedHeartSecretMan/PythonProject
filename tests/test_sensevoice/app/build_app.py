#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice应用程序打包脚本
使用Nuitka将Python应用打包为独立的macOS应用
"""

import os
import sys
import subprocess
from pathlib import Path

def build_app():
    """构建macOS应用程序"""
    
    # 当前目录
    current_dir = Path(__file__).parent
    
    # 应用程序名称
    app_name = "SenseVoice"
    
    # 主程序文件
    main_script = current_dir / "sensevoice_app.py"
    
    # 输出目录
    output_dir = current_dir / "dist"
    
    # 模型目录
    model_dir = current_dir.parent / "stores" / "checkpoints"
    
    # Nuitka命令
    nuitka_cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--onefile",
        "--macos-create-app-bundle",
        f"--macos-app-name={app_name}",
        "--macos-app-mode=gui",
        "--enable-plugin=pyqt5",
        "--include-data-dir=stores=stores",
        "--output-dir=dist",
        "--remove-output",
        "--assume-yes-for-downloads",
        "--warn-implicit-exceptions",
        "--warn-unusual-code",
        str(main_script)
    ]
    
    print(f"开始构建 {app_name} 应用程序...")
    print(f"主程序: {main_script}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 检查主程序文件是否存在
    if not main_script.exists():
        print(f"错误: 主程序文件不存在: {main_script}")
        return False
    
    # 检查模型文件是否存在
    if not model_dir.exists():
        print(f"警告: 模型目录不存在: {model_dir}")
        print("请确保模型文件已下载到正确位置")
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 执行Nuitka构建
        print("执行Nuitka构建命令:")
        print(" ".join(nuitka_cmd))
        print()
        
        result = subprocess.run(nuitka_cmd, cwd=current_dir, check=True)
        
        print(f"\n✅ {app_name} 应用程序构建成功!")
        print(f"应用程序位置: {output_dir / f'{app_name}.app'}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 构建失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 构建过程中发生错误: {e}")
        return False

def install_dependencies():
    """安装依赖包"""
    print("安装应用程序依赖...")
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"错误: 依赖文件不存在: {requirements_file}")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        
        print("✅ 依赖安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def main():
    """主函数"""
    print("SenseVoice macOS应用程序构建工具")
    print("=" * 40)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        return
    
    # 检查是否在macOS上运行
    if sys.platform != "darwin":
        print("警告: 此脚本专为macOS设计")
    
    # 询问是否安装依赖
    install_deps = input("是否安装/更新依赖包? (y/N): ").lower().strip()
    if install_deps in ['y', 'yes']:
        if not install_dependencies():
            print("依赖安装失败，退出构建")
            return
        print()
    
    # 构建应用程序
    if build_app():
        print("\n🎉 构建完成!")
        print("\n使用说明:")
        print("1. 在Finder中打开dist目录")
        print("2. 将SenseVoice.app拖拽到Applications文件夹")
        print("3. 双击运行应用程序")
        print("\n注意: 首次运行可能需要在系统偏好设置中允许运行")
    else:
        print("\n❌ 构建失败")

if __name__ == "__main__":
    main()
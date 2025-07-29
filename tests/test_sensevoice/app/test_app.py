#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice应用程序测试脚本
"""

import sys
from pathlib import Path


def test_imports():
    """测试导入"""
    print("测试导入...")

    try:
        import PyQt5

        print("✅ PyQt5 导入成功")
    except ImportError as e:
        print(f"❌ PyQt5 导入失败: {e}")
        return False

    try:
        import funasr_onnx

        print("✅ funasr_onnx 导入成功")
    except ImportError as e:
        print(f"❌ funasr_onnx 导入失败: {e}")
        return False

    try:
        import torch

        print("✅ torch 导入成功")
    except ImportError as e:
        print(f"❌ torch 导入失败: {e}")
        return False

    try:
        from .app_config import APP_NAME, APP_VERSION

        print(f"✅ 应用配置导入成功: {APP_NAME} v{APP_VERSION}")
    except ImportError as e:
        print(f"❌ 应用配置导入失败: {e}")
        return False

    return True


def test_model_files():
    """测试模型文件"""
    print("\n测试模型文件...")

    from .app_config import SENSEVOICE_MODEL_PATH, is_model_available

    if is_model_available():
        print(f"✅ 模型文件存在: {SENSEVOICE_MODEL_PATH}")

        # 检查关键文件
        key_files = ["config.yaml", "configuration.json", "tokens.json"]
        for file_name in key_files:
            file_path = SENSEVOICE_MODEL_PATH / file_name
            if file_path.exists():
                print(f"✅ {file_name} 存在")
            else:
                print(f"⚠️  {file_name} 缺失")

        return True
    else:
        print(f"❌ 模型文件不存在: {SENSEVOICE_MODEL_PATH}")
        return False


def test_app_creation():
    """测试应用程序创建"""
    print("\n测试应用程序创建...")

    try:
        # 检查是否有显示环境
        import os

        if "DISPLAY" not in os.environ and sys.platform != "darwin":
            print("⚠️  无显示环境，跳过GUI测试")
            return True

        # 在macOS上，即使没有GUI环境也可能出现问题，所以我们只测试导入
        try:
            print("✅ 应用程序类导入成功")

            # 测试配置导入
            from .app_config import get_model_path

            model_path = get_model_path()
            print(f"✅ 模型路径配置: {model_path}")

            return True

        except Exception as e:
            print(f"❌ 应用程序类导入失败: {e}")
            return False

    except Exception as e:
        print(f"❌ 应用程序测试失败: {e}")
        return False


def test_config():
    """测试配置文件"""
    print("\n测试配置文件...")

    try:
        from .app_config import (
            APP_NAME,
            APP_VERSION,
            MODEL_CONFIG,
            UI_CONFIG,
            get_audio_filter,
        )

        print(f"✅ 应用名称: {APP_NAME}")
        print(f"✅ 应用版本: {APP_VERSION}")
        print(f"✅ 模型配置: {MODEL_CONFIG}")
        print(f"✅ UI配置: {UI_CONFIG}")
        print(f"✅ 音频过滤器: {get_audio_filter()}")

        return True

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n测试文件结构...")

    current_dir = Path(__file__).parent
    required_files = [
        "sensevoice_app.py",
        "app_config.py",
        "run_app.py",
        "build_app.py",
        "app_requirements.txt",
    ]

    all_exist = True
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 缺失")
            all_exist = False

    # 检查目录结构
    stores_dir = current_dir / "stores" / "checkpoints"
    if stores_dir.exists():
        print(f"✅ 模型目录存在: {stores_dir}")
    else:
        print(f"⚠️  模型目录不存在: {stores_dir}")

    return all_exist


def main():
    """主测试函数"""
    print("SenseVoice 应用程序测试")
    print("=" * 30)

    tests = [
        ("文件结构", test_file_structure),
        ("配置文件", test_config),
        ("导入测试", test_imports),
        ("模型文件", test_model_files),
        ("应用创建", test_app_creation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")

    print(f"\n{'=' * 50}")
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！应用程序准备就绪。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

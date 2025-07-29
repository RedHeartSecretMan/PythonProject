"""SenseVoice应用程序包

这个包包含了SenseVoice语音转录应用的所有核心组件：
- sensevoice_app.py: 主应用程序GUI界面
- app_config.py: 应用配置文件
- run_app.py: 应用启动脚本
- build_app.py: 应用打包脚本
- test_app.py: 应用测试脚本
"""

__version__ = "1.0.0"
__author__ = "SenseVoice Team"

# 导入主要组件
from .app_config import *

# 定义包的公共接口
__all__ = [
    'APP_NAME',
    'APP_VERSION',
    'MODEL_CONFIG',
    'UI_CONFIG',
    'ERROR_MESSAGES',
    'STATUS_MESSAGES',
    'get_model_path',
    'get_audio_filter'
]
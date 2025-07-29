# SenseVoice 语音转录应用

基于ONNX引擎的独立macOS语音转录工具，使用PyQt5构建用户界面。

## 功能特性

- 🎵 支持多种音频格式（WAV, MP3, FLAC, M4A, AAC等）
- 🚀 基于ONNX引擎的高效推理
- 🎨 简洁现代的用户界面
- 📋 一键复制转录结果
- 📱 适配macOS的原生体验
- 📦 可打包为独立应用程序

## 系统要求

- macOS 10.14 或更高版本
- Python 3.8 或更高版本
- 至少 4GB 内存
- 1GB 可用磁盘空间

## 安装依赖

1. 安装Python依赖包：
```bash
pip install -r app_requirements.txt
```

2. 下载SenseVoice模型文件到 `stores/checkpoints/SenseVoiceSmall` 目录

## 运行应用程序

### 开发模式

```bash
python run_app.py
```

### 直接运行

```bash
python sensevoice_app.py
```

## 打包为独立应用

使用Nuitka将应用程序打包为独立的macOS应用：

```bash
python build_app.py
```

打包完成后，应用程序将位于 `dist/SenseVoice.app`

## 使用说明

1. **选择音频文件**：点击"选择音频文件"按钮，选择要转录的音频文件
2. **开始转录**：点击"开始转录"按钮，等待转录完成
3. **查看结果**：转录结果将显示在文本框中
4. **复制结果**：点击"复制结果"按钮将转录文本复制到剪贴板

## 支持的音频格式

- WAV (*.wav)
- MP3 (*.mp3)
- FLAC (*.flac)
- M4A (*.m4a)
- AAC (*.aac)
- OGG (*.ogg)
- WMA (*.wma)

## 项目结构

```
test_sensevoice/
├── sensevoice_app.py      # 主应用程序
├── app_config.py          # 应用程序配置
├── run_app.py             # 启动脚本
├── build_app.py           # 打包脚本
├── app_requirements.txt   # 依赖列表
├── README.md              # 说明文档
├── stores/                # 模型文件目录
│   └── checkpoints/
│       └── SenseVoiceSmall/
└── dist/                  # 打包输出目录
    └── SenseVoice.app
```

## 配置选项

应用程序的配置选项在 `app_config.py` 中定义：

- **模型配置**：批处理大小、量化设置、语言检测等
- **UI配置**：窗口大小、字体大小、颜色主题等
- **音频格式**：支持的音频文件格式

## 故障排除

### 模型文件缺失

如果出现"模型文件缺失"错误，请确保：
1. SenseVoice模型文件已下载到正确位置
2. 模型文件目录结构正确
3. 模型文件完整且未损坏

### 依赖包问题

如果出现导入错误，请检查：
1. 所有依赖包是否已正确安装
2. Python版本是否符合要求
3. 虚拟环境是否已激活

### 转录失败

如果转录过程失败，请检查：
1. 音频文件格式是否支持
2. 音频文件是否损坏
3. 系统内存是否充足

## 技术栈

- **UI框架**：PyQt5
- **推理引擎**：ONNX Runtime
- **模型**：SenseVoice
- **打包工具**：Nuitka
- **语言**：Python 3.8+

## 许可证

本项目遵循相关开源许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。
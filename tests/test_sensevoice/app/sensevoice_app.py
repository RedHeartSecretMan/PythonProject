#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice语音转录应用
基于ONNX引擎的独立macOS应用程序
"""

import sys
import os
import threading
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
        QWidget, QPushButton, QTextEdit, QLabel, QFileDialog,
        QProgressBar, QMessageBox, QFrame, QSplitter
    )
    from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
    from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
except ImportError:
    print("错误: 未安装PyQt5，请运行: pip install PyQt5")
    sys.exit(1)

try:
    from funasr_onnx import SenseVoiceSmall
    from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    print("错误: 未安装funasr-onnx，请运行: pip install funasr-onnx")
    sys.exit(1)

from .app_config import (
    APP_NAME, APP_VERSION, APP_STYLESHEET, MODEL_CONFIG,
    UI_CONFIG, ERROR_MESSAGES, STATUS_MESSAGES,
    get_model_path, is_model_available, get_audio_filter
)


class TranscriptionWorker(QThread):
    """语音转录工作线程"""
    finished = pyqtSignal(str)  # 转录完成信号
    error = pyqtSignal(str)     # 错误信号
    progress = pyqtSignal(str)  # 进度信号
    
    def __init__(self, model_path, audio_path):
        super().__init__()
        self.model_path = model_path
        self.audio_path = audio_path
        
    def run(self):
        try:
            self.progress.emit(STATUS_MESSAGES['loading_model'])
            # 初始化模型
            model = SenseVoiceSmall(
                self.model_path, 
                batch_size=MODEL_CONFIG['batch_size'], 
                quantize=MODEL_CONFIG['quantize']
            )
            
            self.progress.emit(STATUS_MESSAGES['transcribing'])
            # 执行推理
            result = model(
                self.audio_path, 
                language=MODEL_CONFIG['language'], 
                textnorm=MODEL_CONFIG['textnorm']
            )
            
            # 后处理
            if result and len(result) > 0:
                transcription = rich_transcription_postprocess(result[0])
                self.finished.emit(transcription)
            else:
                self.error.emit(ERROR_MESSAGES['empty_result'])
                
        except Exception as e:
            self.error.emit(f"{ERROR_MESSAGES['transcription_failed']}: {str(e)}")


class SenseVoiceApp(QMainWindow):
    """SenseVoice主应用程序窗口"""
    
    def __init__(self):
        super().__init__()
        self.model_path = get_model_path()
        self.current_audio_path = None
        self.worker = None
        
        self.init_ui()
        self.check_model_availability()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(f"{APP_NAME} 语音转录工具")
        self.setGeometry(100, 100, UI_CONFIG['window_width'], UI_CONFIG['window_height'])
        self.setMinimumSize(UI_CONFIG['window_min_width'], UI_CONFIG['window_min_height'])
        
        # 设置应用程序样式
        self.setStyleSheet(APP_STYLESHEET)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel(f"{APP_NAME} 语音转录工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            font-size: {UI_CONFIG['title_font_size']}px;
            font-weight: bold;
            color: #333333;
            margin-bottom: 10px;
        """)
        main_layout.addWidget(title_label)
        
        # 文件选择区域
        file_frame = QFrame()
        file_layout = QVBoxLayout(file_frame)
        
        # 文件选择按钮
        self.select_file_btn = QPushButton("选择音频文件")
        self.select_file_btn.clicked.connect(self.select_audio_file)
        file_layout.addWidget(self.select_file_btn)
        
        # 选中文件显示
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("color: #666666; font-style: italic;")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        main_layout.addWidget(file_frame)
        
        # 转录按钮
        self.transcribe_btn = QPushButton("开始转录")
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.transcribe_btn.setEnabled(False)
        main_layout.addWidget(self.transcribe_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel(STATUS_MESSAGES['ready'])
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666666;")
        main_layout.addWidget(self.status_label)
        
        # 结果显示区域
        result_frame = QFrame()
        result_layout = QVBoxLayout(result_frame)
        
        result_title = QLabel("转录结果:")
        result_title.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        result_layout.addWidget(result_title)
        
        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText("转录结果将在这里显示...")
        self.result_text.setMinimumHeight(200)
        result_layout.addWidget(self.result_text)
        
        # 复制按钮
        self.copy_btn = QPushButton("复制结果")
        self.copy_btn.clicked.connect(self.copy_result)
        self.copy_btn.setEnabled(False)
        result_layout.addWidget(self.copy_btn)
        
        main_layout.addWidget(result_frame)
        
    def check_model_availability(self):
        """检查模型文件是否可用"""
        if not is_model_available():
            QMessageBox.warning(
                self, 
                "模型文件缺失", 
                ERROR_MESSAGES['model_not_found']
            )
            self.status_label.setText(STATUS_MESSAGES['model_missing'])
            self.status_label.setStyleSheet("color: #ff4444;")
        else:
            self.status_label.setText(STATUS_MESSAGES['model_ready'])
            self.status_label.setStyleSheet("color: #00aa00;")
    
    def select_audio_file(self):
        """选择音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            get_audio_filter()
        )
        
        if file_path:
            self.current_audio_path = file_path
            self.file_label.setText(f"已选择: {os.path.basename(file_path)}")
            self.file_label.setStyleSheet("color: #333333;")
            self.transcribe_btn.setEnabled(True)
            self.result_text.clear()
            self.copy_btn.setEnabled(False)
    
    def start_transcription(self):
        """开始转录"""
        if not self.current_audio_path:
            QMessageBox.warning(self, "警告", ERROR_MESSAGES['no_file_selected'])
            return
            
        if not is_model_available():
            QMessageBox.warning(self, "错误", ERROR_MESSAGES['model_not_found'])
            return
        
        # 禁用按钮，显示进度条
        self.transcribe_btn.setEnabled(False)
        self.select_file_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        
        # 创建并启动工作线程
        self.worker = TranscriptionWorker(self.model_path, self.current_audio_path)
        self.worker.finished.connect(self.on_transcription_finished)
        self.worker.error.connect(self.on_transcription_error)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.start()
    
    def on_progress_update(self, message):
        """更新进度信息"""
        self.status_label.setText(message)
    
    def on_transcription_finished(self, result):
        """转录完成处理"""
        self.result_text.setPlainText(result)
        self.copy_btn.setEnabled(True)
        self.reset_ui_state()
        self.status_label.setText(STATUS_MESSAGES['transcription_complete'])
        self.status_label.setStyleSheet("color: #00aa00;")
    
    def on_transcription_error(self, error_message):
        """转录错误处理"""
        QMessageBox.critical(self, "转录错误", error_message)
        self.reset_ui_state()
        self.status_label.setText(STATUS_MESSAGES['transcription_failed'])
        self.status_label.setStyleSheet("color: #ff4444;")
    
    def reset_ui_state(self):
        """重置UI状态"""
        self.transcribe_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def copy_result(self):
        """复制结果到剪贴板"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.result_text.toPlainText())
        
        # 显示复制成功提示
        self.status_label.setText(STATUS_MESSAGES['copied_to_clipboard'])
        self.status_label.setStyleSheet("color: #00aa00;")
        
        # 3秒后恢复状态
        QTimer.singleShot(3000, lambda: self.status_label.setText(STATUS_MESSAGES['ready']))
        QTimer.singleShot(3000, lambda: self.status_label.setStyleSheet("color: #666666;"))


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName(f"{APP_NAME}转录工具")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_NAME)
    
    # 创建并显示主窗口
    window = SenseVoiceApp()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
# encoding:utf-8
"""
@File    : test_chat_completion_client_for_multimodal.py
@Time    : 2025/03/19 10:33:52
@Author  : WangHao
@Version : 0.1.1

vLLM多模态模型推理客户端
支持以下功能：
1. 仅文本推理
2. 单图像推理（在线URL/本地文件/file协议）
3. 多图像推理（在线URL/本地文件/file协议）
4. 视频推理（在线URL/本地文件/file协议）
5. 音频推理（在线URL/本地文件/file协议）

启动vLLM服务器的示例命令：

(单图像推理 - Llava)
vllm serve llava-hf/llava-1.5-7b-hf --chat-template template_llava.jinja

(多图像推理 - Phi-3.5-vision-instruct)
vllm serve microsoft/Phi-3.5-vision-instruct --task generate \
    --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=2

(视频推理 - Qwen2.5-VL-7B-Instruct-AWQ)
OMP_NUM_THREADS=4 vllm serve /home/wanghao/projects/LargeModels/Qwen2.5-VL-7B-Instruct-AWQ --task generate \
    --tensor-parallel-size 2 --dtype half --quantization awq_marlin \
    --max-model-len 128000 --gpu-memory-utilization 0.8 --limit-mm-per-prompt "image=8,video=1" \
    --allowed-local-media-path /home/wanghao/

(音频推理 - Ultravox)
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b --max-model-len 4096
"""

import argparse
import base64
import os
import sys
import urllib.parse
from typing import Dict, List, Optional

import requests
from openai import OpenAI


class MultimodalClient:
    """多模态模型推理客户端"""

    def __init__(
        self, api_key: str = "EMPTY", api_base: str = "http://localhost:8000/v1"
    ):
        """初始化客户端"""
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.available_models = self._get_available_models()

    def _get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            return []

    @staticmethod
    def encode_base64(path_or_url: str) -> str:
        """从URL获取内容并编码为base64格式"""
        if path_or_url.startswith("file://"):
            file_path = urllib.parse.unquote(path_or_url[7:])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")
        elif path_or_url.startswith(("http://", "https://")):
            with requests.get(path_or_url) as response:
                response.raise_for_status()
                return base64.b64encode(response.content).decode("utf-8")
        else:
            if not os.path.exists(path_or_url):
                raise FileNotFoundError(f"文件不存在: {path_or_url}")
            with open(path_or_url, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")

    @staticmethod
    def get_mime_type(path_or_url: str) -> str:
        """根据文件扩展名获取MIME类型"""
        ext = os.path.splitext(path_or_url)[1].lower()
        mime_map = {
            # 图片格式
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            # 视频格式
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            # 音频格式
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
        }
        return mime_map.get(ext, "application/octet-stream")

    @staticmethod
    def is_url(path_or_url: str) -> bool:
        """判断字符串是否为URL"""
        return path_or_url.startswith(("http://", "https://", "file://"))

    def prepare_content_item(
        self, content_path: str, content_type: str, force_base64: bool = False
    ) -> Dict:
        """准备请求中的内容项"""
        content_type_to_keys = {
            "image": "image_url",
            "video": "video_url",
            "audio": "audio_url",
        }

        content_type_to_key = content_type_to_keys.get(content_type)
        if not content_type_to_key:
            raise ValueError(f"不支持的内容类型: {content_type}")

        mime_type = self.get_mime_type(content_path)

        # 处理URL和文件路径
        if self.is_url(content_path) and not force_base64:
            return {
                "type": content_type_to_key,
                content_type_to_key: {"url": content_path},
            }
        else:
            base64_content = self.encode_base64(content_path)
            return {
                "type": content_type_to_key,
                content_type_to_key: {
                    "url": f"data:{mime_type};base64,{base64_content}"
                },
            }

    def run_inference(
        self,
        model: str,
        prompt: str,
        content_items: Optional[List[Dict]] = None,
        max_tokens: int = 1024,
    ) -> str | None:
        """运行推理并返回结果"""
        if content_items is None:
            content_items = []

        # 验证模型是否可用
        if self.available_models and model not in self.available_models:
            print(f"警告: 模型 '{model}' 不在可用模型列表中")

        message_content = [{"type": "text", "text": prompt}]
        message_content.extend(content_items)

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message_content,
                    }  # type: ignore
                ],
                model=model,
                max_tokens=max_tokens,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"推理失败: {e}")
            return None

    def run_text_only(
        self, model: str, prompt: str = "中国的首都是什么？", max_tokens: int = 1024
    ) -> None:
        """仅文本推理"""
        result = self.run_inference(model, prompt, max_tokens=max_tokens)
        print("仅文本推理结果:", result)

    def run_single_image(
        self,
        model: str,
        image_path_or_url: str,
        prompt: str = "这张图片里有什么？",
        force_base64: bool = False,
        max_tokens: int = 1024,
    ) -> None:
        """单图像推理"""
        image_item = self.prepare_content_item(image_path_or_url, "image", force_base64)
        result = self.run_inference(model, prompt, [image_item], max_tokens=max_tokens)
        print(f"{image_path_or_url}推理结果:", result)

    def run_multi_image(
        self,
        model: str,
        image_paths_or_urls: List[str],
        prompt: str = "这些图片中有什么？",
        force_base64: bool = False,
        max_tokens: int = 1024,
    ) -> None:
        """多图像推理"""
        image_items = [
            self.prepare_content_item(path, "image", force_base64)
            for path in image_paths_or_urls
        ]
        result = self.run_inference(model, prompt, image_items, max_tokens=max_tokens)
        print(f"{image_paths_or_urls}推理结果:", result)

    def run_video(
        self,
        model: str,
        video_path_or_url: str,
        prompt: str = "这个视频里有什么？",
        force_base64: bool = False,
        max_tokens: int = 1024,
    ) -> None:
        """视频推理"""
        video_item = self.prepare_content_item(video_path_or_url, "video", force_base64)
        result = self.run_inference(model, prompt, [video_item], max_tokens=max_tokens)
        print(f"{video_path_or_url}推理结果:", result)

    def run_audio(
        self,
        model: str,
        audio_path_or_url: str,
        prompt: str = "这段音频里有什么？",
        force_base64: bool = False,
        max_tokens: int = 1024,
    ) -> None:
        """音频推理示例"""
        audio_item = self.prepare_content_item(audio_path_or_url, "audio", force_base64)
        result = self.run_inference(model, prompt, [audio_item], max_tokens=max_tokens)
        print(f"{audio_path_or_url}推理结果:", result)


def main():
    """主函数"""
    # 初始化客户端
    client = MultimodalClient()

    parser = argparse.ArgumentParser(
        description="使用OpenAI客户端对多模态语言模型进行推理的演示"
    )
    parser.add_argument("--model", "-m", type=str, help="使用的模型ID", required=True)

    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="text-only",
        choices=["text-only", "single-image", "multi-image", "video", "audio"],
        help="对话类型，支持多种多模态数据",
    )

    parser.add_argument(
        "--path-or-url", type=str, help="单个媒体文件的本地文件路径或URL"
    )

    parser.add_argument(
        "--paths-or-urls",
        nargs="+",
        help="多个媒体文件的本地文件路径或URL组成的列表",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="用于推理的文本提示",
    )

    parser.add_argument(
        "--force-base64",
        action="store_true",
        help="强制使用base64编码内容",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="最大生成token数",
    )

    args = parser.parse_args()

    # 根据类型调用相应的函数
    if args.type == "text-only":
        client.run_text_only(
            args.model, args.prompt or "中国的首都是什么？", args.max_tokens
        )
    elif args.type == "single-image":
        client.run_single_image(
            args.model,
            args.path_or_url
            or "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            args.prompt or "这张图片里有什么？",
            args.force_base64,
            args.max_tokens,
        )
    elif args.type == "multi-image":
        client.run_multi_image(
            args.model,
            args.paths_or_urls
            or [
                "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
            ],
            args.prompt or "这些图片中有什么？",
            args.force_base64,
            args.max_tokens,
        )
    elif args.type == "video":
        client.run_video(
            args.model,
            args.path_or_url
            or "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
            args.prompt or "这个视频里有什么？",
            args.force_base64,
            args.max_tokens,
        )
    elif args.type == "audio":
        client.run_audio(
            args.model,
            args.path_or_url
            or "https://upload.wikimedia.org/wikipedia/commons/2/28/Karplus-strong-played.ogg",
            args.prompt or "这段音频里有什么？",
            args.force_base64,
            args.max_tokens,
        )


if __name__ == "__main__":
    sys.argv.extend(
        [
            "--model",
            "./stores/Qwen2.5-VL-7B-Instruct-AWQ",
            "--type",
            "video",
        ]
    )
    sys.exit(main())

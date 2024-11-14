import subprocess
import os
from threading import current_thread
import signal
import time


class VLLMServerManager:
    def __init__(self):
        self.process = None
        if current_thread().name == "MainThread":
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    # 启动 vLLM API 服务
    def start_vllm_openai_api_server(
        self,
        model_path="./stores/minicpm3/4b",
        device_index="0",
    ):
        if self.process is None or self.process.poll() is not None:
            command = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
                "--served-model-name",
                "minicpm3_4b",
                "--dtype",
                "bfloat16",
                "--kv-cache-dtype",
                "fp8",
                "--tensor-parallel-size",
                "1",
                "--max-model-len",
                "4096",
                "--trust-remote-code",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device_index
            with open("vllm_minicpm3_4b.log", "w") as log_file:
                self.process = subprocess.Popen(
                    command, stdout=log_file, stderr=subprocess.STDOUT, env=env
                )
            print(f"Started vLLM API server with PID {self.process.pid}")
        else:
            print("vLLM API server is already running.")

    # 停止 vLLM API 服务
    def stop_vllm_openai_api_server(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                print(f"Stopped vLLM API server with PID {self.process.pid}")
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                self.process = None
        else:
            print("No vLLM API server is running.")

    # 检查 vLLM API 服务
    def check_vllm_openai_api_server(self):
        if self.process is None or self.process.poll() is not None:
            print("vLLM API server is not running.")
            return False
        else:
            print(f"vLLM API server is running with PID {self.process.pid}")
            return True

    def _signal_handler(self, signal_number, code_frame):
        self.stop_vllm_openai_api_server()


if __name__ == "__main__":
    manager = VLLMServerManager()
    # 启动服务器
    manager.start_vllm_openai_api_server()

    # 检查服务器
    while manager.process is not None:
        time.sleep(10)
        manager.check_vllm_openai_api_server()

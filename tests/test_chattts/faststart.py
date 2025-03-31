import os
import sys

import sounddevice as sd
import soundfile as sf
import torch

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "ChatTTS")
    )
)
import ChatTTS

torch._dynamo.config.cache_size_limit = 64  # type: ignore
torch._dynamo.config.suppress_errors = True  # type: ignore
torch.set_float32_matmul_precision("high")

chat = ChatTTS.Chat()
chat.load_models(source="local", local_path="stores/nosft")

texts = ["我的朋友 ChatTTS，早上好，中午好，晚上好。"]
wavs = chat.infer(texts)

# sounddevice and soundfile input data shape is (num_samples, num_channels)
sd.play(data=wavs[0].T, samplerate=24000)
sd.wait()
sf.write(file="results/faststart.wav", data=wavs[0].T, samplerate=24_000)

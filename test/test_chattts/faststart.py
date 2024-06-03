import os
import sys

import sounddevice as sd
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
chat.load_models(source="local", local_path="chattts/resources/nosft")

texts = ["中午好，我的朋友ChatTTS。"]

wavs = chat.infer(texts)
sd.play(wavs[0].transpose(1, 0), samplerate=24000)
sd.wait()

import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "ChatTTS")
    )
)
import sounddevice as sd
import torch

import ChatTTS

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")

chat = ChatTTS.Chat()
chat.load_models(source="local", local_path="chattts/models/nosft")

texts = ["中午好，我的朋友ChatTTS。"]

wavs = chat.infer(texts, use_decoder=True)
sd.play(wavs[0], samplerate=24000)

from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

model_directory = "./stores/checkpoints/SenseVoiceSmall"
model = SenseVoiceSmall(model_directory, batch_size=10, quantize=False)

# inference
input_path = "./datas/audios/陈杨.wav"
result = model(input_path, language="auto", textnorm="withitn")
print([rich_transcription_postprocess(i) for i in result])

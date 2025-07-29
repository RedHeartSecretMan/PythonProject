from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_directory = "./stores/checkpoints/SenseVoiceSmall"
model = AutoModel(
    model=model_directory,
    trust_remote_code=False,
    vad_model="./stores/checkpoints/fsmn_vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="mps",
)
input_path = "./datas/audios/陈杨.wav"
result = model.generate(
    input=input_path,
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(result[0]["text"])
print(text)

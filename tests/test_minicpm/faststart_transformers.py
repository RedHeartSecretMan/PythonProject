import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
model_path = "./stores/minicpm3/4b"
device = torch.device("mps")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)

texts = """介绍5个北京的景点。
           灵活运用多种修辞手法，字数800左右。"""
responds, history = model.chat(
    tokenizer,
    texts,
    temperature=0.7,
    top_p=0.7,
)
print(responds)

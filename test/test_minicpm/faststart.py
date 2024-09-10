from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.manual_seed(51)

path = "./stores/minicpm3/4b"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
)

texts = """女子提分手竟遭男友杀害嫌犯自杀未果报警自首
           请以故事的形式还原案件发展过程尽量不要平铺直述、矛盾点、冲击点可以场景化，侧重人物心理变化描写
           字数2000左右，三到四行一段，每段间隔一行"""
responds, history = model.chat(
    tokenizer,
    texts,
    temperature=0.7,
    top_p=0.7,
    max_new_tokens=8192*4,
)
print(responds)

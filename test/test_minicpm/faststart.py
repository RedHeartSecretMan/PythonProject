from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


model_name = "./stores/minicpm3/4b"
texts = """女子提分手竟遭男友杀害嫌犯自杀未果报警自首
           请以故事的形式还原案件发展过程尽量不要平铺直述、矛盾点、冲击点可以场景化，侧重人物心理变化描写
           字数2000左右，三到四行一段，每段间隔一行"""
prompt = [
    {
        "role": "user",
        "content": f"{texts}",
    }
]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
input_text = tokenizer.apply_chat_template(
    prompt, tokenize=False, add_generation_prompt=True
)

llm = LLM(
    model=model_name, trust_remote_code=True, tensor_parallel_size=1, max_model_len=5500
)
sampling_params = SamplingParams(top_p=0.7, temperature=0.7, max_tokens=1024)

outputs = llm.generate(prompts=str(input_text), sampling_params=sampling_params)

print(outputs[0].outputs[0].text)

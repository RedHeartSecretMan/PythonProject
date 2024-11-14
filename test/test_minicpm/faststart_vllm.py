from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


model_name = "./stores/minicpm3/4b"
texts = """介绍5个北京的景点。
           灵活运用多种修辞手法，字数800左右。"""
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

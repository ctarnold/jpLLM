from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", 
                                             low_cpu_mem_usage=True)

messages = [
    {"role": "user", "content": "¿Me entiendes cuando escribo en español?"},
    {"role": "assistant", "content": ""},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
# https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
outputs = model.generate(inputs, max_new_tokens=2048, temperature = 0.6)

with open("outputAuto.txt", "a") as f:    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True), file = f)
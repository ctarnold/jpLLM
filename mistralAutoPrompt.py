from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", 
                                             low_cpu_mem_usage=True)

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"},
    {"role": "user", "content": "Dame un ejemplo de code-switching in the Miami-English form"},
    {"role": "user", "content": "Escribeme un cuento en English and Spanish, con los idiomas mixed together"},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
# https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
outputs = model.generate(inputs, max_new_tokens=2048, temperature = 0.6)

with open("outputAuto.txt", "a") as f:    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True), file = f)
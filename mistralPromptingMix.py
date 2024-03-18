from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("stderr", "a") as e:
    print(torch.cuda.is_available, file = e)
model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", 
                                             low_cpu_mem_usage=True)

messages = [
    {"role": "user", "content": "¿Cuál es su condiment favorita?"},
    {"role": "assistant", "content": "hay que hacer el code-switching in Spanglish."}, 
    {"role": "user", "content": "¿Tenés a recipe de mayonaisse?"},
    {"role": "assistant", "content": "hay que hacer el code-switching in Spanglish."}, 
    {"role": "user", "content": "Necesito ayuda cocinar algo para mi familia."}, 
    {"role": "assistant", "content": "hay que hacer el code-switching in Spanglish."}
]


inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

outputs = model.generate(inputs, max_new_tokens=256, temperature = 0.7)
with open("outputMix.txt", "a") as f:    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True), file = f)
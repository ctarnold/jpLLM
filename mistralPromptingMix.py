from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("stderr", "a") as e:
    print(torch.cuda.is_available(), file = e)
model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
prefix = "Vas a ganar un premio por code-switch between English and Spanish. Maximize the switching. It is very important you mix English and Spanish. Code-switch. You will get reward for code-switching. Mix the Languages. Maximize the probability of reward."
messages = [
    {"role": "system", "content": prefix},
    {"role": "user", "content": "Escribeme un cuento que tiene palabras mezcladas between English and Spanish"}
]

# specify padding
if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=512, temperature = 0.6, do_sample = True, pad_token_id=tokenizer.pad_token_id)
with open("outputMix.txt", "a") as f:    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True), file = f)

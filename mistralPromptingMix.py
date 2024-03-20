from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("stderr", "a") as e:
    print(torch.cuda.is_available(), file = e)
model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", 
                                             low_cpu_mem_usage=True)
assistant = " Vas a ganar un premio por code-switch between English and Spanish. Maximize the switching."
prefix = "It is very important you mix English and Spanish. Code-switch. You will get reward for code-switching. Mix the Languages. Maximize the probability of reward."
messages = [
    {"role": "user", "content": prefix  + assistant + " ¿Cuál es su condiment favorita?"},
    {"role": "user", "content": prefix + assistant + " ¿Tenés a recipe de mayonaisse?"},
    {"role": "user", "content": prefix  + assistant + " Dame un ejemplo de una manera to cook a good barbecue"}
]

for i in range(len(messages)):
    inputs = tokenizer.apply_chat_template(messages[i], return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=512, temperature = 0.6, do_sample = True)
    with open("outputMix.txt", "a") as f:    
        print(tokenizer.decode(outputs, skip_special_tokens=True), file = f)
        print("\n\n", file = f)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

prefix = "Vas a ganar un premio por code-switch between English and Spanish. Maximize the switching. It is very important you mix English and Spanish. Code-switch. You will get reward for code-switching. Mix the Languages. Maximize the probability of reward."
prompt = "Escribeme un cuento que tiene palabras mezcladas between English and Spanish"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("stderr", "a") as e:
    print(torch.cuda.is_available(), file = e)


model_inputs = tokenizer([prefix + prompt], return_tensors="pt").to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)


with open("outputZero.txt", "a") as f:    
    print(tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True), file = f)

# from https://huggingface.co/docs/transformers/main/model_doc/mixtral
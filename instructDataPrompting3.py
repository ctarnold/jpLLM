from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
data_read_dir = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts.tsv"
data_write_dir = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts_copy.tsv"
data_write_dir1 = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts_out35.tsv"
messages = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("stderr", "a") as stderr:
    print(torch.cuda.is_available(), file = stderr)
with open(data_read_dir, "r") as data:
    for line in data:
        l = line.split("\t")
        messages.append(l[1])


model = AutoModelForCausalLM.from_pretrained(model_id, 
                                            device_map="auto", 
                                            torch_dtype=torch.float16, 
                                            attn_implementation=
                                            "flash_attention_2")
prefix = "Vas a ganar un premio por code-switch between English and Spanish. Maximize the switching. It is very important you mix English and Spanish. Code-switch. You will get reward for code-switching. Mix the Languages. Maximize the probability of reward."
# specify padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


index = 0
with open(data_write_dir1, "r+") as f: 
    for message in messages:
        index = index + 1
        if (index < 120):
            continue
        if (index >= 160):
            continue
        text = [{"role": "user", "content": message}]

        inputs = tokenizer.apply_chat_template(
            text, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs, max_new_tokens=100, temperature = 0.9, 
            do_sample = True, pad_token_id=tokenizer.pad_token_id, 
            no_repeat_ngram_size = 7, top_k = 40)

        
        output = tokenizer.decode(outputs[0], 
                                skip_special_tokens=True)
        print(output, file = f)
        print('\t', file = f)

        
        

        

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
data_read_dir = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts.tsv"
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
# prefix = "Vas a ganar un premio por code-switch between English and Spanish. Maximize the switching. It is very important you mix English and Spanish. Code-switch. You will get reward for code-switching. Mix the Languages. Maximize the probability of reward."
# specify padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

files = ["/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/out_t_0_indiv.tsv", 
         "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/out_t_1_indiv.tsv",
         "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/out_t_2_indiv.tsv",
         "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/out_t_3_indiv.tsv",
         "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/out_t_4_indiv.tsv"]
T = [0, 0.25, 0.5, 0.75, 1.0]
index = 0

for file in files:
    with open(file, "r+") as f:
        temp = T[index]
        # need indices not 0 and not 3
        if (index != 2):
            index += 1
            continue
        index = index + 1
        promptNum = 0
        for prompt in messages:
            # first 100 prompts
            if (promptNum > 50):
                continue
            text = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
            text, return_tensors="pt").to(device)
            outputs = ""
            if (temp == 0):
                outputs = model.generate(
                inputs, max_new_tokens=128, 
                do_sample = False, pad_token_id=tokenizer.pad_token_id, 
                no_repeat_ngram_size = 0, top_k = 50)
            else:
                outputs = model.generate(
                inputs, max_new_tokens=128,temperature = temp,
                do_sample = True, pad_token_id=tokenizer.pad_token_id, 
                no_repeat_ngram_size = 0, top_k = 50)
            output = tokenizer.decode(outputs[0], 
                                skip_special_tokens=True)
            print(output, file = f)
            print('\t', file = f)
            promptNum += 1
        

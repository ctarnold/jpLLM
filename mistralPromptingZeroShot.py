import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             torch_dtype=torch.float16, 
                                             attn_implementation="flash_attention_2", 
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)


prefix = "Answer this prompt as a bilingual English/Spanish Miami speaker who code-switches:"
prompt = "Escribeme un cuento que tiene palabras mezcladas between English and Spanish."


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("stderr", "a") as e:
    print(torch.cuda.is_available(), file = e)

messages = [
    {"role": "user", "content": prefix + prompt},
    {"role": "assistant", "content": ""}
]

model_inputs = tokenizer.apply_chat_template([messages], return_tensors="pt").to(device)

generated_ids = model.generate(**model_inputs, 
                               max_new_tokens=512, 
                               do_sample=True,
                               no_repeat_ngram_size = 5,
                               temperature = 0.6) 
# do not repeat >=5-grams


with open("outputZero.txt", "a") as f:    
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True), file = f)

# from https://huggingface.co/docs/transformers/main/model_doc/mixtral
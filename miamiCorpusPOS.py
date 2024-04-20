import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-ner-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-ner-lince")
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)

text = "testing the ner of the model."

print(lid_model(text))
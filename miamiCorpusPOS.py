import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-ner-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-ner-lince")
ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

text = "estoy intentando hacer la recognición de entidaded nombeadas. por alguna razón, el resultado no funciona."

print(ner_model(text))
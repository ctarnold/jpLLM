
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-pos-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-ner-lince")
pos_model = pipeline('ner', model=model, tokenizer=tokenizer)

print(pos_model("put any spanish english code-mixed sentence"))

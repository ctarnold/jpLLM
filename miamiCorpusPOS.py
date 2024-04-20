from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-ner-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-ner-lince")

ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

print(ner_model("put any spanish english code-mixed sentence"))

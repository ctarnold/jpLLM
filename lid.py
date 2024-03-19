from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

text = "me dice ella que trabaha en una tienda de furniture, so anyways that's that one. this is Chris' boyfriend."

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)

with open("lidout.txt", "a") as f:
    print(lid_model(text), file = f)

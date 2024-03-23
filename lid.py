from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

text = "me dice ella que trabaja en una tienda de furniture, so anyways that's that one. this is Chris' boyfriend."

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)

out = lid_model(text)


spaCount = 0
engCount = 0
otherCount = 0
for i in range(len(out)):
    language = out[i].get('entity')
    if (language == 'spa'):
        spaCount += 1
    if (language == 'eng'):
        engCount += 1
    if (language != 'eng' and language != 'spa'):
        otherCount += 1

with open("lidout.txt", "a") as f:
    print(spaCount, file = f)
    print(" Spanish Count\n", file = f)
    print(engCount, file = f)
    print(" English Count\n", file = f)
    print(otherCount, file = f)
    print(" Other Count\n", file = f)
    
    
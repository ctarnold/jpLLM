from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

data_dir = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts_out.tsv"
data_dir1 = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts_out1.tsv"
data_dir2 = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts_out2.tsv"
data_dir3 = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts_out3.tsv"
data_dir4 = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts_out4.tsv"

fileNum = 0


text = "me dice ella que trabaja en una tienda de furniture, so anyways that's that one. this is Chris' boyfriend."

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)


messages = []
spaCount = 0
engCount = 0
otherCount = 0

with open("lidout.txt", "a") as f:
    for i in range(fileNum):
        
        if (i == 0):
            data = data_dir
        if (i == 1):
            data = data_dir1
        if (i == 2):
            data = data_dir2
        if (i == 3):
            data = data_dir3
        if (i == 4):
            data = data_dir4
        messages = []
        with open(data, "r") as data:
            for line in data:
                l = line.split("\t")
                messages.append(l[1])
        for message in messages:
            out = lid_model(text)
            for j in range(len(out)):
                language = out[j].get('entity')
                if (language == 'spa'):
                    spaCount += 1
                if (language == 'en'):
                    engCount += 1
                if (language != 'en' and language != 'spa'):
                    otherCount += 1
    
print(spaCount, file = f)
print(" Spanish Count\n", file = f)
print(engCount, file = f)
print(" English Count\n", file = f)
print(otherCount, file = f)
print(" Other Count\n", file = f)

    
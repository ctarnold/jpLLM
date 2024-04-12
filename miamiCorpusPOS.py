import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")
lid_model = pipeline('pos', model=model, tokenizer=tokenizer)

data_dir = "/scratch/gpfs/ca2992/jpLLM/bangor/crowdsourced_bangor"
out = "/scratch/gpfs/ca2992/jpLLM/bangor/test"

# get all the speech of the corpus as a gigantic string
message = ""
lidGround = []
posGround = []

with open(out, "a") as output:
    for file in os.listdir(data_dir):
        if os.path.isdir(data_dir  + '/' + file):
        # Skip directories
            continue
        if(file == "README.md"):
            continue
        with open(data_dir  + '/' + file, "r") as read:
            for line in read:
                values = line.split()
                # skip blank lines
                if (len(values) <= 2):
                    continue
                num = values[0]
                word = values[1]
                lid = values[2]
                pos = values[3]
                message = word + " "
                lidGround.append(lid)
                posGround.append(pos)

posResult = lid_model(message)
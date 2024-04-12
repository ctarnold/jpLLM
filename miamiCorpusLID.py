import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)

data_dir = "/scratch/gpfs/ca2992/jpLLM/bangor/crowdsourced_bangor"
out = "/scratch/gpfs/ca2992/jpLLM/bangor/test"

correctSpa = 0
correctEn = 0
wrongSpa = 0
wrongEn = 0
other = 0

message = ""
lidGround = []
posGround = []

def groundCompare(lidResult):
     for j in range(len(lidResult)):
        language = lidResult[j].get('entity')
        if (language == 'spa'):
            if (lidGround[j] == 'spa'):
                correctSpa += 1
            if (lidGround[j] == 'eng' or lidGround[j] == 'en'):
                wrongSpa += 1
        if (language == 'en' or language == 'eng'):
            if (lidGround[j] == 'eng' or lidGround[j] == 'en'):
                correctEn += 1
            if (lidGround[j] == 'spa'):
                wrongEn += 1
        if (language != 'en' and language != 'spa' and language != 'eng'):
                other += 1

with open(out, "a") as output:
    for file in os.listdir(data_dir):
        if os.path.isdir(data_dir  + '/' + file):
        # Skip directories
            continue
        if(file == "README.md"):
            continue
        with open(data_dir  + '/' + file, "r") as read:
            # get all the speech of the corpus as a gigantic string
            # lid model i/o is capped at length 510 it seems.
            message = ""
            lidGround = []
            posGround = []
            for line in read:
                values = line.split()
                # skip blank lines
                if (len(values) <= 2):
                    continue
                num = values[0]
                word = values[1]
                lid = values[2]
                pos = values[3]
                message = message + " " + word + " "
                lidGround.append(lid)
                posGround.append(pos)
                # with period or question mark get ground truth
                # comparison
                if (word == '?' or word == '.'):
                    lidResult = lid_model(message)
                    groundCompare(lidResult)
                    message = ""
                    lidGround = []
                    posGround = []


print(correctSpa)
print(correctEn)
print(wrongSpa)
print(wrongEn)
print(other)

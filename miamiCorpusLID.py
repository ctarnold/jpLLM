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
words = []

# another way to do this would be to check 
# whether each word is valid against the ground truth,
# if not, remove characters or concatenate as needed.

def groundCompare(lidResult):
    global lidGround
    global posGround
    global correctSpa
    global correctEn
    global wrongSpa
    global wrongEn
    global words
    global other

   
    index = 0
    groundIndex = 0
    # second way
    # for each word in the ground truth
    # use first token tag for ground truth
    for word in range(len(words)):
        lidToken= lidResult[index].get('word')
        language = lidResult[index].get('entity')
        # concatenate lid tokens 
        print(word, lidToken, word == lidToken)
        # report mismatch
        if (word != lidToken and word[0] != lidToken[0]):
            print("MISMATCH", word, lidToken)
            continue
        while (word != lidToken and word[0] == lidToken[0]):
            index += 1
            lidToken = lidToken + lidResult[index].get('word')
        
        if (language == 'spa'):
            if (lidGround[groundIndex] == 'spa'):
                correctSpa += 1
            if (lidGround[groundIndex] == 'eng' or lidGround[groundIndex] == 'en'):
                wrongSpa += 1
        if (language == 'en' or language == 'eng'):
            if (lidGround[groundIndex] == 'eng' or lidGround[groundIndex] == 'en'):
                correctEn += 1
            if (lidGround[groundIndex] == 'spa'):
                wrongEn += 1
        if (language != 'en' and language != 'spa' and language != 'eng'):
                other += 1
        index += 1
        groundIndex += 1

### word: ... index = 0, token .
### word: ... index = 1, token ..
### word: ... index = 2, token ...

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
            words = []
            for line in read:
                values = line.split()
                # skip blank lines
                if (len(values) <= 3):
                    continue
                num = values[0]
                word = values[1]
                lid = values[2]
                pos = values[3]

                acronym = False
                for i in range(len(word)):
                    if (word[i] == '_'):
                        acronym = True


                if (word != '...' and acronym != True and word != '<unintelligible>'):
                    message += (" " + word)
                    lidGround.append(lid)
                    posGround.append(pos)
                    words.append(word)
                # with period or question mark get ground truth
                # comparison
                if (word == '?' or word == '.'):
                    lidResult = lid_model(message)
                    print(len(lidResult))
                    print(len(lidGround))
                    print(file)
                    print(lidResult)
                    print(lidGround)
                    groundCompare(lidResult)
                    message = ""
                    lidGround = []
                    posGround = []
                    words = []


print(correctSpa)
print(correctEn)
print(wrongSpa)
print(wrongEn)
print(other)

import os
import lidCall
import miamiCorpusLID
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline



# model_name = '/scratch/gpfs/ca2992/robertuito-base-cased'
model_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince'
tokenizer_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
out_dir = 'mistral_lid_ratios'
pos_model_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-pos-lince'
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)
pos_model = pipeline('ner', model=model, tokenizer = tokenizer)

dir = '/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/'
files = ['out_t_0_indiv.tsv',
        'out_t_1_indiv.tsv',
        'out_t_2_indiv.tsv', 
        'out_t_3_indiv.tsv',
        'out_t_4_indiv.tsv',]


# given a token with the '#' symbol,
# remove the symbol for preprocessing
def cleanPoundSign(word):
    tempTok = ""
    for i in range(len(word)):
        if (word[i] != '#'):
            tempTok = tempTok + word[i]
    return tempTok

lid_pred = []
# convert token predictions to word predictions
def tokenToWordPred(message, trueWords):
    lidResult = lid_model(message)
    posResult = pos_model(message)
    index = 0
    for word in trueWords:
        lidToken = lidResult[index].get('word')
        # get the lid predicted for this token and append
        # to the lid word level predictions
        lid = lidResult[index].get('entity')

        lid_pred.append([lid])
        # if token word mismatch imlidsible to handle
        if (word != lidToken and word[0] != lidToken[0]):
            print("MISMATCH", word, lidToken)
            continue

        while (word != lidToken and word[0] == lidToken[0]):
            index += 1
            lidToken = lidToken + lidResult[index].get('word')
            # get rid of # symbols added by tokenizer
            lidToken = cleanPoundSign(lidToken)
        index += 1

def cleanInstruct(text):
    instruction = True
    response = False
    almost = False

    message = ""
    for char in text:
        if char == ']' and almost == False:
            almost = True
        elif char == ']' and almost == True:
            almost = False
            response = True
            instruction = False
        elif response == True:
            message = message + char
        assert instruction != response
    return message

spanish_count = 0
english_count = 0

out_dir = 'lang_lid_ratio_7'
switch_verb = 0
switch_noun = 0
switch_conj = 0
switch_count = 0
count = 0
fileNum = 0
for file in files:
    if (fileNum > 0):
        continue
    fileNum += 1
    with open(dir + file, "r+") as f:
        message = ""
        for line in f:
            if (line[0] == '['):
                model_input = ""
                if (message == ""):
                    model_input = cleanInstruct(line)
                else:
                    model_input = cleanInstruct(message)
                lid_results = lid_model(model_input)
                pos_results = pos_model(model_input)
                message = ""
                for i in range(len(lid_results)):
                    count += 1
                    lid = lid_results[i].get('entity')
                    pos = pos_results[i].get('entity')
                    if (i == 0):
                        last_lid = lid
                        last_pos = pos
                    # detect code-switching switch
                    if (last_lid != lid):
                        switch_count += 1
                        if (pos == "VERB"):
                            switch_verb += 1
                        elif (pos == "NOUN"):
                            switch_noun += 1
                        elif (pos == "CONJ"):
                            switch_conj += 1
                    if (lid == 'spa'):
                        spanish_count +=1 
                    if (lid == 'en'):
                        english_count += 1
                    last_lid = lid
                    last_pos = pos
            else:
                message += line
        with open(dir + out_dir, "a") as o:
            print(file, file = o)
            print(spanish_count, "Spanish Count", file = o)
            print(english_count, "English Count", file = o)
            print(switch_count, switch_noun, switch_conj, switch_verb, file = o)
                

            
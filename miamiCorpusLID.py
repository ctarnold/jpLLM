import os
import lidCall
import eval
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# model_name = '/scratch/gpfs/ca2992/robertuito-base-cased'
model_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince'
tokenizer_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

out_dir = '/scratch/gpfs/ca2992/jpLLM/jpLLM/lid_out'
data_dir = '/scratch/gpfs/ca2992/jpLLM/bangor/crowdsourced_bangor'

lid_model = pipeline('ner', model=model, tokenizer=tokenizer)
lid_truth = []
lid_pred = []

# given a token with the '#' symbol,
# remove the symbol for preprocessing
def cleanPoundSign(word):
    tempTok = ""
    for i in range(len(word)):
        if (word[i] != '#'):
            tempTok = tempTok + word[i]
    return tempTok


# words in the annotated Bangor Corpus
# contain ' if a contraction. Check to allow
# concatenation
def isContraction(word):
    for char in word:
        if (char == '\''):
            return True
    return False


# convert token predictions to word predictions
def tokenToWordPred(message, trueWords):
    lidResult = lid_model(message)
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

fileCount = 0
with open(out_dir, "a") as output:
    for file in os.listdir(data_dir):
        if os.path.isdir(data_dir  + '/' + file):
        # Skip directories and readme
            continue
        if(file == "README.md"):
            continue
        # open the current file in the directory
        with open(data_dir  + '/' + file, "r") as read:
            if (fileCount >= 1):
                continue
            fileCount += 1
            numWords = 0
            words = []
            message = ""
            for line in read:
                values = line.split()
                # skip blank lines or placeholder lines
                if (len(values) <= 3):
                    # print(line)
                    continue
                # print(values[0], values[1], values[2], values[3])
                # print(line)
                lid = values[2] #lid at index 2 of each line
                word = values[1] # word at index 1 of each line
                numWords += 1
                # print(lid)
                if isContraction(word):
                    # if is a contraction, implicitly use last truth tag
                    message = message + word
                    lastWord = words.pop()
                    words.append(lastWord + word)
                else:
                    # if it is not a contraction, use the truth tag
                    message = message + " " + word
                    words.append(word)
                    lid_truth.append([lid])
                # at the end of each sentence, pass into the model
                if (word == '.'):
                    tokenToWordPred(message, words)
                    numWords = 0
                    lid = []
                    words = []
                    message = ""
    # print(lid_truth, file = output)
    # print(lid_pred, file = output)
    # print(len(lid_truth), len(lid_pred), file = output)
    print(len(lid_truth), len(lid_pred))
    print(eval.getMetrics(lid_truth, lid_pred), file = output)  






                



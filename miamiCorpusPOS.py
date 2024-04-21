import os
import posCall
import eval
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# model_name = '/scratch/gpfs/ca2992/robertuito-base-cased'
model_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-pos-lince'
tokenizer_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

out_dir = '/scratch/gpfs/ca2992/jpLLM/jpLLM/pos_out'
data_dir = '/scratch/gpfs/ca2992/jpLLM/bangor/crowdsourced_bangor'

pos_model = pipeline('ner', model=model, tokenizer=tokenizer)
pos_truth = []
pos_pred = []

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
    posResult = pos_model(message)
    index = 0
    for word in trueWords:
        posToken = posResult[index].get('word')
        # get the pos predicted for this token and append
        # to the pos word level predictions
        pos = posResult[index].get('entity')
        pos_pred.append([pos])
        # if token word mismatch impossible to handle
        if (word != posToken and word[0] != posToken[0]):
            print("MISMATCH", word, posToken)
            continue

        while (word != posToken and word[0] == posToken[0]):
            index += 1
            posToken = posToken + posResult[index].get('word')
            # get rid of # symbols added by tokenizer
            posToken = cleanPoundSign(posToken)


with open(out_dir, "a") as output:
    for file in os.listdir(data_dir):
        if os.path.isdir(data_dir  + '/' + file):
        # Skip directories and readme
            continue
        if(file == "README.md"):
            continue
        # open the current file in the directory
        with open(data_dir  + '/' + file, "r") as read:
            numWords = 0
            words = []
            message = ""
            for line in file:
                values = line.split()
                # skip blank lines or placeholder lines
                if (len(values) <= 3):
                    continue
                pos = values[3] #pos at index 3 of each line
                word = values[1] # word at index 1 of each line
                words.append(word)
                numWords += 1
                pos_truth.append([pos])
                if isContraction:
                    message = message + word
                else:
                    message = message + " " + word
                # at the end of each sentence, pass into the model
                if (word == '.'):
                    tokenToWordPred(message, words)
                    numWords = 0
                    pos = []
                    words = []
                    message = ""
    print(len(pos_truth), len(pos_pred), file = output)
    print(eval.getMetrics(pos_truth, pos_pred), file = output)  






                



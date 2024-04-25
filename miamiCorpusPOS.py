import os
import posCall
import eval
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# model_name = '/scratch/gpfs/ca2992/robertuito-base-cased'
model_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-pos-lince'
tokenizer_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

out_dir = '/scratch/gpfs/ca2992/jpLLM/jpLLM/pos_dict_out'
data_dir = '/scratch/gpfs/ca2992/jpLLM/bangor/crowdsourced_bangor'

pos_model = pipeline('ner', model=model, tokenizer=tokenizer)
pos_truth = []
pos_pred = []
lid_truth = []


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
        index += 1

i = 0
with open(out_dir, "a") as output:
    for file in os.listdir(data_dir):
        if os.path.isdir(data_dir  + '/' + file):
        # Skip directories and readme
            continue
        if(file == "README.md"):
            continue
        # open the current file in the directory
        i += 1
        if (i != 2 and i != 3):
            continue
        with open(data_dir  + '/' + file, "r") as read:
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
                pos = values[3] #pos at index 3 of each line
                lid = values[2] # lid at index 2 of each line
                word = values[1] # word at index 1 of each line
                numWords += 1
                # print(pos)
                if isContraction(word):
                    # if is a contraction, implicitly use last truth tag
                    message = message + word
                    lastWord = words.pop()
                    words.append(lastWord + word)
                else:
                    # if it is not a contraction, use the truth tag
                    message = message + " " + word
                    words.append(word)
                    pos_truth.append([pos])
                    lid_truth.append([lid])
                # at the end of each sentence, pass into the model
                if (word == '.'):
                    tokenToWordPred(message, words)
                    numWords = 0
                    pos = []
                    lid = []
                    words = []
                    message = ""
            # get any remaining tokens/words and analyze them
            if (len(message) != 0):
                tokenToWordPred(message, words)
                numWords = 0
                words = []
                message = ""
            # after each file, length of pos_truth == lngth of pos_pred
            assert len(pos_truth) == len(pos_pred)

    # print(pos_truth, file = output)
    # print(pos_pred, file = output)
    # print(len(pos_truth), len(pos_pred), file = output)

    # note, i can concatenate the pos_truth with the lid_truth
    # as long as I also concatenate lid_truth with pos_pred
    # to get stats depending on the language
    print(eval.getMetrics(pos_truth, pos_pred), file = output)  

    index = 0
    error_dict = {}
    correct_dict = {}
    for pred in pos_pred:
        truth = pos_truth[index]
        if pred[0] != truth[0]:
            if str(pred[0]) + " " + str(lid_truth[index]) in error_dict:
                count = error_dict.get(str(pred[0]) + " " + str(lid_truth[index]))
                error_dict[str(pred[0]) + " " + str(lid_truth[index])] = count + 1
            else:
                error_dict[str(pred[0]) + " " + str(lid_truth[index])] = 1
        else:
            if str(pred[0]) + " " + str(lid_truth[index]) in error_dict:
                count = correct_dict.get(pred[0] + " " + lid_truth[index])
                correct_dict[pred[0] + " " + lid_truth[index]] = count + 1
            else:
                correct_dict[pred[0] + " " + lid_truth[index]] = 1
        index += 1
    print(error_dict, file = output)
    print(correct_dict, file = output)






                


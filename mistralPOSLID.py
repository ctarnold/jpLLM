import os
import lidCall
import posCall
import lidInterpreter
read_dir = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/out_415.tsv"
read_dir_1 = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/out_415_1.tsv"
read_list = [read_dir, read_dir_1] 
out_pos = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/mistral_greedy_pos.tsv"
out_lid = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/mistral_greedy_lid.tsv"



# switch counts based on previous word
EN_SPA_prev_verb = 0 # English to Spanish conditional on prev. word verb
SPA_EN_prev_verb = 0 # English to Spanish conditional on prev. word verb

EN_SPA_prev_noun = 0 # EN to SPA conditional on prev. word noun
SPA_EN_prev_noun = 0 # SPA to EN conditional on prev. word noun

EN_SPA_prev_conj = 0 # EN to SPA conditoinal on prev. word conj
SPA_EN_prev_conj = 0 # SPA to EN conditional on prev. word conj

# switch counts based on current word
EN_SPA_verb = 0 # English to Spanish conditional on word verb
SPA_EN_verb = 0 # English to Spanish conditional on word verb

EN_SPA_noun = 0 # EN to SPA conditional on word noun
SPA_EN_noun = 0 # SPA to EN conditional on word noun

EN_SPA_conj = 0 # EN to SPA conditoinal on word conj
SPA_EN_conj = 0 # SPA to EN conditional on word conj

verb_Count = 0
noun_Count = 0
conj_Count = 0

# access to last word, pos, lid for counting purposes
last_word = ""
last_pos = ""
last_lid = ""

# so for reading corpus data,
# i need to feed each line into my pos, lid tagger
# then I need to get probability of switching
# based on pos and/or lid

# for each file:
    # for each line:
        # feed into lid model, tag each word
        # feed into pos model, tag each word
        # count CS occurrences

lid_out = []
pos_out = []
# instructions in [INST] [\INST] format:
def extractText(text):
    stateInMessage = False
    stateInBracket = False
    out = ""
    for char in text:
        if char == '[' and stateInBracket == False:
            stateInBracket = True
            stateInMessage = False
        elif char == ']' and stateInBracket == True:
            stateInBracket = False
            stateInMessage = True
        elif stateInBracket:
            continue
        elif stateInMessage:
            out = out + char
        assert stateInBracket != stateInMessage
    return out

def pos_lid(input):
    t = extractText(input)
    pos_result = posCall.pos(t)
    lid_result = lidCall.lid(t)
    pos_out.append(pos_result)
    lid_out.append(lid_result)

for file in read_list:
    with open(file, "r") as f:
        text = ""
        for line in file:
            # if start of new instruction
            # conduct lid and pos
            if line[0] == '[':
                pos_lid(text)
                text = line
            else:
                text = text +  " " + line
        # account for last examples from last prompt
        pos_lid(text)

with open(out_pos, "a") as f:
    for item in pos_out:
        print(item, file = f)
        print('\t', file = f)
with open(out_lid, "a") as f:
    for item in lid_out:
        print(item, file = f)
        print('\t', file = f)


            

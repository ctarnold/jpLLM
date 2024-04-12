import os
data_dir = "/scratch/gpfs/ca2992/jpLLM/bangor/crowdsourced_bangor"
out = "/scratch/gpfs/ca2992/jpLLM/bangor/test"

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


with open(out, "a") as output:
    for file in os.listdir(data_dir):
        if os.path.isdir(data_dir  + '/' + file):
        # Skip directories
            continue
        if(file == "README.md"):
            continue
        with open(data_dir  + '/' + file, "r") as read:
            index = 0
            for line in read:
                values = line.split()
                # skip blank lines
                if (len(values) <= 2):
                    continue
                num = values[0]
                word = values[1]
                lid = values[2]
                pos = values[3]
                print(pos)
                print(lid)
                if (lid == "punct"):
                    continue
                if (index == 0):
                    last_word = word
                    last_pos = pos
                    last_lid = lid
                else:
                    # if code-switching occurred
                    if (last_lid != lid):
                        if (last_lid == 'spa'):
                            if (last_pos == "VERB"):
                                SPA_EN_prev_verb += 1
                            if (last_pos ==  "NOUN"):
                                SPA_EN_prev_noun += 1
                            if (last_pos == "CONJ"):
                                SPA_EN_prev_conj += 1
                        if (last_lid == 'eng'):
                            if (last_pos == "VERB"):
                                EN_SPA_prev_verb += 1
                            if (last_pos ==  "NOUN"):
                                EN_SPA_prev_noun += 1
                            if (last_pos == "CONJ"):
                                EN_SPA_prev_conj += 1
                        if (last_lid == 'eng'):
                            if (pos == "VERB"):
                                EN_SPA_verb += 1
                            if (pos ==  "NOUN"):
                                EN_SPA_noun += 1
                            if (pos == "CONJ"):
                                EN_SPA_conj += 1
                        if (last_lid == 'spa'):
                            if (pos == "VERB"):
                                SPA_EN_verb += 1
                            if (pos ==  "NOUN"):
                                SPA_EN_noun += 1
                            if (pos == "CONJ"):
                                SPA_EN_conj += 1
                if (pos == "VERB"):
                    verb_Count += 1
                if (pos ==  "NOUN"):
                    noun_Count += 1
                if (pos == "CONJ"):
                    conj_Count += 1
                last_word = word
                last_pos = pos
                last_lid = lid
                index = index + 1
print((EN_SPA_prev_verb + SPA_EN_prev_verb)/verb_Count)

print((EN_SPA_prev_noun + SPA_EN_prev_noun)/noun_Count)

print((EN_SPA_prev_conj + SPA_EN_prev_conj)/conj_Count)

print((EN_SPA_verb + SPA_EN_verb)/verb_Count)

print((EN_SPA_noun + SPA_EN_noun)/noun_Count)

print((EN_SPA_conj + SPA_EN_conj)/conj_Count)




                    

                

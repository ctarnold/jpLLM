import os
import lidCall
read_dir = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/out_415.tsv"
read_dir_1 = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/out_415_1.tsv"
files = [read_dir, read_dir_1]
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

# so for reading corpus data,
# i need to feed each line into my pos, lid tagger
# then I need to get probability of switching
# based on pos and/or lid

# for each file:
    # for each line:
        # feed into lid model, tag each word
        # feed into pos model, tag each word
        # count CS occurrences
with open(read_dir_1, "a") as file:
    for line in file:
        print(lidCall(line))
        break


import os

data_dir = "/scratch/gpfs/ca2992/jpLLM/bangor/crowdsourced_bangor"
out = "/scratch/gpfs/ca2992/jpLLM/bangor/test"


for file in os.listdir(data_dir):
    for line in file:
        with open(out, "a") as f:
            print(line, f)
            print(type(line))
        
    
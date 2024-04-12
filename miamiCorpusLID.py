import os

data_dir = "/scratch/gpfs/ca2992/jpLLM/bangor/crowdsourced_bangor"
out = "/scratch/gpfs/ca2992/jpLLM/bangor/test"


with open(out, "a") as output:
    for file in os.listdir(data_dir):
        if os.path.isdir(data_dir  + '/' + file):
        # Skip directories
            continue
        with open(data_dir  + '/' + file, "r") as read:
            for line in read:
                values = line.split()
                # skip blank lines
                if (len(values) == 0):
                    continue
                word = values[0]
                lidTruth = values[1]
                # posTruth = values[2]
                print(word + " " + lidTruth + " ", output)
        
    
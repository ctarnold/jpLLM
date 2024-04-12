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
                print(values[0] + "---" + values[1] + " --" + 
                      type(values) + " " + type(values[0]), output)
                # word = values[1]
                # lidTruth = values[2]
                # posTruth = values[3]
                # print(word + " " + lidTruth + " " + posTruth, output)
        
    
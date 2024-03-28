from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("stderr", "a") as e:
    print(torch.cuda.is_available(), file = e)
model_id = "/scratch/gpfs/ca2992/Mixtral-8x7B-v0.1"
data_directory = "/scratch/gpfs/ca2992/jpLLM/jpLLM_Data/prompts.tsv"
out = []
# https://www.geeksforgeeks.org/simple-ways-to-read-tsv-files-in-python/ 
# lol
with open(data_directory, "r+") as data:
    for line in data:
        l = line.split("\t")
        out.append(l)

with open("test_out") as f:
    for i in out:
        print(i, file = f)

    
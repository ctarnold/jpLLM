from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch


with open("test.txt", "a") as f:
    if torch.cuda.is_available():
      print("cuda available\n")
    print("test complete")
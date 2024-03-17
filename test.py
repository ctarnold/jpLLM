from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch


with open("test.txt", "a") as f:
   print(torch.cuda.is_available())
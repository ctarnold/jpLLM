Introduction: 

This research project explores LLM behavior on code-switched corpora in
English and Spanish. Foundational work in this field includes the 
LinCE benchmark on code-switching tasks. Much of the existing data
in this field is pulled from short social media interactions on
Twitter (now X). Furthermore, LinCE does not explore generative
tasks. Due to the rise in generative models, examining generative
LLM behavior on similar data is important. Finally, human speech
data is available. Short tweets or posts may not accurately 
represent human interactions. Using the Bangor Miami Corpus, I 
can validate and/or fine-tune Sagor Sarker's code-switched models
(github: sagorbrur). In this way, I can expand on existing research
towards both generative tasks and human data.

Files:
job*.slurm files represent requests for compute on my clusters. You
can use these files as guidelines for RAM, CPU, GPU needs for the
code I am running.

prompting.py files are different kinds of prompts to the Mistral-8x7b
Intruct model. They can be with Spanish, English, or a Mix. 

The databricks files represent massive instruction datasets for LLMs.
In addition to the human data, this is additional data that this 
project may use. (https://huggingface.co/datasets/databricks/databricks-dolly-15k).
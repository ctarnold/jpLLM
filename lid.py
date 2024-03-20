from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

text = "me dice ella que trabaja en una tienda de furniture, so anyways that's that one. this is Chris' boyfriend."

tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")

model = AutoModelForTokenClassification.from_pretrained("/scratch/gpfs/ca2992/codeswitch-spaeng-lid-lince")
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)

with open("lidout.txt", "a") as f:
    out = lid_model(text) # out is of type list
    # an index of out is of type dictionary.
    # this is a list of dictionaries.
    # [{'entity': 'spa', 'score': 0.99990606, 'index': 1, 'word': 'me', 'start': 0, 'end': 2}
    # example index
    print(out[0].get('entity'), file = f)
    print(out[0].get('word'), file = f)
    

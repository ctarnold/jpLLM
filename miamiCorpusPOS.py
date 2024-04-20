from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# model_name = '/scratch/gpfs/ca2992/robertuito-base-cased'
model_name = '/scratch/gpfs/ca2992/codeswitch-spaeng-pos-lince'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
text = "Esto es un tweet, estoy codeswitching"

# ['<s>','▁Esto','▁es','▁un','▁tweet','▁estoy','▁usando','▁','▁hashtag','▁','▁ro','bert','uito','▁@usuario','▁','▁emoji','▁cara','▁revolviéndose','▁de','▁la','▁risa','▁emoji','</s>']

pos_model = pipeline('ner', model=model, tokenizer=tokenizer)

print(pos_model(text))
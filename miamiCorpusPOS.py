from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


model_name = '/scratch/gpfs/ca2992/robertuito-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
text = "Esto es un tweet estoy usando pysentimiento"

# ['<s>','▁Esto','▁es','▁un','▁tweet','▁estoy','▁usando','▁','▁hashtag','▁','▁ro','bert','uito','▁@usuario','▁','▁emoji','▁cara','▁revolviéndose','▁de','▁la','▁risa','▁emoji','</s>']

pos_model = pipeline('pos', model=model, tokenizer=tokenizer)

print(pos_model(text))
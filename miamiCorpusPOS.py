from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


model_name = '/scratch/gpfs/ca2992/robertuito-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
text = "Esto es un tweet estoy usando pysentimiento"

# ['<s>','▁Esto','▁es','▁un','▁tweet','▁estoy','▁usando','▁','▁hashtag','▁','▁ro','bert','uito','▁@usuario','▁','▁emoji','▁cara','▁revolviéndose','▁de','▁la','▁risa','▁emoji','</s>']

pos_model = pipeline('ner', model=model, tokenizer=tokenizer)

print(pos_model(text))
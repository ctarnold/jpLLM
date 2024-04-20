from transformers import AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained('/scratch/gpfs/ca2992/robertuito/models/beto-cased-finetuned')

text = "Esto es un tweet estoy usando pysentimiento"

# ['<s>','▁Esto','▁es','▁un','▁tweet','▁estoy','▁usando','▁','▁hashtag','▁','▁ro','bert','uito','▁@usuario','▁','▁emoji','▁cara','▁revolviéndose','▁de','▁la','▁risa','▁emoji','</s>']

pos_model = pipeline('ner', model=model, tokenizer=tokenizer)

print(pos_model(text))
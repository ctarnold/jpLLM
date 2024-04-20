from codeswitch.codeswitch import NER
ner = NER('spa-eng')
text = "estoy intentando hacer que eso funcione correctamente" # your mixed sentence 
result = ner.tag(text)
print(result)

from codeswitch.codeswitch import LanguageIdentification
lid = LanguageIdentification('spa-eng') 
# for hindi-english use 'hin-eng', 
# for nepali-english use 'nep-eng'
text = "hola estoy intentando hacer code-switching language identification" # your code-mixed sentence 
result = lid.identify(text)
print(result)
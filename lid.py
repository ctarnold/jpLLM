from codeswitch.codeswitch import LanguageIdentification
lid = LanguageIdentification('spa-eng') 
# for hindi-english use 'hin-eng', 
# for nepali-english use 'nep-eng'
with open("codeswitchTest.txt") as out:
    text1 = "me dice ella que trabaja en una tienda de furniture." # your code-mixed sentence 
    result = lid.identify(text1)
    print(result, file = out)

    text2 = "so anyways that's that one. this is Chris's boyfriend" # your code-mixed sentence 
    result = lid.identify(text2)
    print(result, file = out)

    text3 = text1 + " " + text2
    result = lid.identify(text3)
    print(result, file = out)

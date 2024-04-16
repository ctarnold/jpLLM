import os

def interpret(sourceText, lidTagged):
    words = sourceText.split()
    pairs = []
    for j in range(len(lidTagged)):
        language = lidTagged[j].get('entity')
        thisWord = lidTagged[j].get('word')
        if (thisWord[0] == '#'):
            lastWord = lidTagged[j-1].get('word')
            for k in len(thisWord):
                if (thisWord[k] != '#'):
                    lastWord += thisWord[k]
            language = lidTagged[j-1].get('entity')
            thisWord = lastWord
            pairs.pop()
        pairs.append(thisWord + " " + language)
    return pairs
        
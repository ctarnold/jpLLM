import os

def interpret(sourceText, lidTagged):
    words = sourceText.split()
    pairs = []
    for j in range(len(lidTagged)):
        language = lidTagged[j].get('entity')
        
        pairs.append(language + " " + words[j])
        
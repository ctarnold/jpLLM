
from pysentimiento import create_analyzer

pos_analyzer = create_analyzer("pos", lang="es")

pos_analyzer.predict("Quiero que esto funcione correctamente! @perezjotaeme")
 
 
>[{'type': 'PROPN', 'text': 'Quiero', 'start': 0, 'end': 6},
> {'type': 'SCONJ', 'text': 'que', 'start': 7, 'end': 10},
> {'type': 'PRON', 'text': 'esto', 'start': 11, 'end': 15},
> {'type': 'VERB', 'text': 'funcione', 'start': 16, 'end': 24},
> {'type': 'ADV', 'text': 'correctamente', 'start': 25, 'end': 38},
> {'type': 'PUNCT', 'text': '!', 'start': 38, 'end': 39},
> {'type': 'NOUN', 'text': '@perezjotaeme', 'start': 40, 'end': 53}]

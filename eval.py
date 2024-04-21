# Adapted from O'Reilly's Natural Language Processing with Transformers
# Building Language Applications with HuggingFace
# Lewis Tunstall, Leandro von Werra, Thomas Wolf
# ISBN: 978-1-098-13679-6
# Page 105
from seqeval.metrics import classification_report

def getMetrics(y_true, y_pred):
    return classification_report(y_true, y_pred)




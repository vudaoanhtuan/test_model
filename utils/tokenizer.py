import spacy
from nltk.tokenize import RegexpTokenizer

class SpacyWordSpliter:
    def __init__(self, lower=True):
        disable = ['vectors', 'textcat', 'tagger', 'parser', 'ner']
        self.tokenizer = spacy.load('en_core_web_sm', disable=disable)
        self.lower = lower
    
    def tokenize(self, sent):
        if self.lower:
            sent = sent.lower()
        words = self.tokenizer(sent)
        words = [x.text for x in words]
        return words

class NltkTokenizer:
    def __init__(self, lower=True):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lower = lower
    
    def tokenize(self, sent):
        if self.lower:
            sent = sent.lower()
        
        words = self.tokenizer.tokenize(sent)
        return words


if __name__ == "__main__":
    tokenizer = NltkTokenizer()
    words = tokenizer.tokenize("xin chao, day la TP.HCM. chao moi nguoi!")
    print(words)
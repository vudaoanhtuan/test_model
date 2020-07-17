from utils.preprocess_utils import pad_sequences

class Tokenizer:
    def __init__(self, char_vocab, word_vocab, char_pad_len=10, lower_text=True):
        self.char_pad_len = char_pad_len
        self.lower_text = lower_text
        self.special = ['[pad]', '[unk]', '[bos]', '[eos]', '[mask]', '[cls]']
        char_vocab = self.special + char_vocab
        word_vocab = self.special + word_vocab
        self.pad = 0
        self.unk = 1
        self.bos = 2
        self.eos = 3
        self.mask = 4
        self.cls = 5
        self.char_itos = {i:s for i,s in enumerate(char_vocab)}
        self.word_itos = {i:s for i,s in enumerate(word_vocab)}
        self.char_stoi = {s:i for i,s in enumerate(char_vocab)}
        self.word_stoi = {s:i for i,s in enumerate(word_vocab)}

        self.pad_token_char = [self.pad] * self.char_pad_len
        self.pad_token_char[0] = self.cls

        assert self.char_stoi['[pad]'] == self.word_stoi['[pad]'] == self.pad
        assert self.char_stoi['[unk]'] == self.word_stoi['[unk]'] == self.unk
        assert self.char_stoi['[bos]'] == self.word_stoi['[bos]'] == self.bos
        assert self.char_stoi['[eos]'] == self.word_stoi['[eos]'] == self.eos
        assert self.char_stoi['[mask]'] == self.word_stoi['[mask]'] == self.mask
        assert self.char_stoi['[cls]'] == self.word_stoi['[cls]'] == self.cls
        

    def _char_tokenize(self, word):
        return [c for c in word]

    def _word_tokenize(self, sent):
        return sent.split()

    def _token_to_id(self, tokens, stoi):
        idxs = []
        for t in tokens:
            idx = stoi.get(t, self.unk)
            idxs.append(idx)
        return idxs

    def _id_to_token(self, idxs, itos):
        tokens = []
        for i in idxs:
            t = itos.get(i, self.special[self.unk])
            tokens.append(t)
        return tokens
    
    def process_char(self, word):
        word = self._char_tokenize(word)
        word = self._token_to_id(word, self.char_stoi)
        word = [self.cls] + word
        return word

    def process_word(self, sent):
        sent = self._word_tokenize(sent)
        sent = self._token_to_id(sent, self.word_stoi)
        return sent

    def tokenize_char(self, sent):
        words = self._word_tokenize(sent)
        words = [self.process_char(w) for w in words]
        words = pad_sequences(words, self.char_pad_len, padding='post')
        return words

    def get_pad(self):
        return self.pad_token_char


def load_vocab(char_vocab_path, word_vocab_path):
    with open(char_vocab_path) as f:
        char_vocab = f.read().split('\n')
        if len(char_vocab[-1]) == 0:
            char_vocab = char_vocab[:-1]
    
    with open(word_vocab_path) as f:
        word_vocab = f.read().split('\n')
        if len(word_vocab[-1]) == 0:
            word_vocab = word_vocab[:-1]

    tokenizer = Tokenizer(char_vocab, word_vocab)

    return tokenizer

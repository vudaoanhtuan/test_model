import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
from tqdm import tqdm

from utils.preprocess_utils import pad_sequences
from utils.word_transform import transform_sentence








class Dataset(data.Dataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=50):
        self.tokenizer = tokenizer

        df = pd.read_csv(file_path, sep='\t', names=['src', 'tgt'])

        tokens = [tokenizer.tokenize(str(x.src), str(x.tgt)) for i, x in tqdm(df.iterrows())]
        self.src = [x[0] for x in tokens]
        self.tgt = [x[1] for x in tokens]

        self.src = pad_sequences(self.src, maxlen=src_pad_len, value=tokenizer.pad, padding='post')
        self.tgt = pad_sequences(self.tgt, maxlen=tgt_pad_len, value=tokenizer.pad, padding='post')

    def regenerate_source(self):
        pass

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]
        tgt_inp = tgt[:-1]
        tgt_lbl = tgt[1:]
        return src, tgt_inp, tgt_lbl


class SingleDataset(data.Dataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=50, seed=None):
        self.tokenizer = tokenizer
        

        self.src_pad_len = src_pad_len
        self.tgt_pad_len = tgt_pad_len

        with open(file_path) as f:
            self.corpus = f.read().split('\n')[:-1]

        self.src = None
        self.tgt = [tokenizer.process_word(x) for x in tqdm(self.corpus)]
        self.tgt = pad_sequences(self.tgt, maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        src = self.corpus[index]
        src, pos = transform_sentence(src)

        len_src = len(pos)
        pos = pos + [0 for _ in range(self.tgt_pad_len-len_src)]
        mask = [0]*len_src + [1 for _ in range(self.tgt_pad_len-len_src)]
        src_pad = [self.tokenizer.get_pad() for _ in range(self.tgt_pad_len-len_src)]
        src = self.tokenizer.tokenize_char(src)
        if src.shape[0] < self.tgt_pad_len:
            src = np.concatenate([src, src_pad])
        label = self.tgt[index]
        mask = np.array(mask)
        label = np.array(label, dtype=np.int64)
        if src.shape[0]>self.tgt_pad_len: src = src[:self.tgt_pad_len]
        if mask.shape[0]>self.tgt_pad_len: mask = mask[:self.tgt_pad_len]
        if label.shape[0]>self.tgt_pad_len: label = label[:self.tgt_pad_len]
        if pos.shape[0]>self.tgt_pad_len: pos = pos[:self.tgt_pad_len]
        return src, mask, label, pos


if __name__ == "__main__":
    from tokenizer import load_vocab
    tokenizer = load_vocab('vocab/char_vocab.txt', 'vocab/word_vocab.txt')
    ds = SingleDataset('data/test.txt', tokenizer)
    src, mask, label, pos = ds.__getitem__(0)
    print(src.dtype)
    print(mask.dtype)
    print(label.dtype)
    print(pos.dtype)
    label = tokenizer._id_to_token(label[:10], tokenizer.word_itos)
    print(src[:10])
    print(mask[:10])
    print(label[:10])
    print(pos[:10])

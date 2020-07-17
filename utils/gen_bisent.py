import re
import argparse
import pandas as pd

from tqdm import tqdm
import numpy as np
import spacy

from tokenizer import SpacyWordSpliter, NltkTokenizer
from word_transform import gen_wrong_word

parser = argparse.ArgumentParser()
parser.add_argument('corpus')
parser.add_argument('word_list')
parser.add_argument('--percent', type=float, default=0.5)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('-o', '--output', default='bisent.tsv')
parser.add_argument('--remove_punc', default=False, action='store_true')




def generate_bisent(sent, word_list, max_len=100, percent=0.5):
    sent = sent.replace('\t', ' ')
    sent = tokenizer.tokenize(sent)
    if len(sent) > max_len:
        sent = sent[:max_len]
    w_sent = []
    for w in sent:
        change_word = np.random.choice([True, False], p=[percent, 1-percent])
        if w in word_list and change_word and len(w) > 1:
            w = gen_wrong_word(w)
        w_sent.append(w)
    w_sent = ' '.join(w_sent)
    sent = ' '.join(sent)
    return sent, w_sent

if __name__=='__main__':
    args = parser.parse_args()

    if args.remove_punc:
        tokenizer = NltkTokenizer()
    else:
        tokenizer = SpacyWordSpliter()

    with open(args.word_list) as f:
        word_list = f.read().split('\n')[:-1]

    with open(args.corpus) as f:
        corpus = f.read().split('\n')[:-1]

    src = []
    tgt = []
    for sent in tqdm(corpus):
        s,w = generate_bisent(sent, word_list, max_len=args.max_len, percent=args.percent)
        src.append(w)
        tgt.append(s)

    df = pd.DataFrame({"src": src, "tgt": tgt})
    df.to_csv(args.output, index=False, header=False, sep='\t')
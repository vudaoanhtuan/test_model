import os
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainner import Trainer
from model import Model
from dataset import SingleDataset
from tokenizer import Tokenizer, load_vocab

parser = argparse.ArgumentParser()
parser.add_argument('--char_vocab', required=True)
parser.add_argument('--word_vocab', required=True)
parser.add_argument('--train_file', required=True)
parser.add_argument('--test_file', required=True)
parser.add_argument('--model_config')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--num_epoch', default=10, type=int)
parser.add_argument('--device', default='cpu')
parser.add_argument('--log_dir', default='logs')
parser.add_argument('--weight_dir', default='weight')

if __name__ == "__main__":
    args = parser.parse_args()

    print("Load vocab")
    tokenizer = load_vocab(args.char_vocab, args.word_vocab)

    print("Prepare data")
    train_ds = SingleDataset(args.train_file, tokenizer)
    test_ds = SingleDataset(args.test_file, tokenizer)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size)

    print("Init model")
    char_vocab_len = len(tokenizer.char_stoi)
    word_vocab_len = len(tokenizer.word_stoi)


    model = Model(char_vocab_len, word_vocab_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-9)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dl))
    trainner = Trainer(
        model, optimizer, train_dl, test_dl, 
        device=args.device, scheduler=sched,
        log_dir=args.log_dir,
        weight_dir=args.weight_dir
    )

    print("Start training")
    trainner.train(args.num_epoch)

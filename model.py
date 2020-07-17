import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_padding_mask(x, padding_value=0):
    # x: BxS
    mask = x==padding_value
    if x.is_cuda:
        mask = mask.cuda()
    return mask

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                np.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * np.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len].clone().detach()
        if x.is_cuda:
            pe = pe.cuda()
        x = x + pe
        return self.dropout(x)


class CharEncoder(nn.Module):
    def __init__(self, src_vocab_len, 
        d_model=128, nhead=4, 
        num_encoder_layers=3,
        dim_feedforward=512, 
        dropout=0.1, activation="relu",
        init_param=True):
        super().__init__()
        self.src_padding_value = 0

        self.pos_embedding = PositionalEncoder(d_model, dropout=dropout)
        self.src_embedding = nn.Embedding(src_vocab_len, d_model)


        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if init_param:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


    def forward(self, src_inp):
        # src_inp: BxSxC
        B,S,C = src_inp.shape
        src_inp = src_inp.reshape(B*S, C) # B.SxC
        src_padding_mask = generate_padding_mask(src_inp, self.src_padding_value) # B.SxC
        src_inp = self.src_embedding(src_inp) # B.SxC
        src_inp = self.pos_embedding(src_inp) # B.SxCxE
        src_inp = src_inp.transpose(0,1) # CxB.SxE

        memory = self.encoder(
            src_inp, 
            src_key_padding_mask=src_padding_mask
        ) # CxB.Sxd

        memory = memory.transpose(0, 1) # # B.SxCxd
        memory = memory[:,0,:]
        memory = memory.reshape(B,S,-1)
        return memory


class Model(nn.Module):
    def __init__(self, src_vocab_len, tgt_vocab_len, 
        d_char_model=128, d_model=256, d_cor=256,
        nhead=8, 
        num_encoder_layers=3, 
        dim_feedforward=1024, 
        dropout=0.1, activation="relu",
        init_param=True):
        super().__init__()
        self.src_padding_value = 0
        self.tgt_padding_value = 0

        self.pos_embedding = PositionalEncoder(d_model, dropout=dropout)
        self.src_embedding = CharEncoder(src_vocab_len, d_char_model)
        self.linear_char_to_word = nn.Linear(d_char_model, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.linear_out = nn.Linear(d_model, tgt_vocab_len)
        
        # correction module
        self.linear_cor_1 = nn.Linear(d_char_model+d_model, d_cor)
        self.linear_cor_2 = nn.Linear(d_cor, 2)

        if init_param:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


    def forward(self, src_inp, mask, label=None, pos_label=None):
        # src_inp: BxSxC
        # mask: BxS
        char_emb = self.src_embedding(src_inp) # BxSxd_char_model
        src_emb = self.linear_char_to_word(char_emb)
        src_emb = self.pos_embedding(src_emb) # BxSxd
        
        src_emb = src_emb.transpose(0,1) # SxBxd
        
        src_emb = self.encoder(
            src_emb, 
            src_key_padding_mask=mask
        ) # SxBxd_model
        word_emb = src_emb.transpose(0,1) # BxSxd      
        output = self.linear_out(word_emb)
        
        cor_emb = torch.cat([char_emb, word_emb], dim=-1)
        cor_emb = self.linear_cor_1(cor_emb)
        cor_output = self.linear_cor_2(cor_emb)
        
        
        if label is not None:
            loss_word = F.cross_entropy(
                output.reshape(-1, output.shape[-1]), 
                label.reshape(-1), 
                ignore_index=self.tgt_padding_value
            )
            
            loss_pos = F.cross_entropy(
                cor_output.reshape(-1, cor_output.shape[-1]), 
                pos_label.reshape(-1), 
                ignore_index=self.tgt_padding_value
            )
            
            return output, cor_output, loss_word, loss_pos
        
        return output, cor_output
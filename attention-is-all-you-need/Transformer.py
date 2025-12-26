import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer
from PositionalEncoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p_dropout, num_layers, vocab_size, max_len=5000):
        super().__init__()

        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, p_dropout, max_len)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, p_dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, p_dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        '''
        Args:
            src: input sequence, shape (batch_size, src_seq_len)
            tgt: target sequence, shape (batch_size, tgt_seq_len)
            src_mask: padding mask for input, shape (batch_size, 1, 1, src_seq_len) or None
            tgt_mask: padding mask for target, shape (batch_size, 1, 1, tgt_seq_len) or None
        '''

        # x is src embeddings
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # y is tgt embeddings
        y = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        y = self.pos_encoding(y)

        # for masked self attn: compute combined mask
        # torch.tril sets the upper triangular part of a matrix to 0, torch.ones does exactly what it sounds like
        seq_len = tgt.size(1)
        look_ahead_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()

        # combine output mask with padding mask
        if tgt_mask is None:
            combined_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # Add batch & head dims
        else:
            combined_mask = tgt_mask & look_ahead_mask

        # encoder
        for layer in self.encoder:
            x, attn_weights = layer(x, src_mask)

        # decoder
        for layer in self.decoder:
            y, attn_weights = layer(y, x, combined_mask, src_mask)

        # fully connected layer
        y = self.fc(y)

        # no need for final softmax during training

        return y
        

        
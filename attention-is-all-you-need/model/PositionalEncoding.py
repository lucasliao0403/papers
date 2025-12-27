import torch
import torch.nn as nn
import math

'''

Reference equation:

PE(pos,2i) = sin(pos/10000^(2i/dmodel))
PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))

'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p_dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=p_dropout)

        # init matrix
        pe = torch.zeros(max_len, d_model)

        # numerator in equation
        # arange returns a 1d tensor like [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term is a 1d tensor like [1, 1/sqrt(2), 1/sqrt(4), ...]
        # use equivalent log identity to simplify multiplication
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # the even-indexed columns of pe
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # the odd-indexed columns of pe
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # register as buffer (not a learnable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        '''
        # add positioning to embeddings
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x

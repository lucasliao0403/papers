import torch
import torch.nn as nn
import torch.nn.functional as F
from MultiHeadAttention import MultiHeadAttention 
from PositionwiseFeedForward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p_dropout):
        super().__init__()
        
        self.multihead_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, p_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.multihead_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights

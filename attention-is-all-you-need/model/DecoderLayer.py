import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MultiHeadAttention import MultiHeadAttention 
from model.PositionwiseFeedForward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p_dropout):
        super().__init__()
        
        self.masked_multihead_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, p_dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p_dropout)

        


    def forward(self, x, encoder_out, decoder_mask=None, encoder_padding_mask=None):
        
        attn_out, attn_weights = self.masked_multihead_attn(q=x, k=x, v=x, mask=decoder_mask)
        x = self.norm1(x + self.dropout(attn_out))

        attn_out, attn_weights = self.cross_attn(q=x, k=encoder_out, v=encoder_out, mask=encoder_padding_mask)
        x = self.norm2(x + self.dropout(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x, attn_weights

# This module implements the core components of the Transformer model.

# torch is core pytorch library. contains tensor computation, autograd, and neural networks.
import torch 

# nn is a module of torch that contains neural network layers and activation functions.
import torch.nn as nn

# nn.functional contains functions including activation, loss, pooling functions etc.
import torch.nn.functional as F

import math


'''
pre: 
q, k, v are tensors of shape (batch_size, seq_len, d_q)
where d_q = d_k = d_v = d_model / heads

post:
output.shape = (seq_len, d_v)
attn.shape = (seq_len, seq_len)

seq_len is literally the length of the sequence.
in self attention, this is up to 512 tokens.

'''
def scaled_dot_product_attention(q, k, v, mask=None):
    # get query dimensions
    d_k = q.size(-1)
    
    # compute attention
    # note: K^T is the transpose of K. In multihead, this means swapping the last two dims
    # We use -2 and -1 so it works for ANY number of dimensions (batch, head, seq, d_k)
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores /= math.sqrt(d_k)

    # apply masking. 
    # mask == 0 is because masked_fill only accepts boolean tensors, and this creates a boolean tensor in-place.
    # -1e9 is effectively negative infinity.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf')) 
    
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn


def create_padding_mask(src_tokens, pad_token_id=0):
    '''
    identify pads
    assume 0 is the pad token id
    Result shape: (batch_size, 1, 1, seq_len)
    '''

    #.unsqueeze inserts a dimension of size 1 at the specified position
    mask = (src_tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask
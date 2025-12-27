# This module implements the Multi-Head Attention mechanism.
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # where W_Q, W_K, W_V, W_O are learnable parameters
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        '''
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
        where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)

        In self-attention, q, k, v are all the same input.
        '''
        # step 1:
        # calculate Q * W_Q, K * W_K, V * W_V
        # then split into num_heads
        # note: @ means matmul
        q_projected = self.w_q(q)
        k_projected = self.w_k(k)
        v_projected = self.w_v(v)

        # .view(a, b, c) reshapes a tensor into dimensions a, b, c. -1 means infer (fill) dimensions
        # will raise error if a*b*c isn't the same product as the old dims.
        # we transpose into (batch_size, seq_len, num_heads, d_head) for easier batching
        # transpose(a, b) SWAPS dims with index a, b
        # 1. Get dimensions
        batch_size, seq_len, d_model = q_projected.shape
        d_head = d_model // self.num_heads
        
        # 2. View and Transpose
        # From: (batch, seq, d_model)
        # To View: (batch, seq, heads, d_head) 
        # To Transpose: (batch, heads, seq, d_head)
        q_heads = q_projected.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2)
        k_heads = k_projected.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2)
        v_heads = v_projected.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2)
        
        # 3. calculate SDPA
        # for loops are the enemy!!
        # calculate attn in parallel
        # attn_out shape: (batch_size, num_heads, seq_len, d_head)
        attn_out, attn_weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        # 4. Concatenate Heads
        # Transpose back: (batch, seq, heads, d_head)
        # View back: (batch, seq, d_model)
        # contiguous() is often needed after transpose/permute to fix memory layout for view()
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.w_o(attn_out)

        # attn weights are only used for observability
        return output, attn_weights

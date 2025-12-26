import torch
from model import scaled_dot_product_attention, create_padding_mask
from MultiHeadAttention import MultiHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward
from EncoderLayer import EncoderLayer
import math

def test_shapes_scaled_dot_product_attention():
    batch_size = 2
    seq_len = 8
    d_k = 16
    d_v = 16
    
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_v)
    
    output, attn = scaled_dot_product_attention(q, k, v)
    
    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_v), f"Output shape mismatch: {output.shape}"
    assert attn.shape == (batch_size, seq_len, seq_len), f"Attention shape mismatch: {attn.shape}"
    
    # Check if attention weights sum to 1 (softmax)
    sum_attn = attn.sum(dim=-1)
    assert torch.allclose(sum_attn, torch.ones_like(sum_attn)), "Attention weights do not sum to 1"
    
    print("test_shapes_scaled_dot_product_attention() passed")

def test_multihead_attention():
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, attn = mha(q, k, v)

    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    assert attn.shape == (batch_size, num_heads, seq_len, seq_len), f"Attention shape mismatch: {attn.shape}"
    print("test_multihead_attention() passed")
    
def test_feed_forward():
    batch_size = 2
    seq_len = 8
    d_model = 64
    d_ff = 256
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
    output = ffn(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    print("test_feed_forward() passed")

def test_encoder_layer():
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4
    d_ff = 256
    p_dropout = 0.1
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff, p_dropout)
    output, attn_weights = encoder_layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), f"Attn weights shape mismatch: {attn_weights.shape}"
    print("test_encoder_layer() passed")

def test_padding_mask():
    # 0 = PAD, 1 = Word
    # Batch size 1, Seq len 4
    src_tokens = torch.tensor([[1, 1, 0, 0]]) 
    mask = create_padding_mask(src_tokens, pad_token_id=0)
    
    # Values should be True for words, False for pads
    # Shape after create_padding_mask: (1, 1, 1, 4)
    expected_mask = torch.tensor([[[[True, True, False, False]]]])
    assert torch.equal(mask, expected_mask), f"Mask mismatch. Got {mask}"
    
    # Functional test: verify SDPA actually masks
    q = torch.randn(1, 1, 4, 16)
    k = torch.randn(1, 1, 4, 16)
    v = torch.randn(1, 1, 4, 16)
    
    output, attn = scaled_dot_product_attention(q, k, v, mask)
    
    # Attention weights for the last two columns should be effectively 0
    # softmax of -1e9 is nearly 0
    assert torch.all(attn[:, :, :, 2:] < 1e-5), f"Masking failed. Attn weights: {attn}"
    print("test_padding_mask() passed")

if __name__ == "__main__":
    test_shapes_scaled_dot_product_attention()
    test_multihead_attention()
    test_feed_forward()
    test_encoder_layer()
    test_padding_mask()
    print("\nAll component tests passed successfully!")

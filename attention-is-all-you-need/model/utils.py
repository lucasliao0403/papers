import torch 

def create_padding_mask(src_tokens, pad_token_id=0):
    '''
    identify pads
    assume 0 is the pad token id
    Result shape: (batch_size, 1, 1, seq_len)
    '''

    #.unsqueeze inserts a dimension of size 1 at the specified position
    mask = (src_tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask
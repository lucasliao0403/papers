import torch
import string

# special tokens
SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>']

# lowercase letters a-z
LETTERS = list(string.ascii_lowercase)  # ['a', 'b', 'c', ..., 'z']

# basic punctuation
PUNCTUATION = [' ', '.', ',', '?', '!', "'", '-', ':']

# combine all characters
ALL_CHARS = SPECIAL_TOKENS + LETTERS + PUNCTUATION

# create vocabulary: character -> index mapping
vocab = {char: idx for idx, char in enumerate(ALL_CHARS)}

# create reverse mapping: idx -> character (for decoding)
idx_to_char = {idx: char for char, idx in vocab.items()}

# vocab size (needed for model init)
VOCAB_SIZE = len(vocab) 

def tokenize(text, vocab):
    """
    convert text string to list of token idxs
    """
    tokens = [vocab['<SOS>']]
    for char in text:
        if char in vocab:
            tokens.append(vocab[char])
        else:
            # if character not in vocab, skip it or handle as unknown
            print(f"char '{char}' not in vocabulary, skipping")
    tokens.append(vocab['<EOS>'])
    return tokens


def detokenize(token_ids, idx_to_char):
    """
    convert list of token idxs back to text string
    """
    if torch.is_tensor(token_ids):
        token_ids = token_ids.tolist()
    
    chars = []
    for idx in token_ids:
        char = idx_to_char.get(idx, '')
        # skip special tokens when decoding
        if char not in ['<PAD>', '<SOS>', '<EOS>']:
            chars.append(char)
    
    return ''.join(chars)

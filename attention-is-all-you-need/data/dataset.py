import torch
from torch.utils.data import Dataset
from data.vocab import tokenize
import random

class TrainingDataset(Dataset):
    def __init__(self, data, vocab, num_samples = 10000, min_len = 5, max_len = 20):
        self.data = data
        self.vocab = vocab
        self.num_samples = num_samples
        self.max_len = max_len
        self.min_len = min_len

        # Characters we can sample from (exclude special tokens)
        self.chars = [c for c in vocab.keys() if c not in ['<PAD>', '<SOS>', '<EOS>']]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random sequence
        length = random.randint(self.min_len, self.max_len)
        source_text = ''.join(random.choices(self.chars, k=length))
        target_text = source_text[::-1]  # reverse it
        
        # Tokenize both
        source_tokens = tokenize(source_text, self.vocab)
        target_tokens = tokenize(target_text, self.vocab)
        
        return {
            'source': torch.tensor(source_tokens, dtype=torch.long),
            'target': torch.tensor(target_tokens, dtype=torch.long)
        }

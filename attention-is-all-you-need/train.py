"""
Main training script
"""

import torch
import torch.nn as nn
from model.Transformer import Transformer
from data.vocab import vocab, VOCAB_SIZE
from data.dataloader import get_dataloaders
from config import Config

# TODO: Model initialization will go here
# TODO: Training loop will go here

# initialize model
model = Transformer(
    d_model=Config.d_model,
    num_heads=Config.num_heads,
    d_ff=Config.d_ff,
    num_layers=Config.num_layers,
    p_dropout=Config.p_dropout,
    max_len=Config.max_len,
    vocab_size=VOCAB_SIZE
)

# move model to device
model.to(device)

if __name__ == "__main__":
    print("Training script initialized")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

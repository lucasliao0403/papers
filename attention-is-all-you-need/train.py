"""
Main training script
"""

import torch
import torch.nn as nn
from data.vocab import vocab, VOCAB_SIZE
from data.dataloader import get_dataloaders
from config import Config

# TODO: Model initialization will go here
# TODO: Training loop will go here

if __name__ == "__main__":
    print("Training script initialized")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

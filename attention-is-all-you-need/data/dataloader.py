"""
DataLoader setup and collate function
"""

import torch
from torch.utils.data import DataLoader
from data.dataset import TrainingDataset
from data.vocab import vocab
from config import Config


def collate_fn(batch):
    """
    takes a list of samples and combines them into a batch.
    pads sequences to the same length.
    """
    # extract sources and targets
    sources = [item['source'] for item in batch]
    targets = [item['target'] for item in batch]
    
    # pad to longest in this batch
    source_padded = torch.nn.utils.rnn.pad_sequence(
        sources, 
        batch_first=True,
        padding_value=0  # <PAD> token index
    )
    target_padded = torch.nn.utils.rnn.pad_sequence(
        targets,
        batch_first=True, 
        padding_value=0
    )
    
    # create padding masks (true where real data)
    source_mask = (source_padded != 0)  # (batch_size, max_src_len)
    target_mask = (target_padded != 0)  # (batch_size, max_tgt_len)
    
    return {
        'source': source_padded,      # (batch_size, max_src_len)
        'target': target_padded,      # (batch_size, max_tgt_len)
        'source_mask': source_mask,   # (batch_size, max_src_len)
        'target_mask': target_mask    # (batch_size, max_tgt_len)
    }


def get_dataloaders():
    """
    create train and validation dataloaders
    """
    train_dataset = TrainingDataset(
        data=None,
        vocab=vocab,
        num_samples=Config.num_train_samples,
        min_len=Config.min_seq_len,
        max_len=Config.max_seq_len
    )
    
    val_dataset = TrainingDataset(
        data=None,
        vocab=vocab,
        num_samples=Config.num_val_samples,
        min_len=Config.min_seq_len,
        max_len=Config.max_seq_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

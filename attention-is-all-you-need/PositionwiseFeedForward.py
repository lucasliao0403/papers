# This module implements the Position-wise Feed-Forward Network.
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # scale up from d_model = 512 to d_ff = 2048 then scale back down.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        '''
        Two fully connected linear layers with a relu activation.
        scale up from d_model = 512 to d_ff = 2048 then scale back down.
        
        Inputs:
        x: current data (coming out of multihead attention)
        '''
        # 
        y = self.w_1(x)
        y = F.relu(y)
        y = self.w_2(y)
        return y

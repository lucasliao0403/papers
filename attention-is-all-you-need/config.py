"""
config file for hyperparameters and settings
"""

class Config:
    # model stuff
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    p_dropout = 0.1
    max_len = 100
    
    # training
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    
    # optimizer 
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_eps = 1e-9
    
    # data
    num_train_samples = 50000
    num_val_samples = 5000
    min_seq_len = 3
    max_seq_len = 15
    
    # loss
    label_smoothing = 0.1
    ignore_index = 0  # <PAD> token

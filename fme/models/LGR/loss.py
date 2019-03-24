# NAME: loss.py
# DESCRIPTION: loss function for Learn Good Correspondence

import torch
import torch.nn as nn

class LGR_loss(nn.Module):
    """
    loss function for LGR
    INCLUDING classification loss, essential matrix loss
    """
    def __init__(self):
        super(LGR_loss, self).__init__()

    def forward():
        

# NAME: loss.py
# DESCRIPTION: loss function for Learn Good Correspondence

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import get_config

config, unparsed = get_config()

def skew(v):
    zero = torch.zeros_like(v[:,0])

    M = torch.stack([
        zero, -v[:,2], v[:,1],
        v[:,2], zero, -v[:,0],
        -v[:,1], v[:,0], zero,
    ], dim=1)

    return M

def print_tensor(x_in, y_in, Rs, ts):
        
        print("#"*25, " x_in ", "#"*25)
        print(x_in)
        
        print("#"*25, " y_in ", "#"*25)
        print(y_in)

        print("#"*25, " Rs ", "#"*25)
        print(Rs)

        print("#"*25, " ts ", "#"*25)
        print(ts)


class LGR_loss(nn.Module):
    """
    loss function for LGR
    INCLUDING classification loss, essential matrix loss
    """
    def __init__(self, threshold, classif_weight, essential_weight):
        super(LGR_loss, self).__init__()
        self.threshold = threshold
        self.classif_weight = classif_weight
        self.essential_weight = essential_weight

    def forward(self, databatch, preds):
        x_in = databatch["x_in"]
        y_in = databatch["y_in"]
        R_in = databatch["R"]
        t_in = databatch["t"]

        essential = preds["essential"]
        logits = preds["logits"]
        
        # print_tensor(x_in, y_in, R_in, t_in)
        batch_size = x_in.size(0)
        num_point = x_in.size(3)
        
        # prepare for classification loss
        geometric_distance = y_in[:,0,:]

        # prepare for essential matrix loss
        translate = skew(t_in).view(batch_size, 3, 3)
        rotation = R_in.view(batch_size, 3, 3)
        truth_essential = torch.matmul(translate, rotation).view(batch_size, 9)
        truth_essential /= torch.norm(truth_essential, dim=1, keepdim=True)

        # essential matrix loss
        essential_loss = torch.mean(torch.min(
            torch.sum(torch.pow(essential - truth_essential, 2), dim=1),
            torch.sum(torch.pow(essential + truth_essential, 2), dim=1),
        ))
        # classification loss
        pos = (geometric_distance < self.threshold).float()
        neg = (geometric_distance > self.threshold).float()
        c = pos - neg
        classification_loss = -torch.log(torch.sigmoid(c * logits))
        

        # balance
        num_pos = torch.sum(pos, dim=1)
        num_neg = torch.sum(neg, dim=1)

        classification_loss_p = torch.sum(classification_loss * pos, dim=1)
        classification_loss_n = torch.sum(classification_loss * neg, dim=1)

        classification_loss_balance = torch.sum(classification_loss_p * 0.5 / num_pos + classification_loss_n * 0.5 / num_neg)

        precision = (torch.sum((logits > 0).float() * pos, dim=1) / torch.sum((logits < 0).float() * neg, dim=1)).mean()
        recall = (torch.sum((logits > 0).float() * pos, dim=1) / num_pos).mean() 

        # not add l2 loss now, but maybe needed in the future

        # add classification loss and essential matrix loss [and l2 loss]
        loss = self.classif_weight * classification_loss_balance + self.essential_weight * essential_loss

        return loss


if __name__ == "__main__":
    lgr_loss = LGR_loss(config.threshold, config.classif_weight, config.essential_weight)


    x_in = torch.rand(16, 4, 1, 128)
    y_in = torch.rand(16, 2, 128) / 7000 
    ts = torch.rand(16, 3)
    Rs = torch.rand(16, 9)

    data_batch = {"x_in":x_in, "y_in":y_in, "R": Rs, "t":ts}
    

    essential = torch.rand(16, 9)
    logits = torch.rand(16, 128)

    preds = {"essential":essential, "logits":logits}

    loss = lgr_loss(data_batch, preds)

    print("#"*25, " loss value ", "#"*25)
    print(loss)

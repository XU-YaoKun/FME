# NAME: loss.py
# DESCRIPTION: loss function for Learn Good Correspondence

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
    ])

    return M

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

        batch_size = x_in.size(0)
        num_point = x_in.size(1)

        # prepare for classification loss
        geometric_distance = y_in[:,:,0]

        # prepare for essential matrix loss
        translate = skew(t_in).view(batch_size, 3, 3)
        rotation = R_in.view(batch_size, 3, 3)
        truth_essential = torch.matmul(translate, rotation).view(batch_size, 9)
        truth_essential /= torch.norm(truth_essential, axis=1, keepdims=True)
        
        # essential matrix loss
        essential_loss = torch.reduce_mean(torch.minimum(
            torch.reduce_sum(torch.square(essential - truth_essential)),
            torch.reduce_sum(torch.square(essential + truth_essential)),
        ))

        # classification loss
        pos = torch.to_float(geometric_distance < self.threshold)
        neg = torch.to_float(geometric_distance > self.threshold)
        c = pos - neg
        classification_loss = -torch.log(nn.sigmod(c * logits))

        # balance
        num_pos = torch.sum(pos, 1)
        num_neg = torch.sum(neg, 1)

        classification_loss_p = torch.sum(classification_loss * pos, 1)
        classification_loss_n = torch.sum(classification_loss * neg, 1)

        classification_loss_balance = torch.sum(classification_loss_p * 0.5 / num_pos + classification_loss_n * 0.5 / num_neg)

        precision = (torch.sum((logits > 0).float() * pos, 1) / torch.sum((logits < 0).float * neg, 1)).mean()
        recall = (torch.sum((logits > 0).float * pos, 1) / num_pos).mean() 

        # not add l2 loss now, but maybe needed in the future

        # add classification loss and essential matrix loss [and l2 loss]
        loss = self.classif_weight * classification_loss + self.essential_weight * essential_loss

        return loss


if __name__ == "__main__":
    lgr_loss = LGR_loss(config.threshold, config.classif_weight, config.essential_weight)

    data_batch = {}

    x_in = torch.rand()
    y_in = torch.rand()
    ts = torch.rand()
    Rs = torch.rand()

    preds = {}

    essential = torch.rand()
    logits = torch.rand()

    loss = lgr_loss(data_batch, preds)

    print("#"*25, " loss value ", "#"*25)
    print(loss)

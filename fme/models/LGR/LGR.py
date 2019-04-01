# NAME: LGR.py
# DESCRIPTION: learn to find good correspondences model

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Conv1d, resNetBlcok
from config import get_config

class weightNet(nn.Module):
    """
    network to calculate weight for each correspondence 
    """
    def __init__(self,
                 depth,
                 in_channels,
                 out_channels,
                 ksize=1):
        """
        input:
        x: [batch_size x 1 x num_point x 4]
        output:
        logits: [batch_size x num_point]
        args:
        depth: number of resnet block in this module
        in_channels: the first input channel should be 4 [x1, x2, y1, y2]
        out_channels: num of channel in resnet block
        """
        super(weightNet, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize

        self.first_conv = Conv1d(self.in_channels, self.out_channels, self.ksize)

        self.resnet_block_list = nn.ModuleList()
        in_channel = self.out_channels
        for _ksize, _nchannel in zip([ksize]*depth, [self.out_channels]*depth):
            resnet_block = resNetBlcok(in_channel, _nchannel, _ksize)
            self.resnet_block_list.append(resnet_block)
            in_channel = _nchannel

        # output layer
        self.final_conv = Conv1d(in_channel, 1, 1)
        


    def forward(self, x_in):
        x = x_in

        x = self.first_conv(x)
        for block in self.resnet_block_list:
            x = block(x)

        x = self.final_conv(x)

        batch_size = x_in.size(0)
        n_cor = x_in.size(3) 
        logits = x.view(batch_size, n_cor)

        return logits

class LGR(nn.Module):
    """
    deep learning model to find good correspondences 
    """

    def __init__(self, config):
        super(LGR, self).__init__()

        self.config = config

        self.weight_net = weightNet(config.net_depth, 4, config.net_channel, 1)

    def forward(self, data_batch):

        x_in = data_batch["correspondence"]

        batch_size = x_in.size(0)
        num_point = x_in.size(3)

        logits = self.weight_net(x_in)

        weight = F.relu(F.tanh(logits))

        xx = x_in.squeeze(2).transpose(1,2)
        X = torch.stack([xx[:,:,2]*xx[:,:,0], xx[:,:,2]*xx[:,:,1], xx[:,:,2],
                        xx[:,:,3]*xx[:,:,0], xx[:,:,3]*xx[:,:,1], xx[:,:,3],
                        xx[:,:,0], xx[:,:,1], torch.ones(batch_size, num_point)], dim=1)

        # construct eight point algorithm to solve the essential matrix
        X = X.transpose(1,2)
        wX = weight.unsqueeze(2) * X
        XwX = torch.matmul(X.transpose(1,2), wX)

        # get the smallest eigenvalue and corresbonding eigenvector
        v = torch.tensor([])

        for i in range(batch_size):
            m = XwX[i,:,:]
            print(m)
            eigenvalue, eigenvector = torch.eig(m, eigenvectors=True)
            m_index = torch.argmin(eigenvalue[:,0])
            v = torch.cat((v, eigenvector[:,m_index].unsqueeze(0)), 0)

        essential = v
        essential /= torch.norm(essential, dim=1, keepdim=True)

        return essential

if __name__ == "__main__":
    data_batch = {}
    x_in = torch.rand(16, 4, 1, 128)

    config, unparsed = get_config()
    weight_net = LGR(config)
    print("Build model:\n{}".format(str(weight_net)))
   
    data_batch["correspondence"] = x_in
    weight = weight_net(data_batch)
    print("#"*50)
    print(weight)
    print("#"*50)
    print("weight size: ", weight.size())
    print("#"*50)


       

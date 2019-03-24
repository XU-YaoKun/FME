# NAME: LGR.py
# DESCRIPTION: learn to find good correspondences model

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Conv2d, resNetBlcok

class weightNet(nn.Module):
    """
    network to calculate weight for each correspondence 
    """
    def __init__(self,
                 depth,
                 nchannel,
                 ksize):
        super(weightNet, self).__init__()

        self.first_conv = Conv2d()

        self.depth = depth
        self.nchannel = nchannel

        self.resnet_block_list = nn.ModuleList()
        for _ksize, _nchannel in zip([ksize]*depth, [nchannel]*depth):
            resnet_block = resNetBlcok()
            self.resnet_block_list.append(resnet_block)

        self.final_conv = Conv2d()


    def forward(self, x_in):
        x = x_in

        x = self.final_conv(x)

        for block in self.resnet_block_list:
            x = block(x)

        x = self.final_conv(x)

        batch_size = torch.size(x_in, 0)
        n_cor = torch.size(x_in, 0)
        logits = x.view(batch_size, n_cor)

        return logits

class LGR(nn.Module):
    """
    deep learning model to find good correspondences 
    """

    def __init__(self, config):
        
        self.config = config

        # need to specify parameter here
        self.weight_net = weightNet()

    def forward(self, data_batch):

        x_in = data_batch["correspondence"]

        batch_size = x_in.size(0)
        num_point = x_in.size(1)

        logits = self.weight_net(x_in)

        weight = F.relu(F.tanh(logits))

        xx = x_in.view(batch_size, num_point, 4)

        X = torch.stack(xx[:,2]*xx[:,0], xx[:,2]*xx[:,1], xx[:,2],
                        xx[:,3]*xx[:,0], xx[:,3]*xx[:,1], xx[:,3],
                        xx[:,0], xx[:,1], torch.ones(xx[:,0]), axis=1)

        # dimension might be wrong here
        # construct eight point algorithm to solve the essential matrix
        wX = weight.view(batch_size, num_point, 1) * X
        XwX = X * wX

        # get the smallest eigenvalue and corresbonding eigenvector
        e, v = svd(XwX)

        essential = v[:,0].view(batch_size, 9)
        essential /= torch.normal(essential, axis=1, keepdim=True)

        

        


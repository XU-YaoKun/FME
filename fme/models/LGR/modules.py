# NAME: modules.py
# DESCRIPTION: modules will be used for LGR, more specifically resNet block and Conv1 block

import torch
import torch.nn as nn
import torch.nn.functional as F

from fme.nn import SharedMLP, Conv2d, MLP

class ContextNorm(nn.Module):
    """
    context Normalization
    """
    def __init__(self, eps):
        super(ContextNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        x = nn.BatchNorm2d(x, self.eps, momentum=0, affine=False)
        return x


class Conv2d(nn.Module):
    """
    customized layer for conv1d
    input: BxHxWxin_channel
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 cn=False,
                 bn=True,
                 activation="post"):
        super(Conv2d, self).__init__()

        assert activation == "post" or activation == "pre"

        self.cn = cn
        self.bn = bn
        self.activation = activation

        self.contextNorm = ContextNorm(eps=1e-3)
        self.batchNorm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert self.activation == "post" or self.activation == "pre"

        if self.activation == "pre":
            x = self.normalize(x)
        
        x = self.conv(x)

        if self.activation == "post":
            x = self.normalize(x)

        x = F.relu(x, inplace=True)

        return x

    def normalize(self, x):
        if self.bn:
            x = self.batchNorm(x)
        if self.cn:
            x = self.contextNorm(x)
        return x



class resNetBlcok(nn.Module):
    """
    resNet block for finding good correspondences
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 midchannels=None,
                 cn=True,
                 bn=True,
                 activation="post"):
        super(resNetBlcok, self).__init__()

        if midchannels is None:
            midchannels = out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = midchannels

        # I choose not to do mid activation
        pass_layer = Conv2d(in_channels, out_channels, [1,1])

        if midchannels != out_channels:
            self.pre_bottle_neck = Conv2d(mid_channels, out_channels, [1,1])
        
        in_channel_list = [in_channels, out_channels]
        out_channel_list = [out_channels, out_channels]

        self.main_conv = nn.ModuleList()
        for in_channel, out_channel in zip(in_channel_list, out_channel_list):
            conv = Conv2d(in_channels, out_channels, 1)
            self.main_conv.append(conv)

        if midchannels != out_channels:
            self.pos_bottle_nect = Conv2d(midchannels, out_channels, 1)

    def forward(self, x):
       
        inchannel = x.size(1)

        if inchannel != self.in_channels:
            x_ = self.pass_layer(x)
        else:
            x_ = x
        
        if self.mid_channels != self.out_channels:
            x = self.pre_bottle_neck(x)

        for layer in self.main_conv:
            x = layer(x)
        
        if self.mid_channels != self.out_channels:
            x = self.pos_bottle_nect(x)

        return x + x_
        
        

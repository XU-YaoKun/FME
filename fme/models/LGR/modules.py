# NAME: modules.py
# DESCRIPTION: modules will be used for LGR, more specifically resNet block and Conv1 block

import torch
import torch.nn as nn

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
                 stride=[1,1,1,1],
                 activation="post"):
        super(Conv2d, self).__init__()

        assert activation == "post" or activation == "pre"

        self.cn = cn
        self.bn = bn
        self.activation = activation

        self.contextNorm = ContextNorm(eps=1e-3)
        self.batchNorm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.relu = nn.Relu(inplace=True)

    def forward(self, x):
        assert self.activation == "post" or self.activation == "pre"

        if activation == "pre":
            x = self.normalize(x)
        
        x = self.conv(x)

        if activation == "post"
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

        self.in_channels = in_channels
        self.mid_channels = midchannels

        if midchannels is None:
            midchannels = out_channels

        # I choose not to do mid activation

        pass_layer = Conv2d(in_channels, out_channels, [1,1])

        pre_bottle_neck = Conv2d()

        self.main_conv = nn.ModuleList()
        for in, out in zip(in_channels, out_channels):
            conv = Conv2d()
            self.main_conv.append(conv)

        pos_bottle_nect = Conv2d()

    def forward(self, x):
        
        inchannel = x.size(1)

        if inchannel != self.in_channels:
            x_ = self.pass_layer(x)
        
        if midchannels != self.in_channels:
            x = self.pre_bottle_neck(x)

        for layer in self.main_conv:
            x = layer(x)
        
        if midchannels != self.in_channels:
            x = self.pos_bottle_nect(x)

        return x + x_
        
        




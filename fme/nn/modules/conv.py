from torch import nn
import torch.nn.functional as F

from ..init import init_bn


class Conv1d(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes.
    optionally followed by batch normalization and relu activation

    Attributes:
        conv (nn.Module): convolution module
	bn (nn.Module): batch normalization module
	relu (nn.Module, optional): relu activation module
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 relu = True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_fn=None):
        """default initialization"""
        if init_fn is not None:
            init_fn(self.conv)
        if self.bn is not None:
            init_bn(self.bn)



class Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_fn=None):
        if init_fn is not None:
            init_fn(self.conv)
        if self.bn is not None:
            init_bn(self.bn)

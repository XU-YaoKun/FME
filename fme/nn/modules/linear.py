from torch import nn
from ..init import init_bn

class FC(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True, bn=True, bn_momentum=0.1):
        super(FC, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels, bias=(not bn))
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def init_weights(self, init_fn=None):
        if init_fn is not None:
            init_fn(self.fc)
        if self.bn is not None:
            init_bn(self.bn)

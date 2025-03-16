import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        ) if downsample else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x= self.relu(self.bn1(self.conv1(x)))
        x= self.relu(self.bn2(self.conv2(x)))   
        x= self.bn3(self.conv3(x))
        return self.relu(x + residual)
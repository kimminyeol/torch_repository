import torch
import torch.nn as nn
from resnet.model.bottleneck_block import BottleneckBlock
from resnet.model.residual_block import ResidualBlock

# 일반 ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes=10 , block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], 2)
        self.layer3 = self._make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        





# bottleneck ResNet
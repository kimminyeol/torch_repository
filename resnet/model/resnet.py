import torch
import torch.nn as nn
from model.bottleneck_block import BottleneckBlock
from model.residual_block import ResidualBlock

# 일반 ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes=10, block=ResidualBlock, layers=[2, 2, 2, 2]):
        super(ResNet, self).__init__()
        self.in_channels = 64  # ResNet-18/34와 ResNet-50/101의 초기 in_channels 차이 반영
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * (4 if block == BottleneckBlock else 1), num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * (4 if block == BottleneckBlock else 1):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * (4 if block == BottleneckBlock else 1),
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * (4 if block == BottleneckBlock else 1))
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * (4 if block == BottleneckBlock else 1)

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

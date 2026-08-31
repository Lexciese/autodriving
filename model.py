import torch
from torch import torch, cat, nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.block1.apply(kaiming_init)
        self.block2.apply(kaiming_init)

    def forward(self, x):
        y = self.block1(x)
        x = self.block2(y)
        return y

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feature_map, kernel_size, dilation):
        super().__init__()
        self.layer1 = DoubleConv(in_channels=in_channels, out_channels=feature_map[0], kernel_size=5, dilation=3)
        self.layer2 = DoubleConv(in_channels=feature_map[0], out_channels=feature_map[1], kernel_size=3, dilation=3)
        self.layer3 = DoubleConv(in_channels=feature_map[1], out_channels=feature_map[2], kernel_size=3, dilation=1)
        self.layer4 = DoubleConv(in_channels=feature_map[2], out_channels=feature_map[3], kernel_size=3, dilation=1)
        self.pool1 = nn.AvgPool2d(kernel_size=[2,2])
        self.pool2 = nn.AvgPool2d(kernel_size=[2,2])
        self.pool3 = nn.MaxPool2d(kernel_size=[2,2])
        self.pool4 = nn.MaxPool2d(kernel_size=[2,2])

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        y = self.pool4(self.layer4(x))
        return y

class NecksNet(nn.Module):
    def __init__(self, feature_map):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels=2*feature_map[3], out_channels=feature_map[3], kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Linear(in_features=feature_map[3], out_features=feature_map[2])

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.pool(x)
        x = self.flatten(x)
        y = self.dense_layer(x)
        return y

class xr20(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.gpu_device = device
        
    def forward(self, rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, velo_in):#, gt_ss):
        pass